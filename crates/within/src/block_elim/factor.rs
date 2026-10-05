//! Reduced-system factor for Schur-complement local solves.
//!
//! [`ReducedFactor`] wraps an `approx-chol` factor, either directly or behind a
//! Gremban cover. [`factor_sddm`] bridges to the `approx-chol` builder.

use approx_chol::low_level::Builder;
use approx_chol::{Factor, Sddm};
use schwarz_precond::LocalSolveError;

use crate::domain::Grounding;
use crate::BuildError;

/// Reduced-system factor for Schur-complement local solves.
#[derive(Clone, serde::Serialize)]
pub(crate) enum ReducedFactor {
    // Postcard encodes discriminants by declaration order and the fixture pins Direct = 0.
    /// Factor of the reduced Schur itself, carrying the gauge applied around the solve.
    Direct {
        /// Factor of the reduced Schur.
        factor: Factor,
        /// Gauge of the reduced system.
        grounding: Grounding,
    },
    /// Gremban cover of a signed reduced Schur, kept here so the operator stays single-sized.
    Cover {
        /// Factor of the doubled cover.
        inner: Factor,
        /// Single signed reduced dimension exposed to the caller.
        m: usize,
    },
}

impl ReducedFactor {
    /// The gauge the operator-level solve applies; `None` for a self-grounding cover.
    pub(crate) fn grounding(&self) -> Option<Grounding> {
        match self {
            Self::Direct { grounding, .. } => Some(*grounding),
            Self::Cover { .. } => None,
        }
    }

    /// The kept block's size, which is all a caller ever hands in or reads back.
    pub(crate) fn n(&self) -> usize {
        match self {
            Self::Direct { factor, .. } => factor.n(),
            // The cover is hidden behind the single signed interface.
            Self::Cover { m, .. } => *m,
        }
    }

    pub(crate) fn scratch_len(&self) -> usize {
        match self {
            Self::Direct { factor, .. } => factor.scratch_len(),
            Self::Cover { inner, .. } => inner.n() + inner.scratch_len(),
        }
    }

    /// `x` spans [`Self::n`]; `scratch` is at least [`Self::scratch_len`] long.
    pub(crate) fn solve_in_place(
        &self,
        x: &mut [f64],
        scratch: &mut [f64],
    ) -> Result<(), LocalSolveError> {
        match self {
            Self::Direct { factor, .. } => solve_approx(factor, x, scratch),
            Self::Cover { inner, m } => {
                let (buf, inner_scratch) = scratch.split_at_mut(inner.n());
                buf[..*m].copy_from_slice(x);
                for (out, &v) in buf[*m..].iter_mut().zip(x.iter()) {
                    *out = -v;
                }
                solve_approx(inner, buf, inner_scratch)?;
                // Read back the antisymmetric solution: x = (x⁺ - x⁻) / 2.
                for (i, out) in x.iter_mut().enumerate() {
                    *out = 0.5 * (buf[i] - buf[*m + i]);
                }
                Ok(())
            }
        }
    }
}

/// Wire mirror of [`ReducedFactor`]; a bare inner [`Factor`] makes `Cover`-of-`Cover` undecodable.
#[derive(serde::Deserialize)]
enum ReducedFactorWire {
    Direct {
        factor: Factor,
        grounding: Grounding,
    },
    Cover {
        inner: Factor,
        m: usize,
    },
}

impl<'de> serde::Deserialize<'de> for ReducedFactor {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::de::Error;
        match ReducedFactorWire::deserialize(deserializer)? {
            ReducedFactorWire::Direct { factor, grounding } => {
                Ok(ReducedFactor::Direct { factor, grounding })
            }
            ReducedFactorWire::Cover { inner, m } => {
                if m.checked_mul(2) != Some(inner.n()) {
                    return Err(D::Error::custom(
                        "Cover inner factor dimension inconsistent with m",
                    ));
                }
                Ok(ReducedFactor::Cover { inner, m })
            }
        }
    }
}

fn solve_approx(f: &Factor, x: &mut [f64], scratch: &mut [f64]) -> Result<(), LocalSolveError> {
    f.solve_in_place(x, scratch)
        .map_err(|e| LocalSolveError::BackendFailed {
            context: "within.local.block_elim.reduced_approx",
            message: e.to_string(),
        })
}

/// Returns the `approx-chol` error unmapped, so the caller can spot an unusable exact pivot.
pub(crate) fn factor_sddm(
    sddm: Sddm,
    config: approx_chol::Config,
) -> Result<Factor, approx_chol::Error> {
    Builder::new(config).build(sddm)
}

pub(crate) fn local_solver_build(e: approx_chol::Error) -> BuildError {
    BuildError::LocalSolverBuild(format!("failed Schur complement factorization: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{ApproxCholConfig, DEFAULT_DENSE_SCHUR_THRESHOLD};
    use approx_chol::{ExactFailure, Grounded, Laplacian};

    #[test]
    fn cover_reduced_factor_solves_signed_system() {
        // Pins the `ReducedFactor::Cover` embed/read-back, not approx-chol's accuracy.
        let edges = Laplacian::new(vec![0, 1, 2, 2, 2], vec![3, 2], vec![1.0, 1.0]).unwrap();
        let cover = Grounded::new(edges, vec![1.0; 4]).unwrap().into();
        let config = ApproxCholConfig {
            seed: 0,
            split_merge: Some(2),
        }
        .to_approx_chol(DEFAULT_DENSE_SCHUR_THRESHOLD, ExactFailure::Error);
        let inner = factor_sddm(cover, config).expect("factor cover");
        let reduced = ReducedFactor::Cover { inner, m: 2 };
        assert_eq!(reduced.n(), 2);

        let b = [1.0, 0.5];
        let mut x = b;
        let mut scratch = vec![0.0; reduced.scratch_len()];
        reduced
            .solve_in_place(&mut x, &mut scratch)
            .expect("cover solve");

        // Residual of the signed system M x = b (M = [[2,1],[1,2]]).
        let r0 = 2.0 * x[0] + x[1] - b[0];
        let r1 = x[0] + 2.0 * x[1] - b[1];
        assert!(r0.hypot(r1) < 1e-9, "residual too large: ({r0}, {r1})");
    }

    /// `[[2, -1], [-1, 2]]`: one edge, unit surplus on both vertices.
    fn approx_2x2() -> Factor {
        let edges = Laplacian::new(vec![0, 1, 1], vec![1], vec![1.0]).unwrap();
        let config = ApproxCholConfig::default()
            .to_approx_chol(DEFAULT_DENSE_SCHUR_THRESHOLD, ExactFailure::Error);
        factor_sddm(Grounded::new(edges, vec![1.0, 1.0]).unwrap().into(), config)
            .expect("factor 2x2")
    }

    #[test]
    fn valid_direct_round_trips() {
        let bytes = postcard::to_stdvec(&ReducedFactor::Direct {
            factor: approx_2x2(),
            grounding: Grounding::Floating,
        })
        .expect("serialize");
        let restored: ReducedFactor = postcard::from_bytes(&bytes).expect("deserialize");
        assert_eq!(restored.n(), 2);
    }

    #[test]
    fn cover_with_undersized_inner_is_rejected() {
        // The inner factor cannot hold the antisymmetric embed for m = 5, which needs 10 nodes.
        let bad = ReducedFactor::Cover {
            inner: approx_2x2(),
            m: 5,
        };
        let bytes = postcard::to_stdvec(&bad).expect("serialize");
        assert!(postcard::from_bytes::<ReducedFactor>(&bytes).is_err());
    }
}
