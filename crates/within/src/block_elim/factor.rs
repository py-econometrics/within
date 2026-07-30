//! Reduced-system factor for Schur-complement local solves.
//!
//! [`ReducedFactor`] wraps an `approx-chol` factor, either directly or behind a
//! Gremban cover. [`factor_sparse`] bridges to the `approx-chol` builder.

use approx_chol::low_level::Builder;
use approx_chol::{CsrRef, Factor};
use schwarz_precond::LocalSolveError;

use super::csr_matrix::CsrMatrix;
use crate::domain::Grounding;
use crate::BuildError;

/// Reduced-system factor for Schur-complement local solves.
#[derive(Clone, serde::Serialize)]
pub(crate) enum ReducedFactor {
    // Postcard encodes discriminants by declaration order and the fixture pins Direct = 0.
    /// Factor of the reduced Schur CSR itself, carrying the gauge applied around the solve.
    Direct {
        /// Factor of the reduced Schur.
        factor: Factor,
        /// Gauge of the reduced system.
        grounding: Grounding,
    },
    /// Gremban cover of a signed reduced Schur, kept here so the operator stays single-sized.
    Cover {
        /// Factor of the doubled cover; the tail past `2*m` is the grounding augmentation.
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

    /// Explicit ground vertex appended past the kept block; `None` for the kept block alone.
    pub(crate) fn explicit_ground_index(&self, n_kept: usize) -> Option<usize> {
        (self.grounding() == Some(Grounding::Grounded) && self.input_dimension() == n_kept + 1)
            .then_some(n_kept)
    }

    /// Whether the factor was built over a kept block of this size, with or without ground.
    pub(crate) fn spans_kept_block(&self, n_kept: usize) -> bool {
        self.input_dimension() == n_kept || self.explicit_ground_index(n_kept).is_some()
    }

    /// Dimension handed to the backend, which adds grounding vertices of its own.
    pub(crate) fn input_dimension(&self) -> usize {
        match self {
            Self::Direct { factor, .. } => factor.original_n(),
            // The cover is hidden behind the single signed interface.
            Self::Cover { m, .. } => *m,
        }
    }

    pub(crate) fn solve_dimension(&self) -> usize {
        match self {
            Self::Direct { factor, .. } => factor.n(),
            Self::Cover { m, .. } => *m,
        }
    }

    /// Extra scratch beyond the reduced buffers; only [`Self::Cover`] embeds into a larger system.
    pub(crate) fn scratch_len(&self) -> usize {
        match self {
            Self::Direct { .. } => 0,
            Self::Cover { inner, .. } => inner.n(),
        }
    }

    /// `x` spans [`Self::solve_dimension`]; `scratch` is at least [`Self::scratch_len`] long.
    pub(crate) fn solve_in_place(
        &self,
        x: &mut [f64],
        scratch: &mut [f64],
    ) -> Result<(), LocalSolveError> {
        match self {
            Self::Direct { factor, .. } => {
                debug_assert_eq!(factor.n(), x.len());
                solve_approx(factor, x)
            }
            Self::Cover { inner, m } => {
                debug_assert_eq!(*m, x.len());
                let cover_n = inner.n();
                let buf = &mut scratch[..cover_n];
                // Grounding vertices past the `2m` cover nodes have zero RHS, so clear the tail.
                buf[..*m].copy_from_slice(x);
                for (out, &v) in buf[*m..2 * *m].iter_mut().zip(x.iter()) {
                    *out = -v;
                }
                for slot in buf[2 * *m..].iter_mut() {
                    *slot = 0.0;
                }
                solve_approx(inner, buf)?;
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
                // Outside `[2m, 2m + 2]` the antisymmetric embed over- or under-runs.
                let two_m = m
                    .checked_mul(2)
                    .ok_or_else(|| D::Error::custom("Cover dimension m too large"))?;
                let cover_dim = inner.n();
                if cover_dim < two_m || cover_dim > two_m.saturating_add(2) {
                    return Err(D::Error::custom(
                        "Cover inner factor dimension inconsistent with m",
                    ));
                }
                Ok(ReducedFactor::Cover { inner, m })
            }
        }
    }
}

fn solve_approx(f: &Factor, x: &mut [f64]) -> Result<(), LocalSolveError> {
    f.solve_in_place(x)
        .map_err(|e| LocalSolveError::BackendFailed {
            context: "within.local.block_elim.reduced_approx",
            message: e.to_string(),
        })
}

/// Returns the `approx-chol` error unmapped, so the caller can spot an unusable exact pivot.
pub(crate) fn factor_sparse(
    matrix: &CsrMatrix,
    config: approx_chol::Config,
) -> Result<Factor, approx_chol::Error> {
    let csr = CsrRef::new(
        matrix.indptr(),
        matrix.indices(),
        matrix.data(),
        u32::try_from(matrix.n()).expect("Schur complement dimension exceeds u32::MAX"),
    )?;
    Builder::new(config).build(csr)
}

pub(crate) fn local_solver_build(e: approx_chol::Error) -> BuildError {
    BuildError::LocalSolverBuild(format!("failed Schur complement factorization: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{ApproxCholConfig, DEFAULT_DENSE_SCHUR_THRESHOLD};
    use approx_chol::ExactFailure;

    #[test]
    fn cover_reduced_factor_solves_signed_system() {
        // Pins the `ReducedFactor::Cover` embed/read-back, not approx-chol's accuracy.
        let cover = CsrMatrix::new(
            vec![0, 2, 4, 6, 8],
            vec![0, 3, 1, 2, 1, 2, 0, 3],
            vec![2.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, 2.0],
            4,
        );
        let config = ApproxCholConfig {
            seed: 0,
            split_merge: Some(2),
        }
        .to_approx_chol(DEFAULT_DENSE_SCHUR_THRESHOLD, ExactFailure::Error);
        let inner = factor_sparse(&cover, config).expect("factor cover");
        let reduced = ReducedFactor::Cover { inner, m: 2 };
        assert_eq!(reduced.input_dimension(), 2);
        assert_eq!(reduced.solve_dimension(), 2);

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

    /// A 2×2 SDDM system; approx-chol grounds its surplus, so the factor is
    /// dimension 3 over an input of 2.
    fn approx_2x2() -> Factor {
        let m = CsrMatrix::new(
            vec![0, 2, 4],
            vec![0, 1, 0, 1],
            vec![2.0, -1.0, -1.0, 2.0],
            2,
        );
        let config = ApproxCholConfig::default()
            .to_approx_chol(DEFAULT_DENSE_SCHUR_THRESHOLD, ExactFailure::Error);
        factor_sparse(&m, config).expect("factor 2x2")
    }

    #[test]
    fn valid_direct_round_trips() {
        let bytes = postcard::to_stdvec(&ReducedFactor::Direct {
            factor: approx_2x2(),
            grounding: Grounding::Floating,
        })
        .expect("serialize");
        let restored: ReducedFactor = postcard::from_bytes(&bytes).expect("deserialize");
        assert_eq!(restored.input_dimension(), 2);
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
