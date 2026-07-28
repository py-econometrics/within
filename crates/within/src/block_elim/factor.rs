//! Reduced-system factor backends for Schur-complement local solves.
//!
//! [`ReducedFactor`] wraps either approximate sparse Cholesky (from `approx-chol`)
//! or dense Cholesky on a principal minor ([`DenseCholesky`]).
//! [`factor_sparse`] bridges to the `approx-chol` builder, surfacing build errors
//! as [`BuildError::LocalSolverBuild`].

use approx_chol::low_level::Builder;
use approx_chol::{CsrRef, Factor};
use faer::linalg::triangular_solve::{
    solve_lower_triangular_in_place, solve_upper_triangular_in_place,
};
use faer::{MatMut, MatRef, Par, Side};
use schwarz_precond::LocalSolveError;

use super::csr_matrix::CsrMatrix;
use crate::config::ApproxCholConfig;
use crate::BuildError;

// ===========================================================================
// ReducedFactor — reduced-system factor backend for Schur-complement solves
// ===========================================================================

/// Reduced-system factor backend for Schur-complement local solves.
#[derive(Clone, serde::Serialize)]
pub(crate) enum ReducedFactor {
    // Postcard encodes enum discriminants by declaration order; the wire
    // fixture pins Approx = 0 and Dense = 1, so new variants append after them.
    /// Approximate sparse Cholesky on the reduced Schur CSR.
    Approx(Factor),
    /// Exact dense Cholesky on a principal minor of the Schur matrix.
    Dense(DenseCholesky),
    /// Gremban double cover of a signed (frustrated) reduced Schur: `inner`
    /// factors the doubled cover system (any backend — sampled approx-chol for
    /// large covers, exact dense for small ones), and `m` is the single signed
    /// dimension the solve presents to the caller. The cover lives only here —
    /// the local operator stays single-sized.
    Cover {
        /// Factor of the doubled cover (dimension
        /// `inner.factor_dimension() >= 2*m`; the tail past `2*m` is the
        /// cover's grounding augmentation).
        inner: Box<ReducedFactor>,
        /// Single signed reduced dimension exposed to the caller.
        m: usize,
    },
}

impl ReducedFactor {
    pub(crate) fn try_dense(minor: Vec<f64>, n: usize) -> Option<Self> {
        DenseCholesky::try_factor(minor, n).map(Self::Dense)
    }

    pub(crate) fn input_dimension(&self) -> usize {
        match self {
            Self::Approx(f) => f.original_n(),
            Self::Dense(f) => f.n(),
            // The cover is hidden behind the single signed interface.
            Self::Cover { m, .. } => *m,
        }
    }

    pub(crate) fn factor_dimension(&self) -> usize {
        match self {
            Self::Approx(f) => f.n(),
            Self::Dense(f) => f.n(),
            Self::Cover { m, .. } => *m,
        }
    }

    /// Extra scratch beyond the reduced buffers that [`Self::solve_in_place`]
    /// needs. Only [`Self::Cover`] embeds into a larger system; the direct
    /// arms solve in place and need none.
    pub(crate) fn scratch_len(&self) -> usize {
        match self {
            Self::Approx(_) | Self::Dense(_) => 0,
            Self::Cover { inner, .. } => inner.factor_dimension() + inner.scratch_len(),
        }
    }

    /// Solve the reduced system in place. `x` has length [`Self::factor_dimension`];
    /// `scratch` is at least [`Self::scratch_len`] long (the [`Self::Cover`]
    /// embed buffer, reused across LSMR iterations).
    pub(crate) fn solve_in_place(
        &self,
        x: &mut [f64],
        scratch: &mut [f64],
    ) -> Result<(), LocalSolveError> {
        match self {
            Self::Approx(f) => {
                debug_assert_eq!(f.n(), x.len());
                f.solve_in_place(x)
                    .map_err(|e| LocalSolveError::BackendFailed {
                        context: "within.local.block_elim.reduced_approx",
                        message: e.to_string(),
                    })?;
                Ok(())
            }
            Self::Dense(f) => {
                f.solve_in_place(x);
                Ok(())
            }
            Self::Cover { inner, m } => {
                debug_assert_eq!(*m, x.len());
                let cover_n = inner.factor_dimension();
                let (buf, rest) = scratch.split_at_mut(cover_n);
                // Embed the antisymmetric RHS [b, -b] into the cover. Any
                // grounding vertex past the 2m cover nodes has zero RHS (it
                // cancels in the antisymmetric sheet difference), so clear the
                // reused scratch tail rather than assume it is zeroed.
                buf[..*m].copy_from_slice(x);
                for (out, &v) in buf[*m..2 * *m].iter_mut().zip(x.iter()) {
                    *out = -v;
                }
                for slot in buf[2 * *m..].iter_mut() {
                    *slot = 0.0;
                }
                inner.solve_in_place(buf, rest)?;
                // Read back the antisymmetric solution: x = (x⁺ - x⁻) / 2.
                for (i, out) in x.iter_mut().enumerate() {
                    *out = 0.5 * (buf[i] - buf[*m + i]);
                }
                Ok(())
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Deserialize — validated reconstruction from untrusted bytes
// ---------------------------------------------------------------------------

/// Wire mirror of [`ReducedFactor`]. `Cover`'s `inner` is a non-recursive
/// [`LeafFactor`], so a `Cover`-of-`Cover` — which no real build produces —
/// fails to decode with an unknown-variant error instead of recursing without
/// bound (a serialized chain of `Cover` discriminants would otherwise overflow
/// the stack, both while decoding and in [`ReducedFactor::scratch_len`]).
#[derive(serde::Deserialize)]
enum ReducedFactorWire {
    Approx(Factor),
    Dense(DenseCholesky),
    Cover { inner: Box<LeafFactor>, m: usize },
}

/// A `Cover`'s inner factor: only ever a direct backend, never another cover.
#[derive(serde::Deserialize)]
enum LeafFactor {
    Approx(Factor),
    Dense(DenseCholesky),
}

/// Reject an `Approx` factor whose dimensions cannot arise from `approx-chol`,
/// whose contract augments the input by at most one Gremban vertex, so
/// `n ∈ {original_n, original_n + 1}`. Without this, a crafted `Factor::n`
/// overflows the caller's scratch arithmetic.
fn checked_approx(f: Factor) -> Result<ReducedFactor, &'static str> {
    let (n, base) = (f.n(), f.original_n());
    if n == base || Some(n) == base.checked_add(1) {
        Ok(ReducedFactor::Approx(f))
    } else {
        Err("Approx factor dimension inconsistent with its original dimension")
    }
}

/// Reject a [`DenseCholesky`] whose factor length is neither the full `n×n`
/// minor nor the anchored `(n-1)×(n-1)` one — the only two shapes
/// [`DenseCholesky::solve_in_place`] can view without reading out of bounds.
fn checked_dense(dc: DenseCholesky) -> Result<ReducedFactor, &'static str> {
    let n = dc.n;
    let full = n.checked_mul(n);
    let anchored = n.saturating_sub(1).checked_mul(n.saturating_sub(1));
    if Some(dc.l_row_major.len()) == full || Some(dc.l_row_major.len()) == anchored {
        Ok(ReducedFactor::Dense(dc))
    } else {
        Err("DenseCholesky factor length inconsistent with its dimension")
    }
}

impl<'de> serde::Deserialize<'de> for ReducedFactor {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::de::Error;
        match ReducedFactorWire::deserialize(deserializer)? {
            ReducedFactorWire::Approx(f) => checked_approx(f).map_err(D::Error::custom),
            ReducedFactorWire::Dense(dc) => checked_dense(dc).map_err(D::Error::custom),
            ReducedFactorWire::Cover { inner, m } => {
                let inner = match *inner {
                    LeafFactor::Approx(f) => checked_approx(f),
                    LeafFactor::Dense(dc) => checked_dense(dc),
                }
                .map_err(D::Error::custom)?;
                // `solve_in_place` embeds the antisymmetric `[b, -b]` RHS into
                // the inner factor, whose dimension is the `2m` doubled cover
                // nodes plus at most two augmentation vertices (the grounded
                // minor's and approx-chol's Gremban vertex). Anything outside
                // `[2m, 2m + 2]` over- or under-runs that embed.
                let two_m = m
                    .checked_mul(2)
                    .ok_or_else(|| D::Error::custom("Cover dimension m too large"))?;
                let cover_dim = inner.factor_dimension();
                if cover_dim < two_m || cover_dim > two_m.saturating_add(2) {
                    return Err(D::Error::custom(
                        "Cover inner factor dimension inconsistent with m",
                    ));
                }
                Ok(ReducedFactor::Cover {
                    inner: Box::new(inner),
                    m,
                })
            }
        }
    }
}

// ===========================================================================
// DenseCholesky — dense Cholesky on a principal minor
// ===========================================================================

/// Dense Cholesky on a principal `m × m` minor of the reduced Schur system.
///
/// `m == n` factors the full (nonsingular) complement; `m == n − 1` anchors
/// the last node (`x = 0`) — the gauge for a floating Laplacian.
/// `m` is recovered from the factor length, keeping the wire format unchanged.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct DenseCholesky {
    /// Lower-triangular factor of the `m × m` minor, row-major.
    l_row_major: Vec<f64>,
    /// Full Schur dimension.
    n: usize,
}

impl DenseCholesky {
    fn try_factor(minor: Vec<f64>, n: usize) -> Option<Self> {
        let anchored = n.saturating_sub(1);
        let m = match minor.len() {
            len if len == n * n => n,
            len if len == anchored * anchored => anchored,
            _ => return None,
        };
        if m == 0 {
            return Some(Self {
                l_row_major: Vec::new(),
                n,
            });
        }

        let mat_ref = MatRef::from_row_major_slice(&minor, m, m);
        let llt = mat_ref.llt(Side::Lower).ok()?;
        let l = llt.L();

        let mut l_row_major = vec![0.0; m * m];
        for r in 0..m {
            for c in 0..=r {
                l_row_major[r * m + c] = l[(r, c)];
            }
        }

        Some(Self { l_row_major, n })
    }

    fn n(&self) -> usize {
        self.n
    }

    /// Solve `L L^T x = b` on the factored minor in-place.
    ///
    /// Expects `x.len() == n`; coordinates past the minor (`x[m..]`) are the
    /// anchored gauge and are written as zero.
    fn solve_in_place(&self, x: &mut [f64]) {
        debug_assert_eq!(x.len(), self.n);
        let m = if self.l_row_major.len() == self.n * self.n {
            self.n
        } else {
            self.n.saturating_sub(1)
        };
        debug_assert_eq!(self.l_row_major.len(), m * m);

        if m > 0 {
            // Delegate both substitutions to faer's blocked kernels, viewing
            // the stored factor and `x` in place: L y = b then Lᵀ x = y.
            let l = MatRef::from_row_major_slice(&self.l_row_major, m, m);
            solve_lower_triangular_in_place(
                l,
                MatMut::from_column_major_slice_mut(&mut x[..m], m, 1),
                Par::Seq,
            );
            solve_upper_triangular_in_place(
                l.transpose(),
                MatMut::from_column_major_slice_mut(&mut x[..m], m, 1),
                Par::Seq,
            );
        }

        for v in &mut x[m..] {
            *v = 0.0;
        }
    }
}

// ===========================================================================
// factor_sparse — bridge into approx_chol for sparse reduced Schur
// ===========================================================================

/// Factor a sparse (SDDM) reduced Schur complement into a direct `Approx` factor.
pub(crate) fn factor_sparse(
    matrix: &CsrMatrix,
    approx_chol: ApproxCholConfig,
) -> Result<ReducedFactor, BuildError> {
    let schur_builder = Builder::new(approx_chol.to_approx_chol());
    let csr = CsrRef::new(
        matrix.indptr(),
        matrix.indices(),
        matrix.data(),
        u32::try_from(matrix.n()).expect("Schur complement dimension exceeds u32::MAX"),
    )
    .map_err(|e| BuildError::LocalSolverBuild(format!("invalid Schur complement CSR: {e}")))?;
    schur_builder
        .build(csr)
        .map(ReducedFactor::Approx)
        .map_err(|e| {
            BuildError::LocalSolverBuild(format!("failed Schur complement factorization: {e}"))
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dense_cholesky_solve_n0() {
        let chol = DenseCholesky {
            l_row_major: Vec::new(),
            n: 0,
        };
        let mut x: Vec<f64> = Vec::new();
        chol.solve_in_place(&mut x); // should be no-op
    }

    #[test]
    fn test_dense_cholesky_solve_n1_anchored() {
        let chol = DenseCholesky {
            l_row_major: Vec::new(),
            n: 1,
        };
        let mut x = vec![42.0];
        chol.solve_in_place(&mut x);
        assert_eq!(x[0], 0.0); // empty anchored minor -> x[0] = 0
    }

    #[test]
    fn test_try_factor_wrong_length() {
        // n=3 admits minors of 9 (full) or 4 (anchored) elements; give 3.
        let result = DenseCholesky::try_factor(vec![1.0, 2.0, 3.0], 3);
        assert!(result.is_none());
    }

    #[test]
    fn test_try_factor_anchored_n1() {
        // n=1, empty minor -> anchored m=0, Some with empty factor.
        let result = DenseCholesky::try_factor(Vec::new(), 1);
        assert!(result.is_some());
        assert_eq!(result.unwrap().n(), 1);
    }

    #[test]
    fn test_try_factor_singular() {
        // Singular 2x2 anchored minor of n=3 (both rows identical).
        let result = DenseCholesky::try_factor(vec![1.0, 1.0, 1.0, 1.0], 3);
        assert!(result.is_none());
    }

    #[test]
    fn test_full_minor_solves_directly() {
        // Full 2x2 SPD minor (m == n): plain LLT solve, no anchored zero.
        let chol = DenseCholesky::try_factor(vec![4.0, 0.0, 0.0, 9.0], 2).unwrap();
        let mut x = vec![8.0, 18.0];
        chol.solve_in_place(&mut x);
        assert_eq!(x, vec![2.0, 2.0]);
    }

    #[test]
    fn cover_reduced_factor_solves_signed_system() {
        // Gremban double cover of the signed M = [[2, 1], [1, 2]] (a positive,
        // frustrated off-diagonal). Node order [0, 1, 0', 1']; the positive
        // edge (0,1) routes to 0-1' and 0'-1 as SDDM off-diagonals, so the
        // cover is strictly diagonally dominant. Verified by hand that
        // Ŝ·[z, -z] = [Mz, -Mz], hence the antisymmetric embed of b recovers
        // M⁻¹b — this pins the ReducedFactor::Cover embed/read-back, not
        // approx-chol's accuracy (checked via the residual M x ≈ b).
        let cover = CsrMatrix::new(
            vec![0, 2, 4, 6, 8],
            vec![0, 3, 1, 2, 1, 2, 0, 3],
            vec![2.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, 2.0],
            4,
        );
        let inner = factor_sparse(
            &cover,
            ApproxCholConfig {
                seed: 0,
                split_merge: Some(2),
                ..Default::default()
            },
        )
        .expect("factor cover");
        let reduced = ReducedFactor::Cover {
            inner: Box::new(inner),
            m: 2,
        };
        assert_eq!(reduced.input_dimension(), 2);
        assert_eq!(reduced.factor_dimension(), 2);

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

    fn dense_2x2() -> ReducedFactor {
        ReducedFactor::try_dense(vec![4.0, 0.0, 0.0, 9.0], 2).expect("spd minor")
    }

    #[test]
    fn valid_dense_round_trips() {
        let bytes = postcard::to_stdvec(&dense_2x2()).expect("serialize");
        let restored: ReducedFactor = postcard::from_bytes(&bytes).expect("deserialize");
        assert_eq!(restored.factor_dimension(), 2);
    }

    #[test]
    fn nested_cover_is_rejected() {
        // A cover whose inner is itself a cover — never built, and the shape
        // whose `scratch_len` recursion overflowed the stack (#166).
        let nested = ReducedFactor::Cover {
            inner: Box::new(ReducedFactor::Cover {
                inner: Box::new(dense_2x2()),
                m: 1,
            }),
            m: 1,
        };
        let bytes = postcard::to_stdvec(&nested).expect("serialize");
        assert!(postcard::from_bytes::<ReducedFactor>(&bytes).is_err());
    }

    #[test]
    fn dense_with_inconsistent_length_is_rejected() {
        // len 3 is neither the full 4×4 nor the anchored 3×3 minor of n = 4.
        let bad = ReducedFactor::Dense(DenseCholesky {
            l_row_major: vec![1.0, 2.0, 3.0],
            n: 4,
        });
        let bytes = postcard::to_stdvec(&bad).expect("serialize");
        assert!(postcard::from_bytes::<ReducedFactor>(&bytes).is_err());
    }

    #[test]
    fn cover_with_undersized_inner_is_rejected() {
        // The inner factor (dim 2) cannot hold the antisymmetric embed for m = 5
        // (which needs 2m = 10 nodes).
        let bad = ReducedFactor::Cover {
            inner: Box::new(dense_2x2()),
            m: 5,
        };
        let bytes = postcard::to_stdvec(&bad).expect("serialize");
        assert!(postcard::from_bytes::<ReducedFactor>(&bytes).is_err());
    }
}
