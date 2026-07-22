//! Reduced-system factor backends for Schur-complement local solves.
//!
//! [`ReducedFactor`] wraps either approximate sparse Cholesky (from `approx-chol`)
//! or dense Cholesky on a principal minor ([`DenseCholesky`]).
//! [`factor_sparse`] bridges to the `approx-chol` builder, surfacing build errors
//! as [`BuildError::LocalSolverBuild`].

use approx_chol::low_level::Builder;
use approx_chol::{CsrRef, Factor};
use faer::{MatRef, Side};
use schwarz_precond::LocalSolveError;

use super::csr_matrix::CsrMatrix;
use crate::config::ApproxCholConfig;
use crate::BuildError;

// ===========================================================================
// ReducedFactor — reduced-system factor backend for Schur-complement solves
// ===========================================================================

/// Reduced-system factor backend for Schur-complement local solves.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
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
        let l = &self.l_row_major;
        let m = if l.len() == self.n * self.n {
            self.n
        } else {
            self.n.saturating_sub(1)
        };
        debug_assert_eq!(l.len(), m * m);

        // Forward solve on the minor: L y = b.
        for i in 0..m {
            // SAFETY: i<m, row bounds and triangular-access bounds are validated by loop limits.
            let mut s = unsafe { *x.get_unchecked(i) };
            for j in 0..i {
                // SAFETY: i<m, j<i<m -> indices are in bounds.
                let lij = unsafe { *l.get_unchecked(i * m + j) };
                // SAFETY: j<i<m -> in bounds.
                let xj = unsafe { *x.get_unchecked(j) };
                s -= lij * xj;
            }
            // SAFETY: i<m -> diagonal index and write index are in bounds.
            let lii = unsafe { *l.get_unchecked(i * m + i) };
            debug_assert!(
                lii.is_finite() && lii != 0.0,
                "Cholesky diagonal must be a finite nonzero pivot"
            );
            unsafe { *x.get_unchecked_mut(i) = s / lii };
        }

        // Backward solve on the minor: L^T x = y.
        for i in (0..m).rev() {
            // SAFETY: i<m -> in bounds.
            let mut s = unsafe { *x.get_unchecked(i) };
            for j in (i + 1)..m {
                // SAFETY: j<m, i<j -> in bounds.
                let lji = unsafe { *l.get_unchecked(j * m + i) };
                // SAFETY: j<m -> in bounds.
                let xj = unsafe { *x.get_unchecked(j) };
                s -= lji * xj;
            }
            // SAFETY: i<m -> diagonal index and write index are in bounds.
            let lii = unsafe { *l.get_unchecked(i * m + i) };
            debug_assert!(
                lii.is_finite() && lii != 0.0,
                "Cholesky diagonal must be a finite nonzero pivot"
            );
            unsafe { *x.get_unchecked_mut(i) = s / lii };
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
}
