//! Reduced-system factor backends for Schur-complement local solves.
//!
//! [`ReducedFactor`] wraps either approximate sparse Cholesky (from `approx-chol`)
//! or dense Cholesky on an anchored principal minor ([`AnchoredDenseCholesky`]).
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
    /// Approximate sparse Cholesky on the reduced Schur CSR.
    Approx(Factor),
    /// Exact dense Cholesky on an anchored principal minor of the Schur matrix.
    Dense(AnchoredDenseCholesky),
}

impl ReducedFactor {
    pub(crate) fn try_dense_laplacian_minor(anchored_minor: Vec<f64>, n: usize) -> Option<Self> {
        AnchoredDenseCholesky::try_from_dense_anchored_minor(anchored_minor, n).map(Self::Dense)
    }

    pub(crate) fn n(&self) -> usize {
        match self {
            Self::Approx(f) => f.n(),
            Self::Dense(f) => f.n(),
        }
    }

    pub(crate) fn solve_in_place(&self, x: &mut [f64]) -> Result<(), LocalSolveError> {
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
        }
    }
}

// ===========================================================================
// AnchoredDenseCholesky — dense Cholesky on anchored principal minor
// ===========================================================================

/// Dense Cholesky on an anchored principal minor of a Laplacian-like matrix.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct AnchoredDenseCholesky {
    /// Lower-triangular factor of the `(n-1) x (n-1)` anchored minor.
    l_row_major: Vec<f64>,
    /// Full Schur dimension before anchoring.
    n: usize,
}

impl AnchoredDenseCholesky {
    fn try_from_dense_anchored_minor(dense_minor: Vec<f64>, n: usize) -> Option<Self> {
        let m = n.saturating_sub(1);
        if m == 0 {
            return Some(Self {
                l_row_major: Vec::new(),
                n,
            });
        }
        if dense_minor.len() != m * m {
            return None;
        }
        Self::factor_dense_minor(dense_minor, n)
    }

    fn factor_dense_minor(dense_minor: Vec<f64>, n: usize) -> Option<Self> {
        let m = n.saturating_sub(1);
        let mat_ref = MatRef::from_row_major_slice(&dense_minor, m, m);
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

    /// Solve `L L^T x = b` on the anchored minor in-place.
    ///
    /// Expects `x.len() == n`; writes the anchored coordinate `x[n-1] = 0`.
    fn solve_in_place(&self, x: &mut [f64]) {
        debug_assert_eq!(x.len(), self.n);
        if self.n == 0 {
            return;
        }
        if self.n == 1 {
            x[0] = 0.0;
            return;
        }

        let m = self.n - 1;
        let l = &self.l_row_major;
        debug_assert_eq!(l.len(), m * m);

        // Forward solve on anchored block: L y = b.
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
            unsafe { *x.get_unchecked_mut(i) = s / lii };
        }

        // Backward solve on anchored block: L^T x = y.
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
            unsafe { *x.get_unchecked_mut(i) = s / lii };
        }

        x[m] = 0.0;
    }
}

// ===========================================================================
// factor_sparse — bridge into approx_chol for sparse reduced Schur
// ===========================================================================

/// Factor a sparse Schur complement matrix via `approx_chol`.
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
    fn test_anchored_dense_cholesky_solve_n0() {
        let chol = AnchoredDenseCholesky {
            l_row_major: Vec::new(),
            n: 0,
        };
        let mut x: Vec<f64> = Vec::new();
        chol.solve_in_place(&mut x); // should be no-op
    }

    #[test]
    fn test_anchored_dense_cholesky_solve_n1() {
        let chol = AnchoredDenseCholesky {
            l_row_major: Vec::new(),
            n: 1,
        };
        let mut x = vec![42.0];
        chol.solve_in_place(&mut x);
        assert_eq!(x[0], 0.0); // n==1 -> x[0] = 0
    }

    #[test]
    fn test_try_from_dense_anchored_minor_wrong_length() {
        // n=3 -> m=2, expects 4 elements, give 3
        let result = AnchoredDenseCholesky::try_from_dense_anchored_minor(vec![1.0, 2.0, 3.0], 3);
        assert!(result.is_none());
    }

    #[test]
    fn test_try_from_dense_anchored_minor_n1() {
        // n=1 -> m=0, should return Some with empty
        let result = AnchoredDenseCholesky::try_from_dense_anchored_minor(Vec::new(), 1);
        assert!(result.is_some());
        assert_eq!(result.unwrap().n(), 1);
    }

    #[test]
    fn test_factor_dense_minor_singular() {
        // Singular 2x2 matrix (both rows identical)
        let result = AnchoredDenseCholesky::factor_dense_minor(vec![1.0, 1.0, 1.0, 1.0], 3);
        assert!(result.is_none());
    }
}
