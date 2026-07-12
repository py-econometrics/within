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
    /// Approximate sparse Cholesky on the reduced Schur CSR.
    Approx(Factor),
    /// Exact dense Cholesky on a principal minor of the Schur matrix.
    Dense(DenseCholesky),
}

impl ReducedFactor {
    pub(crate) fn try_dense(minor: Vec<f64>, n: usize) -> Option<Self> {
        DenseCholesky::try_factor(minor, n).map(Self::Dense)
    }

    pub(crate) fn input_dimension(&self) -> usize {
        match self {
            Self::Approx(f) => f.original_n(),
            Self::Dense(f) => f.n(),
        }
    }

    pub(crate) fn factor_dimension(&self) -> usize {
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
}
