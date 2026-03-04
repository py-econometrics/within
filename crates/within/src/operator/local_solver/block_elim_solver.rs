use approx_chol::Factor;
use faer::{MatRef, Side};
use schwarz_precond::{LocalSolveError, LocalSolver, SparseMatrix};

use crate::operator::gramian::CrossTab;

use super::transforms::{backsub_block, negate_block, subtract_mean};

/// Reduced-system factor backend for Schur-complement local solves.
pub(crate) enum ReducedFactor {
    /// Approximate sparse Cholesky on the reduced Schur CSR.
    Approx(Factor),
    /// Exact dense Cholesky on an anchored principal minor of the Schur matrix.
    Dense(AnchoredDenseCholesky),
}

impl ReducedFactor {
    pub(crate) fn approx(factor: Factor) -> Self {
        Self::Approx(factor)
    }

    /// Try dense anchored Cholesky factorization for a Laplacian-like Schur matrix.
    ///
    /// Uses the top-left `(n-1) x (n-1)` principal minor; the dropped coordinate
    /// is fixed to zero during solves and later re-centered with the full local
    /// mean subtraction already present in `BlockElimSolver`.
    pub(crate) fn try_dense_laplacian(matrix: &SparseMatrix) -> Option<Self> {
        AnchoredDenseCholesky::try_from_sparse_laplacian(matrix).map(Self::Dense)
    }

    pub(crate) fn try_dense_laplacian_minor(anchored_minor: Vec<f64>, n: usize) -> Option<Self> {
        AnchoredDenseCholesky::try_from_dense_anchored_minor(anchored_minor, n).map(Self::Dense)
    }

    fn n(&self) -> usize {
        match self {
            Self::Approx(f) => f.n(),
            Self::Dense(f) => f.n(),
        }
    }

    fn solve_in_place(&self, x: &mut [f64]) -> Result<(), LocalSolveError> {
        match self {
            Self::Approx(f) => {
                debug_assert_eq!(f.n(), x.len());
                f.solve_in_place(x)
                    .map_err(|e| LocalSolveError::ApproxCholSolveFailed {
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

/// Dense Cholesky on an anchored principal minor of a Laplacian-like matrix.
pub(crate) struct AnchoredDenseCholesky {
    /// Lower-triangular factor of the `(n-1) x (n-1)` anchored minor.
    l_row_major: Vec<f64>,
    /// Full Schur dimension before anchoring.
    n: usize,
}

impl AnchoredDenseCholesky {
    fn try_from_sparse_laplacian(matrix: &SparseMatrix) -> Option<Self> {
        let n = matrix.n();
        if n <= 1 {
            return Some(Self {
                l_row_major: Vec::new(),
                n,
            });
        }

        let m = n - 1;
        let mut dense_minor = vec![0.0; m * m];
        for row in 0..m {
            let start = matrix.indptr()[row] as usize;
            let end = matrix.indptr()[row + 1] as usize;
            for idx in start..end {
                let col = matrix.indices()[idx] as usize;
                if col < m {
                    dense_minor[row * m + col] = matrix.data()[idx];
                }
            }
        }
        Self::factor_dense_minor(dense_minor, n)
    }

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

/// Local subdomain solver using block elimination on the bipartite SDDM.
pub struct BlockElimSolver {
    /// Bipartite Gramian structure: C, C^T, diag_q, diag_r.
    cross_tab: CrossTab,
    /// `1 / D_elim[k]` for the eliminated (larger) diagonal block.
    inv_diag_elim: Vec<f64>,
    /// Reduced-system factor backend.
    reduced_factor: ReducedFactor,
    /// True if the q-block was eliminated (n_q >= n_r).
    eliminate_q: bool,
    /// Total DOF count (`n_q + n_r`).
    n_local: usize,
    /// Factor dimension for the reduced solve (may be `n_keep + 1` for sparse AC).
    n_reduced: usize,
}

impl BlockElimSolver {
    pub(crate) fn new(
        cross_tab: CrossTab,
        inv_diag_elim: Vec<f64>,
        reduced_factor: ReducedFactor,
        eliminate_q: bool,
    ) -> Self {
        let n_local = cross_tab.n_local();
        let n_reduced = reduced_factor.n();
        Self {
            cross_tab,
            inv_diag_elim,
            reduced_factor,
            eliminate_q,
            n_local,
            n_reduced,
        }
    }

    #[cfg(test)]
    pub(crate) fn uses_dense_reduced_factor(&self) -> bool {
        matches!(self.reduced_factor, ReducedFactor::Dense(_))
    }
}

impl LocalSolver for BlockElimSolver {
    fn n_local(&self) -> usize {
        self.n_local
    }

    fn scratch_size(&self) -> usize {
        self.n_local + self.n_reduced
    }

    fn solve_local(&self, rhs: &mut [f64], sol: &mut [f64]) -> Result<(), LocalSolveError> {
        let n = self.n_local;
        let n_q = self.cross_tab.n_q();
        let n_r = self.cross_tab.n_r();
        let ct = &self.cross_tab;

        negate_block(&mut rhs[..n], n_q);
        subtract_mean(rhs, n);

        if self.eliminate_q {
            let n_keep = n_r;

            {
                let (main, scratch) = rhs.split_at_mut(n);
                scratch[..n_keep].copy_from_slice(&main[n_q..]);
                ct.ct
                    .spmv_diag_add(&self.inv_diag_elim, &main[..n_q], &mut scratch[..n_keep]);
            }
            if self.n_reduced > n_keep {
                rhs[n + n_keep] = 0.0;
            }
            subtract_mean(&mut rhs[n..], self.n_reduced);

            sol[n_q..n_q + self.n_reduced].copy_from_slice(&rhs[n..n + self.n_reduced]);
            self.reduced_factor
                .solve_in_place(&mut sol[n_q..n_q + self.n_reduced])?;

            {
                let (sol_q, sol_r) = sol.split_at_mut(n_q);
                backsub_block(sol_q, &rhs[..n_q], &ct.c, &self.inv_diag_elim, sol_r);
            }
        } else {
            let n_keep = n_q;

            {
                let (main, scratch) = rhs.split_at_mut(n);
                scratch[..n_keep].copy_from_slice(&main[..n_q]);
                ct.c.spmv_diag_add(&self.inv_diag_elim, &main[n_q..], &mut scratch[..n_keep]);
            }
            if self.n_reduced > n_keep {
                rhs[n + n_keep] = 0.0;
            }
            subtract_mean(&mut rhs[n..], self.n_reduced);

            sol[..self.n_reduced].copy_from_slice(&rhs[n..n + self.n_reduced]);
            self.reduced_factor
                .solve_in_place(&mut sol[..self.n_reduced])?;

            {
                let (sol_q, sol_r) = sol.split_at_mut(n_q);
                backsub_block(
                    &mut sol_r[..n_r],
                    &rhs[n_q..n_q + n_r],
                    &ct.ct,
                    &self.inv_diag_elim,
                    sol_q,
                );
            }
        }

        subtract_mean(sol, n);
        negate_block(&mut sol[..n], n_q);
        Ok(())
    }
}
