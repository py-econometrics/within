//! Approximate Cholesky-backed local solver for Schwarz subdomains.

use std::sync::Arc;

use approx_chol::Factor;
use faer::{MatRef, Side};
use rayon::prelude::*;
use schwarz_precond::{LocalSolveError, LocalSolveOptions, LocalSolver, SparseMatrix};

use crate::operator::csr_block::{CsrBlock, PAR_SPMV_THRESHOLD};
use crate::operator::gramian::CrossTab;

// ===========================================================================
// Transform helpers — sign-flipping, mean subtraction, back-substitution
// ===========================================================================

/// Minimum number of rows to trigger parallel back-substitution.
const PAR_BACKSUB_THRESHOLD: usize = 10_000;
const PAR_BACKSUB_CHUNK: usize = 4096;

/// Negate elements in `slice[from..]`.
#[inline]
fn negate_block(slice: &mut [f64], from: usize) {
    for val in slice[from..].iter_mut() {
        *val = -*val;
    }
}

/// Subtract the mean of `slice[..n]` from those `n` elements.
#[inline]
fn subtract_mean(slice: &mut [f64], n: usize) {
    if n == 0 {
        return;
    }
    let mean: f64 = slice[..n].iter().sum::<f64>() / n as f64;
    for val in slice[..n].iter_mut() {
        *val -= mean;
    }
}

/// Back-substitute for the eliminated block.
fn backsub_block(
    sol_output: &mut [f64],
    rhs_slice: &[f64],
    cross_matrix: &CsrBlock,
    inv_diag: &[f64],
    sol_source: &[f64],
    options: LocalSolveOptions,
) {
    let n = sol_output.len();
    if n > PAR_BACKSUB_THRESHOLD && options.allow_inner_parallelism() {
        sol_output
            .par_chunks_mut(PAR_BACKSUB_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let row_start = chunk_idx * PAR_BACKSUB_CHUNK;
                for (local_i, si) in chunk.iter_mut().enumerate() {
                    let i = row_start + local_i;
                    let start = cross_matrix.indptr[i] as usize;
                    let end = cross_matrix.indptr[i + 1] as usize;
                    let mut sum = 0.0;
                    for idx in start..end {
                        let j = cross_matrix.indices[idx] as usize;
                        sum += cross_matrix.data[idx] * sol_source[j];
                    }
                    *si = inv_diag[i] * (rhs_slice[i] + sum);
                }
            });
    } else {
        for i in 0..n {
            let start = cross_matrix.indptr[i] as usize;
            let end = cross_matrix.indptr[i + 1] as usize;
            let mut sum = 0.0;
            for idx in start..end {
                let j = cross_matrix.indices[idx] as usize;
                sum += cross_matrix.data[idx] * sol_source[j];
            }
            sol_output[i] = inv_diag[i] * (rhs_slice[i] + sum);
        }
    }
}

// ===========================================================================
// ApproxCholSolver — local solver backed by approximate Cholesky
// ===========================================================================

/// Local subdomain solver backed by approximate Cholesky factorization.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct ApproxCholSolver {
    factor: Factor,
    strategy: LocalSolveStrategy,
    n_local: usize,
}

impl ApproxCholSolver {
    pub(crate) fn new(factor: Factor, strategy: LocalSolveStrategy, n_local: usize) -> Self {
        Self {
            factor,
            strategy,
            n_local,
        }
    }
}

impl LocalSolver for ApproxCholSolver {
    fn n_local(&self) -> usize {
        self.n_local
    }

    fn scratch_size(&self) -> usize {
        self.factor.n()
    }

    fn solve_local(
        &self,
        rhs: &mut [f64],
        sol: &mut [f64],
        _options: LocalSolveOptions,
    ) -> Result<(), LocalSolveError> {
        let n_local = self.n_local;
        let (solve_n, fbs) = match &self.strategy {
            LocalSolveStrategy::Laplacian => {
                debug_assert_eq!(self.factor.n(), n_local);
                sol[..n_local].copy_from_slice(&rhs[..n_local]);
                self.factor
                    .solve_in_place(&mut sol[..n_local])
                    .map_err(|e| LocalSolveError::ApproxCholSolveFailed {
                        context: "within.local.approx_chol.laplacian",
                        message: e.to_string(),
                    })?;
                return Ok(());
            }
            LocalSolveStrategy::LaplacianGramian { first_block_size } => {
                (n_local, Some(*first_block_size))
            }
            LocalSolveStrategy::GramianAugmented { first_block_size } => {
                (n_local + 1, Some(*first_block_size))
            }
            LocalSolveStrategy::Sddm => (n_local + 1, None),
        };

        if let Some(fbs) = fbs {
            negate_block(&mut rhs[..n_local], fbs);
        }
        if solve_n > n_local {
            debug_assert_eq!(self.factor.n(), solve_n);
            rhs[n_local] = 0.0;
        }
        subtract_mean(rhs, solve_n);

        sol[..solve_n].copy_from_slice(&rhs[..solve_n]);
        debug_assert_eq!(self.factor.n(), solve_n);
        self.factor
            .solve_in_place(&mut sol[..solve_n])
            .map_err(|e| LocalSolveError::ApproxCholSolveFailed {
                context: "within.local.approx_chol.augmented_or_gramian",
                message: e.to_string(),
            })?;

        if let Some(fbs) = fbs {
            negate_block(&mut sol[..n_local], fbs);
        }
        Ok(())
    }

    fn inner_parallelism_work_estimate(&self) -> usize {
        0
    }
}

// ===========================================================================
// FeLocalSolver — dispatch enum
// ===========================================================================

/// FE-specific local solver, dispatching to either full-SDDM ApproxChol or
/// Schur complement block elimination.
#[derive(serde::Serialize, serde::Deserialize)]
pub enum FeLocalSolver {
    /// Full bipartite SDDM factorized via approximate Cholesky.
    FullSddm { solver: ApproxCholSolver },
    /// Schur complement reduction + reduced-system factor solve
    /// (dense anchored for tiny Schur when enabled, otherwise sparse ApproxChol).
    SchurComplement(Box<BlockElimSolver>),
}

impl LocalSolver for FeLocalSolver {
    fn n_local(&self) -> usize {
        match self {
            Self::FullSddm { solver, .. } => solver.n_local(),
            Self::SchurComplement(s) => s.n_local(),
        }
    }

    fn scratch_size(&self) -> usize {
        match self {
            Self::FullSddm { solver, .. } => solver.scratch_size(),
            Self::SchurComplement(s) => s.scratch_size(),
        }
    }

    fn solve_local(
        &self,
        rhs: &mut [f64],
        sol: &mut [f64],
        options: LocalSolveOptions,
    ) -> Result<(), LocalSolveError> {
        match self {
            Self::FullSddm { solver, .. } => solver.solve_local(rhs, sol, options),
            Self::SchurComplement(s) => s.solve_local(rhs, sol, options),
        }
    }

    fn inner_parallelism_work_estimate(&self) -> usize {
        match self {
            Self::FullSddm { solver, .. } => solver.inner_parallelism_work_estimate(),
            Self::SchurComplement(s) => s.inner_parallelism_work_estimate(),
        }
    }
}

/// Determines how the local Gramian solve is performed for a subdomain.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum LocalSolveStrategy {
    /// The local Gramian is naturally a graph Laplacian (no augmentation needed).
    Laplacian,

    /// The local Gramian is SDDM but not Laplacian, so Gremban augmentation added
    /// an extra node.
    Sddm,

    /// Factor-pair domain where the bipartite Gramian maps to a Laplacian via
    /// sign-flipping the second block. No augmentation was needed.
    LaplacianGramian { first_block_size: usize },

    /// Factor-pair domain where the Gramian needed Gremban augmentation.
    GramianAugmented { first_block_size: usize },
}

impl LocalSolveStrategy {
    /// Map a bipartite hint and augmentation flag to the appropriate local
    /// solve strategy.
    pub fn from_flags(first_block_size: Option<usize>, was_augmented: bool) -> Self {
        match first_block_size {
            Some(fbs) => {
                if was_augmented {
                    Self::GramianAugmented {
                        first_block_size: fbs,
                    }
                } else {
                    Self::LaplacianGramian {
                        first_block_size: fbs,
                    }
                }
            }
            None => {
                if was_augmented {
                    Self::Sddm
                } else {
                    Self::Laplacian
                }
            }
        }
    }
}

// ===========================================================================
// ReducedFactor — reduced-system factor backend for Schur-complement solves
// ===========================================================================

/// Reduced-system factor backend for Schur-complement local solves.
#[derive(serde::Serialize, serde::Deserialize)]
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

// ===========================================================================
// AnchoredDenseCholesky — dense Cholesky on anchored principal minor
// ===========================================================================

/// Dense Cholesky on an anchored principal minor of a Laplacian-like matrix.
#[derive(serde::Serialize, serde::Deserialize)]
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
// BlockElimSolver — local solver using block elimination
// ===========================================================================

/// Local subdomain solver using block elimination on the bipartite SDDM.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct BlockElimSolver {
    /// Bipartite Gramian structure: C, C^T, diag_q, diag_r.
    cross_tab: Arc<CrossTab>,
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
        cross_tab: impl Into<Arc<CrossTab>>,
        inv_diag_elim: Vec<f64>,
        reduced_factor: ReducedFactor,
        eliminate_q: bool,
    ) -> Self {
        let cross_tab = cross_tab.into();
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

    fn estimated_inner_parallel_work(&self) -> usize {
        let max_rows = self.cross_tab.n_q().max(self.cross_tab.n_r());
        if max_rows <= PAR_BACKSUB_THRESHOLD.max(PAR_SPMV_THRESHOLD) {
            return 0;
        }

        let cross_nnz = self.cross_tab.c.nnz();
        (2 * cross_nnz) + self.n_local
    }
}

impl LocalSolver for BlockElimSolver {
    fn n_local(&self) -> usize {
        self.n_local
    }

    fn scratch_size(&self) -> usize {
        self.n_local + self.n_reduced
    }

    fn solve_local(
        &self,
        rhs: &mut [f64],
        sol: &mut [f64],
        options: LocalSolveOptions,
    ) -> Result<(), LocalSolveError> {
        let n = self.n_local;
        let n_q = self.cross_tab.n_q();
        let n_r = self.cross_tab.n_r();
        let ct = &self.cross_tab;

        // Block elimination for the bipartite SDDM system [D_q, C; C^T, D_r]:
        // Step 1: Negate the q-block of rhs to convert from SDDM form to the
        //         signed Laplacian form where C carries a negative sign.
        //         This is equivalent to solving [-D_q, C; C^T, D_r] x = rhs'.
        negate_block(&mut rhs[..n], n_q);
        subtract_mean(rhs, n);

        if self.eliminate_q {
            let n_keep = n_r;

            {
                let (main, scratch) = rhs.split_at_mut(n);
                scratch[..n_keep].copy_from_slice(&main[n_q..]);
                ct.ct.spmv_diag_add(
                    &self.inv_diag_elim,
                    &main[..n_q],
                    &mut scratch[..n_keep],
                    options.allow_inner_parallelism(),
                );
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
                backsub_block(
                    sol_q,
                    &rhs[..n_q],
                    &ct.c,
                    &self.inv_diag_elim,
                    sol_r,
                    options,
                );
            }
        } else {
            let n_keep = n_q;

            {
                let (main, scratch) = rhs.split_at_mut(n);
                scratch[..n_keep].copy_from_slice(&main[..n_q]);
                ct.c.spmv_diag_add(
                    &self.inv_diag_elim,
                    &main[n_q..],
                    &mut scratch[..n_keep],
                    options.allow_inner_parallelism(),
                );
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
                    options,
                );
            }
        }

        subtract_mean(sol, n);
        negate_block(&mut sol[..n], n_q);
        Ok(())
    }

    fn inner_parallelism_work_estimate(&self) -> usize {
        self.estimated_inner_parallel_work()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use schwarz_precond::SparseMatrix;

    #[test]
    fn test_subtract_mean_empty() {
        let mut data = vec![1.0, 2.0, 3.0];
        subtract_mean(&mut data, 0);
        // Should not modify anything
        assert_eq!(data, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_subtract_mean_basic() {
        let mut data = vec![2.0, 4.0, 6.0];
        subtract_mean(&mut data, 3);
        // mean = 4.0
        assert!((data[0] - (-2.0)).abs() < 1e-14);
        assert!((data[1] - 0.0).abs() < 1e-14);
        assert!((data[2] - 2.0).abs() < 1e-14);
    }

    #[test]
    fn test_subtract_mean_partial() {
        let mut data = vec![3.0, 5.0, 100.0];
        subtract_mean(&mut data, 2);
        // mean of first 2 = 4.0
        assert!((data[0] - (-1.0)).abs() < 1e-14);
        assert!((data[1] - 1.0).abs() < 1e-14);
        assert_eq!(data[2], 100.0); // unchanged
    }

    #[test]
    fn test_local_solve_strategy_from_flags_laplacian() {
        let s = LocalSolveStrategy::from_flags(None, false);
        assert!(matches!(s, LocalSolveStrategy::Laplacian));
    }

    #[test]
    fn test_local_solve_strategy_from_flags_sddm() {
        let s = LocalSolveStrategy::from_flags(None, true);
        assert!(matches!(s, LocalSolveStrategy::Sddm));
    }

    #[test]
    fn test_local_solve_strategy_from_flags_laplacian_gramian() {
        let s = LocalSolveStrategy::from_flags(Some(5), false);
        match s {
            LocalSolveStrategy::LaplacianGramian { first_block_size } => {
                assert_eq!(first_block_size, 5);
            }
            other => panic!("expected LaplacianGramian, got: {:?}", other),
        }
    }

    #[test]
    fn test_local_solve_strategy_from_flags_gramian_augmented() {
        let s = LocalSolveStrategy::from_flags(Some(3), true);
        match s {
            LocalSolveStrategy::GramianAugmented { first_block_size } => {
                assert_eq!(first_block_size, 3);
            }
            other => panic!("expected GramianAugmented, got: {:?}", other),
        }
    }

    #[test]
    fn test_anchored_dense_cholesky_n0() {
        // n <= 1: should return Some with empty l_row_major
        let m = SparseMatrix::new(vec![0u32], Vec::new(), Vec::new(), 0);
        let result = AnchoredDenseCholesky::try_from_sparse_laplacian(&m);
        assert!(result.is_some());
        let chol = result.unwrap();
        assert_eq!(chol.n(), 0);
    }

    #[test]
    fn test_anchored_dense_cholesky_n1() {
        // 1x1 matrix: n=1 -> n<=1 early return
        let m = SparseMatrix::new(vec![0u32, 1], vec![0u32], vec![2.0], 1);
        let result = AnchoredDenseCholesky::try_from_sparse_laplacian(&m);
        assert!(result.is_some());
        let chol = result.unwrap();
        assert_eq!(chol.n(), 1);
    }

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
    fn test_anchored_dense_cholesky_solve_3x3_laplacian() {
        // 3-node path Laplacian: [[2,-1,0],[-1,2,-1],[0,-1,1]]
        // Its 2x2 anchored minor = [[2,-1],[-1,2]] -> L = [[sqrt(2), 0], [-1/sqrt(2), sqrt(3/2)]]
        let m = SparseMatrix::new(
            vec![0u32, 2, 5, 7],
            vec![0u32, 1, 0, 1, 2, 1, 2],
            vec![2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 1.0],
            3,
        );
        let chol = AnchoredDenseCholesky::try_from_sparse_laplacian(&m).expect("should factor");
        assert_eq!(chol.n(), 3);

        // Solve L L^T x = [1, 0, ?] on anchored minor
        let mut x = vec![1.0, 0.0, 0.0];
        chol.solve_in_place(&mut x);
        // x[2] should be 0 (anchored)
        assert_eq!(x[2], 0.0);
        // Verify: anchored minor * x[0..2] approx [1, 0]
        let check0 = 2.0 * x[0] - 1.0 * x[1];
        let check1 = -x[0] + 2.0 * x[1];
        assert!((check0 - 1.0).abs() < 1e-10, "check0 = {}", check0);
        assert!((check1 - 0.0).abs() < 1e-10, "check1 = {}", check1);
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

    #[test]
    fn test_negate_block() {
        let mut data = vec![1.0, -2.0, 3.0, -4.0];
        negate_block(&mut data, 2);
        assert_eq!(data, vec![1.0, -2.0, -3.0, 4.0]);
    }
}
