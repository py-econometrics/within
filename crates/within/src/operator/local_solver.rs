//! Local solvers for Schwarz subdomains.
//!
//! Each Schwarz subdomain corresponds to a pair of factors (q, r) and the
//! local Gramian on that subdomain has a 2x2 block structure:
//!
//! ```text
//!     [ D_q   C  ]
//! G = [          ]
//!     [ C^T  D_r ]
//! ```
//!
//! where `D_q`, `D_r` are diagonal (weighted level counts) and `C` is the
//! cross-tabulation matrix.
//!
//! # The bipartite sign-flipping trick
//!
//! The `approx-chol` crate solves SDDM systems (Symmetric Diagonally Dominant
//! M-matrices), where off-diagonal entries are non-positive. But our local
//! Gramian `G` has *positive* off-diagonal entries in `C`. The key insight is
//! that this two-factor Gramian has **bipartite structure**: DOFs split into
//! the q-block and the r-block, and all non-zero off-diagonal entries connect
//! a q-DOF to an r-DOF (never q-to-q or r-to-r). Negating all entries in one
//! block — i.e., applying the similarity transform `S G S` where
//! `S = diag(I, -I)` — flips the sign of `C` without changing the diagonal
//! blocks, producing a valid SDDM matrix. The solution is recovered by
//! negating the corresponding block of the output.
//!
//! # `BlockElimSolver` — block elimination with Schur complement
//!
//! Exploits the diagonal structure of `D_q` and `D_r` to perform (exact or approximate) block
//! elimination of the larger factor. If `n_q >= n_r`, we eliminate the q-block:
//!
//! 1. **Forward eliminate**: solve `D_q z_q = r_q - C z_r` (trivial since
//!    `D_q` is diagonal)
//! 2. **Reduced solve**: solve the Schur complement system
//!    `S z_r = r_r - C^T D_q^{-1} r_q` where `S = D_r - C^T D_q^{-1} C`
//! 3. **Back-substitute**: recover `z_q` from step 1
//!
//! The reduced system `S` is smaller (dimension `n_r` instead of `n_q + n_r`)
//! and is factored via either approximate Cholesky or dense Cholesky,
//! depending on size. See [`schur_complement`](super::schur_complement) for
//! the Schur complement computation.

use std::sync::Arc;

use approx_chol::Factor;
use faer::{MatRef, Side};
use rayon::prelude::*;
use schwarz_precond::{LocalSolveError, LocalSolveInvoker, LocalSolver};

use crate::operator::csr_block::{CsrBlock, PAR_SPMV_THRESHOLD};
use crate::operator::gramian::CrossTab;

// ===========================================================================
// FeLocalSolveInvoker — delegates to BlockElimSolver with parallelism control
// ===========================================================================

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct FeLocalSolveInvoker;

impl LocalSolveInvoker<BlockElimSolver> for FeLocalSolveInvoker {
    fn solve_local(
        &self,
        solver: &BlockElimSolver,
        rhs: &mut [f64],
        sol: &mut [f64],
        allow_inner_parallelism: bool,
    ) -> Result<(), LocalSolveError> {
        solver.solve_local_with_parallelism(rhs, sol, allow_inner_parallelism)
    }
}

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

/// Scale `slice[i] *= diag[i]` for the first `slice.len()` entries.
#[inline]
fn scale_by_diag_in_place(slice: &mut [f64], diag: &[f64]) {
    debug_assert!(diag.len() >= slice.len());
    for (val, &di) in slice.iter_mut().zip(diag.iter()) {
        *val *= di;
    }
}

/// Back-substitute for the eliminated block from a pre-scaled RHS.
fn backsub_block_from_scaled_rhs(
    sol_output: &mut [f64],
    scaled_rhs: &[f64],
    cross_matrix: &CsrBlock,
    inv_diag: &[f64],
    sol_source: &[f64],
    allow_inner_parallelism: bool,
) {
    let n = sol_output.len();
    debug_assert!(scaled_rhs.len() >= n);
    if n > PAR_BACKSUB_THRESHOLD && allow_inner_parallelism {
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
                    *si = scaled_rhs[i] + (inv_diag[i] * sum);
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
            sol_output[i] = scaled_rhs[i] + (inv_diag[i] * sum);
        }
    }
}

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
    pub(crate) fn approx(factor: Factor) -> Self {
        Self::Approx(factor)
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
// BlockElimSolver — local solver using block elimination
// ===========================================================================

/// Local subdomain solver using block elimination on the bipartite SDDM.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
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

impl BlockElimSolver {
    fn solve_local_with_parallelism(
        &self,
        rhs: &mut [f64],
        sol: &mut [f64],
        allow_inner_parallelism: bool,
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
            scale_by_diag_in_place(&mut rhs[..n_q], &self.inv_diag_elim);

            {
                let (main, scratch) = rhs.split_at_mut(n);
                ct.ct.spmv_assign_add(
                    &main[..n_q],
                    &main[n_q..n_q + n_keep],
                    &mut scratch[..n_keep],
                    allow_inner_parallelism,
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
                backsub_block_from_scaled_rhs(
                    sol_q,
                    &rhs[..n_q],
                    &ct.c,
                    &self.inv_diag_elim,
                    sol_r,
                    allow_inner_parallelism,
                );
            }
        } else {
            let n_keep = n_q;
            scale_by_diag_in_place(&mut rhs[n_q..n_q + n_r], &self.inv_diag_elim);

            {
                let (main, scratch) = rhs.split_at_mut(n);
                ct.c.spmv_assign_add(
                    &main[n_q..n_q + n_r],
                    &main[..n_q],
                    &mut scratch[..n_keep],
                    allow_inner_parallelism,
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
                backsub_block_from_scaled_rhs(
                    &mut sol_r[..n_r],
                    &rhs[n_q..n_q + n_r],
                    &ct.ct,
                    &self.inv_diag_elim,
                    sol_q,
                    allow_inner_parallelism,
                );
            }
        }

        subtract_mean(sol, n);
        negate_block(&mut sol[..n], n_q);
        Ok(())
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
        self.solve_local_with_parallelism(rhs, sol, true)
    }

    fn inner_parallelism_work_estimate(&self) -> usize {
        self.estimated_inner_parallel_work()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn test_negate_block() {
        let mut data = vec![1.0, -2.0, 3.0, -4.0];
        negate_block(&mut data, 2);
        assert_eq!(data, vec![1.0, -2.0, -3.0, 4.0]);
    }
}
