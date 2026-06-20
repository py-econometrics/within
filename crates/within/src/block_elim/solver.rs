use std::sync::Arc;

use rayon::prelude::*;
use schwarz_precond::{LocalSolveError, LocalSolver};

use crate::config::LocalSolverConfig;
use crate::csr_block::{CsrBlock, PAR_SPMV_THRESHOLD};
use crate::domain::{BlockDiagonals, CrossTab};
use crate::BuildError;

use super::elimination::Elimination;
use super::factor::{factor_sparse, ReducedFactor};
use super::schur::{ApproxSchurComplement, ExactSchurComplement, SchurComplement, SchurLaplacian};

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
    // Neumaier compensated summation: a flat `iter().sum()` loses precision for
    // large `n`, which biases the mean we subtract. The running compensation
    // recovers the low-order bits dropped on each addition.
    let mut sum = 0.0_f64;
    let mut comp = 0.0_f64;
    for &val in &slice[..n] {
        let t = sum + val;
        if sum.abs() >= val.abs() {
            comp += (sum - t) + val;
        } else {
            comp += (val - t) + sum;
        }
        sum = t;
    }
    let mean = (sum + comp) / n as f64;
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
// BlockElimSolver — local solver using block elimination
// ===========================================================================

/// Local subdomain solver using block elimination on the bipartite SDDM.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct BlockElimSolver {
    /// Bipartite Gramian structure: C and C^T (diagonals are folded into
    /// `inv_diag_elim`/`reduced_factor` at build time and not retained).
    cross_tab: Arc<CrossTab>,
    /// `1 / D_elim[k]` for the eliminated (larger) diagonal block.
    inv_diag_elim: Vec<f64>,
    /// Reduced-system factor backend.
    pub(crate) reduced_factor: ReducedFactor,
    /// True if the q-block was eliminated (n_q >= n_r).
    eliminate_q: bool,
    /// Total DOF count (`n_q + n_r`).
    n_local: usize,
    /// Factor dimension for the reduced solve (may be `n_keep + 1` for sparse AC).
    n_reduced: usize,
}

/// The q/r blocks assigned to their eliminated/kept roles for one solve.
///
/// `BlockElimSolver` eliminates one diagonal block and solves a reduced system
/// on the other; `eliminate_q` fixes which is which at construction. This bundles
/// the per-orientation block ranges and cross operators under named roles —
/// reusing [`super::elimination::Elimination`]'s `keep_to_elim` / `elim_to_keep`
/// vocabulary — so [`BlockElimSolver::eliminate_and_recover`] stays
/// orientation-agnostic.
struct BlockRoles<'a> {
    /// Index range of the eliminated block within the `[q | r]` layout.
    elim: std::ops::Range<usize>,
    /// Index range of the kept (reduced) block.
    keep: std::ops::Range<usize>,
    /// Cross block with kept rows / eliminated columns; applied to the eliminated
    /// block to form the reduced RHS (`Cᵀ` when eliminating q, `C` when r).
    keep_to_elim: &'a CsrBlock,
    /// Cross block with eliminated rows / kept columns; used in back-substitution
    /// to recover the eliminated block (`C` when eliminating q, `Cᵀ` when r).
    elim_to_keep: &'a CsrBlock,
}

impl BlockRoles<'_> {
    /// Borrow the eliminated block (mutable, the back-substitution output) and
    /// the kept block (immutable source) out of `sol`. The two blocks meet at
    /// the q/r boundary; `sol`'s tail past `n_local` is reduced-solve scratch,
    /// so the kept side is clamped to `keep`'s length.
    fn split_sol<'s>(&self, sol: &'s mut [f64]) -> (&'s mut [f64], &'s [f64]) {
        // The split relies on the two blocks being adjacent: the lower one ends
        // exactly where the upper one begins (they meet at the q/r boundary).
        debug_assert!(
            self.elim.end == self.keep.start || self.keep.end == self.elim.start,
            "elim and keep blocks must be adjacent",
        );
        let (lo, hi) = sol.split_at_mut(self.elim.start.max(self.keep.start));
        if self.elim.start < self.keep.start {
            (&mut lo[self.elim.clone()], &hi[..self.keep.len()])
        } else {
            (&mut hi[..self.elim.len()], &lo[self.keep.clone()])
        }
    }
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

    /// Build a `BlockElimSolver` from a [`CrossTab`] and solver config.
    ///
    /// Pipeline: build the [`Elimination`] once; attempt dense factorization on
    /// the anchored minor when below `dense_threshold`; otherwise (or on dense
    /// failure) assemble the sparse Schur complement and factor it via
    /// `approx_chol`. The `Elimination` is consumed at the end to produce
    /// `inv_diag_elim` and `eliminate_q`.
    ///
    /// `diagonals` are the build-time-only diagonal blocks; they are read by
    /// [`Elimination::new`] and dropped once the factor is built.
    pub(crate) fn build(
        cross_tab: CrossTab,
        diagonals: &BlockDiagonals,
        config: &LocalSolverConfig,
    ) -> Result<Self, BuildError> {
        let elim = Elimination::new(&cross_tab, diagonals)?;
        let n_keep = elim.n_keep;
        let prefer_dense = config.dense_threshold > 0 && n_keep <= config.dense_threshold;

        // Below the dense threshold the reduced system is tiny — always use exact
        // Schur complement (cheap at this size) and dense Cholesky factorization.
        let dense_factor = if prefer_dense {
            let anchored_minor = SchurLaplacian::anchored_minor_from_elimination(&elim);
            ReducedFactor::try_dense_laplacian_minor(anchored_minor, n_keep)
        } else {
            None
        };

        // Dense factorization can return None on a singular minor — fall through
        // to the sparse path rather than failing.
        let factor = match dense_factor {
            Some(f) => f,
            None => {
                let schur_csr = match config.approx_schur {
                    None => ExactSchurComplement.compute(&elim),
                    Some(cfg) => ApproxSchurComplement::new(cfg).compute(&elim),
                };
                factor_sparse(&schur_csr, config.approx_chol)?
            }
        };

        let Elimination {
            inv_diag_elim,
            eliminate_q,
            ..
        } = elim;
        Ok(BlockElimSolver::new(
            cross_tab,
            inv_diag_elim,
            factor,
            eliminate_q,
        ))
    }

    /// Eliminate one diagonal block and recover it by back-substitution.
    ///
    /// `roles` assigns the q/r blocks and cross operators to the eliminated/kept
    /// roles; the sequence below is the bipartite-SDDM block-elimination kernel
    /// and runs unchanged for both orientations.
    fn eliminate_and_recover(
        &self,
        roles: &BlockRoles,
        rhs: &mut [f64],
        sol: &mut [f64],
        allow_inner_parallelism: bool,
    ) -> Result<(), LocalSolveError> {
        let n = self.n_local;
        let n_keep = roles.keep.len();

        // Scale the eliminated block by its inverse diagonal.
        scale_by_diag_in_place(&mut rhs[roles.elim.clone()], &self.inv_diag_elim);

        // Apply `keep_to_elim` into the scratch tail to form the reduced RHS.
        {
            let (main, scratch) = rhs.split_at_mut(n);
            roles.keep_to_elim.spmv_assign_add(
                &main[roles.elim.clone()],
                &main[roles.keep.clone()],
                &mut scratch[..n_keep],
                allow_inner_parallelism,
            );
        }
        if self.n_reduced > n_keep {
            rhs[n + n_keep] = 0.0;
        }
        subtract_mean(&mut rhs[n..], self.n_reduced);

        // Solve the reduced system in place.
        let reduced = roles.keep.start..roles.keep.start + self.n_reduced;
        sol[reduced.clone()].copy_from_slice(&rhs[n..n + self.n_reduced]);
        self.reduced_factor.solve_in_place(&mut sol[reduced])?;

        // Back-substitute to recover the eliminated block.
        let (sol_output, sol_source) = roles.split_sol(sol);
        backsub_block_from_scaled_rhs(
            sol_output,
            &rhs[roles.elim.clone()],
            roles.elim_to_keep,
            &self.inv_diag_elim,
            sol_source,
            allow_inner_parallelism,
        );
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

    fn inner_parallelism_work_estimate(&self) -> usize {
        let max_rows = self.cross_tab.n_q().max(self.cross_tab.n_r());
        if max_rows <= PAR_BACKSUB_THRESHOLD.max(PAR_SPMV_THRESHOLD) {
            return 0;
        }

        let cross_nnz = self.cross_tab.c.nnz();
        (2 * cross_nnz) + self.n_local
    }

    fn solve_local(
        &self,
        rhs: &mut [f64],
        sol: &mut [f64],
        allow_inner_parallelism: bool,
    ) -> Result<(), LocalSolveError> {
        let n = self.n_local;
        let n_q = self.cross_tab.n_q();
        let ct = &self.cross_tab;

        // Block elimination for the bipartite SDDM system [D_q, C; C^T, D_r]:
        // negate the q-block to convert from SDDM form to the signed-Laplacian
        // form where C carries a negative sign (equivalent to solving
        // [-D_q, C; C^T, D_r] x = rhs').
        negate_block(&mut rhs[..n], n_q);
        subtract_mean(rhs, n);

        // The eliminated/kept roles are fixed by `eliminate_q`; name the swap once.
        let roles = if self.eliminate_q {
            BlockRoles {
                elim: 0..n_q,
                keep: n_q..n,
                keep_to_elim: &ct.ct,
                elim_to_keep: &ct.c,
            }
        } else {
            BlockRoles {
                elim: n_q..n,
                keep: 0..n_q,
                keep_to_elim: &ct.c,
                elim_to_keep: &ct.ct,
            }
        };
        self.eliminate_and_recover(&roles, rhs, sol, allow_inner_parallelism)?;

        subtract_mean(sol, n);
        negate_block(&mut sol[..n], n_q);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::config::ApproxCholConfig;
    use crate::csr_block::CsrBlock;

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
    fn test_negate_block() {
        let mut data = vec![1.0, -2.0, 3.0, -4.0];
        negate_block(&mut data, 2);
        assert_eq!(data, vec![1.0, -2.0, -3.0, 4.0]);
    }

    /// Build a CrossTab with `n_q < r` so that `eliminate_q == false`, plus its
    /// build-time diagonals.
    fn make_cross_tab_q_lt_r() -> (CrossTab, BlockDiagonals) {
        let c_dense = vec![
            // row 0
            1.0, 0.0, 0.0, 0.0, 0.0, // row 1
            0.0, 1.0, 0.0, 0.0, 0.0,
        ];
        let c = CsrBlock::from_dense_table(&c_dense, 2, 5);
        let ct = c.transpose();
        let diagonals = BlockDiagonals {
            q: vec![2.0, 3.0],
            r: vec![2.0, 3.0, 1.0, 1.0, 1.0],
        };
        (CrossTab { c, ct }, diagonals)
    }

    #[test]
    fn test_block_elim_solver_eliminate_q_false() {
        let (cross_tab, diagonals) = make_cross_tab_q_lt_r();
        assert_eq!(cross_tab.n_q(), 2);
        assert_eq!(cross_tab.n_r(), 5);

        let config = LocalSolverConfig {
            approx_chol: ApproxCholConfig::default(),
            approx_schur: None,
            dense_threshold: 0, // disable dense fast path to ensure sparse path is covered
        };
        let solver = BlockElimSolver::build(cross_tab, &diagonals, &config)
            .expect("block-elim build failed");

        assert!(
            !solver.eliminate_q,
            "expected eliminate_q=false when n_q < n_r",
        );
        // n_local = n_q + n_r = 2 + 5 = 7
        assert_eq!(solver.n_local(), 7);
    }

    #[test]
    fn test_block_elim_solver_eliminate_q_false_solve_residual() {
        let (cross_tab, diagonals) = make_cross_tab_q_lt_r();
        let n_local = cross_tab.n_q() + cross_tab.n_r(); // 7

        let config = LocalSolverConfig {
            approx_chol: ApproxCholConfig::default(),
            approx_schur: None,
            dense_threshold: 0,
        };
        let solver = BlockElimSolver::build(cross_tab, &diagonals, &config)
            .expect("block-elim build failed");
        assert!(!solver.eliminate_q);

        let scratch_sz = solver.scratch_size();
        let mut rhs = vec![0.0; scratch_sz];
        for (i, v) in rhs[..n_local].iter_mut().enumerate() {
            *v = (i as f64 + 1.0) * 0.5;
        }
        let mut sol = vec![0.0; scratch_sz];

        solver
            .solve_local(&mut rhs, &mut sol, true)
            .expect("solve_local should succeed");

        for (i, &v) in sol[..n_local].iter().enumerate() {
            assert!(v.is_finite(), "sol[{i}] = {v} is not finite");
        }
        let sol_norm: f64 = sol[..n_local].iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(sol_norm > 1e-15, "solution is unexpectedly all-zero");
    }
}
