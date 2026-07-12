use std::sync::Arc;

use rayon::prelude::*;
use schwarz_precond::{LocalSolveError, LocalSolver};

use crate::config::LocalSolverConfig;
use crate::csr_block::{CsrBlock, PAR_SPMV_THRESHOLD};
use crate::domain::{CoordinateMap, CrossTab, SddmComponent, SolveSpace};
use crate::BuildError;

use super::compensated_sum;
use super::elimination::Elimination;
use super::factor::{factor_sparse, ReducedFactor};
use super::schur;

// ===========================================================================
// Solve helpers
// ===========================================================================

/// Minimum number of rows to trigger parallel back-substitution.
const PAR_BACKSUB_THRESHOLD: usize = 10_000;
const PAR_BACKSUB_CHUNK: usize = 4096;

/// Subtract the mean of `slice[..n]` from those `n` elements.
#[inline]
fn subtract_mean(slice: &mut [f64], n: usize) {
    if n == 0 {
        return;
    }
    let mean = compensated_sum(&slice[..n]) / n as f64;
    for val in slice[..n].iter_mut() {
        *val -= mean;
    }
}

#[inline]
fn scale_by_diag_in_place(slice: &mut [f64], diagonal: &[f64]) {
    debug_assert!(diagonal.len() >= slice.len());
    for (value, &scale) in slice.iter_mut().zip(diagonal.iter()) {
        *value *= scale;
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
    /// Internal factor dimension, including backend-added auxiliary vertices.
    n_reduced: usize,
    /// Original-to-SDDM coordinate map; in-memory only until the wire-format bump.
    #[serde(skip)]
    coordinates: CoordinateMap,
    /// Whether the augmented Laplacian has a ground vertex.
    #[serde(skip)]
    solve_space: SolveSpace,
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
    fn explicit_ground_index(&self, n_keep: usize) -> Option<usize> {
        (self.solve_space == SolveSpace::Grounded
            && self.reduced_factor.input_dimension() == n_keep + 1)
            .then_some(n_keep)
    }

    pub(crate) fn new(
        cross_tab: impl Into<Arc<CrossTab>>,
        inv_diag_elim: Vec<f64>,
        reduced_factor: ReducedFactor,
        eliminate_q: bool,
        coordinates: CoordinateMap,
        solve_space: SolveSpace,
    ) -> Self {
        let cross_tab = cross_tab.into();
        let n_local = cross_tab.n_local();
        let n_reduced = reduced_factor.factor_dimension();
        Self {
            cross_tab,
            inv_diag_elim,
            reduced_factor,
            eliminate_q,
            n_local,
            n_reduced,
            coordinates,
            solve_space,
        }
    }

    /// Build a `BlockElimSolver` from a [`CrossTab`] and solver config.
    ///
    /// Pipeline: build the [`Elimination`] once; attempt dense factorization of
    /// the exact Schur minor when below `dense_threshold`; otherwise (or on
    /// dense failure) assemble the sparse Schur complement — sampled unless
    /// configured exact — and factor it via `approx_chol`. The `Elimination` is
    /// consumed at the end to produce `inv_diag_elim` and `eliminate_q`.
    ///
    /// `diagonals` are the build-time-only diagonal blocks; they are read by
    /// [`Elimination::new`] and dropped once the factor is built.
    pub(crate) fn build(
        component: SddmComponent,
        config: &LocalSolverConfig,
    ) -> Result<Self, BuildError> {
        let SddmComponent {
            cross_tab,
            diagonals,
            ground_edges,
            coordinates,
            solve_space,
        } = component;
        let elim = Elimination::new(&cross_tab, &diagonals, &ground_edges, solve_space)?;
        let n_keep = elim.n_keep;

        // Below the dense threshold the reduced system is tiny — factor the
        // exact Schur minor densely. Floating systems anchor one node; grounded
        // systems factor the full nonsingular complement.
        let dense_factor = if n_keep == 0 {
            // Trivial 1×1 component: nothing kept to reduce, so the solve
            // degenerates to `x = r/d`; never send an empty system to approx-chol.
            ReducedFactor::try_dense(Vec::new(), 0)
        } else if config.dense_threshold > 0 && n_keep <= config.dense_threshold {
            let m = match solve_space {
                SolveSpace::Floating => n_keep - 1,
                SolveSpace::Grounded => n_keep,
            };
            ReducedFactor::try_dense(schur::dense_minor(&elim, m), n_keep)
        } else {
            None
        };

        // Dense factorization can return None on a singular minor — fall through
        // to the sparse path rather than failing.
        let factor = match dense_factor {
            Some(f) => f,
            None => {
                let schur_csr = match config.approx_schur {
                    Some(cfg) => schur::sampled(&elim, &cfg),
                    None => schur::exact_for_factor(&elim),
                };
                factor_sparse(&schur_csr, config.approx_chol)?
            }
        };
        let reduced_input_dimension = factor.input_dimension();
        debug_assert!(
            reduced_input_dimension == n_keep
                || (solve_space == SolveSpace::Grounded && reduced_input_dimension == n_keep + 1)
        );
        debug_assert!(factor.factor_dimension() >= reduced_input_dimension);

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
            coordinates,
            solve_space,
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
        let explicit_ground = self.explicit_ground_index(n_keep);

        // Scale the eliminated block by its inverse diagonal.
        scale_by_diag_in_place(&mut rhs[roles.elim.clone()], &self.inv_diag_elim);

        // Apply `keep_to_elim` into the scratch tail to form the reduced RHS.
        {
            let (main, scratch) = rhs.split_at_mut(n);
            scratch[n_keep..self.n_reduced].fill(0.0);
            roles.keep_to_elim.spmv_assign_add(
                &main[roles.elim.clone()],
                &main[roles.keep.clone()],
                &mut scratch[..n_keep],
                allow_inner_parallelism,
            );
        }
        match self.solve_space {
            SolveSpace::Floating => {
                subtract_mean(&mut rhs[n..], self.n_reduced);
            }
            // Grounded-Laplacian reduction of the nonsingular SDD reduced
            // system: the ground node absorbs the injected current, and its
            // potential is the gauge subtracted after the solve.
            SolveSpace::Grounded => {
                if let Some(ground) = explicit_ground {
                    rhs[n + ground] = -compensated_sum(&rhs[n..n + n_keep]);
                }
            }
        }

        // Solve the reduced system in place.
        let reduced = roles.keep.start..roles.keep.start + self.n_reduced;
        sol[reduced.clone()].copy_from_slice(&rhs[n..n + self.n_reduced]);
        self.reduced_factor.solve_in_place(&mut sol[reduced])?;
        if let Some(ground) = explicit_ground {
            let ground = sol[roles.keep.start + ground];
            for v in &mut sol[roles.keep.start..roles.keep.start + n_keep] {
                *v -= ground;
            }
        }

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

        self.coordinates.apply(&mut rhs[..n], n_q);
        if self.solve_space == SolveSpace::Floating {
            subtract_mean(rhs, n);
        }

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

        if self.solve_space == SolveSpace::Floating {
            subtract_mean(sol, n);
        }
        self.coordinates.apply(&mut sol[..n], n_q);
        Ok(())
    }
}

#[cfg(test)]
mod tests;
