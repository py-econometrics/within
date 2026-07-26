use std::sync::Arc;

use rayon::prelude::*;
use schwarz_precond::{LocalSolveError, LocalSolver};

use crate::config::{LocalSolverConfig, SchurMode};
use crate::csr_block::{CsrBlock, PAR_SPMV_THRESHOLD};
use crate::domain::{
    BlockDiagonals, CoordinateMap, CrossTab, Grounding, LocalComponent, Reduction,
};
use crate::BuildError;

use super::compensated_sum;
use super::elimination::Elimination;
use super::factor::{factor_sparse, CoverFactor, DirectFactor, ReducedSystem};
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
    let recover = |i: usize| -> f64 {
        let start = cross_matrix.indptr[i] as usize;
        let end = cross_matrix.indptr[i + 1] as usize;
        let mut sum = 0.0;
        for idx in start..end {
            let j = cross_matrix.indices[idx] as usize;
            sum += cross_matrix.data[idx] * sol_source[j];
        }
        scaled_rhs[i] + (inv_diag[i] * sum)
    };
    if n > PAR_BACKSUB_THRESHOLD && allow_inner_parallelism {
        sol_output
            .par_chunks_mut(PAR_BACKSUB_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let row_start = chunk_idx * PAR_BACKSUB_CHUNK;
                for (local_i, si) in chunk.iter_mut().enumerate() {
                    *si = recover(row_start + local_i);
                }
            });
    } else {
        for (i, out) in sol_output.iter_mut().enumerate() {
            *out = recover(i);
        }
    }
}

// ===========================================================================
// BlockElimSolver — local solver using block elimination
// ===========================================================================

/// Local subdomain solver using block elimination on the bipartite SDDM.
#[derive(Clone, serde::Serialize)]
pub struct BlockElimSolver {
    /// Bipartite Gramian structure: C and C^T (diagonals are folded into
    /// `inv_diag_elim`/`reduced_system` at build time and not retained).
    cross_tab: Arc<CrossTab>,
    /// `1 / D_elim[k]` for the eliminated (larger) diagonal block.
    inv_diag_elim: Vec<f64>,
    pub(crate) reduced_system: ReducedSystem,
    /// Original-to-SDDM coordinate map.
    coordinates: CoordinateMap,
}

impl<'de> serde::Deserialize<'de> for BlockElimSolver {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::de::Error;

        #[derive(serde::Deserialize)]
        struct Helper {
            cross_tab: CrossTab,
            inv_diag_elim: Vec<f64>,
            reduced_system: ReducedSystem,
            coordinates: CoordinateMap,
        }

        let h = Helper::deserialize(deserializer)?;

        let CrossTab { c, ct } = h.cross_tab;
        if !c.is_structurally_valid() {
            return Err(D::Error::custom(
                "cross_tab.c is not a structurally valid CSR block",
            ));
        }
        if ct.nrows != c.ncols || ct.ncols != c.nrows || !ct.is_structurally_valid() {
            return Err(D::Error::custom("cross_tab.ct shape disagrees with c"));
        }
        let (n_q, n_r) = (c.nrows, c.ncols);
        let n_internal = n_q
            .checked_add(n_r)
            .ok_or_else(|| D::Error::custom("cross_tab dimensions overflow usize"))?;

        let eliminate_q = n_q >= n_r;
        let (elim_size, n_keep) = if eliminate_q { (n_q, n_r) } else { (n_r, n_q) };
        if h.inv_diag_elim.len() != elim_size {
            return Err(D::Error::custom(
                "inv_diag_elim length disagrees with eliminated block",
            ));
        }
        if let CoordinateMap::Scaled(factors) = &h.coordinates {
            if factors.len() != n_internal {
                return Err(D::Error::custom(
                    "Scaled coordinate map length disagrees with n_internal",
                ));
            }
        }

        h.reduced_system
            .validate_keep_dimension(n_keep)
            .map_err(D::Error::custom)?;

        let ct = c.transpose();
        Ok(BlockElimSolver::new(
            CrossTab { c, ct },
            h.inv_diag_elim,
            h.reduced_system,
            h.coordinates,
        ))
    }
}

struct BlockRoles<'a> {
    elim: std::ops::Range<usize>,
    keep: std::ops::Range<usize>,
    keep_to_elim: &'a CsrBlock,
    elim_to_keep: &'a CsrBlock,
}

impl BlockRoles<'_> {
    fn split_sol<'s>(&self, sol: &'s mut [f64]) -> (&'s mut [f64], &'s [f64]) {
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

/// Factor one operator's reduced Schur complement, choosing the backend by
/// size and configuration: an exact dense minor below `dense_threshold`,
/// otherwise a sparse Schur — sampled unless `approx_schur` is disabled —
/// factored via `approx_chol`. `grounding` gauges the reduced system:
/// `Floating` anchors one node, `Grounded` factors the full complement.
fn build_reduced_factor(
    elim: &Elimination,
    grounding: &Grounding,
    config: &LocalSolverConfig,
) -> Result<DirectFactor, BuildError> {
    let n_keep = elim.n_keep;
    let dense_factor = if n_keep == 0 {
        // Trivial 1×1 operator: nothing kept to reduce, so the solve degenerates
        // to `x = r/d`; never send an empty system to approx-chol.
        DirectFactor::try_dense(Vec::new(), 0)
    } else if config.dense_threshold > 0 && n_keep <= config.dense_threshold {
        let m = match grounding {
            Grounding::Floating => n_keep - 1,
            Grounding::Grounded(_) => n_keep,
        };
        DirectFactor::try_dense(schur::dense_minor(elim, m), n_keep)
    } else {
        None
    };
    // Dense factorization returns None on a singular minor — fall through to the
    // sparse path rather than failing.
    match dense_factor {
        Some(factor) => Ok(factor),
        None => {
            let schur_csr = match &config.schur {
                SchurMode::Approximate(cfg) => schur::sampled(elim, cfg),
                SchurMode::Exact => schur::exact_for_factor(elim),
            };
            factor_sparse(&schur_csr, config.approx_chol)
        }
    }
}

/// Build the Gremban double cover of a signed subdomain operator: each
/// off-diagonal `M_ij` becomes a same-sheet copy when nonnegative and a
/// cross-sheet copy when negative, both of magnitude `|M_ij|`, so the
/// 2×-sized cover is SDDM and acts on the antisymmetric `[z, -z]` subspace as
/// the original signed operator. Diagonals (and the caller's ground edges)
/// duplicate across sheets. Built transiently in [`BlockElimSolver::build`] to
/// factor a [`Reduction::Cover`] reduction, then discarded.
fn assemble_bipartite_cover(
    cross_tab: &CrossTab,
    diagonals: &BlockDiagonals,
) -> (CrossTab, BlockDiagonals) {
    let c = &cross_tab.c;
    let n_q = c.nrows;
    let n_r = c.ncols;
    let n_r_u32 = u32::try_from(n_r).expect("cover columns exceed u32::MAX");
    // The cover doubles both dimensions: transpose stores source rows (0..2*n_q)
    // as u32 indices and the cross-sheet shift emits columns up to 2*n_r - 1, so
    // both doubled sizes — not just n_r — must fit the u32 index.
    u32::try_from(2 * n_q).expect("doubled cover rows exceed u32::MAX");
    u32::try_from(2 * n_r).expect("doubled cover columns exceed u32::MAX");

    let mut indptr = Vec::with_capacity(2 * n_q + 1);
    let mut indices = Vec::with_capacity(2 * c.nnz());
    let mut data = Vec::with_capacity(2 * c.nnz());
    indptr.push(0u32);
    for copy_shifted in [false, true] {
        for i in 0..n_q {
            let start = c.indptr[i] as usize;
            let end = c.indptr[i + 1] as usize;
            // Same-sheet columns (base 0) precede cross-sheet columns (base
            // n_r) so each output row stays column-sorted.
            for column_shifted in [false, true] {
                let column_base = if column_shifted { n_r_u32 } else { 0 };
                let select_negative = column_shifted != copy_shifted;
                for idx in start..end {
                    let value = c.data[idx];
                    if (value < 0.0) != select_negative {
                        continue;
                    }
                    indices.push(c.indices[idx] + column_base);
                    data.push(value.abs());
                }
            }
            indptr.push(u32::try_from(indices.len()).expect("cover nonzeros exceed u32::MAX"));
        }
    }
    let cover_c = CsrBlock {
        indptr,
        indices,
        data,
        nrows: 2 * n_q,
        ncols: 2 * n_r,
    };
    let cover_ct = cover_c.transpose();
    let cover_diagonals = BlockDiagonals {
        q: diagonals.q.repeat(2),
        r: diagonals.r.repeat(2),
    };
    (
        CrossTab {
            c: cover_c,
            ct: cover_ct,
        },
        cover_diagonals,
    )
}

impl BlockElimSolver {
    fn n_internal(&self) -> usize {
        self.cross_tab.n_local()
    }

    fn n_reduced(&self) -> usize {
        self.reduced_system.factor_dimension()
    }

    fn eliminate_q(&self) -> bool {
        self.cross_tab.n_q() >= self.cross_tab.n_r()
    }

    fn explicit_ground_index(&self, n_keep: usize) -> Option<usize> {
        match &self.reduced_system {
            ReducedSystem::Grounded(factor) if factor.input_dimension() == n_keep + 1 => {
                Some(n_keep)
            }
            _ => None,
        }
    }

    pub(crate) fn new(
        cross_tab: impl Into<Arc<CrossTab>>,
        inv_diag_elim: Vec<f64>,
        reduced_system: ReducedSystem,
        coordinates: CoordinateMap,
    ) -> Self {
        Self {
            cross_tab: cross_tab.into(),
            inv_diag_elim,
            reduced_system,
            coordinates,
        }
    }

    /// Build a `BlockElimSolver` from a [`LocalComponent`] and solver config.
    ///
    /// Pipeline: build the [`Elimination`] on the single stored operator, then
    /// factor its reduced Schur per [`Reduction`]. A `Direct` component
    /// factors the reduced Schur straight away; a `Cover` (frustrated, signed)
    /// component rebuilds its Gremban double cover transiently, factors the
    /// cover's reduced Schur, and keeps only that factor — the stored operator
    /// and `inv_diag_elim` stay single-sized (#91). The `Elimination` is
    /// consumed at the end to produce `inv_diag_elim`.
    ///
    /// `diagonals` are the build-time-only diagonal blocks; they are read by
    /// [`Elimination::new`] and dropped once the factor is built.
    pub(crate) fn build(
        component: LocalComponent,
        config: &LocalSolverConfig,
    ) -> Result<Self, BuildError> {
        let LocalComponent {
            cross_tab,
            diagonals,
            coordinates,
            reduction,
        } = component;
        let elim = Elimination::new(&cross_tab, &diagonals, reduction.grounding())?;

        let reduced_system = match &reduction {
            Reduction::Direct(grounding) => {
                let factor = build_reduced_factor(&elim, grounding, config)?;
                match grounding {
                    Grounding::Floating => ReducedSystem::Floating(factor),
                    Grounding::Grounded(_) => ReducedSystem::Grounded(factor),
                }
            }
            Reduction::Cover(grounding) => {
                // Cover the single signed operator, factor the cover's reduced
                // Schur (now SDDM, hence sampleable), then discard the cover —
                // only the factor is retained.
                let (cover_cross, cover_diag) = assemble_bipartite_cover(&cross_tab, &diagonals);
                let cover_grounding = grounding.doubled();
                let cover_elim = Elimination::new(&cover_cross, &cover_diag, &cover_grounding)?;
                let inner = build_reduced_factor(&cover_elim, &cover_grounding, config)?;
                let cover = CoverFactor::try_new(inner, elim.n_keep)
                    .map_err(|message| BuildError::LocalSolverBuild(message.to_owned()))?;
                ReducedSystem::Signed(cover)
            }
        };

        let Elimination { inv_diag_elim, .. } = elim;
        Ok(BlockElimSolver::new(
            cross_tab,
            inv_diag_elim,
            reduced_system,
            coordinates,
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
        let n = self.n_internal();
        let n_reduced = self.n_reduced();
        let n_keep = roles.keep.len();
        let explicit_ground = self.explicit_ground_index(n_keep);

        // Scale the eliminated block by its inverse diagonal.
        scale_by_diag_in_place(&mut rhs[roles.elim.clone()], &self.inv_diag_elim);

        // Apply `keep_to_elim` into the scratch tail to form the reduced RHS.
        {
            let (main, scratch) = rhs.split_at_mut(n);
            scratch[n_keep..n_reduced].fill(0.0);
            roles.keep_to_elim.spmv_assign_add(
                &main[roles.elim.clone()],
                &main[roles.keep.clone()],
                &mut scratch[..n_keep],
                allow_inner_parallelism,
            );
        }
        match &self.reduced_system {
            ReducedSystem::Floating(_) => {
                subtract_mean(&mut rhs[n..], n_reduced);
            }
            // Grounded-Laplacian reduction of the nonsingular SDD reduced
            // system: the ground node absorbs the injected current, and its
            // potential is the gauge subtracted after the solve.
            ReducedSystem::Grounded(_) => {
                if let Some(ground) = explicit_ground {
                    rhs[n + ground] = -compensated_sum(&rhs[n..n + n_keep]);
                }
            }
            // Signed operator: the reduced `Cover` factor grounds itself via
            // the antisymmetric `[b, -b]` embed, so no gauge handling here.
            ReducedSystem::Signed(_) => {}
        }

        // Anchored at the kept block, the reduced solve spills past it into the
        // eliminated block (eliminate-r) or scratch tail (eliminate-q); those slots
        // are dead except the grounded gauge slot, a transient the subtraction below
        // consumes. The `rhs` tail is the factor's embed scratch (unused unless Cover).
        let reduced = roles.keep.start..roles.keep.start + n_reduced;
        debug_assert!(
            n_keep <= n_reduced,
            "reduced region must cover the kept block",
        );
        debug_assert!(
            reduced.end <= sol.len() && n + n_reduced <= rhs.len(),
            "reduced solve and its RHS copy must fit sol and rhs",
        );
        debug_assert!(
            explicit_ground.is_none_or(|g| n_keep <= g && g < n_reduced),
            "grounded gauge slot must spill past the kept block into a solved slot",
        );
        sol[reduced.clone()].copy_from_slice(&rhs[n..n + n_reduced]);
        let embed = &mut rhs[n + n_reduced..];
        self.reduced_system
            .solve_in_place(&mut sol[reduced], embed)?;
        if let Some(ground) = explicit_ground {
            let ground = sol[roles.keep.start + ground];
            for v in &mut sol[roles.keep.start..roles.keep.start + n_keep] {
                *v -= ground;
            }
        }

        // Back-substitute for the eliminated block; it reads the gauged kept block,
        // so the gauge subtraction above must run first.
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
        self.n_internal()
    }

    fn scratch_size(&self) -> usize {
        self.n_internal() + self.n_reduced() + self.reduced_system.scratch_len()
    }

    fn inner_parallelism_work_estimate(&self) -> usize {
        let max_rows = self.cross_tab.n_q().max(self.cross_tab.n_r());
        if max_rows <= PAR_BACKSUB_THRESHOLD.max(PAR_SPMV_THRESHOLD) {
            return 0;
        }

        let cross_nnz = self.cross_tab.c.nnz();
        (2 * cross_nnz) + self.n_internal()
    }

    fn solve_local(
        &self,
        rhs: &mut [f64],
        sol: &mut [f64],
        allow_inner_parallelism: bool,
    ) -> Result<(), LocalSolveError> {
        let n = self.n_internal();
        let n_q = self.cross_tab.n_q();
        let ct = &self.cross_tab;
        let floating = matches!(self.reduced_system, ReducedSystem::Floating(_));

        self.coordinates.fold(&mut rhs[..n], n_q);
        if floating {
            subtract_mean(rhs, n);
        }

        let roles = if self.eliminate_q() {
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

        if floating {
            subtract_mean(sol, n);
        }
        self.coordinates.unfold(&mut sol[..n], n_q);
        Ok(())
    }
}

#[cfg(test)]
mod tests;
