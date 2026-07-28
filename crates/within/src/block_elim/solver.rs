use std::sync::Arc;

use approx_chol::{ExactFailure, Factor};
use rayon::prelude::*;
use schwarz_precond::{LocalSolveError, LocalSolver};

use crate::config::{LocalSolverConfig, SchurMode};
use crate::csr_block::{CsrBlock, PAR_SPMV_THRESHOLD};
use crate::domain::{
    CoordinateMap, CrossTab, Grounding, LocalComponent, OperatorForm, SddmOperator,
};
use crate::BuildError;

use super::compensated_sum;
use super::elimination::invert_eliminated_diagonal;
use super::factor::{factor_sparse, local_solver_build, ReducedFactor};
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
    /// `inv_diag_elim`/`reduced_factor` at build time and not retained).
    cross_tab: Arc<CrossTab>,
    /// `1 / D_elim[k]` for the eliminated (larger) diagonal block.
    inv_diag_elim: Vec<f64>,
    /// Reduced-system factor backend.
    reduced_factor: ReducedFactor,
    /// Internal DOF count (`n_rows + n_cols`) — the operator is always single-sized;
    /// a frustrated component's cover lives inside `reduced_factor`.
    #[serde(skip)]
    n_internal: usize,
    /// Internal factor dimension, including backend-added auxiliary vertices.
    #[serde(skip)]
    n_reduced: usize,
    /// Original-to-SDDM coordinate map.
    coordinates: CoordinateMap,
}

impl<'de> serde::Deserialize<'de> for BlockElimSolver {
    /// Reconstruct from bytes that may be untrusted (a pickle cache, another
    /// machine, a tampered file), validating every cross-field invariant the
    /// infallible [`Self::new`] takes for granted. Each dimension is pinned to a
    /// witness that is itself bounded by the input length, so no accepted solver
    /// can overflow its scratch arithmetic or index out of bounds when applied.
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::de::Error;

        #[derive(serde::Deserialize)]
        struct Helper {
            cross_tab: CrossTab,
            inv_diag_elim: Vec<f64>,
            reduced_factor: ReducedFactor,
            coordinates: CoordinateMap,
        }

        let h = Helper::deserialize(deserializer)?;

        // `c` bounds n_rows by its `indptr` length; the stored transpose's row
        // count (validated the same way) is the only witness that bounds n_cols,
        // without which recomputing the transpose below could allocate wildly.
        let CrossTab { c, ct } = h.cross_tab;
        if !c.is_structurally_valid() {
            return Err(D::Error::custom(
                "cross_tab.c is not a structurally valid CSR block",
            ));
        }
        if ct.nrows != c.ncols || ct.ncols != c.nrows || !ct.is_structurally_valid() {
            return Err(D::Error::custom("cross_tab.ct shape disagrees with c"));
        }
        let (n_rows, n_cols) = (c.nrows, c.ncols);
        let n_internal = n_rows + n_cols;

        // Components are eliminated-major: the first block is scaled by
        // `inv_diag_elim` and the second is what the reduced factor solves.
        let (elim_size, n_keep) = (n_rows, n_cols);
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

        // The reduced factor is driven over the kept block, plus the explicit
        // ground slot a grounded complement appends.
        let input_dim = h.reduced_factor.input_dimension();
        let ground_slot = h.reduced_factor.grounding() == Some(Grounding::Grounded);
        if input_dim != n_keep && !(ground_slot && input_dim == n_keep + 1) {
            return Err(D::Error::custom(
                "reduced factor input dimension disagrees with the kept block",
            ));
        }

        // Rebuild the transpose from the validated `c` so it cannot disagree;
        // `new` derives the counts, which never travel on the wire.
        let ct = c.transpose();
        Ok(BlockElimSolver::new(
            CrossTab { c, ct },
            h.inv_diag_elim,
            h.reduced_factor,
            h.coordinates,
        ))
    }
}

/// Factor one operator's reduced Schur complement. At or below
/// `dense_threshold` the reduced system is small enough that fill-in does not
/// matter, so the exact complement goes to approx-chol under an exact-only
/// backend; a pivot it cannot use falls through to the sampled Schur, as does
/// any larger system. approx-chol picks exact or approximate elimination per
/// connected block either way.
fn build_reduced_factor(
    operator: &SddmOperator,
    inv_diagonal_eliminated: &[f64],
    config: &LocalSolverConfig,
) -> Result<Factor, BuildError> {
    let exact_below = config.dense_threshold;
    if exact_below > 0 && operator.n_kept() <= exact_below {
        let exact = schur::exact_for_factor(operator, inv_diagonal_eliminated);
        let ac = config
            .approx_chol
            .to_approx_chol(exact_below, ExactFailure::Error);
        match factor_sparse(&exact, ac) {
            Err(approx_chol::Error::DenseFactorizationFailed { .. }) => {}
            result => return result.map_err(local_solver_build),
        }
    }
    let schur_csr = match &config.schur {
        SchurMode::Approximate(cfg) => schur::sampled(operator, cfg),
        SchurMode::Exact => schur::exact_for_factor(operator, inv_diagonal_eliminated),
    };
    let ac = config
        .approx_chol
        .to_approx_chol(exact_below, ExactFailure::FallBackToApproximate);
    factor_sparse(&schur_csr, ac).map_err(local_solver_build)
}

/// Build the Gremban double cover of a signed subdomain operator: each
/// off-diagonal `M_ij` becomes a same-sheet copy when nonnegative and a
/// cross-sheet copy when negative, both of magnitude `|M_ij|`, so the
/// 2×-sized cover is SDDM and acts on the antisymmetric `[z, -z]` subspace as
/// the original signed operator. Diagonals (and the caller's ground edges)
/// duplicate across sheets. Built transiently in [`BlockElimSolver::build`] to
/// factor an [`OperatorForm::SignedPendingCover`] component, then discarded.
fn assemble_bipartite_cover(operator: &SddmOperator) -> SddmOperator {
    let c = &operator.cross_tab.c;
    let n_rows = c.nrows;
    let n_cols = c.ncols;
    let n_cols_u32 = u32::try_from(n_cols).expect("cover columns exceed u32::MAX");
    // The cover doubles both dimensions: transpose stores source rows (0..2*n_rows)
    // as u32 indices and the cross-sheet shift emits columns up to 2*n_cols - 1, so
    // both doubled sizes — not just n_cols — must fit the u32 index.
    u32::try_from(2 * n_rows).expect("doubled cover rows exceed u32::MAX");
    u32::try_from(2 * n_cols).expect("doubled cover columns exceed u32::MAX");

    let mut indptr = Vec::with_capacity(2 * n_rows + 1);
    let mut indices = Vec::with_capacity(2 * c.nnz());
    let mut data = Vec::with_capacity(2 * c.nnz());
    indptr.push(0u32);
    for copy_shifted in [false, true] {
        for i in 0..n_rows {
            let start = c.indptr[i] as usize;
            let end = c.indptr[i + 1] as usize;
            // Same-sheet columns (base 0) precede cross-sheet columns (base
            // n_cols) so each output row stays column-sorted.
            for column_shifted in [false, true] {
                let column_base = if column_shifted { n_cols_u32 } else { 0 };
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
        nrows: 2 * n_rows,
        ncols: 2 * n_cols,
    };
    let cover_ct = cover_c.transpose();
    SddmOperator {
        cross_tab: CrossTab {
            c: cover_c,
            ct: cover_ct,
        },
        diagonal: double_for_cover(&operator.diagonal, n_rows),
        ground_edges: double_for_cover(&operator.ground_edges, n_rows),
        grounding: operator.grounding,
    }
}

/// Each Gremban sheet carries a copy of every vertex, so a flat per-vertex
/// array doubles within each side rather than end to end.
fn double_for_cover(values: &[f64], n_rows: usize) -> Vec<f64> {
    let (rows, cols) = values.split_at(n_rows);
    [rows, rows, cols, cols].concat()
}

impl BlockElimSolver {
    fn explicit_ground_index(&self, n_keep: usize) -> Option<usize> {
        (self.reduced_factor.grounding() == Some(Grounding::Grounded)
            && self.reduced_factor.input_dimension() == n_keep + 1)
            .then_some(n_keep)
    }

    pub(crate) fn new(
        cross_tab: impl Into<Arc<CrossTab>>,
        inv_diag_elim: Vec<f64>,
        reduced_factor: ReducedFactor,
        coordinates: CoordinateMap,
    ) -> Self {
        let cross_tab = cross_tab.into();
        let n_internal = cross_tab.n_local();
        let n_reduced = reduced_factor.factor_dimension();
        Self {
            cross_tab,
            inv_diag_elim,
            reduced_factor,
            n_internal,
            n_reduced,
            coordinates,
        }
    }

    /// Build a `BlockElimSolver` from a [`LocalComponent`] and solver config.
    ///
    /// Pipeline: fold the eliminated diagonal on the single stored operator,
    /// then factor its reduced Schur per [`OperatorForm`]. A `Laplacian`
    /// component factors the reduced Schur straight away; a signed one rebuilds
    /// its Gremban double cover transiently, factors the cover's reduced Schur,
    /// and keeps only that factor — the stored operator and its inverse diagonal
    /// stay single-sized (#91).
    ///
    /// The operator's diagonal and ground edges are build-time-only inputs and
    /// are dropped once the factor is built.
    pub(crate) fn build(
        component: LocalComponent,
        config: &LocalSolverConfig,
    ) -> Result<Self, BuildError> {
        let LocalComponent {
            operator,
            form,
            coordinates,
        } = component;
        let inv_diagonal_eliminated = invert_eliminated_diagonal(&operator)?;

        let factor = match form {
            OperatorForm::Laplacian => {
                let factor = ReducedFactor::Approx {
                    factor: build_reduced_factor(&operator, &inv_diagonal_eliminated, config)?,
                    grounding: operator.grounding,
                };
                let reduced_input_dimension = factor.input_dimension();
                debug_assert!(
                    reduced_input_dimension == operator.n_kept()
                        || (operator.grounding == Grounding::Grounded
                            && reduced_input_dimension == operator.n_kept() + 1)
                );
                debug_assert!(factor.factor_dimension() >= reduced_input_dimension);
                factor
            }
            // Cover the single signed operator, factor the cover's reduced
            // Schur (now SDDM, hence sampleable), then discard the cover —
            // only the factor is retained. Surplus survives the cover, so it
            // grounds exactly as the signed operator did.
            OperatorForm::SignedPendingCover => {
                let cover = assemble_bipartite_cover(&operator);
                let cover_inv_diagonal = invert_eliminated_diagonal(&cover)?;
                let inner = build_reduced_factor(&cover, &cover_inv_diagonal, config)?;
                ReducedFactor::Cover {
                    inner,
                    m: operator.n_kept(),
                }
            }
        };

        Ok(BlockElimSolver::new(
            operator.cross_tab,
            inv_diagonal_eliminated,
            factor,
            coordinates,
        ))
    }

    /// Eliminate one diagonal block and recover it by back-substitution.
    ///
    /// Components arrive oriented, so the row block is the eliminated side.
    fn eliminate_and_recover(
        &self,
        rhs: &mut [f64],
        sol: &mut [f64],
        allow_inner_parallelism: bool,
    ) -> Result<(), LocalSolveError> {
        let n = self.n_internal;
        let n_elim = self.cross_tab.n_rows();
        let n_keep = n - n_elim;
        let explicit_ground = self.explicit_ground_index(n_keep);

        // Scale the eliminated block by its inverse diagonal.
        scale_by_diag_in_place(&mut rhs[..n_elim], &self.inv_diag_elim);

        // Apply `keep_to_elim` into the scratch tail to form the reduced RHS.
        {
            let (main, scratch) = rhs.split_at_mut(n);
            scratch[n_keep..self.n_reduced].fill(0.0);
            self.cross_tab.ct.spmv_assign_add(
                &main[..n_elim],
                &main[n_elim..],
                &mut scratch[..n_keep],
                allow_inner_parallelism,
            );
        }
        match self.reduced_factor.grounding() {
            Some(Grounding::Floating) => {
                subtract_mean(&mut rhs[n..], self.n_reduced);
            }
            // Grounded-Laplacian reduction of the nonsingular SDD reduced
            // system: the ground node absorbs the injected current, and its
            // potential is the gauge subtracted after the solve.
            Some(Grounding::Grounded) => {
                if let Some(ground) = explicit_ground {
                    rhs[n + ground] = -compensated_sum(&rhs[n..n + n_keep]);
                }
            }
            // A cover grounds itself via the antisymmetric `[b, -b]` embed.
            None => {}
        }

        // Anchored at the kept block, the reduced solve spills past it into the
        // eliminated block (eliminate-r) or scratch tail (eliminate-q); those slots
        // are dead except the grounded gauge slot, a transient the subtraction below
        // consumes. The `rhs` tail is the factor's embed scratch (unused unless Cover).
        let reduced = n_elim..n_elim + self.n_reduced;
        debug_assert!(
            n_keep <= self.n_reduced,
            "reduced region must cover the kept block",
        );
        debug_assert!(
            reduced.end <= sol.len() && n + self.n_reduced <= rhs.len(),
            "reduced solve and its RHS copy must fit sol and rhs",
        );
        debug_assert!(
            explicit_ground.is_none_or(|g| n_keep <= g && g < self.n_reduced),
            "grounded gauge slot must spill past the kept block into a solved slot",
        );
        sol[reduced.clone()].copy_from_slice(&rhs[n..n + self.n_reduced]);
        let embed = &mut rhs[n + self.n_reduced..];
        self.reduced_factor
            .solve_in_place(&mut sol[reduced], embed)?;
        if let Some(ground) = explicit_ground {
            let ground = sol[n_elim + ground];
            for v in &mut sol[n_elim..n_elim + n_keep] {
                *v -= ground;
            }
        }

        // Back-substitute for the eliminated block; it reads the gauged kept block,
        // so the gauge subtraction above must run first.
        let (sol_output, sol_source) = sol.split_at_mut(n_elim);
        let sol_source = &sol_source[..n_keep];
        backsub_block_from_scaled_rhs(
            sol_output,
            &rhs[..n_elim],
            &self.cross_tab.c,
            &self.inv_diag_elim,
            sol_source,
            allow_inner_parallelism,
        );
        Ok(())
    }
}

impl LocalSolver for BlockElimSolver {
    fn n_local(&self) -> usize {
        self.n_internal
    }

    fn scratch_size(&self) -> usize {
        self.n_internal + self.n_reduced + self.reduced_factor.scratch_len()
    }

    fn inner_parallelism_work_estimate(&self) -> usize {
        let max_rows = self.cross_tab.n_rows().max(self.cross_tab.n_cols());
        if max_rows <= PAR_BACKSUB_THRESHOLD.max(PAR_SPMV_THRESHOLD) {
            return 0;
        }

        let cross_nnz = self.cross_tab.c.nnz();
        (2 * cross_nnz) + self.n_internal
    }

    fn solve_local(
        &self,
        rhs: &mut [f64],
        sol: &mut [f64],
        allow_inner_parallelism: bool,
    ) -> Result<(), LocalSolveError> {
        let n = self.n_internal;
        let n_rows = self.cross_tab.n_rows();

        self.coordinates.fold(&mut rhs[..n], n_rows);
        if self.reduced_factor.grounding() == Some(Grounding::Floating) {
            subtract_mean(rhs, n);
        }

        self.eliminate_and_recover(rhs, sol, allow_inner_parallelism)?;

        if self.reduced_factor.grounding() == Some(Grounding::Floating) {
            subtract_mean(sol, n);
        }
        self.coordinates.unfold(&mut sol[..n], n_rows);
        Ok(())
    }
}

#[cfg(test)]
mod tests;
