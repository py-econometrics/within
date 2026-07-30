use std::sync::Arc;

use approx_chol::{ExactFailure, Factor};
use rayon::prelude::*;
use schwarz_precond::{LocalSolveError, LocalSolver};

use crate::config::{LocalSolverConfig, SchurMode};
use crate::csr_block::{CsrBlock, PAR_SPMV_THRESHOLD};
use crate::domain::{CoordinateMap, CrossTab, Grounding, LocalComponent, MatrixForm, SddmMatrix};
use crate::BuildError;

use super::compensated_sum;
use super::factor::{factor_sparse, local_solver_build, ReducedFactor};
use super::schur;

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

/// Local subdomain solver using block elimination on the bipartite SDDM.
#[derive(Clone, serde::Serialize)]
pub struct BlockElimSolver {
    /// Bipartite Gramian `C` and `Cᵀ`; diagonals are folded in at build time and not retained.
    cross_tab: Arc<CrossTab>,
    /// `1 / D_elim[k]` for the eliminated (larger) diagonal block.
    inv_diag_elim: Vec<f64>,
    /// Reduced-system factor backend.
    reduced_factor: ReducedFactor,
    /// Internal DOF count; the matrix is single-sized, a cover living in `reduced_factor`.
    #[serde(skip)]
    n_internal: usize,
    /// Internal factor dimension, including backend-added auxiliary vertices.
    #[serde(skip)]
    n_reduced: usize,
    /// Original-to-SDDM coordinate map.
    coordinates: CoordinateMap,
}

impl<'de> serde::Deserialize<'de> for BlockElimSolver {
    /// Validates every cross-field invariant [`Self::new`] assumes, against untrusted bytes.
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

        // `ct`'s row count is the only witness bounding `c.ncols`.
        let CrossTab { c, ct } = h.cross_tab;
        if !c.is_structurally_valid() {
            return Err(D::Error::custom(
                "cross_tab.c is not a structurally valid CSR block",
            ));
        }
        if ct.nrows != c.ncols || ct.ncols != c.nrows || !ct.is_structurally_valid() {
            return Err(D::Error::custom("cross_tab.ct shape disagrees with c"));
        }
        // Components are eliminated-major: the first block is scaled by `inv_diag_elim`.
        let (n_eliminated, n_kept) = (c.nrows, c.ncols);
        let n_internal = n_eliminated + n_kept;

        if h.inv_diag_elim.len() != n_eliminated {
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

        if !h.reduced_factor.spans_kept_block(n_kept) {
            return Err(D::Error::custom(
                "reduced factor input dimension disagrees with the kept block",
            ));
        }

        // Rebuild from the validated `c` so the transpose cannot disagree.
        let ct = c.transpose();
        Ok(BlockElimSolver::new(
            CrossTab { c, ct },
            h.inv_diag_elim,
            h.reduced_factor,
            h.coordinates,
        ))
    }
}

/// An eliminated-major SDDM plus its eliminated-diagonal reciprocals, which it does not carry.
struct Eliminated {
    matrix: SddmMatrix,
    inv_diagonal: Vec<f64>,
}

impl Eliminated {
    fn new(matrix: SddmMatrix) -> Result<Self, BuildError> {
        debug_assert!(
            matrix.n_eliminated() >= matrix.n_kept(),
            "component is not eliminated-major"
        );
        let inv_diagonal = matrix.diagonal[..matrix.n_eliminated()]
            .iter()
            .enumerate()
            .map(|(i, &d)| {
                if d > 0.0 {
                    Ok(1.0 / d)
                } else {
                    Err(BuildError::SingularDiagonal { index: i })
                }
            })
            .collect::<Result<Vec<f64>, BuildError>>()?;
        Ok(Self {
            matrix,
            inv_diagonal,
        })
    }

    /// Fold of the signed matrix's Gremban cover; transient, dropped once its factor is built.
    fn cover(&self) -> Result<Self, BuildError> {
        Self::new(assemble_bipartite_cover(&self.matrix))
    }

    /// At or below `dense_threshold` fill-in does not matter, so the exact complement is used.
    fn factor_reduced(&self, config: &LocalSolverConfig) -> Result<Factor, BuildError> {
        let exact_below = config.dense_threshold;
        if exact_below > 0 && self.matrix.n_kept() <= exact_below {
            let exact = schur::exact_for_factor(&self.matrix, &self.inv_diagonal);
            let ac = config
                .approx_chol
                .to_approx_chol(exact_below, ExactFailure::Error);
            match factor_sparse(&exact, ac) {
                Err(approx_chol::Error::DenseFactorizationFailed { .. }) => {}
                result => return result.map_err(local_solver_build),
            }
        }
        let schur_csr = match &config.schur {
            SchurMode::Approximate(cfg) => schur::sampled(&self.matrix, cfg),
            SchurMode::Exact => schur::exact_for_factor(&self.matrix, &self.inv_diagonal),
        };
        let ac = config
            .approx_chol
            .to_approx_chol(exact_below, ExactFailure::FallBackToApproximate);
        factor_sparse(&schur_csr, ac).map_err(local_solver_build)
    }
}

/// Gremban cover: SDDM, and acts on the antisymmetric `[z, -z]` subspace as the original.
fn assemble_bipartite_cover(matrix: &SddmMatrix) -> SddmMatrix {
    let c = &matrix.cross_tab.c;
    let n_rows = c.nrows;
    let n_cols = c.ncols;
    let n_cols_u32 = u32::try_from(n_cols).expect("cover columns exceed u32::MAX");
    // Both doubled sizes must fit u32: the cross-sheet shift emits columns up to `2*n_cols - 1`.
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
            // Same-sheet columns precede cross-sheet ones so each output row stays column-sorted.
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
    SddmMatrix {
        cross_tab: CrossTab {
            c: cover_c,
            ct: cover_ct,
        },
        diagonal: double_for_cover(&matrix.diagonal, n_rows),
        ground_edges: double_for_cover(&matrix.ground_edges, n_rows),
        grounding: matrix.grounding,
    }
}

/// Each Gremban sheet copies every vertex, so a per-vertex array doubles within each side.
fn double_for_cover(values: &[f64], n_rows: usize) -> Vec<f64> {
    let (rows, cols) = values.split_at(n_rows);
    [rows, rows, cols, cols].concat()
}

impl BlockElimSolver {
    /// Size of the block the elimination removes, the leading one.
    fn n_eliminated(&self) -> usize {
        self.cross_tab.n_rows()
    }

    /// Size of the block the reduced factor solves.
    fn n_kept(&self) -> usize {
        self.cross_tab.n_cols()
    }

    pub(crate) fn new(
        cross_tab: impl Into<Arc<CrossTab>>,
        inv_diag_elim: Vec<f64>,
        reduced_factor: ReducedFactor,
        coordinates: CoordinateMap,
    ) -> Self {
        let cross_tab = cross_tab.into();
        debug_assert!(reduced_factor.spans_kept_block(cross_tab.n_cols()));
        let n_internal = cross_tab.n_local();
        let n_reduced = reduced_factor.solve_dimension();
        Self {
            cross_tab,
            inv_diag_elim,
            reduced_factor,
            n_internal,
            n_reduced,
            coordinates,
        }
    }

    /// A signed component keeps only the cover's factor, keeping the matrix single-sized (#91).
    pub(crate) fn build(
        component: LocalComponent,
        config: &LocalSolverConfig,
    ) -> Result<Self, BuildError> {
        let LocalComponent {
            matrix,
            form,
            coordinates,
        } = component;
        let eliminated = Eliminated::new(matrix)?;

        let factor = match form {
            MatrixForm::Laplacian => {
                let factor = ReducedFactor::Direct {
                    factor: eliminated.factor_reduced(config)?,
                    grounding: eliminated.matrix.grounding,
                };
                debug_assert!(factor.solve_dimension() >= factor.input_dimension());
                factor
            }
            // Surplus survives the cover, so it grounds as the signed matrix did.
            MatrixForm::SignedPendingCover => {
                let cover = eliminated.cover()?;
                ReducedFactor::Cover {
                    inner: cover.factor_reduced(config)?,
                    m: eliminated.matrix.n_kept(),
                }
            }
        };

        Ok(BlockElimSolver::new(
            eliminated.matrix.cross_tab,
            eliminated.inv_diagonal,
            factor,
            coordinates,
        ))
    }

    /// Components arrive oriented, so the row block is the eliminated side.
    fn eliminate_and_recover(
        &self,
        rhs: &mut [f64],
        sol: &mut [f64],
        allow_inner_parallelism: bool,
    ) -> Result<(), LocalSolveError> {
        let n = self.n_internal;
        let (n_elim, n_keep) = (self.n_eliminated(), self.n_kept());
        let explicit_ground = self.reduced_factor.explicit_ground_index(n_keep);

        scale_by_diag_in_place(&mut rhs[..n_elim], &self.inv_diag_elim);

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
            // The ground node's potential is the gauge subtracted after the solve.
            Some(Grounding::Grounded) => {
                if let Some(ground) = explicit_ground {
                    rhs[n + ground] = -compensated_sum(&rhs[n..n + n_keep]);
                }
            }
            // A cover grounds itself via the antisymmetric `[b, -b]` embed.
            None => {}
        }

        // The reduced solve spills into slots that are dead except the grounded gauge.
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

        // Back-substitution reads the gauged kept block, so the gauge subtraction runs first.
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
        let n_eliminated = self.n_eliminated();

        self.coordinates.fold(&mut rhs[..n], n_eliminated);
        if self.reduced_factor.grounding() == Some(Grounding::Floating) {
            subtract_mean(rhs, n);
        }

        self.eliminate_and_recover(rhs, sol, allow_inner_parallelism)?;

        if self.reduced_factor.grounding() == Some(Grounding::Floating) {
            subtract_mean(sol, n);
        }
        self.coordinates.unfold(&mut sol[..n], n_eliminated);
        Ok(())
    }
}

#[cfg(test)]
mod tests;
