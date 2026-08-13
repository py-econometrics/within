//! Schur complement of the eliminated diagonal block in a bipartite SDDM
//! system — SDDM in, SDDM on the kept levels out.
//!
//! [`exact`] accumulates the true reduced rows; [`sampled`] approximates each
//! eliminated star's clique on the explicit augmented graph.

use super::compensated_sum;
use super::csr_matrix::CsrMatrix;
use approx_chol::low_level::CliqueTreeSampler;
use rayon::prelude::*;

use crate::config::ApproxSchurConfig;
use crate::csr_block::to_u32;
use crate::domain::{Grounding, SddmMatrix};

/// Whether the kept rows are split across threads. The subdomain loop reaching here is
/// already parallel, so splitting pays only once the reduction itself is large enough to
/// outweigh the tasks it adds for that loop to steal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RowSplit {
    Sequential,
    Parallel,
}

/// Multiply-adds below which splitting the kept rows costs more than it saves.
const PAR_ROW_SPLIT_THRESHOLD: usize = 100_000;

/// Exact Schur complement, accumulated per keep-row without materializing intermediate edges.
pub(crate) fn exact(
    matrix: &SddmMatrix,
    inv_diagonal_eliminated: &[f64],
    split: RowSplit,
) -> CsrMatrix {
    let n_keep = matrix.n_kept();
    // `extract_sparse_row` zeroes what it read, so one workspace serves a whole run.
    let row = |i: usize, work: &mut Vec<f64>, touched: &mut Vec<usize>| {
        compute_schur_row_dense(matrix, inv_diagonal_eliminated, i, work, touched);
        let result = extract_sparse_row(i, work, touched);
        touched.clear();
        result
    };

    let rows: Vec<(Vec<u32>, Vec<f64>)> = match split {
        RowSplit::Parallel => (0..n_keep)
            .into_par_iter()
            .map_init(
                || (vec![0.0f64; n_keep], Vec::new()),
                |(work, touched), i| row(i, work, touched),
            )
            .collect(),
        RowSplit::Sequential => {
            let mut work = vec![0.0f64; n_keep];
            let mut touched = Vec::new();
            (0..n_keep)
                .map(|i| row(i, &mut work, &mut touched))
                .collect()
        }
    };

    assemble_schur_csr(rows, n_keep)
}

/// Sampled Schur complement as a Laplacian; a ground vertex stays an ordinary final vertex.
pub(crate) fn sampled(matrix: &SddmMatrix, config: &ApproxSchurConfig) -> CsrMatrix {
    let edges = par_emit(matrix, config);
    let n = matrix.n_kept() + usize::from(matrix.grounding == Grounding::Grounded);
    build_laplacian_csr(&edges, n)
}

pub(crate) fn exact_for_factor(matrix: &SddmMatrix, inv_diagonal_eliminated: &[f64]) -> CsrMatrix {
    let split = if exact_flops(matrix) < PAR_ROW_SPLIT_THRESHOLD {
        RowSplit::Sequential
    } else {
        RowSplit::Parallel
    };
    let principal = exact(matrix, inv_diagonal_eliminated, split);
    let surplus = reduced_surplus(matrix, inv_diagonal_eliminated);
    build_explicit_laplacian(&principal, &surplus, matrix.grounding)
}

/// Multiply-adds [`exact`] performs: `Σ_k nnz(k)²`, quadratic in a width `n_kept` cannot see.
fn exact_flops(matrix: &SddmMatrix) -> usize {
    matrix
        .cross_tab
        .c
        .indptr
        .windows(2)
        .map(|bounds| {
            let width = (bounds[1] - bounds[0]) as usize;
            width * width
        })
        .sum()
}

/// Scatter Schur row `i` into a dense workspace, recording touched columns.
fn compute_schur_row_dense(
    matrix: &SddmMatrix,
    inv_diagonal_eliminated: &[f64],
    i: usize,
    work: &mut [f64],
    touched: &mut Vec<usize>,
) {
    work[i] = matrix.diagonal_kept()[i];
    touched.push(i);

    for (k, w) in matrix.cross_tab.ct.row(i) {
        let inv = inv_diagonal_eliminated[k];
        for (j, v) in matrix.cross_tab.c.row(k) {
            if work[j] == 0.0 && j != i {
                touched.push(j);
            }
            // Pairing the loadings before the reciprocal makes (i, j) and (j, i) bitwise equal.
            work[j] -= (w * v) * inv;
        }
    }
}

/// Extract non-zeros, keeping a numerically-zero diagonal so SDDM structure survives.
fn extract_sparse_row(i: usize, work: &mut [f64], touched: &mut [usize]) -> (Vec<u32>, Vec<f64>) {
    touched.sort_unstable();
    let mut row_indices = Vec::with_capacity(touched.len());
    let mut row_data = Vec::with_capacity(touched.len());
    for &j in touched.iter() {
        let v = work[j];
        if v != 0.0 || j == i {
            row_indices.push(to_u32(j));
            row_data.push(v);
        }
        work[j] = 0.0;
    }
    (row_indices, row_data)
}

/// Assemble a CSR matrix from per-row sparse results.
fn assemble_schur_csr(rows: Vec<(Vec<u32>, Vec<f64>)>, n_keep: usize) -> CsrMatrix {
    let mut s_indptr = vec![0u32; n_keep + 1];
    let total_nnz: usize = rows.iter().map(|(ri, _)| ri.len()).sum();
    let mut s_indices = Vec::with_capacity(total_nnz);
    let mut s_data = Vec::with_capacity(total_nnz);
    for (i, (ri, rd)) in rows.into_iter().enumerate() {
        s_indices.extend_from_slice(&ri);
        s_data.extend_from_slice(&rd);
        s_indptr[i + 1] = to_u32(s_indices.len());
    }
    CsrMatrix::new(s_indptr, s_indices, s_data, n_keep)
}

/// Edges sorted by `(lo, hi)` land both triangles in column order without per-row sorting.
fn build_laplacian_csr(edges: &[Edge], n: usize) -> CsrMatrix {
    debug_assert!(edges.iter().all(|&(lo, hi, _)| lo < hi));

    let mut lower_count = vec![0u32; n];
    let mut upper_count = vec![0u32; n];
    let mut diag = vec![0.0; n];
    for &(lo, hi, w) in edges {
        diag[lo as usize] += w;
        upper_count[lo as usize] += 1; // row lo gets col hi (upper)
        lower_count[hi as usize] += 1; // row hi gets col lo (lower)
        diag[hi as usize] += w;
    }

    // Row layout: [lower entries | diagonal | upper entries].
    let mut offsets = vec![0u32; n + 1];
    for i in 0..n {
        offsets[i + 1] = offsets[i] + lower_count[i] + 1 + upper_count[i];
    }
    let total_nnz = offsets[n] as usize;
    let mut indices = vec![0u32; total_nnz];
    let mut data = vec![0.0f64; total_nnz];

    let mut lower_cursor: Vec<u32> = (0..n).map(|i| offsets[i]).collect();
    let mut upper_cursor: Vec<u32> = (0..n).map(|i| offsets[i] + lower_count[i] + 1).collect();
    for i in 0..n {
        let pos = (offsets[i] + lower_count[i]) as usize;
        indices[pos] = to_u32(i);
        data[pos] = diag[i];
    }

    for &(lo, hi, w) in edges {
        let lo_idx = lo as usize;
        let hi_idx = hi as usize;
        // Upper triangle: row lo, column hi.
        let pos = upper_cursor[lo_idx] as usize;
        indices[pos] = hi;
        data[pos] = -w;
        upper_cursor[lo_idx] += 1;
        // Lower triangle: row hi, column lo.
        let pos = lower_cursor[hi_idx] as usize;
        indices[pos] = lo;
        data[pos] = -w;
        lower_cursor[hi_idx] += 1;
    }

    CsrMatrix::new(offsets, indices, data, n)
}

fn reduced_surplus(matrix: &SddmMatrix, inv_diagonal_eliminated: &[f64]) -> Vec<f64> {
    if matrix.grounding == Grounding::Floating {
        return vec![0.0; matrix.n_kept()];
    }
    let scaled: Vec<f64> = inv_diagonal_eliminated
        .iter()
        .zip(matrix.surplus_eliminated())
        .map(|(&inv_diag, &surplus)| inv_diag * surplus)
        .collect();
    let mut surplus = vec![0.0; matrix.n_kept()];
    matrix
        .cross_tab
        .ct
        .spmv_assign_add(&scaled, matrix.surplus_kept(), &mut surplus, false);
    surplus
}

pub(super) fn build_explicit_laplacian(
    principal: &CsrMatrix,
    surplus: &[f64],
    grounding: Grounding,
) -> CsrMatrix {
    let n_keep = principal.n();
    let ground = to_u32(n_keep);
    let grounded = grounding == Grounding::Grounded;
    let n = n_keep + usize::from(grounded);
    let mut indptr = Vec::with_capacity(n + 1);
    let mut indices = Vec::new();
    let mut data = Vec::new();
    indptr.push(0);

    for (i, &row_surplus) in surplus.iter().enumerate().take(n_keep) {
        let start = principal.indptr()[i] as usize;
        let end = principal.indptr()[i + 1] as usize;
        let mut adjacency = 0.0;
        let mut diagonal_position = None;
        for (&j, &value) in principal.indices()[start..end]
            .iter()
            .zip(&principal.data()[start..end])
        {
            if j as usize == i {
                diagonal_position = Some(indices.len());
                indices.push(j);
                data.push(0.0);
            } else {
                adjacency -= value;
                indices.push(j);
                data.push(value);
            }
        }
        let diagonal_position = diagonal_position.expect("exact Schur row must contain a diagonal");
        data[diagonal_position] = adjacency + row_surplus;
        if grounded && row_surplus > 0.0 {
            indices.push(ground);
            data.push(-row_surplus);
        }
        indptr.push(to_u32(indices.len()));
    }

    if grounded {
        for (i, &value) in surplus.iter().enumerate() {
            if value > 0.0 {
                indices.push(to_u32(i));
                data.push(-value);
            }
        }
        indices.push(ground);
        data.push(compensated_sum(surplus));
        indptr.push(to_u32(indices.len()));
    }
    CsrMatrix::new(indptr, indices, data, n)
}

/// Undirected fill edge: `(lo_col, hi_col, weight)` with `lo_col < hi_col`.
type Edge = (u32, u32, f64);

/// One eliminated vertex's keep-block neighbors, referencing the cross-tab CSR for zero copy.
struct Star<'a> {
    /// Eliminated vertex index (used for deterministic seeding).
    index: usize,
    /// Neighbor columns in the keep-block.
    col_indices: &'a [u32],
    /// Edge weights to each neighbor.
    weights: &'a [f64],
}

impl Star<'_> {
    fn degree(&self) -> usize {
        self.col_indices.len()
    }
}

/// One rayon task's sampling state. The sampler carries scratch of its own, so it is
/// built per task alongside the buffers rather than per star.
struct StarWorkspace {
    edges: Vec<Edge>,
    entries: Vec<(u32, f64)>,
    sampler: CliqueTreeSampler,
}

impl StarWorkspace {
    fn new(config: &ApproxSchurConfig) -> Self {
        Self {
            edges: Vec::new(),
            entries: Vec::new(),
            sampler: CliqueTreeSampler::new(config.seed, Some(config.split)),
        }
    }

    /// Ground surplus is one more incident edge, so a sampled star's capacity is its diagonal.
    fn sample_star(&mut self, star: &Star, ground: Option<(u32, f64)>) {
        self.entries.clear();
        for (&col, &w) in star.col_indices.iter().zip(star.weights) {
            self.entries.push((col, w));
        }
        if let Some((ground, surplus)) = ground {
            if surplus > 0.0 {
                self.entries.push((ground, surplus));
            }
        }
        if self.entries.len() <= 1 {
            return;
        }
        self.sampler
            .sample(star.index as u64, &self.entries, &mut self.edges);
    }
}

/// Create a zero-copy [`Star`] view for eliminated vertex `k`.
fn star(matrix: &SddmMatrix, k: usize) -> Star<'_> {
    let elim_to_keep = &matrix.cross_tab.c;
    let start = elim_to_keep.indptr[k] as usize;
    let end = elim_to_keep.indptr[k + 1] as usize;
    Star {
        index: k,
        col_indices: &elim_to_keep.indices[start..end],
        weights: &elim_to_keep.data[start..end],
    }
}

fn par_emit(matrix: &SddmMatrix, config: &ApproxSchurConfig) -> Vec<Edge> {
    // The total order fixes per-`(lo, hi)` summation, so the result is scheduling-independent.
    let n_kept = matrix.n_kept();
    let surplus_eliminated = matrix.surplus_eliminated();
    let ground_vertex = (matrix.grounding == Grounding::Grounded)
        .then(|| u32::try_from(n_kept).expect("ground vertex exceeds u32::MAX"));
    let mut edges = (0..matrix.n_eliminated())
        .into_par_iter()
        .fold(
            || StarWorkspace::new(config),
            |mut work, k| {
                let star = star(matrix, k);
                if star.degree() > 0 {
                    let ground = ground_vertex.map(|g| (g, surplus_eliminated[k]));
                    work.sample_star(&star, ground);
                }
                work
            },
        )
        .map(|work| work.edges)
        // Pre-sized single copy; a `reduce` tree of `append`s recopies each edge per level.
        .collect::<Vec<_>>()
        .concat();
    if let Some(ground) = ground_vertex {
        edges.extend(
            matrix
                .surplus_kept()
                .iter()
                .enumerate()
                .filter(|(_, surplus)| **surplus > 0.0)
                .map(|(i, &surplus)| (i as u32, ground, surplus)),
        );
    }
    sort_and_dedup(&mut edges, n_kept);
    edges
}

/// Sort into total `(lo, hi, weight)` order; the weight tiebreak makes the Schur reproducible.
fn sort_and_dedup(edges: &mut Vec<Edge>, n_kept: usize) {
    if edges.len() <= 1 {
        return;
    }
    // Dense counting sort or sparse comparison sort — both produce the same total order.
    if edges.len() >= n_kept {
        counting_sort_by_lo(edges, n_kept);
        let by_hi_weight = |a: &Edge, b: &Edge| a.1.cmp(&b.1).then_with(|| a.2.total_cmp(&b.2));
        edges
            .par_chunk_by_mut(|a, b| a.0 == b.0)
            .for_each(|run| run.sort_unstable_by(by_hi_weight));
    } else {
        edges.par_sort_unstable_by(|a, b| {
            a.0.cmp(&b.0)
                .then(a.1.cmp(&b.1))
                .then_with(|| a.2.total_cmp(&b.2))
        });
    }
    let mut write = 0;
    for read in 1..edges.len() {
        if edges[write].0 == edges[read].0 && edges[write].1 == edges[read].1 {
            edges[write].2 += edges[read].2;
        } else {
            write += 1;
            edges[write] = edges[read];
        }
    }
    edges.truncate(write + 1);
}

/// Stable counting sort of `edges` by `lo` in O(E + n_kept).
fn counting_sort_by_lo(edges: &mut Vec<Edge>, n_kept: usize) {
    let mut cursors = vec![0usize; n_kept + 1];
    for e in edges.iter() {
        // `lo < n_kept` always holds: the ground vertex carries the maximum id and lands on `hi`.
        debug_assert!(
            (e.0 as usize) < n_kept,
            "counting sort key `lo` must be a kept-block id (< n_kept)"
        );
        cursors[e.0 as usize + 1] += 1;
    }
    for i in 1..cursors.len() {
        cursors[i] += cursors[i - 1];
    }
    let mut out = vec![(0u32, 0u32, 0.0f64); edges.len()];
    for &e in edges.iter() {
        let cursor = &mut cursors[e.0 as usize];
        out[*cursor] = e;
        *cursor += 1;
    }
    *edges = out;
}

#[cfg(test)]
mod tests;
