//! Schur complement of the eliminated diagonal block in a bipartite SDDM
//! system — SDDM in, SDDM on the kept levels out.
//!
//! [`exact`] accumulates the true reduced rows; [`sampled`] approximates each
//! eliminated star's clique on the explicit augmented graph.

#[cfg(test)]
use super::csr_matrix::CsrMatrix;
use approx_chol::low_level::CliqueTreeSampler;
use approx_chol::{Laplacian, Sddm};
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

struct UpperSchurRow {
    columns: Vec<u32>,
    values: Vec<f64>,
}

/// Multiply-adds below which splitting the kept rows costs more than it saves.
const PAR_ROW_SPLIT_THRESHOLD: usize = 100_000;

/// Exact Schur complement as a mirrored CSR, which only the tests compare against.
#[cfg(test)]
pub(crate) fn exact(
    matrix: &SddmMatrix,
    inv_diagonal_eliminated: &[f64],
    split: RowSplit,
) -> CsrMatrix {
    assemble_schur_csr(
        exact_rows(matrix, inv_diagonal_eliminated, split),
        matrix.n_kept(),
    )
}

/// Exact Schur rows, diagonal and upper, accumulated without materializing intermediate edges.
fn exact_rows(
    matrix: &SddmMatrix,
    inv_diagonal_eliminated: &[f64],
    split: RowSplit,
) -> Vec<UpperSchurRow> {
    let n_keep = matrix.n_kept();
    // `extract_sparse_row` zeroes what it read, so one workspace serves a whole run.
    let row = |i: usize, work: &mut Vec<f64>, touched: &mut Vec<usize>| {
        compute_schur_row_dense(matrix, inv_diagonal_eliminated, i, work, touched);
        let result = extract_sparse_row(i, work, touched);
        touched.clear();
        result
    };

    let rows: Vec<UpperSchurRow> = match split {
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
    rows
}

/// Sampled Schur complement; fill reaching the ground pseudo-vertex is the kept rows' surplus.
pub(crate) fn sampled(matrix: &SddmMatrix, config: &ApproxSchurConfig) -> Sddm {
    let edges = par_emit(matrix, config);
    let n_kept = matrix.n_kept();
    match matrix.grounding {
        Grounding::Floating => sampled_laplacian(&edges, n_kept, |_, _| {
            unreachable!("a floating matrix has no ground pseudo-vertex to fill toward")
        })
        .into(),
        Grounding::Grounded => {
            let mut surplus = vec![0.0; n_kept];
            let laplacian = sampled_laplacian(&edges, n_kept, |row, weight| surplus[row] += weight);
            // Surplus can miss every kept row, which leaves the reduced system floating.
            Sddm::with_surplus(laplacian, surplus).expect("sampled surplus is non-negative")
        }
    }
}

/// `ground` receives each edge into vertex `n_kept`, which is not part of the Laplacian.
fn sampled_laplacian(
    edges: &[Edge],
    n_kept: usize,
    mut ground: impl FnMut(usize, f64),
) -> Laplacian {
    let mut row_ptrs = vec![0u32; n_kept + 1];
    let mut neighbors = Vec::with_capacity(edges.len());
    let mut weights = Vec::with_capacity(edges.len());
    // Sorted by `(lo, hi)`, so each row's neighbors arrive ascending.
    for &(lo, hi, weight) in edges {
        if hi as usize == n_kept {
            ground(lo as usize, weight);
            continue;
        }
        neighbors.push(hi);
        weights.push(weight);
        row_ptrs[lo as usize + 1] += 1;
    }
    for i in 0..n_kept {
        row_ptrs[i + 1] += row_ptrs[i];
    }
    Laplacian::new(row_ptrs, neighbors, weights).expect("sampled Schur is a Laplacian")
}

pub(crate) fn exact_for_factor(matrix: &SddmMatrix, inv_diagonal_eliminated: &[f64]) -> Sddm {
    let split = if exact_flops(matrix) < PAR_ROW_SPLIT_THRESHOLD {
        RowSplit::Sequential
    } else {
        RowSplit::Parallel
    };
    let rows = exact_rows(matrix, inv_diagonal_eliminated, split);
    let mut row_ptrs = Vec::with_capacity(rows.len() + 1);
    let mut neighbors = Vec::new();
    let mut weights = Vec::new();
    row_ptrs.push(0u32);
    for (i, row) in rows.iter().enumerate() {
        for (&j, &value) in row.columns.iter().zip(&row.values) {
            if j as usize != i {
                neighbors.push(j);
                weights.push(-value);
            }
        }
        row_ptrs.push(to_u32(neighbors.len()));
    }
    let laplacian =
        Laplacian::new(row_ptrs, neighbors, weights).expect("exact Schur is a Laplacian");
    match matrix.grounding {
        Grounding::Floating => laplacian.into(),
        Grounding::Grounded => {
            Sddm::with_surplus(laplacian, reduced_surplus(matrix, inv_diagonal_eliminated))
                .expect("reduced surplus is non-negative")
        }
    }
}

/// Upper-triangle multiply-adds: `Σ_k nnz(k)(nnz(k)+1)/2`.
fn exact_flops(matrix: &SddmMatrix) -> usize {
    matrix
        .cross_tab
        .c
        .indptr
        .windows(2)
        .map(|bounds| {
            let width = (bounds[1] - bounds[0]) as usize;
            width * (width + 1) / 2
        })
        .sum()
}

/// Scatter the diagonal-and-upper part of Schur row `i`, recording touched columns.
fn compute_schur_row_dense(
    matrix: &SddmMatrix,
    inv_diagonal_eliminated: &[f64],
    i: usize,
    work: &mut [f64],
    touched: &mut Vec<usize>,
) {
    work[i] = matrix.diagonal_kept()[i];
    touched.push(i);

    let elim_to_keep = &matrix.cross_tab.c;
    for (k, w) in matrix.cross_tab.ct().row(i) {
        let inv = inv_diagonal_eliminated[k];
        let start = elim_to_keep.indptr[k] as usize;
        let end = elim_to_keep.indptr[k + 1] as usize;
        let columns = &elim_to_keep.indices[start..end];
        let upper_start = columns.partition_point(|&j| (j as usize) < i);
        for position in start + upper_start..end {
            let j = elim_to_keep.indices[position] as usize;
            let v = elim_to_keep.data[position];
            if work[j] == 0.0 && j != i {
                touched.push(j);
            }
            work[j] -= w * v * inv;
        }
    }
}

/// Extract non-zeros, keeping a numerically-zero diagonal so SDDM structure survives.
fn extract_sparse_row(i: usize, work: &mut [f64], touched: &mut [usize]) -> UpperSchurRow {
    touched.sort_unstable();
    let mut columns = Vec::with_capacity(touched.len());
    let mut values = Vec::with_capacity(touched.len());
    for &j in touched.iter() {
        let v = work[j];
        if v != 0.0 || j == i {
            columns.push(to_u32(j));
            values.push(v);
        }
        work[j] = 0.0;
    }
    UpperSchurRow { columns, values }
}

/// Assemble full rows from diagonal-and-upper parts; the strict upper mirrors into the lower.
#[cfg(test)]
fn assemble_schur_csr(rows: Vec<UpperSchurRow>, n_keep: usize) -> CsrMatrix {
    let mut row_offsets = vec![0u32; n_keep + 1];
    for (i, row) in rows.iter().enumerate() {
        row_offsets[i + 1] += to_u32(row.columns.len());
        for &j in &row.columns {
            if j as usize != i {
                row_offsets[j as usize + 1] += 1;
            }
        }
    }
    for i in 0..n_keep {
        row_offsets[i + 1] += row_offsets[i];
    }
    let total_nnz = row_offsets[n_keep] as usize;
    let mut column_indices = vec![0u32; total_nnz];
    let mut values = vec![0.0f64; total_nnz];

    // Row layout: [mirrored lower | diagonal and upper]; ascending source rows keep it sorted.
    let mut lower_write_positions = row_offsets[..n_keep].to_vec();
    for (i, row) in rows.into_iter().enumerate() {
        let upper_start = row_offsets[i + 1] as usize - row.columns.len();
        column_indices[upper_start..upper_start + row.columns.len()].copy_from_slice(&row.columns);
        values[upper_start..upper_start + row.values.len()].copy_from_slice(&row.values);
        for (&j, &v) in row.columns.iter().zip(&row.values) {
            if j as usize != i {
                let position = lower_write_positions[j as usize] as usize;
                column_indices[position] = to_u32(i);
                values[position] = v;
                lower_write_positions[j as usize] += 1;
            }
        }
    }
    CsrMatrix::new(row_offsets, column_indices, values, n_keep)
}

fn reduced_surplus(matrix: &SddmMatrix, inv_diagonal_eliminated: &[f64]) -> Vec<f64> {
    let scaled: Vec<f64> = inv_diagonal_eliminated
        .iter()
        .zip(matrix.surplus_eliminated())
        .map(|(&inv_diag, &surplus)| inv_diag * surplus)
        .collect();
    let mut surplus = vec![0.0; matrix.n_kept()];
    matrix
        .cross_tab
        .ct()
        .spmv_assign_add(&scaled, matrix.surplus_kept(), &mut surplus, false);
    surplus
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
    let mut chunks = (0..matrix.n_eliminated())
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
        .collect::<Vec<_>>();
    if let Some(ground) = ground_vertex {
        chunks.push(
            matrix
                .surplus_kept()
                .iter()
                .enumerate()
                .filter(|(_, surplus)| **surplus > 0.0)
                .map(|(i, &surplus)| (i as u32, ground, surplus))
                .collect(),
        );
    }
    sort_and_dedup(chunks, n_kept)
}

/// Sort into total `(lo, hi, weight)` order; the weight tiebreak makes the Schur reproducible.
fn sort_and_dedup(chunks: Vec<Vec<Edge>>, n_kept: usize) -> Vec<Edge> {
    let total = chunks.iter().map(Vec::len).sum::<usize>();
    if total <= 1 {
        return chunks.concat();
    }
    // Same order either way; under two edges per kept vertex the parallel comparison sort wins.
    let dense = total >= 2 * n_kept;
    let mut edges = if dense {
        counting_sort_by_lo(&chunks, n_kept)
    } else {
        chunks.concat()
    };
    drop(chunks);
    if dense {
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
    edges
}

/// Stable counting sort by `lo` in O(E + n_kept), scattered once from the per-task chunks.
fn counting_sort_by_lo(chunks: &[Vec<Edge>], n_kept: usize) -> Vec<Edge> {
    let mut cursors = vec![0usize; n_kept + 1];
    // Nested loops, not `flatten()`: the flat iterator cost ~10% on both passes at 1M+ edges.
    for chunk in chunks {
        for e in chunk {
            // `lo < n_kept` always holds: the ground vertex has the maximum id and lands on `hi`.
            debug_assert!(
                (e.0 as usize) < n_kept,
                "counting sort key `lo` must be a kept-block id (< n_kept)"
            );
            cursors[e.0 as usize + 1] += 1;
        }
    }
    for i in 1..cursors.len() {
        cursors[i] += cursors[i - 1];
    }
    let mut out = vec![(0u32, 0u32, 0.0f64); cursors[n_kept]];
    for chunk in chunks {
        for &e in chunk {
            let cursor = &mut cursors[e.0 as usize];
            out[*cursor] = e;
            *cursor += 1;
        }
    }
    out
}

#[cfg(test)]
mod tests;
