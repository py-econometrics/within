//! Star iteration and sampled fill edges for Schur complement assembly.
//!
//! For the bipartite SDDM `[D_e, -C; -C^T, D_k]`, eliminating the larger
//! diagonal block (exact since it's diagonal) yields a reduced SDDM system on
//! the smaller block. Components arrive eliminated-major, so the block choice
//! is already made; this module owns the inverse-diagonal fold, zero-copy
//! [`Star`] views, and the sampled fill edges on the explicit augmented graph.

use approx_chol::low_level::{clique_tree_sample, clique_tree_sample_multi};
use rayon::prelude::*;

use crate::config::ApproxSchurConfig;
use crate::domain::{Grounding, SddmMatrix};
use crate::BuildError;

/// Undirected fill edge: `(lo_col, hi_col, weight)` with `lo_col < hi_col`.
pub(crate) type Edge = (u32, u32, f64);

// ===========================================================================
// Star — zero-copy neighborhood view
// ===========================================================================

// Eliminating a diagonal vertex contributes a rank-1 clique (star) to the
// Schur fill graph: every pair of its keep-block neighbors gets a fill edge.
// `sample_star` approximates high-degree stars with GKS 2023 clique-tree
// sampling — O(deg) edges instead of O(deg^2) — keeping the Schur complement
// spectrally close to the exact (row-workspace) path.

/// One eliminated vertex's neighbors in the keep-block.
///
/// References into the cross-tab's CSR arrays for zero-copy access.
pub(crate) struct Star<'a> {
    /// Eliminated vertex index (used for deterministic seeding).
    index: usize,
    /// Neighbor columns in the keep-block.
    col_indices: &'a [u32],
    /// Edge weights to each neighbor.
    weights: &'a [f64],
}

impl Star<'_> {
    pub(crate) fn degree(&self) -> usize {
        self.col_indices.len()
    }
}

/// Sample clique-tree fill edges for one eliminated star.
///
/// Ground surplus is one more incident edge, so the sampled star's capacity is
/// exactly its eliminated diagonal.
fn sample_star(
    star: &Star,
    ground: Option<(u32, f64)>,
    config: &ApproxSchurConfig,
    edges: &mut Vec<Edge>,
    scratch: &mut Vec<(u32, f64)>,
) {
    scratch.clear();
    for (&col, &w) in star.col_indices.iter().zip(star.weights) {
        scratch.push((col, w));
    }
    if let Some((ground, surplus)) = ground {
        if surplus > 0.0 {
            scratch.push((ground, surplus));
        }
    }
    if scratch.len() <= 1 {
        return;
    }
    let seed = config.seed.wrapping_add(star.index as u64);
    if config.split <= 1 {
        clique_tree_sample(scratch, seed, edges);
    } else {
        clique_tree_sample_multi(scratch, config.split, seed, edges);
    }
}

// ===========================================================================
// Star iteration over the eliminated block
// ===========================================================================

/// Fold the eliminated block's diagonal to its reciprocals, the one value
/// Schur assembly needs that the matrix does not already carry.
pub(crate) fn invert_eliminated_diagonal(matrix: &SddmMatrix) -> Result<Vec<f64>, BuildError> {
    debug_assert!(
        matrix.n_eliminated() >= matrix.n_kept(),
        "component is not eliminated-major"
    );
    matrix.diagonal[..matrix.n_eliminated()]
        .iter()
        .enumerate()
        .map(|(i, &d)| {
            if d > 0.0 {
                Ok(1.0 / d)
            } else {
                Err(BuildError::SingularDiagonal { index: i })
            }
        })
        .collect()
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

pub(crate) fn par_emit(matrix: &SddmMatrix, config: &ApproxSchurConfig) -> Vec<Edge> {
    // Emit in parallel (one scratch buffer reused per fold chunk), concatenate,
    // then a single total-order `sort_and_dedup`. The total order fixes the
    // per-`(lo, hi)` weight summation order, so the result is independent of
    // thread scheduling (the concatenation order no longer matters).
    let n_kept = matrix.n_kept();
    let surplus_eliminated = matrix.surplus_eliminated();
    let ground_vertex = (matrix.grounding == Grounding::Grounded)
        .then(|| u32::try_from(n_kept).expect("ground vertex exceeds u32::MAX"));
    let mut edges = (0..matrix.n_eliminated())
        .into_par_iter()
        .fold(
            || (Vec::new(), Vec::<(u32, f64)>::new()),
            |(mut edges, mut scratch), k| {
                let star = star(matrix, k);
                if star.degree() > 0 {
                    let ground = ground_vertex.map(|g| (g, surplus_eliminated[k]));
                    sample_star(&star, ground, config, &mut edges, &mut scratch);
                }
                (edges, scratch)
            },
        )
        .map(|(edges, _)| edges)
        .reduce(Vec::new, |mut a, mut b| {
            a.append(&mut b);
            a
        });
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

/// Sort edges into a total `(lo, hi, weight)` order and merge duplicates by
/// summing weights. The weight tiebreak fixes the per-`(lo, hi)` summation
/// order, making the assembled Schur complement reproducible across runs and
/// thread counts.
fn sort_and_dedup(edges: &mut Vec<Edge>, n_kept: usize) {
    if edges.len() <= 1 {
        return;
    }
    // Dense: linear counting sort by `lo` + parallel per-`lo`-run sorts by
    // `(hi, weight)`. Sparse: one parallel comparison sort. Same total order.
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
        // `lo < n_kept` always holds: the ground vertex (the only id that can
        // equal `n_kept`) carries the maximum id and lands on `hi`.
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
