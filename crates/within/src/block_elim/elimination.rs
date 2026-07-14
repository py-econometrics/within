//! Block-elimination metadata and star iteration for Schur complement assembly.
//!
//! For the bipartite SDDM `[D_q, -C; -C^T, D_r]`, eliminating the larger
//! diagonal block (exact since it's diagonal) yields a reduced SDDM system on
//! the smaller block. This module owns the block-selection decision,
//! precomputed inverse-diagonals, zero-copy [`Star`] views, and sampled fill
//! edges on the explicit augmented graph.

use approx_chol::low_level::{clique_tree_sample, clique_tree_sample_multi};
use rayon::prelude::*;

use crate::config::ApproxSchurConfig;
use crate::csr_block::CsrBlock;
use crate::domain::{BlockDiagonals, CrossTab, GroundEdges, SolveSpace};
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
/// References into [`CsrBlock`]'s arrays for zero-copy access.
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
// Elimination — block selection + star iteration
// ===========================================================================

/// Block-selection decision and star iteration for Schur elimination.
///
/// Encapsulates which block to eliminate, precomputed inverse-diagonals,
/// and provides zero-copy [`Star`] views for each eliminated vertex.
pub(crate) struct Elimination<'a> {
    pub(crate) eliminate_q: bool,
    pub(crate) n_keep: usize,
    pub(crate) n_elim: usize,
    pub(crate) inv_diag_elim: Vec<f64>,
    pub(crate) diag_keep: &'a [f64],
    pub(crate) surplus_keep: &'a [f64],
    pub(crate) surplus_elim: &'a [f64],
    pub(crate) solve_space: SolveSpace,
    pub(crate) keep_to_elim: &'a CsrBlock,
    pub(crate) elim_to_keep: &'a CsrBlock,
}

impl<'a> Elimination<'a> {
    /// Select which block to eliminate and precompute inverse-diagonals.
    ///
    /// The diagonal blocks are build-time-only inputs ([`BlockDiagonals`]); they
    /// are folded into `inv_diag_elim` and borrowed as `diag_keep` here and not
    /// retained past the build.
    pub(crate) fn new(
        cross_tab: &'a CrossTab,
        diagonals: &'a BlockDiagonals,
        ground_edges: &'a GroundEdges,
        solve_space: SolveSpace,
    ) -> Result<Self, BuildError> {
        let n_q = cross_tab.n_q();
        let n_r = cross_tab.n_r();
        // Eliminate the larger block to minimize the reduced system size.
        let eliminate_q = n_q >= n_r;
        let (n_keep, n_elim) = if eliminate_q { (n_r, n_q) } else { (n_q, n_r) };

        let diag_elim = if eliminate_q {
            &diagonals.q
        } else {
            &diagonals.r
        };
        let inv_diag_elim = diag_elim
            .iter()
            .enumerate()
            .map(|(i, &d)| {
                if d > 0.0 {
                    Ok(1.0 / d)
                } else {
                    Err(BuildError::SingularDiagonal {
                        block: if eliminate_q { "q (elim)" } else { "r (elim)" },
                        index: i,
                    })
                }
            })
            .collect::<Result<_, _>>()?;

        let diag_keep = if eliminate_q {
            &diagonals.r
        } else {
            &diagonals.q
        };
        let (surplus_keep, surplus_elim) = if eliminate_q {
            (&ground_edges.r[..], &ground_edges.q[..])
        } else {
            (&ground_edges.q[..], &ground_edges.r[..])
        };

        let (keep_to_elim, elim_to_keep) = if eliminate_q {
            (&cross_tab.ct, &cross_tab.c)
        } else {
            (&cross_tab.c, &cross_tab.ct)
        };

        Ok(Self {
            eliminate_q,
            n_keep,
            n_elim,
            inv_diag_elim,
            diag_keep,
            surplus_keep,
            surplus_elim,
            solve_space,
            keep_to_elim,
            elim_to_keep,
        })
    }

    /// Create a zero-copy [`Star`] view for eliminated vertex `k`.
    fn star(&self, k: usize) -> Star<'_> {
        let start = self.elim_to_keep.indptr[k] as usize;
        let end = self.elim_to_keep.indptr[k + 1] as usize;
        Star {
            index: k,
            col_indices: &self.elim_to_keep.indices[start..end],
            weights: &self.elim_to_keep.data[start..end],
        }
    }

    pub(crate) fn par_emit(&self, config: &ApproxSchurConfig) -> Vec<Edge> {
        // Emit in parallel (one scratch buffer reused per fold chunk), concatenate,
        // then a single total-order `sort_and_dedup`. The total order fixes the
        // per-`(lo, hi)` weight summation order, so the result is independent of
        // thread scheduling (the concatenation order no longer matters).
        let ground_vertex = (self.solve_space == SolveSpace::Grounded)
            .then(|| u32::try_from(self.n_keep).expect("ground vertex exceeds u32::MAX"));
        let mut edges = (0..self.n_elim)
            .into_par_iter()
            .fold(
                || (Vec::new(), Vec::<(u32, f64)>::new()),
                |(mut edges, mut scratch), k| {
                    let star = self.star(k);
                    if star.degree() > 0 {
                        let ground = ground_vertex.map(|g| (g, self.surplus_elim[k]));
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
                self.surplus_keep
                    .iter()
                    .enumerate()
                    .filter(|(_, surplus)| **surplus > 0.0)
                    .map(|(i, &surplus)| (i as u32, ground, surplus)),
            );
        }
        self.sort_and_dedup(&mut edges);
        edges
    }

    /// Sort edges into a total `(lo, hi, weight)` order and merge duplicates by
    /// summing weights. The weight tiebreak fixes the per-`(lo, hi)` summation
    /// order, making the assembled Schur complement reproducible across runs and
    /// thread counts.
    fn sort_and_dedup(&self, edges: &mut Vec<Edge>) {
        if edges.len() <= 1 {
            return;
        }
        // Dense: linear counting sort by `lo` + parallel per-`lo`-run sorts by
        // `(hi, weight)`. Sparse: one parallel comparison sort. Same total order.
        if edges.len() >= self.n_keep {
            self.counting_sort_by_lo(edges);
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

    /// Stable counting sort of `edges` by `lo` in O(E + n_keep).
    fn counting_sort_by_lo(&self, edges: &mut Vec<Edge>) {
        let mut cursors = vec![0usize; self.n_keep + 1];
        for e in edges.iter() {
            // `lo < n_keep` always holds: the ground vertex (the only id that can
            // equal `n_keep`) carries the maximum id and lands on `hi`.
            debug_assert!(
                (e.0 as usize) < self.n_keep,
                "counting sort key `lo` must be a keep-block id (< n_keep)"
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
}
