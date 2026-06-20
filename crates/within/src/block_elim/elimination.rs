//! Block-elimination metadata and star iteration for Schur complement assembly.
//!
//! For the bipartite SDDM `[D_q, -C; -C^T, D_r]`, eliminating the larger
//! diagonal block (exact since it's diagonal) yields a reduced Laplacian-style
//! system on the smaller block. This module owns the block-selection decision,
//! precomputed inverse-diagonals, and zero-copy [`Star`] views used by both
//! Schur complement strategies.

use approx_chol::low_level::{clique_tree_sample, clique_tree_sample_multi};
use rayon::prelude::*;

use crate::config::ApproxSchurConfig;
use crate::csr_block::CsrBlock;
use crate::domain::{BlockDiagonals, CrossTab};
use crate::BuildError;

/// Undirected fill edge: `(lo_col, hi_col, weight)` with `lo_col < hi_col`.
pub(crate) type Edge = (u32, u32, f64);

// ===========================================================================
// Star — zero-copy neighborhood view
// ===========================================================================

// Eliminating a diagonal vertex contributes a rank-1 clique (star) to the
// Schur fill graph: every pair of its keep-block neighbors gets a fill edge.
// `SampledCliqueEmitter` approximates high-degree stars with GKS 2023
// clique-tree sampling — O(deg) edges instead of O(deg^2) — keeping the Schur
// complement spectrally close to the exact (row-workspace) path.

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

/// Emits sampled clique-tree fill edges for every star.
pub(crate) struct SampledCliqueEmitter {
    seed: u64,
    split: u32,
}

impl SampledCliqueEmitter {
    pub(crate) fn new(config: &ApproxSchurConfig) -> Self {
        Self {
            seed: config.seed,
            split: config.split,
        }
    }

    fn emit(&self, star: &Star, edges: &mut Vec<Edge>, scratch: &mut Vec<(u32, f64)>) {
        scratch.clear();
        for (&col, &w) in star.col_indices.iter().zip(star.weights) {
            scratch.push((col, w));
        }
        let seed = self.seed.wrapping_add(star.index as u64);
        if self.split <= 1 {
            clique_tree_sample(scratch, seed, edges);
        } else {
            clique_tree_sample_multi(scratch, self.split, seed, edges);
        }
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

    pub(crate) fn par_emit(&self, emitter: &SampledCliqueEmitter) -> Vec<Edge> {
        // Emit in parallel (one scratch buffer reused per fold chunk), concatenate,
        // then a single total-order `sort_and_dedup`. The total order fixes the
        // per-`(lo, hi)` weight summation order, so the result is independent of
        // thread scheduling (the concatenation order no longer matters).
        let mut edges = (0..self.n_elim)
            .into_par_iter()
            .fold(
                || (Vec::new(), Vec::<(u32, f64)>::new()),
                |(mut edges, mut scratch), k| {
                    let star = self.star(k);
                    if star.degree() > 1 {
                        emitter.emit(&star, &mut edges, &mut scratch);
                    }
                    (edges, scratch)
                },
            )
            .map(|(edges, _)| edges)
            .reduce(Vec::new, |mut a, mut b| {
                a.append(&mut b);
                a
            });
        Self::sort_and_dedup(&mut edges);
        edges
    }

    /// Sort edges into a total `(lo, hi, weight)` order and merge duplicates by
    /// summing weights. The weight tiebreak fixes the per-`(lo, hi)` summation
    /// order, making the assembled Schur complement reproducible across runs and
    /// thread counts.
    fn sort_and_dedup(edges: &mut Vec<Edge>) {
        edges.sort_unstable_by(|a, b| {
            a.0.cmp(&b.0)
                .then(a.1.cmp(&b.1))
                .then_with(|| a.2.total_cmp(&b.2))
        });
        if edges.len() <= 1 {
            return;
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
}
