//! Block-selection, star neighborhoods, and clique edge emission.

use approx_chol::low_level::clique_tree_sample;
use rayon::prelude::*;

use super::Edge;
use crate::operator::csr_block::CsrBlock;
use crate::operator::gramian::CrossTab;

// ---------------------------------------------------------------------------
// EliminationInfo — public output
// ---------------------------------------------------------------------------

/// Metadata from the block-elimination step needed by `BlockElimSolver`.
pub(crate) struct EliminationInfo {
    /// 1 / D_elim[k] for the eliminated diagonal block.
    pub inv_diag_elim: Vec<f64>,
    /// True if the q-block was eliminated (n_q >= n_r).
    pub eliminate_q: bool,
}

// ---------------------------------------------------------------------------
// Star — zero-copy neighborhood view
// ---------------------------------------------------------------------------

/// One eliminated vertex's neighbors in the keep-block.
///
/// References into [`CsrBlock`]'s arrays for zero-copy access.
pub(super) struct Star<'a> {
    /// Eliminated vertex index (used for deterministic seeding).
    index: usize,
    /// Neighbor columns in the keep-block.
    col_indices: &'a [u32],
    /// Edge weights to each neighbor.
    weights: &'a [f64],
    /// `D_elim[k]` (needed by `clique_tree_sample`).
    diag: f64,
}

impl Star<'_> {
    fn degree(&self) -> usize {
        self.col_indices.len()
    }
}

// ---------------------------------------------------------------------------
// CliqueEmitter — strategy for edge emission
// ---------------------------------------------------------------------------

/// Strategy for producing fill edges from a star neighborhood.
pub(super) trait CliqueEmitter {
    /// Per-thread reusable scratch state.
    type Scratch: Default + Send;

    /// Emit fill edges from `star` into `edges`, using `scratch` for temporaries.
    fn emit(&self, star: &Star, edges: &mut Vec<Edge>, scratch: &mut Self::Scratch);
}

/// Emits sampled clique-tree fill edges for every star.
pub(super) struct SampledCliqueEmitter {
    seed: u64,
}

impl SampledCliqueEmitter {
    pub(super) fn new(seed: u64) -> Self {
        Self { seed }
    }
}

/// Thread-local scratch for [`SampledCliqueEmitter`].
#[derive(Default)]
pub(super) struct SampledScratch {
    /// AoS neighbor copy for `clique_tree_sample`.
    buf: Vec<(u32, f64)>,
}

impl CliqueEmitter for SampledCliqueEmitter {
    type Scratch = SampledScratch;

    fn emit(&self, star: &Star, edges: &mut Vec<Edge>, scratch: &mut SampledScratch) {
        scratch.buf.clear();
        for (&col, &w) in star.col_indices.iter().zip(star.weights) {
            scratch.buf.push((col, w));
        }
        let seed = self.seed.wrapping_add(star.index as u64);
        clique_tree_sample(&mut scratch.buf, star.diag, seed, edges);
    }
}

// ---------------------------------------------------------------------------
// Elimination — block selection + star iteration
// ---------------------------------------------------------------------------

/// Block-selection decision and star iteration for Schur elimination.
///
/// Encapsulates which block to eliminate, precomputed inverse-diagonals,
/// and provides zero-copy [`Star`] views for each eliminated vertex.
pub(super) struct Elimination<'a> {
    pub(super) eliminate_q: bool,
    pub(super) n_keep: usize,
    n_elim: usize,
    pub(super) inv_diag_elim: Vec<f64>,
    diag_elim: &'a [f64],
    pub(super) diag_keep: &'a [f64],
    pub(super) keep_to_elim: &'a CsrBlock,
    pub(super) elim_to_keep: &'a CsrBlock,
}

impl<'a> Elimination<'a> {
    /// Select which block to eliminate and precompute inverse-diagonals.
    pub(super) fn new(cross_tab: &'a CrossTab) -> Self {
        let n_q = cross_tab.n_q();
        let n_r = cross_tab.n_r();
        // Eliminate the larger block to minimize the reduced system size.
        let eliminate_q = n_q >= n_r;
        let (n_keep, n_elim) = if eliminate_q { (n_r, n_q) } else { (n_q, n_r) };

        let diag_elim = if eliminate_q {
            &cross_tab.diag_q
        } else {
            &cross_tab.diag_r
        };
        let inv_diag_elim: Vec<f64> = diag_elim
            .iter()
            .map(|&d| if d > 0.0 { 1.0 / d } else { 0.0 })
            .collect();

        let diag_keep = if eliminate_q {
            &cross_tab.diag_r
        } else {
            &cross_tab.diag_q
        };

        let (keep_to_elim, elim_to_keep) = if eliminate_q {
            (&cross_tab.ct, &cross_tab.c)
        } else {
            (&cross_tab.c, &cross_tab.ct)
        };

        Self {
            eliminate_q,
            n_keep,
            n_elim,
            inv_diag_elim,
            diag_elim,
            diag_keep,
            keep_to_elim,
            elim_to_keep,
        }
    }

    /// Create a zero-copy [`Star`] view for eliminated vertex `k`.
    fn star(&self, k: usize) -> Star<'_> {
        let start = self.elim_to_keep.indptr[k] as usize;
        let end = self.elim_to_keep.indptr[k + 1] as usize;
        Star {
            index: k,
            col_indices: &self.elim_to_keep.indices[start..end],
            weights: &self.elim_to_keep.data[start..end],
            diag: self.diag_elim[k],
        }
    }

    /// Parallel edge emission over all stars using the given [`CliqueEmitter`].
    pub(super) fn par_emit<E: CliqueEmitter + Sync>(&self, emitter: &E) -> Vec<Edge> {
        (0..self.n_elim)
            .into_par_iter()
            .fold(
                || (Vec::new(), E::Scratch::default()),
                |(mut edges, mut scratch), k| {
                    let star = self.star(k);
                    if star.degree() > 1 {
                        emitter.emit(&star, &mut edges, &mut scratch);
                    }
                    (edges, scratch)
                },
            )
            .map(|(edges, _)| edges)
            .reduce(Vec::new, |mut a, b| {
                a.extend(b);
                a
            })
    }

    /// Package elimination metadata into [`EliminationInfo`] for the solver.
    pub(super) fn into_info(self) -> EliminationInfo {
        EliminationInfo {
            inv_diag_elim: self.inv_diag_elim,
            eliminate_q: self.eliminate_q,
        }
    }
}
