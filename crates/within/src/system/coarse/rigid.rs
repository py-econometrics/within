//! The rigid-component coarse space. A spell (one eliminated level × one anchor level) whose Gram
//! is within a ratio of both levels' own Grams glues the two levels; each glued component's
//! anchor-side gauges span one coarse column per channel. With the eliminated term fitted exactly,
//! `U = ÃZ` is the gauges' energy-minimal extension.

mod glue;
mod lift;

use std::ops::Range;

use super::{CoarseDecline, CoarseSolve, DiagonalGramBasis};
use crate::design::LevelSegments;
use crate::system::{SplitDesign, TermBlock};
use glue::Gauges;

/// The coarse solve over the unknowns; declines when gluing leaves no coarse space worth a column
/// or its factor exceeds `n_obs` values.
pub(super) fn build(split: &SplitDesign<'_>) -> Result<CoarseSolve, CoarseDecline> {
    let rows = Rows::new(split);
    let gauges = Gauges::glue(split, &rows)?;
    lift::coarse_solve(split, &rows, &gauges)
}

/// `diag(D̃_tᵀ W D̃_t)` of term `term`, each level's whole Gram, column-major (`c · L + l`).
///
/// PROTOTYPE(#607): main keeps `diag(S)` but not `G`'s own diagonal, so the coarse build sums it.
fn term_gram(split: &SplitDesign<'_>, term: usize) -> Vec<f64> {
    let block = split.term(term);
    let (width, n_levels) = (block.layout().n_columns(), block.layout().n_levels());
    let mut gram = vec![0.0; width * n_levels];
    for (i, &l) in block.levels().iter().enumerate() {
        let w = split.weights.weight(i);
        for c in 0..width {
            let d = block.loading(c).map_or(1.0, |z| z[i]);
            gram[c * n_levels + l as usize] += w * d * d;
        }
    }
    gram
}

/// The eliminated term's rows grouped by level, read in the solve basis.
struct Rows<'p> {
    segments: &'p LevelSegments,
    eliminated: TermBlock<'p>,
    n_levels: usize,
    /// `diag(A_eᵀA_e)`, each level's whole Gram block after whitening, column-major.
    gram: Vec<f64>,
    sqrt_weights: Option<&'p [f64]>,
}

impl<'p> Rows<'p> {
    fn new(split: &'p SplitDesign<'_>) -> Self {
        let e = split.eliminated();
        let eliminated = split.term(e);
        Self {
            segments: split.incidence.segments(),
            n_levels: eliminated.layout().n_levels(),
            eliminated,
            gram: term_gram(split, e),
            sqrt_weights: split.weights.sqrt(),
        }
    }

    fn n_levels(&self) -> usize {
        self.n_levels
    }

    fn width(&self) -> usize {
        self.eliminated.layout().n_columns()
    }

    /// `√w` of row `i`.
    fn root(&self, i: usize) -> f64 {
        self.sqrt_weights.map_or(1.0, |s| s[i])
    }

    fn level(&self, level: usize) -> EliminatedLevel<'_> {
        EliminatedLevel {
            rows: self,
            level,
            positions: self.segments.rows(level),
        }
    }
}

/// One level's columns of `A_e`, `W½ D_e`; their row `k` is the level's `k`-th row.
struct EliminatedLevel<'r> {
    rows: &'r Rows<'r>,
    level: usize,
    /// The level's rows: the rows are sorted by the eliminated term.
    positions: Range<usize>,
}

impl EliminatedLevel<'_> {
    /// The rows of positive weight, with their position in the level.
    fn weighted(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        let rows = self.rows;
        (self.positions.clone().enumerate()).filter(|&(_, r)| rows.root(r) > 0.0)
    }

    /// Its rows, weighted or not.
    fn len(&self) -> usize {
        self.positions.len()
    }
}

impl DiagonalGramBasis for EliminatedLevel<'_> {
    fn entry(&self, k: usize, c: usize) -> f64 {
        let r = self.positions.start + k;
        self.rows.root(r) * self.rows.eliminated.loading(c).map_or(1.0, |z| z[r])
    }

    fn norm2(&self, c: usize) -> f64 {
        self.rows.gram[c * self.rows.n_levels() + self.level]
    }
}
