//! The rigid-component coarse space. A spell (one eliminated level × one anchor level) whose Gram
//! is within a ratio of both levels' own Grams glues the two levels; each glued component's
//! anchor-side gauges span one coarse column per channel. With the eliminated term fitted exactly,
//! `U = ÃZ` is the gauges' energy-minimal extension; on the full system `Z` carries that fit `−β`.

mod glue;
mod lift;

use std::borrow::Cow;
use std::ops::Range;

use super::CoarseSolve;
use crate::domain::{LevelRows, PreparedDesign, TermRows, Unknowns};
use crate::linalg::DiagonalGramBasis;
use crate::CoarseDecline;
use glue::Gauges;

/// The coarse solve over `unknowns`; declines when gluing leaves no coarse space worth a column
/// or its factor exceeds `n_obs` values.
pub(super) fn build(
    prepared: &PreparedDesign<'_>,
    unknowns: Unknowns,
) -> Result<CoarseSolve, CoarseDecline> {
    let rows = Rows::new(prepared);
    let gauges = Gauges::glue(prepared, unknowns, &rows)?;
    lift::coarse_solve(prepared, unknowns, &rows, &gauges)
}

/// The eliminated term's rows grouped by level, read in the solve basis.
struct Rows<'p> {
    by_level: Cow<'p, LevelRows>,
    eliminated: TermRows<'p>,
    /// `diag(A_eᵀA_e)`, each level's whole Gram block after whitening, column-major.
    gram: &'p [f64],
    sqrt_weights: Option<&'p [f64]>,
}

impl<'p> Rows<'p> {
    fn new(prepared: &'p PreparedDesign<'_>) -> Self {
        let dominant = prepared.design.dominant;
        Self {
            by_level: prepared.level_rows(dominant),
            eliminated: TermRows::new(&prepared.term(dominant)),
            gram: prepared.diagonal(dominant),
            sqrt_weights: prepared.sqrt_weights(),
        }
    }

    fn n_levels(&self) -> usize {
        self.eliminated.n_levels
    }

    fn width(&self) -> usize {
        self.eliminated.width()
    }

    /// `√w` of row `i`.
    fn root(&self, i: usize) -> f64 {
        self.sqrt_weights.map_or(1.0, |s| s[i])
    }

    fn level(&self, level: usize) -> EliminatedLevel<'_> {
        EliminatedLevel {
            rows: self,
            level,
            positions: self.by_level.positions(level),
        }
    }
}

/// One level's columns of `A_e`, `W½ D_e`; their row `k` is the level's `k`-th row.
struct EliminatedLevel<'r> {
    rows: &'r Rows<'r>,
    level: usize,
    positions: Range<usize>,
}

impl EliminatedLevel<'_> {
    /// The rows of positive weight, with their position in the level.
    fn weighted(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        let rows = self.rows;
        (rows.by_level.rows(self.level).enumerate()).filter(|&(_, r)| rows.root(r) > 0.0)
    }

    /// Its rows, weighted or not.
    fn len(&self) -> usize {
        self.positions.len()
    }
}

impl DiagonalGramBasis for EliminatedLevel<'_> {
    fn entry(&self, k: usize, c: usize) -> f64 {
        let r = self.rows.by_level.row(self.positions.start + k);
        self.rows.root(r) * self.rows.eliminated.column(r, c)
    }

    fn norm2(&self, c: usize) -> f64 {
        self.rows.gram[c * self.rows.n_levels() + self.level]
    }
}
