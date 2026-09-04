//! Cross-term collinearity screen (#281): a slope covariate that is (nearly) a per-level
//! combination of another term's columns adds a null direction whitening cannot see.

use std::ops::Range;

use rayon::prelude::*;

use super::{Design, SlopeBasis};
use crate::channel::Channel;
use crate::BuildWarning;

/// Residual share below which a covariate counts as reproduced by the other term.
const COLLINEARITY_TOL: f64 = 1e-3;

/// Cross-moment table bytes the screen may hold at once, over all terms together.
const TABLE_BUDGET_BYTES: usize = 64 << 20;

/// Rows one residual task claims; small enough that work stealing balances the tail.
const ROWS_PER_TASK: usize = 1 << 16;

pub(crate) fn detect_collinear_slopes(
    design: &Design<'_>,
    weights: Option<&[f64]>,
    basis: &SlopeBasis,
) -> Vec<BuildWarning> {
    if design.n_factors() < 2 {
        return Vec::new();
    }
    let budget = TABLE_BUDGET_BYTES / std::mem::size_of::<f64>() / design.n_factors();
    (0..design.n_factors())
        .into_par_iter()
        .flat_map_iter(move |term| {
            let targets = screened_covariates(design, term);
            residual_shares(design, weights, basis, term, &targets, budget)
                .into_iter()
                .zip(targets)
                .filter(|&(share, _)| share <= COLLINEARITY_TOL)
                .map(
                    move |(relative_residual, (slope, _))| BuildWarning::CollinearSlopeCovariate {
                        slope,
                        term,
                        relative_residual,
                    },
                )
        })
        .collect()
}

/// Every other term's slope covariates, as `(channel, frame column)`.
fn screened_covariates(design: &Design<'_>, term: usize) -> Vec<(Channel, u32)> {
    (0..design.n_factors())
        .filter(|&t| t != term)
        .flat_map(|t| design.channels(t))
        .filter_map(|slope| Some((slope, *design.loading(slope).covariate()?)))
        .collect()
}

/// Two passes, so the share resolves below the `1e-16` a subtraction floors at.
fn residual_shares(
    design: &Design<'_>,
    weights: Option<&[f64]>,
    basis: &SlopeBasis,
    term: usize,
    targets: &[(Channel, u32)],
    budget: usize,
) -> Vec<f64> {
    let m = targets.len();
    if m == 0 {
        return Vec::new();
    }
    let meta = &design.terms[term];
    let levels = design.level_column(term);
    let us: Vec<&[f64]> = meta
        .columns
        .iter()
        .filter_map(|c| c.covariate())
        .map(|&c| basis.loading_column(c as usize))
        .collect();
    let intercept = us.len() < meta.columns.len();
    let columns: Vec<&[f64]> = targets
        .iter()
        .map(|&(_, c)| design.loading_column(c as usize))
        .collect();
    let stride = columns.len() * (us.len() + 1);
    let n_levels = meta.n_levels();
    let plan = ScreenPlan::new(budget, n_levels, stride);
    let screen = Screen {
        levels,
        weights,
        us,
        columns,
        intercept,
        stride,
        // Grouping gathers every column, so it must buy back more than the one block.
        order: match meta.sorted || plan.per_block == n_levels {
            true => RowOrder::AsIs,
            false => RowOrder::Grouped(super::stable_argsort(levels, n_levels)),
        },
    };

    let mut table = vec![0.0f64; plan.per_block * stride];
    let mut level_weight = vec![0.0f64; plan.per_block];
    let mut residual = vec![0.0f64; m];
    let mut variations = Variations {
        w_sum: 0.0,
        stats: vec![Variation::default(); m],
    };
    for levels_block in plan.blocks() {
        let block = Block {
            rows: screen.order.rows(levels, &levels_block),
            levels: levels_block,
        };
        let table = &mut table[..block.levels.len() * stride];
        let level_weight = &mut level_weight[..block.levels.len()];
        table.fill(0.0);
        level_weight.fill(0.0);
        screen.accumulate_cross(&block, table, level_weight, &mut variations);
        screen.to_coefficients(table, level_weight);
        screen.accumulate_residual(&block, table, &mut residual);
    }

    residual
        .iter()
        .zip(&variations.stats)
        .map(
            |(&residual, stat)| match stat.total(variations.w_sum, intercept) {
                // No variation means no slope: a within-term degeneracy the rank test reports.
                total if total > 0.0 => residual / total,
                _ => 1.0,
            },
        )
        .collect()
}

struct Screen<'a> {
    levels: &'a [u32],
    weights: Option<&'a [f64]>,
    /// The term's slope columns in the solve basis.
    us: Vec<&'a [f64]>,
    columns: Vec<&'a [f64]>,
    intercept: bool,
    order: RowOrder,
    /// One level's row of the table: `[Σw·c, Σw·u·c]` per target.
    stride: usize,
}

impl Screen<'_> {
    /// `(row, weight, table row)` for positions `rows` of the term's row order.
    fn active_rows(
        &self,
        first: usize,
        rows: Range<usize>,
    ) -> impl Iterator<Item = (usize, f64, usize)> + '_ {
        rows.filter_map(move |i| {
            let obs = self.order.obs(i);
            let w = self.weights.map_or(1.0, |w| w[obs]);
            (w > 0.0).then(|| (obs, w, self.levels[obs] as usize - first))
        })
    }

    /// Each target's total variation rides this pass, since the rows are already loaded.
    fn accumulate_cross(
        &self,
        block: &Block,
        table: &mut [f64],
        level_weight: &mut [f64],
        variations: &mut Variations,
    ) {
        let v = self.us.len();
        let stride = self.stride;
        let Variations { w_sum, stats } = variations;
        let mut running = *w_sum;
        for (obs, w, row) in self.active_rows(block.levels.start, block.rows.clone()) {
            running += w;
            level_weight[row] += w;
            // Every target shares the running weight, so the Welford ratio is taken once.
            let ratio = w / running;
            for ((slot, column), stat) in table[row * stride..][..stride]
                .chunks_exact_mut(v + 1)
                .zip(&self.columns)
                .zip(&mut *stats)
            {
                let c = column[obs];
                stat.observe(c, w, ratio);
                let wc = w * c;
                slot[0] += wc;
                for (s, u) in slot[1..].iter_mut().zip(&self.us) {
                    *s += wc * u[obs];
                }
            }
        }
        *w_sum = running;
    }

    /// `Σw·c` becomes the fit's constant `c̄`; `Σw·u·c` already is its slope, `u` being orthonormal.
    fn to_coefficients(&self, table: &mut [f64], level_weight: &[f64]) {
        let v = self.us.len();
        for (row, &w_sum) in table.chunks_exact_mut(self.stride).zip(level_weight) {
            for slot in row.chunks_exact_mut(v + 1) {
                slot[0] = match self.intercept && w_sum > 0.0 {
                    true => slot[0] / w_sum,
                    false => 0.0,
                };
            }
        }
    }

    /// Adds each target's `Σw·(c − ĉ)²` from the coefficients left in `table`.
    ///
    /// Read-only on `table`, so the rows split freely; only the `m` sums reduce.
    fn accumulate_residual(&self, block: &Block, table: &[f64], residual: &mut [f64]) {
        if block.rows.is_empty() {
            return;
        }
        let v = self.us.len();
        let stride = self.stride;
        let first = block.levels.start;
        let tasks = block.rows.len().div_ceil(ROWS_PER_TASK);
        let sums: Vec<Vec<f64>> = (0..tasks)
            .into_par_iter()
            .map_init(
                || vec![0.0f64; v],
                |u_row, task| {
                    let mut totals = vec![0.0f64; self.columns.len()];
                    let start = block.rows.start + task * ROWS_PER_TASK;
                    let rows = start..(start + ROWS_PER_TASK).min(block.rows.end);
                    for (obs, w, row) in self.active_rows(first, rows) {
                        for (uj, u) in u_row.iter_mut().zip(&self.us) {
                            *uj = u[obs];
                        }
                        for ((slot, column), total) in table[row * stride..][..stride]
                            .chunks_exact(v + 1)
                            .zip(&self.columns)
                            .zip(&mut totals)
                        {
                            let r = column[obs] - slot[0] - crate::linalg::dot(&slot[1..], u_row);
                            *total += w * r * r;
                        }
                    }
                    totals
                },
            )
            .collect();
        // Combined in task order, so the sum does not depend on how rayon split the rows.
        for task_sums in sums {
            for (total, sum) in residual.iter_mut().zip(task_sums) {
                *total += sum;
            }
        }
    }
}

/// One slice of a term's work: the levels the table holds and the rows they own.
struct Block {
    levels: Range<usize>,
    rows: Range<usize>,
}

/// The order a term's rows are walked in, grouping each level's rows into one run.
enum RowOrder {
    /// The term's rows already run in level order, or the one block spans every level.
    AsIs,
    /// Row ids counting-sorted by level, for a term whose own order scatters them.
    Grouped(Vec<u32>),
}

impl RowOrder {
    fn obs(&self, i: usize) -> usize {
        match self {
            Self::AsIs => i,
            Self::Grouped(perm) => perm[i] as usize,
        }
    }

    /// The positions `block`'s levels own — one contiguous run, so each row is walked once.
    fn rows(&self, levels: &[u32], block: &Range<usize>) -> Range<usize> {
        let bound = |level: usize| match self {
            Self::AsIs => levels.partition_point(|&l| (l as usize) < level),
            Self::Grouped(perm) => {
                perm.partition_point(|&obs| (levels[obs as usize] as usize) < level)
            }
        };
        bound(block.start)..bound(block.end)
    }
}

/// How a term's levels are cut to fit the table budget; one level's row is the floor.
#[derive(Clone, Copy)]
struct ScreenPlan {
    per_block: usize,
    n_levels: usize,
}

impl ScreenPlan {
    /// `budget` and `stride` both count doubles.
    fn new(budget: usize, n_levels: usize, stride: usize) -> Self {
        Self {
            per_block: (budget / stride).clamp(1, n_levels),
            n_levels,
        }
    }

    fn blocks(self) -> impl Iterator<Item = Range<usize>> {
        (0..self.n_levels)
            .step_by(self.per_block)
            .map(move |start| start..(start + self.per_block).min(self.n_levels))
    }
}

/// The targets' total variation, against the one running weight they all share.
struct Variations {
    w_sum: f64,
    stats: Vec<Variation>,
}

/// Welford, so the share's denominator survives a covariate whose mean dwarfs its spread.
#[derive(Clone, Default)]
struct Variation {
    mean: f64,
    m2: f64,
}

impl Variation {
    fn observe(&mut self, c: f64, w: f64, ratio: f64) {
        let delta = c - self.mean;
        self.mean += ratio * delta;
        self.m2 += w * delta * (c - self.mean);
    }

    /// An intercept puts a constant in every level's span; without one, against `Σw·c²`.
    fn total(&self, w_sum: f64, intercept: bool) -> f64 {
        match intercept {
            true => self.m2,
            false => self.m2 + w_sum * self.mean * self.mean,
        }
    }
}

#[cfg(test)]
mod tests;
