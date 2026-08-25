//! Cross-term collinearity screen (#281): a slope covariate that is (nearly) a per-level
//! combination of another term's columns adds a null direction whitening cannot see.

use std::ops::Range;

use rayon::prelude::*;

use super::level_moments::{BasisScratch, LevelMoments, TermMoments};
use super::Design;
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
    moments: &TermMoments,
) -> Vec<BuildWarning> {
    if design.n_factors() < 2 {
        return Vec::new();
    }
    let budget = TABLE_BUDGET_BYTES / std::mem::size_of::<f64>() / design.n_factors();
    (0..design.n_factors())
        .into_par_iter()
        .flat_map_iter(move |term| {
            let targets = screened_covariates(design, term);
            residual_shares(design, weights, moments, term, &targets, budget)
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
    moments: &TermMoments,
    term: usize,
    targets: &[(Channel, u32)],
    budget: usize,
) -> Vec<f64> {
    let m = targets.len();
    if m == 0 {
        return Vec::new();
    }
    let moments = &moments[term];
    let levels = design.level_column(term);
    let zs: Vec<&[f64]> = moments
        .covariates()
        .iter()
        .map(|&c| design.loading_column(c as usize))
        .collect();
    let columns: Vec<&[f64]> = targets
        .iter()
        .map(|&(_, c)| design.loading_column(c as usize))
        .collect();
    let stride = columns.len() * (zs.len() + 1);
    let n_levels = moments.n_levels();
    let plan = ScreenPlan::new(budget, n_levels, stride);
    let screen = Screen {
        levels,
        weights,
        zs,
        columns,
        zeros: vec![0.0; moments.n_slopes()],
        moments,
        stride,
        // Grouping gathers every column, so it must buy back more than the one block.
        order: match design.terms()[term].sorted || plan.per_block == n_levels {
            true => RowOrder::AsIs,
            false => RowOrder::Grouped(super::stable_argsort(levels, n_levels)),
        },
    };

    let mut table = vec![0.0f64; plan.per_block * stride];
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
        table.fill(0.0);
        screen.accumulate_cross(&block, table, &mut variations);
        screen.to_coefficients(&block, table);
        screen.accumulate_residual(&block, table, &mut residual);
    }

    let intercept = moments.intercept();
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
    zs: Vec<&'a [f64]>,
    columns: Vec<&'a [f64]>,
    zeros: Vec<f64>,
    moments: &'a LevelMoments,
    order: RowOrder,
    /// One level's row of the table: `[Σw·c, Σw·z·c]` per target.
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

    /// Without an intercept the level's span holds no constant, so `z` enters uncentered.
    fn center(&self, level: usize) -> &[f64] {
        match self.moments.intercept() {
            true => self.moments.mean(level),
            false => &self.zeros,
        }
    }

    /// Each target's total variation rides this pass, since the rows are already loaded.
    fn accumulate_cross(&self, block: &Block, table: &mut [f64], variations: &mut Variations) {
        let v = self.zs.len();
        let stride = self.stride;
        let Variations { w_sum, stats } = variations;
        let mut running = *w_sum;
        for (obs, w, row) in self.active_rows(block.levels.start, block.rows.clone()) {
            running += w;
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
                for (s, z) in slot[1..].iter_mut().zip(&self.zs) {
                    *s += wc * z[obs];
                }
            }
        }
        *w_sum = running;
    }

    /// Rewrites a level's cross moments as the coefficients `[c̄, β]` of `c`'s fit on its span.
    fn to_coefficients(&self, block: &Block, table: &mut [f64]) {
        let v = self.zs.len();
        let intercept = self.moments.intercept();
        table
            .par_chunks_exact_mut(self.stride)
            .enumerate()
            .for_each_init(
                || (BasisScratch::new(v), vec![0.0f64; v]),
                |(scratch, d), (offset, row)| {
                    let level = block.levels.start + offset;
                    let w_sum = self.moments.w_sum(level);
                    if w_sum <= 0.0 {
                        debug_assert!(
                            row.iter().all(|&x| x == 0.0),
                            "level {level} carries cross moments the term's weights deny"
                        );
                        return;
                    }
                    if v > 0 {
                        self.moments.basis(level, scratch);
                    }
                    let center = self.center(level);
                    for slot in row.chunks_exact_mut(v + 1) {
                        let sum_wc = slot[0];
                        for ((dj, &xj), &cj) in d.iter_mut().zip(&slot[1..]).zip(center) {
                            *dj = xj - sum_wc * cj;
                        }
                        slot[0] = match intercept {
                            true => sum_wc / w_sum,
                            false => 0.0,
                        };
                        slot[1..].fill(0.0);
                        if v > 0 {
                            for q in scratch.basis.chunks_exact(v) {
                                let projection = crate::linalg::dot(q, d);
                                for (b, &qj) in slot[1..].iter_mut().zip(q) {
                                    *b += projection * qj;
                                }
                            }
                        }
                    }
                },
            );
    }

    /// Adds each target's `Σw·(c − ĉ)²` from the coefficients left in `table`.
    ///
    /// Read-only on `table`, so the rows split freely; only the `m` sums reduce.
    fn accumulate_residual(&self, block: &Block, table: &[f64], residual: &mut [f64]) {
        if block.rows.is_empty() {
            return;
        }
        let v = self.zs.len();
        let stride = self.stride;
        let first = block.levels.start;
        let tasks = block.rows.len().div_ceil(ROWS_PER_TASK);
        let sums: Vec<Vec<f64>> = (0..tasks)
            .into_par_iter()
            .map_init(
                || vec![0.0f64; v],
                |dz, task| {
                    let mut totals = vec![0.0f64; self.columns.len()];
                    let start = block.rows.start + task * ROWS_PER_TASK;
                    let rows = start..(start + ROWS_PER_TASK).min(block.rows.end);
                    for (obs, w, row) in self.active_rows(first, rows) {
                        for (dzj, (z, &cj)) in dz
                            .iter_mut()
                            .zip(self.zs.iter().zip(self.center(first + row)))
                        {
                            *dzj = z[obs] - cj;
                        }
                        for ((slot, column), total) in table[row * stride..][..stride]
                            .chunks_exact(v + 1)
                            .zip(&self.columns)
                            .zip(&mut totals)
                        {
                            let r = column[obs] - slot[0] - crate::linalg::dot(&slot[1..], dz);
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
