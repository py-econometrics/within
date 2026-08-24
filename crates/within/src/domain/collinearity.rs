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
    let levels = design.frame.level_column(term);
    let screen = Screen {
        levels,
        weights,
        zs: moments
            .covariates()
            .iter()
            .map(|&c| design.frame.loading_column(c as usize))
            .collect(),
        columns: targets
            .iter()
            .map(|&(_, c)| design.frame.loading_column(c as usize))
            .collect(),
        zeros: vec![0.0; moments.n_slopes()],
        moments,
    };

    let plan = ScreenPlan::new(budget, moments.n_levels(), screen.stride());
    let mut table = vec![0.0f64; plan.per_block * screen.stride()];
    let mut residual = vec![0.0f64; m];
    let mut variations = Variations {
        w_sum: 0.0,
        stats: vec![Variation::default(); m],
    };
    for block in plan.blocks(levels, design.terms[term].sorted) {
        let table = &mut table[..block.levels.len() * screen.stride()];
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
}

impl Screen<'_> {
    /// One level's row of the table: `[Σw·c, Σw·z·c]` per target.
    fn stride(&self) -> usize {
        self.columns.len() * (self.zs.len() + 1)
    }

    /// `(row, weight, table row)` for `rows`; an unsorted term is offered rows it must skip.
    fn active_rows(
        &self,
        levels: &Range<usize>,
        rows: Range<usize>,
    ) -> impl Iterator<Item = (usize, f64, usize)> + '_ {
        let (first, span) = (levels.start, levels.len());
        rows.filter_map(move |obs| {
            let w = self.weights.map_or(1.0, |w| w[obs]);
            let row = (self.levels[obs] as usize).wrapping_sub(first);
            (w > 0.0 && row < span).then_some((obs, w, row))
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
        let stride = self.stride();
        let Variations { w_sum, stats } = variations;
        let mut running = *w_sum;
        for (obs, w, row) in self.active_rows(&block.levels, block.rows.clone()) {
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
            .par_chunks_exact_mut(self.stride())
            .enumerate()
            .for_each_init(
                || (BasisScratch::new(v), vec![0.0f64; v]),
                |(scratch, d), (offset, row)| {
                    let level = block.levels.start + offset;
                    let w_sum = self.moments.w_sum(level);
                    if w_sum <= 0.0 {
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
        let stride = self.stride();
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
                    for (obs, w, row) in self.active_rows(&block.levels, rows) {
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

/// One slice of a term's work: the levels the table holds and the rows offered for them.
struct Block {
    levels: Range<usize>,
    rows: Range<usize>,
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

    fn blocks(self, levels: &[u32], sorted: bool) -> impl Iterator<Item = Block> + '_ {
        (0..self.n_levels)
            .step_by(self.per_block)
            .map(move |start| {
                let end = (start + self.per_block).min(self.n_levels);
                Block {
                    rows: match sorted {
                        true => {
                            levels.partition_point(|&l| (l as usize) < start)
                                ..levels.partition_point(|&l| (l as usize) < end)
                        }
                        false => 0..levels.len(),
                    },
                    levels: start..end,
                }
            })
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
mod tests {
    use super::*;
    use crate::Effect;

    /// Wide enough that no design in these tests splits its table.
    const FULL_TABLE: usize = 1 << 24;

    fn warn_pairs(design: &Design<'_>, weights: Option<&[f64]>) -> Vec<(Channel, usize)> {
        let moments = TermMoments::build(design, weights).expect("design carries slopes");
        detect_collinear_slopes(design, weights, &moments)
            .into_iter()
            .map(|w| match w {
                BuildWarning::CollinearSlopeCovariate { slope, term, .. } => (slope, term),
                other => panic!("unexpected warning {other:?}"),
            })
            .collect()
    }

    /// The shares `term`'s screen reports, in target-channel order.
    fn shares(design: &Design<'_>, term: usize, budget: usize) -> Vec<f64> {
        let moments = TermMoments::build(design, None).expect("design carries slopes");
        let targets = screened_covariates(design, term);
        residual_shares(design, None, &moments, term, &targets, budget)
    }

    fn two_factor_levels(n: usize) -> (Vec<u32>, Vec<u32>) {
        let a = (0..n).map(|i| (i % 40) as u32).collect();
        let b = (0..n).map(|i| ((i / 40) % 25) as u32).collect();
        (a, b)
    }

    fn pseudo_noise(n: usize, seed: u64) -> Vec<f64> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
            })
            .collect()
    }

    #[test]
    fn shared_covariate_across_two_terms_warns_both_ways() {
        let n = 4000;
        let (a, b) = two_factor_levels(n);
        let z = pseudo_noise(n, 3);
        let design = Design::new(vec![
            Effect::new(&a, true, [&z[..]]).unwrap(),
            Effect::new(&b, true, [&z[..]]).unwrap(),
        ])
        .unwrap();
        let pairs = warn_pairs(&design, None);
        assert!(
            pairs.contains(&(Channel { term: 0, column: 1 }, 1)),
            "{pairs:?}"
        );
        assert!(
            pairs.contains(&(Channel { term: 1, column: 1 }, 0)),
            "{pairs:?}"
        );
    }

    /// The disease is scale-invariant, so the screen must be too: the raw Gram's
    /// intercept diagonal dwarfs a small covariate's.
    #[test]
    fn a_rescaled_shared_covariate_still_warns_both_ways() {
        let n = 4000;
        let (a, b) = two_factor_levels(n);
        let z = pseudo_noise(n, 3);
        for scale in [1.0, 1e-6, 1e-9] {
            let scaled: Vec<f64> = z.iter().map(|v| v * scale).collect();
            let design = Design::new(vec![
                Effect::new(&a, true, [&scaled[..]]).unwrap(),
                Effect::new(&b, true, [&scaled[..]]).unwrap(),
            ])
            .unwrap();
            assert_eq!(warn_pairs(&design, None).len(), 2, "scale {scale:e}");
        }
    }

    #[test]
    fn per_level_affine_transform_of_a_shared_covariate_warns() {
        let n = 4000;
        let (a, b) = two_factor_levels(n);
        let z = pseudo_noise(n, 5);
        let scaled: Vec<f64> = z
            .iter()
            .zip(&b)
            .map(|(&v, &level)| v * (1.0 + level as f64) - 0.25 * level as f64)
            .collect();
        let design = Design::new(vec![
            Effect::new(&a, true, [&z[..]]).unwrap(),
            Effect::new(&b, true, [&scaled[..]]).unwrap(),
        ])
        .unwrap();
        let pairs = warn_pairs(&design, None);
        assert!(
            pairs.contains(&(Channel { term: 0, column: 1 }, 1)),
            "{pairs:?}"
        );
    }

    #[test]
    fn factor_measurable_covariate_warns_against_the_intercept_term() {
        let n = 4000;
        let (a, b) = two_factor_levels(n);
        let of_b: Vec<f64> = b.iter().map(|&level| level as f64 + 1.0).collect();
        let design = Design::new(vec![
            Effect::new(&a, true, [&of_b[..]]).unwrap(),
            Effect::new(&b, true, []).unwrap(),
        ])
        .unwrap();
        let pairs = warn_pairs(&design, None);
        assert_eq!(pairs, vec![(Channel { term: 0, column: 1 }, 1)]);
    }

    /// The span holds every level's constant, so a covariate's offset must not
    /// enter the share: measured against `Σw·c²`, `mean=2005, sd=6` reads 7.6e-7.
    #[test]
    fn an_independent_covariate_stays_silent_at_any_offset() {
        let n = 4000;
        let (a, b) = two_factor_levels(n);
        let z = pseudo_noise(n, 5);
        for mean in [0.0, 1.0, 100.0, 2005.0, 1e6] {
            let shifted: Vec<f64> = z.iter().map(|v| mean + 6.0 * v).collect();
            let design = Design::new(vec![
                Effect::new(&a, true, [&shifted[..]]).unwrap(),
                Effect::new(&b, true, []).unwrap(),
            ])
            .unwrap();
            assert_eq!(warn_pairs(&design, None), Vec::new(), "mean {mean:e}");
        }
    }

    #[test]
    fn independent_covariates_stay_silent() {
        let n = 4000;
        let (a, b) = two_factor_levels(n);
        let z1 = pseudo_noise(n, 7);
        let z2 = pseudo_noise(n, 11);
        let design = Design::new(vec![
            Effect::new(&a, true, [&z1[..]]).unwrap(),
            Effect::new(&b, true, [&z2[..]]).unwrap(),
        ])
        .unwrap();
        assert_eq!(warn_pairs(&design, None), Vec::new());
    }

    #[test]
    fn zero_weight_rows_are_excluded_from_the_screen() {
        // The covariates disagree only on rows whose weight is zero (row removal).
        let n = 4000;
        let (a, b) = two_factor_levels(n);
        let z = pseudo_noise(n, 13);
        let mut spoiled = z.clone();
        let mut weights = vec![1.0; n];
        for i in (0..n).step_by(7) {
            spoiled[i] += 10.0;
            weights[i] = 0.0;
        }
        let design = Design::new(vec![
            Effect::new(&a, true, [&z[..]]).unwrap(),
            Effect::new(&b, true, [&spoiled[..]]).unwrap(),
        ])
        .unwrap();
        // The screen reads frame order, so caller weights go through the locality permutation.
        let weights = design.permute_obs_in(&weights).into_owned();
        assert!(!warn_pairs(&design, Some(&weights)).is_empty());
        assert_eq!(warn_pairs(&design, None), Vec::new());
    }

    /// Severity peaks where a covariate is *nearly* shared: `eps²` down to 1e-24.
    #[test]
    fn the_share_resolves_a_perturbation_of_a_shared_covariate() {
        let n = 4000;
        let (a, b) = two_factor_levels(n);
        let z = pseudo_noise(n, 3);
        let noise = pseudo_noise(n, 17);
        let share_at = |eps: f64| {
            let perturbed: Vec<f64> = z.iter().zip(&noise).map(|(&v, &e)| v + eps * e).collect();
            let design = Design::new(vec![
                Effect::new(&a, true, [&z[..]]).unwrap(),
                Effect::new(&b, true, [&perturbed[..]]).unwrap(),
            ])
            .unwrap();
            shares(&design, 1, FULL_TABLE)[0]
        };

        // An exactly shared covariate is reproduced to the last bit of the fit itself.
        let exact = share_at(0.0);
        assert!(exact < 1e-28, "{exact:e}");
        for eps in [1e-12, 1e-9, 1e-6, 1e-3] {
            let share = share_at(eps);
            let ratio = share / (eps * eps);
            assert!((0.2..5.0).contains(&ratio), "eps {eps:e}: share {share:e}");
        }
    }

    /// Without an intercept both the fit and the denominator must read the covariate raw.
    #[test]
    fn a_shared_covariate_warns_against_a_term_without_an_intercept() {
        let n = 4000;
        let (a, b) = two_factor_levels(n);
        let z = pseudo_noise(n, 3);
        // Offset far past its spread, so a centred denominator would read the share above 1.
        let other: Vec<f64> = pseudo_noise(n, 19)
            .iter()
            .map(|v| 100.0 + 6.0 * v)
            .collect();
        let design = Design::new(vec![
            Effect::new(&a, false, [&z[..]]).unwrap(),
            Effect::new(&b, true, [&z[..], &other[..]]).unwrap(),
        ])
        .unwrap();
        let shares = shares(&design, 0, FULL_TABLE);
        assert!(shares[0] < 1e-28, "{:e}", shares[0]);
        assert!((0.5..1.0 + 1e-9).contains(&shares[1]), "{:e}", shares[1]);
    }

    /// More rows than one residual task, checked against a direct within-level sum so a
    /// partition that drops or repeats even a few rows cannot pass.
    #[test]
    fn shares_hold_across_several_residual_tasks() {
        let n = 3 * ROWS_PER_TASK;
        let n_levels = 1000;
        let a: Vec<u32> = (0..n).map(|i| (i % n_levels) as u32).collect();
        let b: Vec<u32> = (0..n).map(|i| ((i / 7) % 137) as u32).collect();
        let c = pseudo_noise(n, 31);
        let design = Design::new(vec![
            Effect::new(&a, true, []).unwrap(),
            Effect::new(&b, true, [&c[..]]).unwrap(),
        ])
        .unwrap();

        // Term 0 carries no slope, so `c`'s fit on a level's span is that level's mean.
        let (mut sums, mut counts) = (vec![0.0f64; n_levels], vec![0.0f64; n_levels]);
        for (&level, &cv) in a.iter().zip(&c) {
            sums[level as usize] += cv;
            counts[level as usize] += 1.0;
        }
        let within: f64 = a
            .iter()
            .zip(&c)
            .map(|(&l, &cv)| {
                let d = cv - sums[l as usize] / counts[l as usize];
                d * d
            })
            .sum();
        let mean = c.iter().sum::<f64>() / n as f64;
        let total: f64 = c.iter().map(|&cv| (cv - mean) * (cv - mean)).sum();

        let share = shares(&design, 0, FULL_TABLE)[0];
        let expected = within / total;
        assert!(
            (share - expected).abs() <= 1e-9 * expected,
            "{share:e} vs {expected:e}"
        );
    }

    /// Splitting the table changes what it holds, never the arithmetic — to rounding.
    #[test]
    fn a_minimal_table_budget_reproduces_the_full_table_shares() {
        let n = 4000;
        let (a, b) = two_factor_levels(n);
        let z1 = pseudo_noise(n, 7);
        let z2 = pseudo_noise(n, 11);
        let design = Design::new(vec![
            Effect::new(&a, true, [&z1[..], &z2[..]]).unwrap(),
            Effect::new(&b, true, [&z1[..]]).unwrap(),
        ])
        .unwrap();
        // Only term 0's rows come out sorted, so the two terms take the two blocking paths.
        assert!(design.terms[0].sorted && !design.terms[1].sorted);
        assert_eq!(shares(&design, 0, 1), shares(&design, 0, FULL_TABLE));
        for (split, whole) in shares(&design, 1, 1)
            .into_iter()
            .zip(shares(&design, 1, FULL_TABLE))
        {
            assert!(
                (split - whole).abs() <= 1e-12 * split.max(whole),
                "{split:e} vs {whole:e}"
            );
        }
    }

    /// A 32-double level row, so 4000 doubles buys 125 levels.
    #[test]
    fn the_plan_keeps_the_table_inside_the_budget() {
        let table = |plan: ScreenPlan| plan.per_block * 32;
        let fits = ScreenPlan::new(4000, 125, 32);
        assert_eq!((fits.per_block, table(fits)), (125, 4000));
        // A level count the budget cannot hold whole is cut, however large it is.
        for n_levels in [1_000, 1_000_000, 100_000_000] {
            let plan = ScreenPlan::new(4000, n_levels, 32);
            assert_eq!(plan.per_block, 125, "{n_levels}");
            assert!(table(plan) <= 4000, "{n_levels}");
        }
        // One level's row is the floor: the stride, however many levels the term has.
        let starved = ScreenPlan::new(0, 100_000_000, 32);
        assert_eq!((starved.per_block, table(starved)), (1, 32));
    }

    /// Every row lands in exactly one block: a sorted term slices its rows, an unsorted
    /// term is offered all of them and skips the ones outside the block.
    #[test]
    fn level_blocks_cover_every_row_exactly_once() {
        let sorted: Vec<u32> = (0..100u32).flat_map(|l| [l; 3]).collect();
        let plan = ScreenPlan::new(7 * 3, 100, 3);
        assert_eq!(plan.per_block, 7);
        let mut next = 0;
        for block in plan.blocks(&sorted, true) {
            assert_eq!(block.rows.start, next);
            assert_eq!(block.rows.end, block.levels.end.min(100) * 3);
            next = block.rows.end;
        }
        assert_eq!(next, sorted.len());

        let scattered: Vec<u32> = (0..300u32).map(|i| (i * 7) % 100).collect();
        let plan = ScreenPlan::new(7, 100, 7);
        assert_eq!(plan.per_block, 1);
        let mut seen = vec![0u32; scattered.len()];
        for block in plan.blocks(&scattered, false) {
            assert_eq!(block.rows, 0..scattered.len());
            for (obs, &l) in scattered.iter().enumerate() {
                seen[obs] += u32::from(block.levels.contains(&(l as usize)));
            }
        }
        assert!(seen.iter().all(|&n| n == 1), "{seen:?}");
    }
}
