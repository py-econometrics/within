use super::*;
use crate::test_rng::pseudo_noise;
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

/// Every row is walked in exactly one block, under either row order.
#[test]
fn level_blocks_cover_every_row_exactly_once() {
    let sorted: Vec<u32> = (0..100u32).flat_map(|l| [l; 3]).collect();
    let plan = ScreenPlan::new(7 * 3, 100, 3);
    assert_eq!(plan.per_block, 7);
    let mut next = 0;
    for levels in plan.blocks() {
        let rows = RowOrder::AsIs.rows(&sorted, &levels);
        assert_eq!(rows.start, next);
        assert_eq!(rows.end, levels.end.min(100) * 3);
        next = rows.end;
    }
    assert_eq!(next, sorted.len());

    let scattered: Vec<u32> = (0..300u32).map(|i| (i * 7) % 100).collect();
    let plan = ScreenPlan::new(7, 100, 7);
    assert_eq!(plan.per_block, 1);
    let order = RowOrder::Grouped(super::super::stable_argsort(&scattered, 100));
    let mut seen = vec![0u32; scattered.len()];
    for levels in plan.blocks() {
        for i in order.rows(&scattered, &levels) {
            let obs = order.obs(i);
            assert!(levels.contains(&(scattered[obs] as usize)), "{obs}");
            seen[obs] += 1;
        }
    }
    assert!(seen.iter().all(|&n| n == 1), "{seen:?}");
}

/// Gaps leave empty levels inside a block; the row order must group only codes that occur.
#[test]
fn gappy_level_codes_screen_the_same_at_any_budget() {
    let n = 4000;
    let a: Vec<u32> = (0..n).map(|i| (i % 40) as u32 * 2).collect();
    let b: Vec<u32> = (0..n).map(|i| ((i / 40) % 25) as u32 * 3).collect();
    let z1 = pseudo_noise(n, 7);
    let z2 = pseudo_noise(n, 11);
    let design = Design::new(vec![
        Effect::new(&a, true, [&z1[..]]).unwrap(),
        Effect::new(&b, true, [&z2[..]]).unwrap(),
    ])
    .unwrap();
    assert!(!design.terms[1].sorted);
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
