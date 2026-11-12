//! Seam tests for the per-term independence of the multi-term whitening,
//! not reachable through the public API.

use crate::channel::Channel;
use crate::domain::{row_weight, Effect, PreparedDesign};
use rstest::rstest;

use super::*;

const F0: [u32; 6] = [0, 0, 0, 1, 1, 1];
const Z0: [f64; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
const Z1: [f64; 6] = [1.0, 4.0, 9.0, 16.0, 25.0, 36.0];
const G: [u32; 6] = [0, 1, 2, 0, 1, 2];
const F2: [u32; 6] = [0, 1, 0, 1, 0, 1];
const Z2: [f64; 6] = [2.0, 7.0, 1.0, 8.0, 3.0, 9.0];
const YEAR: [f64; 11] = [6.0, 8.0, 10.0, 7.0, 5.0, 3.0, 0.0, 4.0, 1.0, 9.0, 2.0];

/// Two slope-bearing terms around a plain one; term 0 is dominant and sorted
/// so the locality sort stays a no-op.
fn three_term_effects() -> Vec<Effect<'static>> {
    vec![
        Effect::new(&F0, true, [&Z0[..], &Z1[..]]).unwrap(),
        Effect::new(&G, true, []).unwrap(),
        Effect::new(&F2, true, [&Z2[..]]).unwrap(),
    ]
}

#[test]
fn build_whitens_each_slope_bearing_term() {
    let prepared = PreparedDesign::unweighted_for_test(Design::new(three_term_effects()).unwrap());
    assert!(prepared.unidentified().next().is_none());
    for term in [0, 2] {
        assert_level_grams_orthonormal(&prepared, term, 1e-12);
    }
}

#[test]
fn slope_free_term_has_no_reparam() {
    let prepared = PreparedDesign::from_levels_for_test(vec![vec![0, 1, 0], vec![0, 0, 1]]);
    assert!(prepared.terms().all(|t| t.slopes.is_empty()));
}

#[test]
fn unidentified_directions_ascend_across_terms() {
    // Index-order term iteration keeps the list ascending without any sort.
    let f0 = [0u32, 0, 1, 1, 2, 2];
    let z0 = [1.0, 2.0, 5.0, 5.0, 3.0, 7.0];
    let f1 = [0u32, 1, 0, 1, 0, 1];
    let z1 = [4.0, 1.0, 4.0, 2.0, 4.0, 3.0];
    let effects = vec![
        Effect::new(&f0, true, [&z0[..]]).unwrap(),
        Effect::new(&f1, true, [&z1[..]]).unwrap(),
    ];
    let prepared = PreparedDesign::unweighted_for_test(Design::new(effects).unwrap());
    assert_eq!(
        prepared.unidentified().collect::<Vec<_>>(),
        vec![
            CoefficientPosition {
                channel: Channel { term: 0, column: 1 },
                level: 1,
            },
            CoefficientPosition {
                channel: Channel { term: 1, column: 1 },
                level: 0,
            },
        ]
    );
}

#[test]
fn back_transform_leaves_other_terms_untouched() {
    let prepared = PreparedDesign::unweighted_for_test(Design::new(three_term_effects()).unwrap());
    let design = &prepared.design;

    let mut x: Vec<f64> = (0..design.n_dofs).map(|i| 1.0 + i as f64).collect();
    let before = x.clone();
    prepared.back_transform(&mut x);

    // Plain term 1 sits between the two slope-bearing blocks.
    let (t1, t2) = (design.terms[1].offset, design.terms[2].offset);
    assert_eq!(x[t1..t2], before[t1..t2]);
    assert_ne!(x[..t1], before[..t1]);
    assert_ne!(x[t2..], before[t2..]);
}

/// Welford's mean of this order is `5 + 2⁻⁵⁰`, so plain centering leaves the mean year a noise load.
#[test]
fn an_observation_at_its_level_mean_loads_exactly_zero() {
    let level = [0u32; 11];
    let effects = vec![Effect::new(&level, true, [&YEAR[..]]).unwrap()];
    let prepared = PreparedDesign::unweighted_for_test(Design::new(effects).unwrap());
    let t = prepared.term(0);
    let mean_year = (0..t.term.levels().len())
        .find(|&i| t.term.raw_slopes().next().unwrap()[i] == 5.0)
        .unwrap();
    assert_eq!(t.slopes[0][mean_year], 0.0);
}

#[rstest]
#[case::level_sum(|_| 1e306)]
#[case::spread(|i| if i % 2 == 0 { 1e307 } else { -1e307 })]
fn a_sum_past_the_float_range_still_whitens_finitely(#[case] huge_at: fn(usize) -> f64) {
    let n = 200;
    let level = vec![0u32; n];
    let huge: Vec<f64> = (0..n).map(huge_at).collect();
    let year: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let effects = vec![Effect::new(&level, true, [&huge[..], &year[..]]).unwrap()];
    let prepared = PreparedDesign::unweighted_for_test(Design::new(effects).unwrap());
    assert!(prepared
        .term(0)
        .slopes
        .iter()
        .flatten()
        .all(|u| u.is_finite()));
}

/// Slopes spread over ~1e-9 of their size; a running mean's rounding would show in the Gram.
#[test]
fn a_large_offset_still_whitens_to_the_identity() {
    let z0: Vec<f64> = YEAR.iter().map(|t| 1e9 + 0.1 * t).collect();
    let z1: Vec<f64> = YEAR
        .iter()
        .map(|t| 1e9 + 0.1 * t + 0.03 * (t * t % 7.0))
        .collect();
    let level = [0u32; 11];
    let effects = vec![Effect::new(&level, true, [&z0[..], &z1[..]]).unwrap()];
    let prepared = PreparedDesign::unweighted_for_test(Design::new(effects).unwrap());
    assert!(prepared.unidentified().next().is_none());
    assert_level_grams_orthonormal(&prepared, 0, 1e-12);
}

/// Per level of `term`: an identified slope has unit norm, a dropped one is zero, and columns,
/// intercept included, are orthogonal to `tol` relative to their norms.
fn assert_level_grams_orthonormal(prepared: &PreparedDesign<'_>, term: usize, tol: f64) {
    let t = prepared.term(term);
    let (levels, width) = (t.term.levels(), t.term.n_columns());
    let first_slope = usize::from(t.term.intercept);
    let w = |i: usize| row_weight(prepared.sqrt_weights(), i);
    let a = |i: usize, c: usize| t.loading(c).map_or(1.0, |z| z[i]);
    let mut grams = vec![0.0; t.term.n_levels() * width * width];
    for (i, &level) in levels.iter().enumerate() {
        let g = &mut grams[level as usize * width * width..][..width * width];
        for j in 0..width {
            for k in 0..width {
                g[j * width + k] += w(i) * a(i, j) * a(i, k);
            }
        }
    }
    for (level, g) in grams.chunks_exact(width * width).enumerate() {
        let dropped = prepared
            .unidentified()
            .filter(|p| {
                p.channel.term == term && p.level == level && p.channel.column >= first_slope
            })
            .count();
        let diag = |j: usize| g[j * width + j];
        for j in first_slope..width {
            let expected = if j < width - dropped { 1.0 } else { 0.0 };
            assert!(
                (diag(j) - expected).abs() <= tol,
                "term {term} level {level}: ‖u{j}‖² = {}",
                diag(j)
            );
        }
        for j in 0..width {
            for k in (0..width).filter(|&k| k != j && diag(j) > 0.0 && diag(k) > 0.0) {
                let cosine = g[j * width + k] / (diag(j) * diag(k)).sqrt();
                assert!(
                    cosine.abs() <= tol,
                    "term {term} level {level}: cos(u{j}, u{k}) = {cosine:e}"
                );
            }
        }
    }
}

#[test]
fn three_near_collinear_slopes_in_one_level_stay_orthonormal() {
    let d = 3e-5;
    let z0 = [1.0, 1.0, 1.0, 1.0];
    let z1 = [1.0 + d, 1.0 - d, 1.0, 1.0];
    let z2 = [1.0, 1.0, 1.0 + d, 1.0 - d];
    let effects = vec![Effect::new(&[0u32; 4], false, [&z0[..], &z1[..], &z2[..]]).unwrap()];
    let prepared = PreparedDesign::unweighted_for_test(Design::new(effects).unwrap());
    assert!(prepared.unidentified().next().is_none());
    assert_level_grams_orthonormal(&prepared, 0, 1e-14);
}

/// Rounded to one `f64`, both means put the columns `1 ulp` apart; exactly centred, they coincide.
#[test]
fn a_spread_at_the_mean_rounding_is_centred_exactly() {
    let eps = f64::EPSILON;
    let z0 = [1.0, 1.0 + eps];
    let z1 = [1.0 + eps, 1.0 + 2.0 * eps];
    let effects = vec![Effect::new(&[0u32; 2], true, [&z0[..], &z1[..]]).unwrap()];
    let prepared = PreparedDesign::unweighted_for_test(Design::new(effects).unwrap());
    let unidentified: Vec<_> = prepared.unidentified().collect();
    assert_eq!(unidentified.len(), 1, "{unidentified:?}");
    assert_ne!(unidentified[0].channel.column, 0);
    assert_level_grams_orthonormal(&prepared, 0, 1e-14);
}

/// Without an intercept a common mean is itself a slope direction, so only centred slopes get one.
#[rstest]
fn near_collinear_large_mean_slopes_stay_orthonormal(
    #[values(true, false)] intercept: bool,
    #[values("unweighted", "weighted", "zero weights")] weighting: &str,
) {
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};
    let (n_levels, rows) = (40, 25);
    let n = n_levels * rows;
    let mean = if intercept { 1e6 } else { 0.0 };
    let mut rng = SmallRng::seed_from_u64(514);
    let levels: Vec<u32> = (0..n).map(|i| (i / rows) as u32).collect();
    let base: Vec<f64> = (0..n).map(|_| rng.random::<f64>() - 0.5).collect();
    let zs: Vec<Vec<f64>> = (0..3)
        .map(|j| {
            (0..n)
                .map(|i| {
                    let noise = if j == 0 {
                        0.0
                    } else {
                        1e-4 * (rng.random::<f64>() - 0.5)
                    };
                    mean + 1.7 * base[i] + noise
                })
                .collect()
        })
        .collect();
    let weights: Option<Vec<f64>> = match weighting {
        "unweighted" => None,
        "weighted" => Some((0..n).map(|_| 0.2 + rng.random::<f64>()).collect()),
        _ => Some(
            (0..n)
                .map(|i| {
                    if i < rows || i % rows < 2 {
                        0.0
                    } else {
                        0.2 + rng.random::<f64>()
                    }
                })
                .collect(),
        ),
    };
    let effects = vec![Effect::new(&levels, intercept, zs.iter().map(Vec::as_slice)).unwrap()];
    let prepared = PreparedDesign::new(Design::new(effects).unwrap(), weights.as_deref()).unwrap();
    let expected: Vec<CoefficientPosition> = if weighting == "zero weights" {
        (0..prepared.term(0).term.n_columns())
            .map(|column| CoefficientPosition {
                channel: Channel { term: 0, column },
                level: 0,
            })
            .collect()
    } else {
        Vec::new()
    };
    assert_eq!(prepared.unidentified().collect::<Vec<_>>(), expected);
    assert_level_grams_orthonormal(&prepared, 0, 1e-14);
}
