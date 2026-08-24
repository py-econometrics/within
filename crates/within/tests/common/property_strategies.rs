//! Shared proptest strategies for the property-test binaries.
#![allow(dead_code)]

use ndarray::Array2;
use proptest::prelude::*;
use within::PreconditionerConfig;

/// Generate a random FE problem: (categories Array2<u32>, y Vec<f64>).
/// 2–3 factors, 2–30 levels each, 50–500 observations.
pub fn random_fe_problem_strategy() -> impl Strategy<Value = (Array2<u32>, Vec<f64>)> {
    (2..=3u32).prop_flat_map(|n_factors| {
        let levels = proptest::collection::vec(2..=30u32, n_factors as usize);
        levels.prop_flat_map(move |n_levels| {
            let n_obs_range = 50..=500usize;
            n_obs_range.prop_flat_map(move |n_obs| {
                let n_levels_clone = n_levels.clone();
                let cat_cols: Vec<_> = n_levels_clone
                    .iter()
                    .map(|&nl| proptest::collection::vec(0..nl, n_obs))
                    .collect();
                let y_vec = proptest::collection::vec(-10.0f64..10.0, n_obs);
                (cat_cols, y_vec).prop_map(move |(cols, y)| {
                    let n_f = cols.len();
                    let n = cols[0].len();
                    let mut cats = Array2::<u32>::zeros((n, n_f));
                    for (f, col) in cols.iter().enumerate() {
                        for (i, &val) in col.iter().enumerate() {
                            cats[[i, f]] = val;
                        }
                    }
                    (cats, y)
                })
            })
        })
    })
}

/// Default additive Schwarz preconditioner config for property tests.
pub fn additive_precond() -> PreconditionerConfig {
    PreconditionerConfig::default()
}

/// One factor's owned inputs, from which the test body borrows to build an
/// [`within::Effect`] (which holds slices, so the data must outlive it).
#[derive(Debug, Clone)]
pub struct FactorData {
    pub levels: Vec<u32>,
    pub intercept: bool,
    pub slopes: Vec<Vec<f64>>,
}

/// A random varying-slopes least-squares problem: 1–3 factors (each an
/// optional intercept plus 0–2 slope covariates), positive weights, and a
/// response — all sharing one observation count.
#[derive(Debug, Clone)]
pub struct SlopesProblem {
    pub factors: Vec<FactorData>,
    pub weights: Vec<f64>,
    pub y: Vec<f64>,
}

fn populated_level_column(n_levels: u32, n_obs: usize) -> impl Strategy<Value = Vec<u32>> {
    proptest::collection::vec(0u32..n_levels, n_obs - n_levels as usize)
        .prop_map(move |mut levels| {
            levels.extend(0..n_levels);
            levels
        })
        .prop_shuffle()
}

pub fn random_slopes_problem_strategy() -> impl Strategy<Value = SlopesProblem> {
    (60usize..=300, 1usize..=3).prop_flat_map(|(n_obs, n_factors)| {
        // Per factor: level count, intercept flag, slope count. An effect with
        // neither an intercept nor a slope is invalid, so force an intercept in
        // that case.
        proptest::collection::vec((2u32..=15, any::<bool>(), 0usize..=2), n_factors).prop_flat_map(
            move |specs| {
                let factor_gens: Vec<_> = specs
                    .into_iter()
                    .map(|(n_levels, intercept, n_slopes)| {
                        let intercept = intercept || n_slopes == 0;
                        (
                            populated_level_column(n_levels, n_obs),
                            proptest::collection::vec(-3.0f64..3.0, n_obs),
                            proptest::collection::vec(
                                proptest::collection::vec(-0.5f64..0.5, n_obs),
                                n_slopes,
                            ),
                        )
                            .prop_map(move |(levels, base, noises)| {
                                // Slopes are near-collinear (shared base + small
                                // per-column noise) so per-level slope Grams are
                                // ill-conditioned — the solver's documented hard case.
                                let slopes = noises
                                    .into_iter()
                                    .enumerate()
                                    .map(|(j, noise)| {
                                        base.iter()
                                            .zip(&noise)
                                            .map(|(b, e)| b * (0.6 + 0.2 * j as f64) + e)
                                            .collect()
                                    })
                                    .collect();
                                FactorData {
                                    levels,
                                    intercept,
                                    slopes,
                                }
                            })
                    })
                    .collect();
                let weights = proptest::collection::vec(0.2f64..3.0, n_obs);
                let y = proptest::collection::vec(-10.0f64..10.0, n_obs);
                (factor_gens, weights, y).prop_map(|(factors, weights, y)| SlopesProblem {
                    factors,
                    weights,
                    y,
                })
            },
        )
    })
}
