//! Cross-term collinearity screen (#281): a slope covariate that is (nearly) a per-level
//! combination of another term's columns adds a null direction whitening cannot see.

use rayon::prelude::*;

use super::{Design, Loading};
use crate::channel::Channel;
use crate::BuildWarning;

/// Residual share below which a covariate counts as reproduced by the other term.
const COLLINEARITY_TOL: f64 = 1e-3;

/// Pivots below this share of the largest diagonal count as a dependent column.
const PIVOT_FLOOR: f64 = 1e-12;

pub(crate) fn detect_collinear_slopes(
    design: &Design<'_>,
    weights: Option<&[f64]>,
) -> Vec<BuildWarning> {
    let mut pairs: Vec<(Channel, u32, usize)> = Vec::new();
    for term in 0..design.n_factors() {
        for slope in design.channels(term) {
            let Loading::Covariate(column) = design.loading(slope) else {
                continue;
            };
            pairs.extend(
                (0..design.n_factors())
                    .filter(|&t| t != term)
                    .map(|t| (slope, column, t)),
            );
        }
    }
    pairs
        .par_iter()
        .filter_map(|&(slope, covariate, term)| {
            let relative_residual = relative_residual(design, weights, covariate, term);
            (relative_residual <= COLLINEARITY_TOL).then_some(
                BuildWarning::CollinearSlopeCovariate {
                    slope,
                    term,
                    relative_residual,
                },
            )
        })
        .collect()
}

/// Share of the covariate's (weighted) sum of squares outside `term`'s column span.
fn relative_residual(
    design: &Design<'_>,
    weights: Option<&[f64]>,
    covariate: u32,
    term: usize,
) -> f64 {
    let target = design.frame.loading_column(covariate as usize);
    let levels = design.frame.level_column(term);
    let columns = &design.terms[term].columns;
    let p = columns.len();
    let stride = p * p + p;

    let covariate_slots: Vec<(usize, &[f64])> = columns
        .iter()
        .enumerate()
        .filter_map(|(slot, c)| {
            c.covariate()
                .map(|&k| (slot, design.frame.loading_column(k as usize)))
        })
        .collect();

    let mut per_level = vec![0.0f64; design.terms[term].n_levels * stride];
    let mut ss_total = 0.0;
    let mut x = vec![1.0f64; p];
    for (obs, (&value, &level)) in target.iter().zip(levels).enumerate() {
        let w = weights.map_or(1.0, |w| w[obs]);
        ss_total += w * value * value;
        for &(slot, loadings) in &covariate_slots {
            x[slot] = loadings[obs];
        }
        let (gram, cross) =
            per_level[level as usize * stride..(level as usize + 1) * stride].split_at_mut(p * p);
        for q in 0..p {
            let wx = w * x[q];
            cross[q] += wx * value;
            for r in 0..p {
                gram[q * p + r] += wx * x[r];
            }
        }
    }
    if ss_total <= 0.0 {
        return 1.0;
    }

    let mut ss_projected = 0.0;
    let mut active = Vec::with_capacity(p);
    for level in per_level.chunks_exact_mut(stride) {
        let (gram, cross) = level.split_at_mut(p * p);
        ss_projected += projected_ss(gram, cross, &mut active);
    }
    ((ss_total - ss_projected) / ss_total).max(0.0)
}

/// `cᵀ G⁺ c` for one level's tiny PSD Gram, by pivoted outer-product Cholesky.
fn projected_ss(gram: &mut [f64], cross: &mut [f64], active: &mut Vec<usize>) -> f64 {
    let p = cross.len();
    let floor = PIVOT_FLOOR * (0..p).map(|q| gram[q * p + q]).fold(0.0f64, f64::max);
    active.clear();
    active.extend(0..p);
    let mut ss = 0.0;
    while let Some((position, &pivot)) = active
        .iter()
        .enumerate()
        .max_by(|a, b| gram[a.1 * p + a.1].total_cmp(&gram[b.1 * p + b.1]))
    {
        let diagonal = gram[pivot * p + pivot];
        if diagonal <= floor {
            break;
        }
        active.swap_remove(position);
        ss += cross[pivot] * cross[pivot] / diagonal;
        for &row in active.iter() {
            let multiplier = gram[row * p + pivot] / diagonal;
            cross[row] -= multiplier * cross[pivot];
            for &col in active.iter() {
                gram[row * p + col] -= multiplier * gram[pivot * p + col];
            }
        }
    }
    ss
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Effect;

    fn warn_pairs(design: &Design<'_>, weights: Option<&[f64]>) -> Vec<(Channel, usize)> {
        detect_collinear_slopes(design, weights)
            .into_iter()
            .map(|w| match w {
                BuildWarning::CollinearSlopeCovariate { slope, term, .. } => (slope, term),
                other => panic!("unexpected warning {other:?}"),
            })
            .collect()
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
}
