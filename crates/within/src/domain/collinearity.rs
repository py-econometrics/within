//! Cross-term collinearity screen (#281): a slope covariate that is (nearly) a per-level
//! combination of another term's columns adds a null direction whitening cannot see.

use rayon::prelude::*;

use super::level_moments::{BasisScratch, TermMoments};
use super::{Design, Loading};
use crate::channel::Channel;
use crate::BuildWarning;

/// Residual share below which a covariate counts as reproduced by the other term.
const COLLINEARITY_TOL: f64 = 1e-3;

pub(crate) fn detect_collinear_slopes(
    design: &Design<'_>,
    weights: Option<&[f64]>,
    moments: &TermMoments,
) -> Vec<BuildWarning> {
    (0..design.n_factors())
        .into_par_iter()
        .flat_map_iter(|term| {
            let targets: Vec<(Channel, u32)> = (0..design.n_factors())
                .filter(|&t| t != term)
                .flat_map(|t| design.channels(t))
                .filter_map(|slope| match design.loading(slope) {
                    Loading::Covariate(column) => Some((slope, column)),
                    Loading::Constant => None,
                })
                .collect();
            residual_shares(design, weights, moments, term, &targets)
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

/// Each target's share of (weighted) variation outside `term`'s column span.
///
/// One pass accumulates only the per-level cross moments `Σw·c` and `Σw·z·c`; the
/// span's own geometry comes from the shared [`TermMoments`], so the projection is
/// `Σ_q (W_q·d)²` against an orthonormal basis rather than a per-level solve.
fn residual_shares(
    design: &Design<'_>,
    weights: Option<&[f64]>,
    moments: &TermMoments,
    term: usize,
    targets: &[(Channel, u32)],
) -> Vec<f64> {
    let moments = &moments[term];
    let m = targets.len();
    let v = moments.n_slopes();
    let intercept = moments.intercept();
    if m == 0 || (v == 0 && !intercept) {
        return vec![1.0; m];
    }

    let columns: Vec<&[f64]> = targets
        .iter()
        .map(|&(_, c)| design.frame.loading_column(c as usize))
        .collect();
    let zs: Vec<&[f64]> = moments
        .covariates()
        .iter()
        .map(|&c| design.frame.loading_column(c as usize))
        .collect();

    // Per level and target: [Σw·c, Σw·z·c].
    let stride = m * (v + 1);
    let mut cross = vec![0.0f64; moments.n_levels() * stride];
    let mut ss_total = vec![0.0f64; m];
    for (obs, &level) in design.frame.level_column(term).iter().enumerate() {
        let w = weights.map_or(1.0, |w| w[obs]);
        if w <= 0.0 {
            continue;
        }
        let level = &mut cross[level as usize * stride..][..stride];
        for ((slot, column), ss) in level
            .chunks_exact_mut(v + 1)
            .zip(&columns)
            .zip(&mut ss_total)
        {
            let wc = w * column[obs];
            *ss += wc * column[obs];
            slot[0] += wc;
            for (s, z) in slot[1..].iter_mut().zip(&zs) {
                *s += wc * z[obs];
            }
        }
    }

    let ss_projected = cross
        .par_chunks_exact(stride)
        .enumerate()
        .fold(
            || (vec![0.0f64; m], BasisScratch::new(v), vec![0.0f64; v]),
            |(mut ss_projected, mut scratch, mut d), (level, cross)| {
                let w_sum = moments.w_sum(level);
                if w_sum > 0.0 {
                    if v > 0 {
                        moments.basis(level, &mut scratch);
                    }
                    let mean = moments.mean(level);
                    for (slot, ss) in cross.chunks_exact(v + 1).zip(&mut ss_projected) {
                        if intercept {
                            *ss += slot[0] * slot[0] / w_sum;
                            for ((dj, &xj), &mj) in d.iter_mut().zip(&slot[1..]).zip(mean) {
                                *dj = xj - slot[0] * mj;
                            }
                        } else {
                            d.copy_from_slice(&slot[1..]);
                        }
                        if v > 0 {
                            for row in scratch.basis.chunks_exact(v) {
                                let projection = crate::linalg::dot(row, &d);
                                *ss += projection * projection;
                            }
                        }
                    }
                }
                (ss_projected, scratch, d)
            },
        )
        .map(|(ss_projected, ..)| ss_projected)
        .reduce(
            || vec![0.0f64; m],
            |mut a, b| {
                for (aj, bj) in a.iter_mut().zip(b) {
                    *aj += bj;
                }
                a
            },
        );

    // An intercept puts every level's constant in the span, so the share has to be
    // measured against the covariate's variation — against `Σw·c²` it would only
    // report how far the covariate's mean sits from zero.
    let (mut sum_wc, mut w_total) = (vec![0.0f64; m], 0.0);
    for (level, chunk) in cross.chunks_exact(stride).enumerate() {
        w_total += moments.w_sum(level);
        for (slot, sum) in chunk.chunks_exact(v + 1).zip(&mut sum_wc) {
            *sum += slot[0];
        }
    }

    ss_projected
        .into_iter()
        .zip(ss_total)
        .zip(sum_wc)
        .map(|((projected, total), sum_wc)| {
            let variation = match intercept {
                true => total - sum_wc * sum_wc / w_total,
                false => total,
            };
            // A covariate with no variation carries no slope at all; that is a
            // within-term degeneracy the whitening's rank test already reports.
            match variation > 0.0 {
                true => ((total - projected) / variation).max(0.0),
                false => 1.0,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Effect;

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
}
