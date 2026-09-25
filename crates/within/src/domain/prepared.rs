use std::sync::OnceLock;

use rayon::prelude::*;

use super::{row_weight, Design, WhitenedTerm};
use crate::channel::{Channel, CoefficientPosition};
use crate::BuildError;

/// A [`Design`] plus all state one weight vector determines.
pub(crate) struct PreparedDesign<'a> {
    pub(crate) design: Design<'a>,
    /// `W^{1/2}` in the design's internal observation order; `None` is unweighted.
    sqrt_weights: Option<Vec<f64>>,
    /// Indexed like `design.terms`.
    terms: Vec<PreparedTerm>,
}

/// One term's weight-dependent state.
enum PreparedTerm {
    /// Slope-free: its `diag(AᵀA)` block is the per-level weight sums, computed on first read.
    Plain(OnceLock<Vec<f64>>),
    Whitened(WhitenedTerm),
}

impl<'a> PreparedDesign<'a> {
    pub(crate) fn new(design: Design<'a>, weights: Option<&[f64]>) -> Result<Self, BuildError> {
        let sqrt_weights = weights
            .map(|weights| prepare_sqrt_weights(&design, weights))
            .transpose()?;
        let terms = (0..design.terms.len())
            .into_par_iter()
            .map(|term| {
                if design.terms[term].has_slopes() {
                    PreparedTerm::Whitened(WhitenedTerm::build(
                        &design,
                        term,
                        sqrt_weights.as_deref(),
                    ))
                } else {
                    PreparedTerm::Plain(OnceLock::new())
                }
            })
            .collect();
        Ok(Self {
            design,
            sqrt_weights,
            terms,
        })
    }

    fn term_diagonal(&self, term: usize) -> &[f64] {
        let cell = match &self.terms[term] {
            PreparedTerm::Whitened(whitened) => return &whitened.diagonal,
            PreparedTerm::Plain(cell) => cell,
        };
        cell.get_or_init(|| {
            let meta = &self.design.terms[term];
            let levels = self.design.frame.level_column(term);
            let sqrt_weights = self.sqrt_weights();
            let mut block = vec![0.0; meta.n_levels()];
            if !meta.sorted {
                for (obs, &level) in levels.iter().enumerate() {
                    block[level as usize] += row_weight(sqrt_weights, obs);
                }
                return block;
            }
            // A sorted level is one run, so a register sum keeps its order without a store per row.
            let mut start = 0;
            for run in levels.chunk_by(|a, b| a == b) {
                let rows = start..start + run.len();
                start = rows.end;
                // A sum of ones is exact, so the unweighted run sum is its length.
                block[run[0] as usize] += match sqrt_weights {
                    None => run.len() as f64,
                    Some(s) => s[rows].iter().fold(0.0, |sum, &si| sum + si * si),
                };
            }
            block
        })
    }

    /// `diag(AᵀA)` in solve coordinates.
    pub(crate) fn gram_diagonal(&self) -> Vec<f64> {
        let blocks: Vec<&[f64]> = (0..self.design.terms.len())
            .into_par_iter()
            .map(|term| self.term_diagonal(term))
            .collect();
        blocks.concat()
    }

    /// One channel's `n_levels` block of [`Self::gram_diagonal`].
    pub(crate) fn channel_diagonal(&self, channel: Channel) -> &[f64] {
        let n_levels = self.design.terms[channel.term].n_levels();
        &self.term_diagonal(channel.term)[channel.column * n_levels..][..n_levels]
    }

    /// A term's loading columns in the solve basis, in slope-column order; empty if slope-free.
    pub(crate) fn term_loadings(&self, term: usize) -> &[Vec<f64>] {
        match &self.terms[term] {
            PreparedTerm::Whitened(whitened) => &whitened.loadings,
            PreparedTerm::Plain(_) => &[],
        }
    }

    /// A channel's loading column in the solve basis; `None` for an intercept.
    pub(crate) fn channel_loading(&self, channel: Channel) -> Option<&[f64]> {
        let meta = &self.design.terms[channel.term];
        meta.columns[channel.column].covariate().map(|_| {
            &*self.term_loadings(channel.term)[channel.column - meta.has_intercept() as usize]
        })
    }

    /// Directions the data cannot identify, ascending in `(term, level, column)`.
    pub(crate) fn unidentified(&self) -> impl Iterator<Item = CoefficientPosition> + '_ {
        self.whitened_terms()
            .flat_map(|(_, whitened)| whitened.unidentified.iter().copied())
    }

    /// Map solve-basis coefficients back to the user's parametrization.
    pub(crate) fn back_transform(&self, x: &mut [f64]) {
        for (term, whitened) in self.whitened_terms() {
            whitened.back_transform(&self.design.terms[term], x);
        }
    }

    fn whitened_terms(&self) -> impl Iterator<Item = (usize, &WhitenedTerm)> {
        self.terms
            .iter()
            .enumerate()
            .filter_map(|(term, prepared)| match prepared {
                PreparedTerm::Whitened(whitened) => Some((term, whitened)),
                PreparedTerm::Plain(_) => None,
            })
    }

    /// `W^{1/2}` in internal observation order; `None` is unweighted.
    #[inline]
    pub(crate) fn sqrt_weights(&self) -> Option<&[f64]> {
        self.sqrt_weights.as_deref()
    }

    /// The Gram weight of row `obs`: the operator applies `s`, so its normal matrix carries `s²`.
    #[inline]
    pub(crate) fn row_weight(&self, obs: usize) -> f64 {
        row_weight(self.sqrt_weights(), obs)
    }
}

/// Validate caller-order weights, then return `√w` in the design's internal order.
fn prepare_sqrt_weights(design: &Design<'_>, weights: &[f64]) -> Result<Vec<f64>, BuildError> {
    if weights.len() != design.n_obs {
        return Err(BuildError::WeightCountMismatch {
            expected: design.n_obs,
            got: weights.len(),
        });
    }
    // `wi >= 0.0` already rejects NaN; `is_finite` additionally rejects `+∞`.
    if let Some((index, &value)) = weights
        .iter()
        .enumerate()
        .find(|&(_, &wi)| !(wi >= 0.0 && wi.is_finite()))
    {
        return Err(BuildError::InvalidWeight { index, value });
    }
    Ok(design
        .permute_obs_in(weights)
        .iter()
        .map(|weight| weight.sqrt())
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::Effect;

    impl<'a> PreparedDesign<'a> {
        /// Unweighted, whitened like a solver would.
        pub(crate) fn unweighted_for_test(design: Design<'a>) -> Self {
            Self::new(design, None).expect("unweighted preparation cannot fail")
        }
    }

    impl PreparedDesign<'static> {
        pub(crate) fn from_levels_for_test(columns: Vec<Vec<u32>>) -> Self {
            Self::unweighted_for_test(Design::from_levels_for_test(columns))
        }
    }

    #[test]
    fn owns_sqrt_weights_in_internal_observation_order() {
        // Dominant factor [2,0,1,0] argsorts to caller positions [1,3,2,0].
        let design = Design::from_levels_for_test(vec![vec![2, 0, 1, 0]]);
        let prepared = PreparedDesign::new(design, Some(&[1.0, 4.0, 9.0, 16.0])).unwrap();

        assert_eq!(prepared.sqrt_weights(), Some(&[2.0, 4.0, 3.0, 1.0][..]));
        assert_eq!(
            (0..4)
                .map(|obs| prepared.row_weight(obs))
                .collect::<Vec<_>>(),
            [4.0, 16.0, 9.0, 1.0]
        );
    }

    /// `Σ (w·l)·l` per column and level, over the internal observation order.
    fn observation_sums(prepared: &PreparedDesign<'_>) -> Vec<f64> {
        let design = &prepared.design;
        let mut diag = vec![0.0; design.n_dofs];
        for (index, term) in design.terms.iter().enumerate() {
            let levels = design.frame.level_column(index);
            for column in 0..term.columns.len() {
                let base = term.column_base(column);
                let z = prepared.channel_loading(Channel {
                    term: index,
                    column,
                });
                for (obs, &level) in levels.iter().enumerate() {
                    let w = prepared.row_weight(obs);
                    diag[base + level as usize] += z.map_or(w, |z| w * z[obs] * z[obs]);
                }
            }
        }
        diag
    }

    #[test]
    fn gram_diagonal_is_the_observation_sums_bitwise() {
        // Level 1 of `f` has z1 = 2·z0 (a rank drop); z1's larger spread pivots it first.
        let f = [0u32, 1, 0, 1, 0, 1, 2, 2];
        let z0 = [1.0, 2.0, 3.0, 4.0, 5.5, 6.0, 0.3, 0.7];
        let z1 = [9.0, 4.0, 1.0, 8.0, 2.0, 12.0, 5.0, 1.1];
        let g = [2u32, 0, 1, 1, 0, 2, 0, 1];
        let h = [1u32, 0, 0, 1, 1, 0, 1, 0];
        let zh = [0.5, 1.5, -2.0, 3.0, 0.25, -1.0, 7.0, 2.0];
        // Nested in `f`, so it comes out sorted: a slope-free term summed by runs.
        let p = [0u32, 1, 0, 1, 0, 1, 1, 1];
        let weights = [2.0, 0.0, 3.0, 0.1, 1.0, 7.0, 0.3, 5.0];
        let effects = || {
            vec![
                Effect::new(&f, true, [&z0[..], &z1[..]]).unwrap(),
                Effect::new(&g, true, []).unwrap(),
                Effect::new(&h, false, [&zh[..]]).unwrap(),
                Effect::new(&p, true, []).unwrap(),
            ]
        };
        for weights in [None, Some(&weights[..])] {
            let prepared = PreparedDesign::new(Design::new(effects()).unwrap(), weights).unwrap();
            assert!(prepared.unidentified().next().is_some());
            assert!(prepared.design.terms[3].sorted && !prepared.design.terms[1].sorted);
            let expected = observation_sums(&prepared);
            assert_eq!(
                prepared
                    .gram_diagonal()
                    .iter()
                    .map(|d| d.to_bits())
                    .collect::<Vec<_>>(),
                expected.iter().map(|d| d.to_bits()).collect::<Vec<_>>(),
                "weights {weights:?}"
            );
        }
    }

    #[test]
    fn gram_diagonal_matches_hand_computed_values() {
        // `z` is constant in level 1 of `f`, so that level's slope is dropped.
        let f = [0u32, 0, 0, 1, 1, 2, 2];
        let z = [1.0, 2.0, 4.0, 3.0, 3.0, 5.0, 7.0];
        let g = [0u32, 1, 0, 1, 0, 1, 1];
        // Perfect squares, so `(√w)²` is exact.
        let weights = [1.0, 4.0, 9.0, 0.25, 4.0, 1.0, 16.0];
        for (weights, f_counts, g_counts) in [
            (None, [3.0, 2.0, 2.0], [3.0, 4.0]),
            (Some(&weights[..]), [14.0, 4.25, 17.0], [14.0, 21.25]),
        ] {
            let effects = vec![
                Effect::new(&f, true, [&z[..]]).unwrap(),
                Effect::new(&g, true, []).unwrap(),
            ];
            let prepared = PreparedDesign::new(Design::new(effects).unwrap(), weights).unwrap();
            let channel = |term, column| prepared.channel_diagonal(Channel { term, column });
            assert_eq!(channel(0, 0), f_counts);
            assert_eq!(channel(1, 0), g_counts);
            // A kept whitened slope has unit weighted norm; a dropped one is an exact-zero column.
            let slope = channel(0, 1);
            assert_eq!(slope[1], 0.0, "weights {weights:?}");
            for level in [0, 2] {
                assert!(
                    (slope[level] - 1.0).abs() < 1e-12,
                    "{slope:?}, weights {weights:?}"
                );
            }
        }
    }

    #[test]
    fn rejects_invalid_weights_before_permuting_them() {
        let design = Design::from_levels_for_test(vec![vec![2, 0, 1, 0]]);

        assert!(PreparedDesign::new(design.clone(), None).is_ok());
        assert!(PreparedDesign::new(design.clone(), Some(&[1.0, 2.0, 3.0, 4.0])).is_ok());
        // Zero weights are valid (an excluded observation).
        assert!(PreparedDesign::new(design.clone(), Some(&[0.0, 1.0, 2.0, 3.0])).is_ok());
        assert!(matches!(
            PreparedDesign::new(design.clone(), Some(&[1.0, 2.0])),
            Err(BuildError::WeightCountMismatch {
                expected: 4,
                got: 2
            })
        ));
        assert!(matches!(
            PreparedDesign::new(design.clone(), Some(&[1.0, -2.0, 3.0, 4.0])),
            Err(BuildError::InvalidWeight { index: 1, .. })
        ));
        assert!(matches!(
            PreparedDesign::new(design.clone(), Some(&[1.0, 2.0, f64::NAN, 4.0])),
            Err(BuildError::InvalidWeight { index: 2, .. })
        ));
        assert!(matches!(
            PreparedDesign::new(design, Some(&[1.0, 2.0, 3.0, f64::INFINITY])),
            Err(BuildError::InvalidWeight { index: 3, .. })
        ));
    }
}
