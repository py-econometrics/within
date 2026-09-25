use rayon::prelude::*;

use super::level_moments::LevelMoments;
use super::{row_weight, Design, TermLayout, TermReparam};
use crate::channel::CoefficientPosition;
use crate::BuildError;

/// A [`Design`] plus all state one weight vector determines.
pub(crate) struct PreparedDesign<'a> {
    pub(crate) design: Design<'a>,
    /// `W^{1/2}` in the design's internal observation order; `None` is unweighted.
    sqrt_weights: Option<Vec<f64>>,
    /// Per design term, in order; `None` for a slope-free term.
    reparams: Vec<Option<TermReparam>>,
}

/// A term in the solve basis: the design's rows plus this preparation's whitened slopes.
#[derive(Clone, Copy)]
pub(crate) struct PreparedTerm<'p> {
    pub(crate) layout: &'p TermLayout,
    pub(crate) levels: &'p [u32],
    pub(crate) sorted: bool,
    /// Solve-basis slopes in coefficient-column order; empty for a slope-free term.
    pub(crate) slopes: &'p [Vec<f64>],
}

impl<'p> PreparedTerm<'p> {
    /// Column `column`'s solve-basis loading; `None` is the intercept.
    pub(crate) fn loading(&self, column: usize) -> Option<&'p [f64]> {
        let j = column.checked_sub(self.layout.has_intercept() as usize)?;
        Some(&self.slopes[j])
    }
}

impl<'a> PreparedDesign<'a> {
    pub(crate) fn new(design: Design<'a>, weights: Option<&[f64]>) -> Result<Self, BuildError> {
        let sqrt_weights = weights
            .map(|weights| prepare_sqrt_weights(&design, weights))
            .transpose()?;
        let slope_terms: Vec<usize> = (0..design.n_factors())
            .filter(|&t| design.terms[t].layout.has_slopes())
            .collect();
        let whitened: Vec<TermReparam> = slope_terms
            .par_iter()
            .map(|&t| {
                let moments = LevelMoments::build(&design, t, sqrt_weights.as_deref());
                TermReparam::build(&design, t, &moments)
            })
            .collect();
        let mut reparams: Vec<Option<TermReparam>> =
            (0..design.n_factors()).map(|_| None).collect();
        for (&t, reparam) in slope_terms.iter().zip(whitened) {
            reparams[t] = Some(reparam);
        }
        Ok(Self {
            design,
            sqrt_weights,
            reparams,
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

    pub(crate) fn term(&self, term: usize) -> PreparedTerm<'_> {
        let (t, reparam) = (&self.design.terms[term], &self.reparams[term]);
        let slopes: &[Vec<f64>] = reparam.as_ref().map_or(&[], |r| &r.slopes);
        // Kernels read column `c` as slope `c - has_intercept`.
        debug_assert_eq!(slopes.len(), t.layout.covariates().count());
        debug_assert!(t.layout.columns[1..]
            .iter()
            .all(|c| c.covariate().is_some()));
        PreparedTerm {
            layout: &t.layout,
            levels: t.levels(),
            sorted: t.sorted(),
            slopes,
        }
    }

    pub(crate) fn terms(&self) -> impl ExactSizeIterator<Item = PreparedTerm<'_>> {
        (0..self.design.n_factors()).map(|term| self.term(term))
    }

    /// Map solve-basis coefficients back to the user's parametrization.
    pub(crate) fn back_transform(&self, x: &mut [f64]) {
        for (t, reparam) in self.design.terms.iter().zip(&self.reparams) {
            if let Some(reparam) = reparam {
                reparam.back_transform(&t.layout, x);
            }
        }
    }

    /// Directions the data cannot identify, ascending in `(term, level, column)`.
    pub(crate) fn unidentified(&self) -> impl Iterator<Item = CoefficientPosition> + '_ {
        self.reparams
            .iter()
            .flatten()
            .flat_map(|r| r.unidentified.iter().copied())
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
