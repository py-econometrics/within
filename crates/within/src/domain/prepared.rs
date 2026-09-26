use rayon::prelude::*;

use super::{row_weight, Column, Design, Term, TermReparam};
use crate::channel::CoefficientPosition;
use crate::BuildError;

/// A [`Design`] plus all state one weight vector determines.
pub(crate) struct PreparedDesign<'a> {
    pub(crate) design: Design<'a>,
    /// `W^{1/2}` in the design's internal observation order; `None` is unweighted.
    sqrt_weights: Option<Vec<f64>>,
    /// Per design term, in order.
    reparams: Vec<TermReparam>,
}

/// A term in the solve basis: the design's rows plus this preparation's whitened slopes.
pub(crate) struct PreparedTerm<'p> {
    pub(crate) term: &'p Term<'p>,
    /// Solve-basis slopes in coefficient-column order; empty for a slope-free term.
    pub(crate) slopes: &'p [Vec<f64>],
}

impl<'p> PreparedTerm<'p> {
    /// Column `column`'s solve-basis loading; `None` is the intercept.
    pub(crate) fn loading(&self, column: usize) -> Option<&'p [f64]> {
        match self.term.column(column) {
            Column::Intercept => None,
            Column::Slope(j) => Some(&self.slopes[j]),
        }
    }
}

impl<'a> PreparedDesign<'a> {
    pub(crate) fn new(design: Design<'a>, weights: Option<&[f64]>) -> Result<Self, BuildError> {
        let sqrt_weights = weights
            .map(|weights| prepare_sqrt_weights(&design, weights))
            .transpose()?;
        let reparams: Vec<TermReparam> = (0..design.n_factors())
            .into_par_iter()
            .map(|t| TermReparam::build(&design, t, sqrt_weights.as_deref()))
            .collect();
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
        let (t, slopes) = (&self.design.terms[term], &self.reparams[term].slopes);
        debug_assert_eq!(slopes.len(), t.raw_slopes().len());
        PreparedTerm { term: t, slopes }
    }

    pub(crate) fn terms(&self) -> impl ExactSizeIterator<Item = PreparedTerm<'_>> {
        (0..self.design.n_factors()).map(|term| self.term(term))
    }

    /// Term `term`'s block of [`Self::gram_diagonal`], laid out as its `dofs()`.
    pub(crate) fn term_diagonal(&self, term: usize) -> &[f64] {
        // Serial fill: a rayon job stolen inside it could re-enter this cell and deadlock.
        self.reparams[term]
            .diagonal
            .get_or_init(|| term_gram_diagonal(self.term(term), self.sqrt_weights()))
    }

    /// `diag(AᵀA)` in solve coordinates.
    pub(crate) fn gram_diagonal(&self) -> Vec<f64> {
        let blocks: Vec<&[f64]> = (0..self.design.n_factors())
            .into_par_iter()
            .map(|term| self.term_diagonal(term))
            .collect();
        let diag = blocks.concat();
        debug_assert_eq!(diag.len(), self.design.n_dofs);
        diag
    }

    /// Map solve-basis coefficients back to the user's parametrization.
    pub(crate) fn back_transform(&self, x: &mut [f64]) {
        for (t, reparam) in self.design.terms.iter().zip(&self.reparams) {
            reparam.back_transform(t, x);
        }
    }

    /// Directions the data cannot identify, ascending in `(term, level, column)`.
    pub(crate) fn unidentified(&self) -> impl Iterator<Item = CoefficientPosition> + '_ {
        self.reparams
            .iter()
            .flat_map(|r| r.unidentified.iter().copied())
    }
}

/// One term's squared column norms, each accumulated in observation order.
fn term_gram_diagonal(prepared: PreparedTerm<'_>, sqrt_weights: Option<&[f64]>) -> Vec<f64> {
    let term = prepared.term;
    let n_levels = term.n_levels();
    let mut diag = vec![0.0; term.n_dofs()];
    for column in 0..term.n_columns() {
        let block = &mut diag[column * n_levels..][..n_levels];
        let z = prepared.loading(column);
        for (obs, &level) in term.levels().iter().enumerate() {
            let w = row_weight(sqrt_weights, obs);
            // Keep `w * z * z` left-to-right: a zero weight kills a huge `z` first.
            block[level as usize] += z.map_or(w, |z| w * z[obs] * z[obs]);
        }
    }
    diag
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
    fn gram_diagonal_matches_hand_computed_values() {
        // `z` is constant in level 1 of `f`, so that level's slope is dropped.
        let f = [0u32, 0, 0, 1, 1, 2, 2];
        let z = [1.0, 2.0, 4.0, 3.0, 3.0, 5.0, 7.0];
        let g = [0u32, 1, 0, 1, 0, 1, 1];
        // Perfect squares, so `(√w)²` is exact.
        let weights = [1.0, 4.0, 9.0, 0.25, 4.0, 1.0, 16.0];
        for (weights, f_sums, g_sums) in [
            (None, [3.0, 2.0, 2.0], [3.0, 4.0]),
            (Some(&weights[..]), [14.0, 4.25, 17.0], [14.0, 21.25]),
        ] {
            let effects = vec![
                crate::Effect::new(&f, true, [&z[..]]).unwrap(),
                crate::Effect::new(&g, true, []).unwrap(),
            ];
            let prepared = PreparedDesign::new(Design::new(effects).unwrap(), weights).unwrap();
            let diag = prepared.gram_diagonal();
            let block =
                |term: usize, column| &diag[prepared.design.terms[term].column_dofs(column)];
            assert_eq!(block(0, 0), f_sums);
            assert_eq!(block(1, 0), g_sums);
            // A kept whitened slope has unit weighted norm; a dropped one is an exact-zero column.
            let slope = block(0, 1);
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
