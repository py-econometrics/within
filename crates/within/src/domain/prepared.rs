use std::borrow::Cow;
use std::sync::Arc;

use rayon::prelude::*;

use super::{Column, Design, Gramian, RowWeights, TermLayout, TermReparam, TermRows};
use crate::channel::CoefficientPosition;
use crate::BuildError;

/// A [`Design`] plus the solve state it and one weight vector determine.
pub(crate) struct PreparedDesign<'a> {
    /// The design in the solve basis: the caller's rows with whitened slopes.
    pub(crate) design: Design<'a>,
    /// `W^{1/2}` in the design's internal observation order.
    pub(crate) weights: RowWeights,
    /// Per design term, its change of basis; `None` is the identity.
    reparams: Vec<Option<TermReparam>>,
}

/// A term in the solve basis: the design's layout and rows plus this preparation's whitened slopes.
pub(crate) struct PreparedTerm<'p> {
    pub(crate) layout: &'p TermLayout,
    pub(crate) rows: &'p TermRows<'p>,
    /// Solve-basis slopes in coefficient-column order; empty for a slope-free term.
    pub(crate) slopes: &'p [Cow<'p, [f64]>],
}

impl Design<'_> {
    /// Term `term` with this design's slopes.
    pub(crate) fn term(&self, term: usize) -> PreparedTerm<'_> {
        let layout = self.layout.term(term);
        PreparedTerm {
            layout,
            rows: &self.terms[term],
            slopes: &self.slopes[layout.slopes()],
        }
    }

    pub(crate) fn terms(&self) -> impl ExactSizeIterator<Item = PreparedTerm<'_>> {
        (0..self.n_factors()).map(|term| self.term(term))
    }
}

impl<'p> PreparedTerm<'p> {
    /// Column `column`'s solve-basis loading; `None` is the intercept.
    pub(crate) fn loading(&self, column: usize) -> Option<&'p [f64]> {
        match self.layout.column(column) {
            Column::Intercept => None,
            Column::Slope(j) => Some(&self.slopes[j][..]),
        }
    }
}

impl<'a> PreparedDesign<'a> {
    pub(crate) fn new(design: Design<'a>, weights: Option<&[f64]>) -> Result<Self, BuildError> {
        let weights = RowWeights::new(&design, weights)?;
        let built: Vec<_> = (0..design.n_factors())
            .into_par_iter()
            .map(|t| TermReparam::build(&design, t, &weights))
            .collect();
        let mut slopes = Vec::with_capacity(design.slopes.len());
        let reparams = built
            .into_iter()
            .map(|built| {
                built.map(|(reparam, whitened)| {
                    slopes.extend(whitened);
                    reparam
                })
            })
            .collect();
        // The raw slopes are dropped here unless the caller's design still shares them.
        let design = Design {
            slopes: Arc::new(slopes.into_iter().map(Cow::Owned).collect()),
            ..design
        };
        Ok(Self {
            design,
            weights,
            reparams,
        })
    }

    /// `G = AᵀA`, read from this design's factors.
    pub(crate) fn gramian(&self) -> Gramian<'_> {
        Gramian {
            design: &self.design,
            weights: &self.weights,
        }
    }

    /// Map solve-basis coefficients back to the user's parametrization.
    pub(crate) fn back_transform(&self, x: &mut [f64]) {
        for (layout, reparam) in self.design.layout.terms().iter().zip(&self.reparams) {
            if let Some(reparam) = reparam {
                reparam.back_transform(layout, &mut x[layout.dofs()]);
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

        assert_eq!(prepared.weights.sqrt(), Some(&[2.0, 4.0, 3.0, 1.0][..]));
        assert_eq!(
            (0..4)
                .map(|obs| prepared.weights.weight(obs))
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
