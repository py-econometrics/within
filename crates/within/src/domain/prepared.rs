use super::{row_weight, Design, SlopeReparam};
use crate::BuildError;

/// A [`Design`] plus all state one weight vector determines.
pub(crate) struct PreparedDesign<'a> {
    pub(crate) design: Design<'a>,
    /// `W^{1/2}` in the design's internal observation order; `None` is unweighted.
    sqrt_weights: Option<Vec<f64>>,
    /// `None` for slope-free designs.
    pub(crate) reparam: Option<SlopeReparam>,
}

impl<'a> PreparedDesign<'a> {
    pub(crate) fn new(design: Design<'a>, weights: Option<&[f64]>) -> Result<Self, BuildError> {
        let sqrt_weights = weights
            .map(|weights| prepare_sqrt_weights(&design, weights))
            .transpose()?;
        let reparam = SlopeReparam::build(&design, sqrt_weights.as_deref());
        Ok(Self {
            design,
            sqrt_weights,
            reparam,
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

    /// Loading column `column` in the solve basis.
    pub(crate) fn loading_column(&self, column: usize) -> &[f64] {
        // Only slope terms reference loading columns, and any slope term makes `reparam` `Some`.
        self.reparam
            .as_ref()
            .expect("loading column on a slope-free design")
            .loading_column(column)
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
