use super::{Design, SlopeBasis};
use crate::BuildError;

/// A [`Design`] plus what one weight vector determines: the row scaling and the slope basis.
pub(crate) struct PreparedDesign<'a> {
    pub(crate) design: Design<'a>,
    /// `W^{1/2}` in internal observation order; `None` is unweighted.
    pub(crate) sqrt_weights: Option<Vec<f64>>,
    pub(crate) basis: SlopeBasis,
}

impl<'a> PreparedDesign<'a> {
    pub(crate) fn new(design: Design<'a>, weights: Option<&[f64]>) -> Result<Self, BuildError> {
        let sqrt_weights = weights.map(|w| sqrt_weights(&design, w)).transpose()?;
        let basis = SlopeBasis::build(&design, sqrt_weights.as_deref());
        Ok(Self {
            design,
            sqrt_weights,
            basis,
        })
    }
}

/// Validated `√w` in internal observation order.
fn sqrt_weights(design: &Design<'_>, w: &[f64]) -> Result<Vec<f64>, BuildError> {
    if w.len() != design.n_obs {
        return Err(BuildError::WeightCountMismatch {
            expected: design.n_obs,
            got: w.len(),
        });
    }
    // `wi >= 0.0` already rejects NaN; `is_finite` additionally rejects `+∞`.
    if let Some((index, &value)) = w
        .iter()
        .enumerate()
        .find(|&(_, &wi)| !(wi >= 0.0 && wi.is_finite()))
    {
        return Err(BuildError::InvalidWeight { index, value });
    }
    Ok(match &design.obs_perm {
        None => w.iter().map(|wi| wi.sqrt()).collect(),
        Some(perm) => perm.iter().map(|&i| w[i as usize].sqrt()).collect(),
    })
}

/// The Gram weight of row `obs`: the operator applies `s`, so its normal matrix carries `s²`.
pub(crate) fn row_weight(sqrt_weights: Option<&[f64]>, obs: usize) -> f64 {
    sqrt_weights.map_or(1.0, |s| s[obs] * s[obs])
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
    fn weights_are_checked_for_count_and_finiteness() {
        let design = Design::from_levels_for_test(vec![vec![0, 0, 0, 0, 0]]);
        let check = |w: &[f64]| sqrt_weights(&design, w);
        assert!(check(&[1.0, 2.0, 3.0, 4.0, 5.0]).is_ok());
        // Zero weights are valid (an excluded observation).
        assert!(check(&[0.0, 1.0, 2.0, 3.0, 4.0]).is_ok());
        assert!(matches!(
            check(&[1.0, 2.0]),
            Err(BuildError::WeightCountMismatch {
                expected: 5,
                got: 2
            })
        ));
        // Negative / non-finite weights are rejected with the offending index.
        assert!(matches!(
            check(&[1.0, -2.0, 3.0, 4.0, 5.0]),
            Err(BuildError::InvalidWeight { index: 1, .. })
        ));
        assert!(matches!(
            check(&[1.0, 2.0, f64::NAN, 4.0, 5.0]),
            Err(BuildError::InvalidWeight { index: 2, .. })
        ));
        assert!(matches!(
            check(&[1.0, 2.0, 3.0, f64::INFINITY, 5.0]),
            Err(BuildError::InvalidWeight { index: 3, .. })
        ));
    }

    #[test]
    fn sqrt_weights_follow_the_locality_permutation() {
        // Dominant factor [2,0,1,0] argsorts to [1,3,2,0].
        let design = Design::from_levels_for_test(vec![vec![2, 0, 1, 0]]);
        let s = sqrt_weights(&design, &[1.0, 4.0, 9.0, 16.0]).unwrap();
        assert_eq!(s, [2.0, 4.0, 3.0, 1.0]);
    }
}
