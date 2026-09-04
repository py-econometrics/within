//! A design fixed to one weight vector: the state every solve on it shares.

use super::{Design, SlopeBasis};
use crate::BuildError;

/// A [`Design`] plus what one weight vector determines: the weights and the slope basis.
pub(crate) struct PreparedDesign<'a> {
    pub(crate) design: Design<'a>,
    /// Raw weights in internal observation order; `None` is unweighted.
    pub(crate) weights: Option<Vec<f64>>,
    /// `W^{1/2}`, the operator's row scaling; kept beside `weights` since `s·s` is not bitwise `w`.
    pub(crate) sqrt_weights: Option<Vec<f64>>,
    pub(crate) basis: SlopeBasis,
}

impl<'a> PreparedDesign<'a> {
    pub(crate) fn new(design: Design<'a>, weights: Option<Vec<f64>>) -> Result<Self, BuildError> {
        let weights = design.permute_weights(weights)?;
        let basis = SlopeBasis::build(&design, weights.as_deref());
        let sqrt_weights = weights
            .as_ref()
            .map(|w| w.iter().map(|wi| wi.sqrt()).collect());
        Ok(Self {
            design,
            weights,
            sqrt_weights,
            basis,
        })
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
}
