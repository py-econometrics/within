//! A design fixed to one weight vector: the state every solve on it shares.

use super::{Design, SlopeBasis};
use crate::BuildError;

/// A [`Design`] plus what one weight vector determines: the weights and the slope basis.
pub(crate) struct PreparedDesign<'a> {
    pub(crate) design: Design<'a>,
    /// Raw weights in internal observation order; `None` is unweighted.
    pub(crate) weights: Option<Vec<f64>>,
    pub(crate) basis: SlopeBasis,
}

impl<'a> PreparedDesign<'a> {
    pub(crate) fn new(design: Design<'a>, weights: Option<Vec<f64>>) -> Result<Self, BuildError> {
        let weights = design.permute_weights(weights)?;
        let basis = SlopeBasis::build(&design, weights.as_deref());
        Ok(Self {
            design,
            weights,
            basis,
        })
    }

    /// `W^{1/2}` in internal observation order, the operator's row scaling.
    pub(crate) fn sqrt_weights(&self) -> Option<Vec<f64>> {
        self.weights
            .as_ref()
            .map(|w| w.iter().map(|wi| wi.sqrt()).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    impl<'a> PreparedDesign<'a> {
        /// Unweighted, on the caller's own loading columns.
        pub(crate) fn with_caller_loadings_for_test(design: Design<'a>) -> Self {
            let basis = SlopeBasis::with_caller_loadings_for_test(&design);
            Self {
                design,
                weights: None,
                basis,
            }
        }

        /// Unweighted, whitened like a solver would.
        pub(crate) fn unweighted_for_test(design: Design<'a>) -> Self {
            Self::new(design, None).expect("unweighted preparation cannot fail")
        }
    }
}
