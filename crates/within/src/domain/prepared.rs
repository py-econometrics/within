//! A design fixed to one weight vector: the state every solve on it shares.

use std::sync::Arc;

use super::collinearity::detect_collinear_slopes;
use super::{Design, SlopeBasis};
use crate::{BuildError, BuildWarning};

/// A [`Design`] plus everything one weight vector determines: weights, slope basis, screening.
pub(crate) struct PreparedDesign<'a> {
    pub(crate) design: Arc<Design<'a>>,
    /// Raw weights in internal observation order; `None` is unweighted.
    pub(crate) weights: Option<Vec<f64>>,
    pub(crate) basis: SlopeBasis,
    pub(crate) warnings: Vec<BuildWarning>,
}

impl<'a> PreparedDesign<'a> {
    pub(crate) fn new(
        design: Arc<Design<'a>>,
        weights: Option<Vec<f64>>,
    ) -> Result<Self, BuildError> {
        let weights = design.permute_weights(weights)?;
        let basis = SlopeBasis::build(&design, weights.as_deref());
        let warnings = detect_collinear_slopes(&design, weights.as_deref(), &basis);
        Ok(Self {
            design,
            weights,
            basis,
            warnings,
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
                design: Arc::new(design),
                weights: None,
                basis,
                warnings: Vec::new(),
            }
        }

        /// Unweighted, whitened like a solver would.
        pub(crate) fn unweighted_for_test(design: Design<'a>) -> Self {
            Self::new(Arc::new(design), None).expect("unweighted preparation cannot fail")
        }
    }
}
