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
    pub(crate) fn new(design: Design<'a>, weights: Option<Vec<f64>>) -> Result<Self, BuildError> {
        let sqrt_weights = design.permute_weights(weights)?.map(|mut w| {
            w.iter_mut().for_each(|wi| *wi = wi.sqrt());
            w
        });
        let basis = SlopeBasis::build(&design, sqrt_weights.as_deref());
        Ok(Self {
            design,
            sqrt_weights,
            basis,
        })
    }
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
}
