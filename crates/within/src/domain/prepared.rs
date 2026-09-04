use super::{Design, SlopeReparam};

/// A [`Design`] plus the whitened slope basis one weight vector determines.
pub(crate) struct PreparedDesign<'a> {
    pub(crate) design: Design<'a>,
    /// `None` for slope-free designs.
    pub(crate) reparam: Option<SlopeReparam>,
}

impl<'a> PreparedDesign<'a> {
    pub(crate) fn new(design: Design<'a>, sqrt_weights: Option<&[f64]>) -> Self {
        let reparam = SlopeReparam::build(&design, sqrt_weights);
        Self { design, reparam }
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

#[cfg(test)]
mod tests {
    use super::*;

    impl<'a> PreparedDesign<'a> {
        /// Unweighted, whitened like a solver would.
        pub(crate) fn unweighted_for_test(design: Design<'a>) -> Self {
            Self::new(design, None)
        }
    }

    impl PreparedDesign<'static> {
        pub(crate) fn from_levels_for_test(columns: Vec<Vec<u32>>) -> Self {
            Self::unweighted_for_test(Design::from_levels_for_test(columns))
        }
    }
}
