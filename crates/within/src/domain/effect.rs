use crate::BuildError;

/// One factor's contribution to a design: level codes, an optional per-level
/// intercept, and per-level slope covariates.
#[derive(Clone, Debug)]
pub struct Effect<'a> {
    levels: &'a [u32],
    intercept: bool,
    slopes: Box<[&'a [f64]]>,
}

impl<'a> Effect<'a> {
    /// Validates the effect is non-empty and every slope matches the level count.
    pub fn new(
        levels: &'a [u32],
        intercept: bool,
        slopes: impl IntoIterator<Item = &'a [f64]>,
    ) -> Result<Self, BuildError> {
        let slopes: Box<[&[f64]]> = slopes.into_iter().collect();
        if !intercept && slopes.is_empty() {
            return Err(BuildError::EmptyEffect);
        }
        let n = levels.len();
        for (slope, s) in slopes.iter().enumerate() {
            if s.len() != n {
                return Err(BuildError::SlopeLengthMismatch {
                    slope,
                    expected: n,
                    got: s.len(),
                });
            }
        }
        Ok(Self {
            levels,
            intercept,
            slopes,
        })
    }

    pub(crate) fn levels(&self) -> &'a [u32] {
        self.levels
    }

    pub(crate) fn intercept(&self) -> bool {
        self.intercept
    }

    pub(crate) fn slopes(&self) -> &[&'a [f64]] {
        &self.slopes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_rejects_slope_length_mismatch_naming_the_offending_slope() {
        let levels = [0u32, 1, 0, 1];
        let ok = [1.0, 2.0, 3.0, 4.0];
        let short = [1.0, 2.0];
        let err = Effect::new(&levels, true, [&ok[..], &short[..]]).unwrap_err();
        assert!(matches!(
            err,
            BuildError::SlopeLengthMismatch {
                slope: 1,
                expected: 4,
                got: 2
            }
        ));
    }

    #[test]
    fn new_rejects_effect_with_no_intercept_and_no_slopes() {
        let levels = [0u32, 1, 0, 1];
        let err = Effect::new(&levels, false, []).unwrap_err();
        assert!(matches!(err, BuildError::EmptyEffect));
    }
}
