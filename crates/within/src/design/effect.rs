use std::borrow::Cow;

use crate::BuildError;

/// One factor's contribution to a design: level codes plus per-level coefficient columns.
#[derive(Clone, Debug)]
pub struct Effect<'a> {
    pub(super) levels: Cow<'a, [u32]>,
    pub(super) intercept: bool,
    pub(super) slopes: Vec<Cow<'a, [f64]>>,
}

impl<'a> Effect<'a> {
    /// Validates the effect is non-empty and every slope finite per observation.
    pub fn new(
        levels: &'a [u32],
        intercept: bool,
        slopes: impl IntoIterator<Item = &'a [f64]>,
    ) -> Result<Self, BuildError> {
        let n = levels.len();
        let slopes: Vec<Cow<'a, [f64]>> = slopes.into_iter().map(Cow::Borrowed).collect();
        for (slope, s) in slopes.iter().enumerate() {
            if s.len() != n {
                return Err(BuildError::SlopeLengthMismatch {
                    slope,
                    expected: n,
                    got: s.len(),
                });
            }
            if let Some((index, &value)) = s.iter().enumerate().find(|&(_, &v)| !v.is_finite()) {
                return Err(BuildError::InvalidLoading {
                    slope,
                    index,
                    value,
                });
            }
        }
        if !intercept && slopes.is_empty() {
            return Err(BuildError::EmptyEffect);
        }
        Ok(Self {
            levels: Cow::Borrowed(levels),
            intercept,
            slopes,
        })
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

    #[test]
    fn new_rejects_non_finite_slope_loading_naming_the_offending_slope() {
        let levels = [0u32, 1, 0, 1];
        let ok = [1.0, 2.0, 3.0, 4.0];
        let bad = [1.0, f64::NAN, 3.0, 4.0];
        let err = Effect::new(&levels, true, [&ok[..], &bad[..]]).unwrap_err();
        assert!(matches!(
            err,
            BuildError::InvalidLoading {
                slope: 1,
                index: 1,
                ..
            }
        ));
    }
}
