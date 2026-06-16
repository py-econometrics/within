//! Fixed-effects term inputs: the [`Fe`] factor specification.

use crate::error::BuildError;

/// One fixed-effects factor term: a per-observation level index, optional
/// varying-slope loadings, and whether to include the level intercept.
// Bridge: fields are read once `Fe` lowers to a store; remove when wired.
#[allow(dead_code)]
pub struct Fe {
    levels: Vec<u32>,
    slopes: Vec<Vec<f64>>,
    intercept: bool,
}

impl Fe {
    /// Builds a fixed-effects term from per-observation level indices, optional
    /// varying-slope loadings, and whether to include the level intercept.
    pub fn new(
        levels: Vec<u32>,
        slopes: Vec<Vec<f64>>,
        intercept: bool,
    ) -> Result<Self, BuildError> {
        if !intercept && slopes.is_empty() {
            return Err(BuildError::EmptyTerm);
        }
        for (slope, column) in slopes.iter().enumerate() {
            if column.len() != levels.len() {
                return Err(BuildError::SlopeCountMismatch {
                    slope,
                    expected: levels.len(),
                    got: column.len(),
                });
            }
        }
        Ok(Self {
            levels,
            slopes,
            intercept,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_fe_is_ok_and_stores_fields() {
        let levels = vec![0u32, 1, 0, 2];
        let fe = Fe::new(levels.clone(), vec![], true).expect("plain FE is valid");
        assert_eq!(fe.levels, levels);
        assert!(fe.slopes.is_empty());
        assert!(fe.intercept);
    }

    #[test]
    fn empty_term_is_rejected() {
        let result = Fe::new(vec![0u32, 1, 0], vec![], false);
        assert!(matches!(result, Err(BuildError::EmptyTerm)));
    }

    #[test]
    fn slope_length_mismatch_is_rejected() {
        let result = Fe::new(vec![0u32, 1, 0], vec![vec![1.0, 2.0]], true);
        assert!(matches!(
            result,
            Err(BuildError::SlopeCountMismatch {
                slope: 0,
                expected: 3,
                got: 2
            })
        ));
    }

    #[test]
    fn slope_only_term_is_ok_and_stores_slope() {
        let slope = vec![1.0, 2.0];
        let fe = Fe::new(vec![0u32, 1], vec![slope.clone()], false).expect("slope-only is valid");
        assert!(!fe.intercept);
        assert_eq!(fe.slopes, vec![slope]);
    }
}
