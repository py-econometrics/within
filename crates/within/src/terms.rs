//! Fixed-effects term inputs: the [`Fe`] factor specification.

use crate::error::BuildError;
use crate::observation::FactorMajorStore;
use crate::solver::IntoDesign;
use crate::Design;

/// One fixed-effects factor term: a per-observation level index, optional
/// varying-slope loadings, and whether to include the level intercept.
pub struct Fe {
    levels: Vec<u32>,
    slopes: Vec<Vec<f64>>,
    // Bridge: only consumed once slope-only terms (intercept = false) lower;
    // until then every term that passes the slopes gate has intercept = true.
    #[allow(dead_code)]
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

impl IntoDesign<'_> for Vec<Fe> {
    type Store = FactorMajorStore;

    fn into_design(self) -> Result<Design<FactorMajorStore>, BuildError> {
        let factor_levels = self
            .into_iter()
            .map(|Fe { levels, slopes, .. }| {
                if slopes.is_empty() {
                    Ok(levels)
                } else {
                    Err(BuildError::Unsupported("varying slopes"))
                }
            })
            .collect::<Result<Vec<Vec<u32>>, BuildError>>()?;
        let n_obs = factor_levels.first().map_or(0, Vec::len);
        Design::from_store(FactorMajorStore::new(factor_levels, n_obs)?)
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

    #[test]
    fn plain_fe_lowers_to_single_factor_design() {
        let fe = Fe::new(vec![0u32, 1, 0, 2], vec![], true).expect("plain FE is valid");
        let design = vec![fe].into_design().expect("plain term list lowers");
        assert_eq!(design.n_factors(), 1);
        assert_eq!(design.n_obs(), 4);
        assert_eq!(design.n_dofs(), 3); // levels 0..=2 ⇒ 3 DOFs
    }

    #[test]
    fn plain_term_list_lowers_to_one_factor_per_term() {
        // term 0: levels 0..=2 ⇒ 3 DOFs; term 1: levels 0..=1 ⇒ 2 DOFs.
        let fe0 = Fe::new(vec![0u32, 1, 0, 2], vec![], true).expect("term 0 valid");
        let fe1 = Fe::new(vec![0u32, 0, 1, 1], vec![], true).expect("term 1 valid");
        let design = vec![fe0, fe1]
            .into_design()
            .expect("plain term list lowers");
        assert_eq!(design.n_factors(), 2);
        assert_eq!(design.n_obs(), 4);
        assert_eq!(design.n_dofs(), 5); // 3 + 2, summed across factors
    }

    #[test]
    fn slope_bearing_term_is_gated_not_supported() {
        let fe = Fe::new(vec![0u32, 1, 0], vec![vec![1.0, 2.0, 3.0]], true)
            .expect("slope term is valid input");
        let result = vec![fe].into_design();
        assert!(matches!(result, Err(BuildError::Unsupported(msg)) if msg.contains("slope")));
    }
}
