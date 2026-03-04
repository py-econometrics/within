use crate::observation::{FactorMajorStore, ObservationWeights};
use crate::{WithinError, WithinResult};

use super::WeightedDesign;

impl WeightedDesign<FactorMajorStore> {
    #[inline]
    fn categories_to_levels(
        categories: Vec<Vec<i64>>,
        n_levels: &[usize],
    ) -> WithinResult<Vec<Vec<u32>>> {
        if categories.len() != n_levels.len() {
            return Err(WithinError::FactorCountMismatch {
                category_factors: categories.len(),
                level_factors: n_levels.len(),
            });
        }

        let mut factor_levels = Vec::with_capacity(categories.len());
        for (factor, cats) in categories.into_iter().enumerate() {
            let max_level = n_levels[factor];
            let mut levels = Vec::with_capacity(cats.len());
            for (observation, c) in cats.into_iter().enumerate() {
                if c < 0 {
                    return Err(WithinError::NegativeCategory {
                        factor,
                        observation,
                        level: c,
                    });
                }
                let level = c as u32;
                if (level as usize) >= max_level {
                    return Err(WithinError::LevelOutOfRange {
                        factor,
                        observation,
                        level,
                        n_levels: max_level,
                    });
                }
                levels.push(level);
            }
            factor_levels.push(levels);
        }
        Ok(factor_levels)
    }

    /// Create a new unweighted design from i64 category arrays.
    ///
    /// Categories arrive as `i64` from numpy and are converted to `u32` here.
    pub fn new(
        categories: Vec<Vec<i64>>,
        n_levels: Vec<usize>,
        n_rows: usize,
    ) -> WithinResult<Self> {
        let factor_levels = Self::categories_to_levels(categories, &n_levels)?;
        let store = FactorMajorStore::new(factor_levels, ObservationWeights::Unit, n_rows)?;
        Self::from_store(store, &n_levels)
    }

    /// Create a weighted design from i64 category arrays and per-observation weights.
    pub fn new_weighted(
        categories: Vec<Vec<i64>>,
        n_levels: Vec<usize>,
        n_rows: usize,
        weights: Vec<f64>,
    ) -> WithinResult<Self> {
        let factor_levels = Self::categories_to_levels(categories, &n_levels)?;
        let store =
            FactorMajorStore::new(factor_levels, ObservationWeights::Dense(weights), n_rows)?;
        Self::from_store(store, &n_levels)
    }
}
