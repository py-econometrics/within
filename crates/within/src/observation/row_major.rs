#[cfg(feature = "ndarray")]
use ndarray::ArrayView2;

use super::{flatten_factor_major_levels, ObservationStore, ObservationWeights};

/// Row-major observation storage: `levels[i * n_factors + q]` is the level
/// for observation `i` in factor `q`.
///
/// Optimal for row-major matvec (D·x, D^T·r) where the inner loop iterates
/// over all factors for a single observation — all levels for obs `i` are in
/// a contiguous cache line.
#[derive(Debug, Clone)]
pub struct RowMajorStore {
    levels: Vec<u32>,
    weights: ObservationWeights,
    n_obs: usize,
    n_factors: usize,
}

impl RowMajorStore {
    /// Construct from factor-major data by transposing `factor_levels[q][i]` → `levels[i * Q + q]`.
    ///
    /// Used in tests where data originates as `Vec<Vec<u32>>`. Production code
    /// should use [`from_array`](Self::from_array) which avoids the transpose when
    /// the input is already C-contiguous.
    pub fn from_factor_major(
        factor_levels: Vec<Vec<u32>>,
        weights: ObservationWeights,
        n_obs: usize,
    ) -> Self {
        let n_factors = factor_levels.len();
        debug_assert!(
            factor_levels.iter().all(|col| col.len() == n_obs),
            "all factor columns must have n_obs entries"
        );
        weights.debug_assert_valid_for(n_obs);
        let levels = flatten_factor_major_levels(&factor_levels, n_obs);
        Self {
            levels,
            weights,
            n_obs,
            n_factors,
        }
    }

    /// Construct from a 2D array view (n_obs × n_factors, usize).
    /// If the array is C-contiguous (row-major), copies directly; otherwise flattens.
    #[cfg(feature = "ndarray")]
    pub fn from_array(categories: ArrayView2<usize>, weights: ObservationWeights) -> Self {
        let n_obs = categories.nrows();
        let n_factors = categories.ncols();
        weights.debug_assert_valid_for(n_obs);
        let levels: Vec<u32> = if let Some(slice) = categories.as_slice() {
            slice.iter().map(|&v| v as u32).collect()
        } else {
            categories.iter().map(|&v| v as u32).collect()
        };
        Self {
            levels,
            weights,
            n_obs,
            n_factors,
        }
    }
}

impl ObservationStore for RowMajorStore {
    #[inline]
    fn n_obs(&self) -> usize {
        self.n_obs
    }

    #[inline]
    fn n_factors(&self) -> usize {
        self.n_factors
    }

    #[inline]
    fn level(&self, obs: usize, factor: usize) -> u32 {
        self.levels[obs * self.n_factors + factor]
    }

    #[inline]
    fn weight(&self, obs: usize) -> f64 {
        self.weights.get(obs)
    }

    #[inline]
    fn is_unweighted(&self) -> bool {
        self.weights.is_unit()
    }

    #[inline]
    fn prefers_row_major_iteration(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observation::sample_factor_levels;
    use crate::observation::FactorMajorStore;

    #[test]
    fn test_row_major_matches_factor_major() {
        let fl = sample_factor_levels();
        let fm = FactorMajorStore::new(fl.clone(), ObservationWeights::Unit, 4)
            .expect("valid factor-major store");
        let rm = RowMajorStore::from_factor_major(fl, ObservationWeights::Unit, 4);

        assert_eq!(rm.n_obs(), fm.n_obs());
        assert_eq!(rm.n_factors(), fm.n_factors());
        for i in 0..4 {
            for q in 0..2 {
                assert_eq!(
                    rm.level(i, q),
                    fm.level(i, q),
                    "mismatch at obs={i}, factor={q}"
                );
            }
            assert_eq!(rm.weight(i), fm.weight(i));
        }
        assert!(rm.is_unweighted());
    }

    #[test]
    fn test_row_major_weighted() {
        let w = vec![0.5, 1.0, 2.0, 3.0];
        let rm = RowMajorStore::from_factor_major(
            sample_factor_levels(),
            ObservationWeights::Dense(w.clone()),
            4,
        );
        assert!(!rm.is_unweighted());
        for (i, &expected) in w.iter().enumerate() {
            assert_eq!(rm.weight(i), expected);
        }
    }

    #[test]
    fn test_row_major_compressed_defaults() {
        let rm =
            RowMajorStore::from_factor_major(sample_factor_levels(), ObservationWeights::Unit, 4);
        assert_eq!(rm.n_unique(), 4);
        assert_eq!(rm.unique_level(1, 0), rm.level(1, 0));
    }
}
