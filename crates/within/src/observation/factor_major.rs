use crate::error::{WithinError, WithinResult};

use super::{ObservationStore, ObservationWeights};

/// Factor-major observation storage: `factor_levels[q][i]` is the level
/// for observation `i` in factor `q`.
///
/// Construction is nearly free — just convert i64 to usize from Python input.
/// Factor-column access is sequential, making it optimal for Gramian build
/// and domain decomposition (which iterate per-factor).
#[derive(Debug, Clone)]
pub struct FactorMajorStore {
    factor_levels: Vec<Vec<u32>>,
    weights: ObservationWeights,
    n_obs: usize,
}

impl FactorMajorStore {
    pub fn new(
        factor_levels: Vec<Vec<u32>>,
        weights: ObservationWeights,
        n_obs: usize,
    ) -> WithinResult<Self> {
        for (factor, col) in factor_levels.iter().enumerate() {
            if col.len() != n_obs {
                return Err(WithinError::ObservationCountMismatch {
                    factor,
                    expected: n_obs,
                    got: col.len(),
                });
            }
        }
        weights.validate_for(n_obs)?;
        Ok(Self {
            factor_levels,
            weights,
            n_obs,
        })
    }

    /// Direct access to the level column for a factor (contiguous slice).
    #[inline]
    pub fn factor_column(&self, factor: usize) -> &[u32] {
        &self.factor_levels[factor]
    }
}

impl ObservationStore for FactorMajorStore {
    #[inline]
    fn n_obs(&self) -> usize {
        self.n_obs
    }

    #[inline]
    fn n_factors(&self) -> usize {
        self.factor_levels.len()
    }

    #[inline]
    fn level(&self, obs: usize, factor: usize) -> u32 {
        self.factor_levels[factor][obs]
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
    fn factor_column(&self, factor: usize) -> Option<&[u32]> {
        Some(self.factor_column(factor))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observation::sample_factor_levels;

    #[test]
    fn test_factor_major_store_basic() {
        let store = FactorMajorStore::new(sample_factor_levels(), ObservationWeights::Unit, 4)
            .expect("valid factor-major store");
        assert_eq!(store.n_obs(), 4);
        assert_eq!(store.n_factors(), 2);
        assert_eq!(store.level(0, 0), 0);
        assert_eq!(store.level(1, 0), 1);
        assert_eq!(store.level(2, 1), 0);
        assert_eq!(store.weight(0), 1.0);
        assert!(store.is_unweighted());
    }

    #[test]
    fn test_factor_major_store_weighted() {
        let store = FactorMajorStore::new(
            vec![vec![0u32, 1, 2]],
            ObservationWeights::Dense(vec![0.5, 1.0, 2.0]),
            3,
        )
        .expect("valid weighted factor-major store");
        assert!(!store.is_unweighted());
        assert_eq!(store.weight(0), 0.5);
        assert_eq!(store.weight(2), 2.0);
    }

    #[test]
    fn test_factor_column() {
        let store = FactorMajorStore::new(
            vec![vec![0u32, 1, 2, 0], vec![3, 2, 1, 0]],
            ObservationWeights::Unit,
            4,
        )
        .expect("valid factor-major store");
        assert_eq!(store.factor_column(0), &[0u32, 1, 2, 0]);
        assert_eq!(store.factor_column(1), &[3u32, 2, 1, 0]);
    }

    #[test]
    fn test_factor_major_compressed_defaults() {
        let store = FactorMajorStore::new(sample_factor_levels(), ObservationWeights::Unit, 4)
            .expect("valid factor-major store");
        assert_eq!(store.n_unique(), 4);
        assert_eq!(store.unique_level(0, 0), store.level(0, 0));
        assert_eq!(store.unique_total_weight(0), store.weight(0));
        assert!(store.is_unique_unweighted());
    }
}
