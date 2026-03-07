//! Core observation data types: weights, factor metadata,
//! the ObservationStore trait, and the default factor-major backend.

use crate::error::{WithinError, WithinResult};

// ---------------------------------------------------------------------------
// ObservationWeights — zero-cost unweighted path
// ---------------------------------------------------------------------------

/// Observation weights: Unit (all 1.0) or Dense (per-observation).
///
/// The `is_unit()` check happens *outside* inner loops, so the hot path
/// sees either a constant `1.0` or a sequential array read — no per-element branch.
#[derive(Debug, Clone)]
pub enum ObservationWeights {
    /// All weights = 1.0, no storage.
    Unit,
    /// Per-observation weights.
    Dense(Vec<f64>),
}

impl ObservationWeights {
    #[inline]
    pub fn get(&self, obs: usize) -> f64 {
        match self {
            ObservationWeights::Unit => 1.0,
            ObservationWeights::Dense(w) => w[obs],
        }
    }

    #[inline]
    pub fn is_unit(&self) -> bool {
        matches!(self, ObservationWeights::Unit)
    }

    /// Debug-assert that this weight vector is compatible with `n_obs` observations.
    ///
    /// Unit weights are always valid; Dense weights must have exactly `n_obs` entries.
    #[inline]
    pub fn debug_assert_valid_for(&self, n_obs: usize) {
        if let ObservationWeights::Dense(w) = self {
            debug_assert_eq!(w.len(), n_obs, "weights must have n_obs entries");
        }
    }

    /// Validate that this weight vector is compatible with `n_obs` observations.
    pub fn validate_for(&self, n_obs: usize) -> WithinResult<()> {
        if let ObservationWeights::Dense(w) = self {
            if w.len() != n_obs {
                return Err(WithinError::WeightCountMismatch {
                    expected: n_obs,
                    got: w.len(),
                });
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// FactorMeta — per-factor metadata (no observation data)
// ---------------------------------------------------------------------------

/// Per-factor metadata: level count and global DOF offset.
///
/// Separated from observation data — the factor no longer "owns" categories.
#[derive(Debug, Clone, Copy)]
pub struct FactorMeta {
    /// Number of levels (groups) in this factor.
    pub n_levels: usize,
    /// Starting index in coefficient space for this factor.
    pub offset: usize,
}

// ---------------------------------------------------------------------------
// ObservationStore trait
// ---------------------------------------------------------------------------

/// Core abstraction: how observation data is stored and accessed.
///
/// Each backend optimizes for different data characteristics.
/// All implementors must be `Send + Sync` for Rayon parallelism.
pub trait ObservationStore: Send + Sync {
    /// Number of observations.
    fn n_obs(&self) -> usize;

    /// Number of factors.
    fn n_factors(&self) -> usize;

    /// Level index for observation `obs` in factor `factor`.
    fn level(&self, obs: usize, factor: usize) -> u32;

    /// Weight for observation `obs`.
    fn weight(&self, obs: usize) -> f64;

    /// Whether all weights are 1.0 (enables optimized unweighted code paths).
    fn is_unweighted(&self) -> bool;

    /// Optional fast-path access to a factor-major column of levels.
    ///
    /// Stores that naturally keep `level(obs, factor)` as contiguous
    /// `levels[factor][obs]` should return `Some(&levels[factor])`.
    /// Others should return `None` (default).
    fn factor_column(&self, _factor: usize) -> Option<&[u32]> {
        None
    }
}

// ---------------------------------------------------------------------------
// FactorMajorStore
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

#[cfg(test)]
pub fn sample_factor_levels() -> Vec<Vec<u32>> {
    vec![vec![0, 1, 2, 0], vec![0, 1, 0, 1]]
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
