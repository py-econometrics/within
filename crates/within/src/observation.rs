//! Core observation data types: weights, factor metadata,
//! the ObservationStore trait, and the default factor-major backend.

mod factor_major;

use crate::error::{WithinError, WithinResult};

pub use factor_major::FactorMajorStore;

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
///
/// The `n_unique` / `unique_level` / `unique_total_weight` methods support
/// compressed backends that deduplicate identical observation tuples. The
/// defaults provide identity behavior (n_unique == n_obs) so uncompressed
/// backends inherit them for free.
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

    // -- Compressed-aware defaults (identity for uncompressed stores) --------

    /// Number of unique observation tuples. Defaults to `n_obs()`.
    fn n_unique(&self) -> usize {
        self.n_obs()
    }

    /// Level for unique tuple `uid` in factor `factor`. Defaults to `level(uid, factor)`.
    fn unique_level(&self, uid: usize, factor: usize) -> u32 {
        self.level(uid, factor)
    }

    /// Total aggregated weight for unique tuple `uid`. Defaults to `weight(uid)`.
    fn unique_total_weight(&self, uid: usize) -> f64 {
        self.weight(uid)
    }

    /// Whether all unique weights are 1.0 (for uncompressed unweighted stores).
    fn is_unique_unweighted(&self) -> bool {
        self.is_unweighted()
    }

    /// Whether this store benefits from row-major iteration (outer loop on
    /// observations, inner loop on factors). Defaults to `false`.
    ///
    /// Override to `true` when `level(obs, factor)` has stride-1 access across
    /// factors for a fixed observation.
    fn prefers_row_major_iteration(&self) -> bool {
        false
    }

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
// Test helpers
// ---------------------------------------------------------------------------

#[cfg(test)]
pub fn sample_factor_levels() -> Vec<Vec<u32>> {
    vec![vec![0, 1, 2, 0], vec![0, 1, 0, 1]]
}
