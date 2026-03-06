//! Weighted design matrix: `WeightedDesign<S>` generic over `ObservationStore`.
//!
//! Stores per-factor metadata via [`FactorMeta`] and delegates observation
//! data access to the pluggable store backend `S`. Provides design matrix
//! operations (D·x, D^T·r, D^T·W·r, gramian diagonal) as methods.
//!
//! `FixedEffectsDesign` is a type alias for the unweighted `FactorMajorStore`
//! backend, preserving backward compatibility.
use crate::observation::{FactorMajorStore, FactorMeta, ObservationStore};
use crate::{WithinError, WithinResult};

mod factor_major;
mod ops;

#[cfg(test)]
mod tests;

/// Weighted fixed-effects design matrix, generic over observation storage.
///
/// `store` holds per-observation data (levels, weights); `factors` holds
/// per-factor metadata (n_levels, offset). The generic parameter `S` is
/// monomorphized — no vtable, no per-element branch.
pub struct WeightedDesign<S: ObservationStore> {
    pub store: S,
    pub factors: Vec<FactorMeta>,
    pub n_rows: usize,
    pub n_dofs: usize,
}

impl<S: ObservationStore + std::fmt::Debug> std::fmt::Debug for WeightedDesign<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WeightedDesign")
            .field("store", &self.store)
            .field("factors", &self.factors)
            .field("n_rows", &self.n_rows)
            .field("n_dofs", &self.n_dofs)
            .finish()
    }
}

/// Backward-compatible alias: unweighted, factor-major storage.
pub type FixedEffectsDesign = WeightedDesign<FactorMajorStore>;

impl<S: ObservationStore> WeightedDesign<S> {
    /// Construct from a store and per-factor level counts.
    pub fn from_store(store: S, n_levels: &[usize]) -> WithinResult<Self> {
        if store.n_factors() != n_levels.len() {
            return Err(WithinError::FactorCountMismatch {
                category_factors: store.n_factors(),
                level_factors: n_levels.len(),
            });
        }

        for (factor, &nl) in n_levels.iter().enumerate() {
            if nl == 0 {
                return Err(WithinError::EmptyLevelSet { factor });
            }
        }

        for (factor, &n_levels_factor) in n_levels.iter().enumerate().take(store.n_factors()) {
            for uid in 0..store.n_unique() {
                let level = store.unique_level(uid, factor);
                if (level as usize) >= n_levels_factor {
                    return Err(WithinError::LevelOutOfRange {
                        factor,
                        observation: uid,
                        level,
                        n_levels: n_levels_factor,
                    });
                }
            }
        }

        let mut factors = Vec::with_capacity(n_levels.len());
        let mut offset = 0;
        for &nl in n_levels {
            factors.push(FactorMeta {
                n_levels: nl,
                offset,
            });
            offset += nl;
        }
        let n_rows = store.n_obs();
        Ok(WeightedDesign {
            store,
            factors,
            n_rows,
            n_dofs: offset,
        })
    }

    /// Weight for a unique observation tuple, respecting the store's weighting mode.
    #[inline]
    pub fn uid_weight(&self, uid: usize) -> f64 {
        if self.store.is_unique_unweighted() {
            1.0
        } else {
            self.store.unique_total_weight(uid)
        }
    }

    #[inline]
    pub fn n_factors(&self) -> usize {
        self.factors.len()
    }

    /// Block offsets including the trailing total (length = n_factors + 1).
    pub fn block_offsets(&self) -> Vec<usize> {
        let mut bo: Vec<usize> = self.factors.iter().map(|f| f.offset).collect();
        bo.push(self.n_dofs);
        bo
    }

    /// Level index for observation `i` in factor `q`.
    #[inline]
    pub fn level(&self, i: usize, q: usize) -> u32 {
        self.store.level(i, q)
    }

    /// Weight for observation `i`.
    #[inline]
    pub fn weight(&self, i: usize) -> f64 {
        self.store.weight(i)
    }
}
