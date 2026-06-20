//! Domain layer: [`Design`] (design-matrix metadata) and factor-pair [`Subdomain`] construction.

pub(crate) mod cross_tab;
pub(crate) mod factor_pairs;

pub(crate) use cross_tab::{find_all_active_levels, BlockDiagonals, CrossTab};

pub(crate) use factor_pairs::{build_local_domains, LocalDomain};

// ===========================================================================
// Design — categorical fixed-effects design (data + layout)
// ===========================================================================

use crate::observation::{factor_columns, level_at, Store};
use crate::BuildError;

/// Per-factor metadata: level count and global DOF offset.
///
/// Pure design-space layout derived from the store's raw levels — the store
/// holds categories, this records where each factor lands in coefficient space.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FactorMeta {
    /// Number of levels (groups) in this factor.
    pub n_levels: usize,
    /// Starting index in coefficient space for this factor.
    pub offset: usize,
}

/// Fixed-effects design, generic over observation storage.
///
/// `store` holds per-observation factor levels; `factors` holds per-factor
/// metadata (n_levels, offset). The `Design` itself is pure data + layout —
/// matrix-vector products live in the internal operator layer.
#[derive(Clone, Debug)]
pub struct Design<S: Store> {
    /// Observation storage backend (owns or borrows the raw factor levels).
    pub(crate) store: S,
    /// Per-factor metadata: level count and global DOF offset.
    pub(crate) factors: Vec<FactorMeta>,
    /// Number of observations (rows of D).
    pub(crate) n_obs: usize,
    /// Total degrees of freedom (columns of D = sum of levels across factors).
    pub(crate) n_dofs: usize,
}

impl<S: Store> Design<S> {
    /// Construct from a store, inferring the number of levels per factor
    /// from the maximum observed level in each column (`max + 1`).
    pub fn from_store(store: S) -> Result<Self, BuildError> {
        if store.n_obs() == 0 {
            return Err(BuildError::EmptyObservations);
        }

        let cols = factor_columns(&store);
        let mut factors = Vec::with_capacity(store.n_factors());
        let mut offset = 0;
        for (q, &cols_q) in cols.iter().enumerate() {
            let n_levels = (0..store.n_obs())
                .map(|uid| level_at(&store, cols_q, uid, q) + 1)
                .max()
                .unwrap(); // safe: n_obs > 0
            factors.push(FactorMeta { n_levels, offset });
            offset += n_levels;
        }
        let n_obs = store.n_obs();
        Ok(Design {
            store,
            factors,
            n_obs,
            n_dofs: offset,
        })
    }

    /// Number of categorical factors in the design.
    #[inline]
    pub fn n_factors(&self) -> usize {
        self.factors.len()
    }

    /// Number of observations (rows of D).
    #[inline]
    pub fn n_obs(&self) -> usize {
        self.n_obs
    }

    /// Total degrees of freedom (columns of D = sum of levels across factors).
    #[inline]
    pub fn n_dofs(&self) -> usize {
        self.n_dofs
    }
}
