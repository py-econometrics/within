//! Domain layer: design matrix layout and factor-pair subdomain construction.
//!
//! This module sits between raw observation storage ([`crate::observation`]) and
//! the linear-algebra operators ([`crate::operator`]).  It answers two questions:
//!
//! 1. **What does the design matrix look like?** — [`Design`] wraps a
//!    [`Store`] with per-factor metadata ([`FactorMeta`]). It is pure data
//!    plus layout: dimensions and the per-factor offset table that the
//!    operators iterate over. Observation weights live *outside* the design
//!    and are carried by the operators that need them.
//!
//! 2. **How is the problem decomposed into subdomains?** — The `factor_pairs`
//!    submodule builds one [`Subdomain`] per connected component of each factor
//!    pair, with partition-of-unity weights that ensure the additive Schwarz
//!    preconditioner is mathematically correct. (Partition-of-unity weights
//!    are *geometric* and unrelated to observation weights.)
//!
//! # Design matrix structure
//!
//! The design matrix **D** is a block matrix with one block per factor. Each
//! block is a "one-hot" matrix: observation (row) *i* has a single 1
//! corresponding to its level in that factor. With Q factors and `n_q` levels
//! each, D has shape `(n_obs, sum(n_q))` and exactly Q nonzeros per row.
//!
//! ```text
//! D = [ D_1 | D_2 | ... | D_Q ]     (n_obs × n_dofs)
//!
//! where D_q[i, j] = 1  if observation i has level j in factor q
//!                    0  otherwise
//! ```
//!
//! The coefficient vector **x** is laid out as `[x_1, x_2, ..., x_Q]` where
//! `x_q` starts at `factors[q].offset` and has length `factors[q].n_levels`.
//!
//! # Domain decomposition and factor pairs
//!
//! The normal-equation Gramian `G = D^T W D` has a natural block structure:
//! diagonal blocks are diagonal matrices (weighted level counts) and off-diagonal
//! blocks `D_q^T W D_r` capture the co-occurrence between each pair of factors.
//! Each factor pair `(q, r)` defines a subdomain whose DOFs are the union of
//! active levels in factors q and r. When the factor-pair bipartite graph has
//! multiple connected components, each component becomes a separate subdomain.

pub(crate) mod factor_pairs;

pub(crate) use factor_pairs::{
    build_domains_and_gramian_blocks, build_gramian_blocks, build_local_domains, PairBlockData,
};

// Re-exports from schwarz-precond
pub use schwarz_precond::PartitionWeights;
pub use schwarz_precond::SubdomainCore;

use crate::observation::{FactorMeta, Store};
use crate::{WithinError, WithinResult};

/// A local subdomain corresponding to a pair of factors.
#[derive(Clone)]
pub struct Subdomain {
    /// Indices `(q, r)` of the two factors this subdomain covers.
    pub factor_pair: (usize, usize),
    /// Generic subdomain core: global DOF indices, restriction, and partition-of-unity weights.
    pub core: SubdomainCore,
}

impl std::fmt::Debug for Subdomain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Subdomain")
            .field("factor_pair", &self.factor_pair)
            .field("n_dofs", &self.core.n_local())
            .finish()
    }
}

// ===========================================================================
// Design — pure data + factor layout
// ===========================================================================

/// Fixed-effects design matrix layout, generic over observation storage.
///
/// `store` holds per-observation level data; `factors` holds per-factor
/// metadata (n_levels, offset). Observation weights are *not* part of
/// `Design` — they are carried by the operators that need them.
pub struct Design<S: Store> {
    /// Observation storage backend (owns or borrows the raw level data).
    pub store: S,
    /// Per-factor metadata: level count and global DOF offset.
    pub factors: Vec<FactorMeta>,
    /// Number of observations (rows of D).
    pub n_rows: usize,
    /// Total degrees of freedom (columns of D = sum of levels across factors).
    pub n_dofs: usize,
}

impl<S: Store + Clone> Clone for Design<S> {
    fn clone(&self) -> Self {
        Self {
            store: self.store.clone(),
            factors: self.factors.clone(),
            n_rows: self.n_rows,
            n_dofs: self.n_dofs,
        }
    }
}

impl<S: Store + std::fmt::Debug> std::fmt::Debug for Design<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Design")
            .field("store", &self.store)
            .field("factors", &self.factors)
            .field("n_rows", &self.n_rows)
            .field("n_dofs", &self.n_dofs)
            .finish()
    }
}

impl<S: Store> Design<S> {
    /// Construct from a store, inferring the number of levels per factor
    /// from the maximum observed level in each column (`max + 1`).
    pub fn from_store(store: S) -> WithinResult<Self> {
        if store.n_obs() == 0 {
            return Err(WithinError::EmptyObservations);
        }

        let mut factors = Vec::with_capacity(store.n_factors());
        let mut offset = 0;
        for q in 0..store.n_factors() {
            let n_levels = (0..store.n_obs())
                .map(|uid| store.level(uid, q) as usize + 1)
                .max()
                .unwrap(); // safe: n_obs > 0
            factors.push(FactorMeta { n_levels, offset });
            offset += n_levels;
        }
        let n_rows = store.n_obs();
        Ok(Design {
            store,
            factors,
            n_rows,
            n_dofs: offset,
        })
    }

    /// Number of categorical factors in the design.
    #[inline]
    pub fn n_factors(&self) -> usize {
        self.factors.len()
    }

    /// Pre-compute factor column slices for all factors.
    ///
    /// Returns a vec where entry `q` is the store's contiguous column for factor `q`,
    /// or `None` if the store doesn't support direct column access.
    pub fn factor_columns(&self) -> Vec<Option<&[u32]>> {
        self.factors
            .iter()
            .enumerate()
            .map(|(q, _)| self.store.factor_column(q))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observation::FactorMajorStore;

    fn make_test_design() -> Design<FactorMajorStore> {
        let categories = vec![vec![0, 1, 2, 0, 1], vec![0, 1, 2, 3, 0]];
        let store = FactorMajorStore::new(categories, 5).expect("valid factor-major store");
        Design::from_store(store).expect("valid test design")
    }

    #[test]
    fn test_construction() {
        let dm = make_test_design();
        assert_eq!(dm.n_factors(), 2);
        assert_eq!(dm.n_dofs, 7);
        assert_eq!(dm.n_rows, 5);
        assert_eq!(dm.factors[0].offset, 0);
        assert_eq!(dm.factors[1].offset, 3);
        let block_offsets: Vec<usize> = dm
            .factors
            .iter()
            .map(|f| f.offset)
            .chain(std::iter::once(dm.n_dofs))
            .collect();
        assert_eq!(block_offsets, vec![0, 3, 7]);
    }

    #[test]
    fn test_factor_meta() {
        let dm = make_test_design();
        assert_eq!(dm.factors[0].n_levels, 3);
        assert_eq!(dm.factors[1].n_levels, 4);
        // Categories are in the store, not in FactorMeta
        assert_eq!(dm.store.level(0, 0), 0);
        assert_eq!(dm.store.level(1, 0), 1);
        assert_eq!(dm.store.level(2, 0), 2);
        assert_eq!(dm.store.level(3, 0), 0);
        assert_eq!(dm.store.level(4, 0), 1);
        assert_eq!(dm.store.level(0, 1), 0);
        assert_eq!(dm.store.level(1, 1), 1);
        assert_eq!(dm.store.level(4, 1), 0);
    }
}
