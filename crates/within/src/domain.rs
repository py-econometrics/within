//! Domain layer: [`Design`] (design-matrix metadata) and factor-pair [`Subdomain`] construction.

pub(crate) mod cross_tab;
pub(crate) mod factor_pairs;

pub(crate) use cross_tab::{find_all_active_levels, BlockDiagonals, CrossTab};

pub(crate) use factor_pairs::{build_local_domains, LocalDomain};

// ===========================================================================
// Design — categorical fixed-effects design (data + layout)
// ===========================================================================

use crate::observation::{level_at, FactorMajorStore, Store};
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
    /// Whether this factor's level column is non-decreasing in the design's
    /// internal observation order (fixed at construction).
    pub sorted: bool,
}

/// The observation data in the design's internal row order: the caller's
/// store passed through untouched, or an owned locality-sorted copy of it
/// (built by [`Design::from_store`]). Reads delegate to whichever is held,
/// so all matvec/cross-tab code consumes it through the [`Store`] trait.
#[derive(Clone, Debug)]
pub(crate) enum InternalStore<S> {
    /// Caller's store as provided — internal order equals caller order.
    AsProvided(S),
    /// Locality-sorted copy; `Design::obs_perm` maps back to caller order.
    Sorted(FactorMajorStore),
}

impl<S: Store> Store for InternalStore<S> {
    #[inline]
    fn n_obs(&self) -> usize {
        match self {
            InternalStore::AsProvided(s) => s.n_obs(),
            InternalStore::Sorted(s) => s.n_obs(),
        }
    }

    #[inline]
    fn n_factors(&self) -> usize {
        match self {
            InternalStore::AsProvided(s) => s.n_factors(),
            InternalStore::Sorted(s) => s.n_factors(),
        }
    }

    #[inline]
    fn level(&self, obs: usize, factor: usize) -> u32 {
        match self {
            InternalStore::AsProvided(s) => s.level(obs, factor),
            InternalStore::Sorted(s) => s.level(obs, factor),
        }
    }

    #[inline]
    fn factor_column(&self, factor: usize) -> Option<&[u32]> {
        match self {
            InternalStore::AsProvided(s) => s.factor_column(factor),
            InternalStore::Sorted(s) => s.factor_column(factor),
        }
    }
}

/// Fixed-effects design, generic over observation storage.
///
/// `store` holds per-observation factor levels; `factors` holds per-factor
/// metadata (n_levels, offset). The `Design` itself is pure data + layout —
/// matrix-vector products live in the internal operator layer.
#[derive(Clone, Debug)]
pub struct Design<S: Store> {
    /// Observation data in internal row order (see [`InternalStore`]).
    pub(crate) store: InternalStore<S>,
    /// Per-factor metadata: level count and global DOF offset.
    pub(crate) factors: Vec<FactorMeta>,
    /// Number of observations (rows of D).
    pub(crate) n_obs: usize,
    /// Total degrees of freedom (columns of D = sum of levels across factors).
    pub(crate) n_dofs: usize,
    /// Locality permutation applied at construction, if any: `obs_perm[k]` is
    /// the caller's original index of the observation at internal position `k`.
    pub(crate) obs_perm: Option<Vec<u32>>,
}

impl<S: Store> Design<S> {
    /// Construct from a store, inferring the number of levels per factor
    /// from the maximum observed level in each column (`max + 1`).
    ///
    /// If the highest-cardinality factor is unsorted, the observations are
    /// copied into an owned locality-sorted store so its gather/scatter runs
    /// sequentially; `obs_perm` records the permutation and the `Solver`
    /// translates per-observation I/O across it. The caller's store is never
    /// mutated.
    pub fn from_store(store: S) -> Result<Self, BuildError> {
        Self::build(store, true)
    }

    /// [`from_store`](Self::from_store) without the locality sort: rows stay
    /// in caller order regardless of sortedness.
    ///
    /// Escape hatch for measuring the sort's effect (profiling baselines,
    /// transparency oracles); production solves want `from_store`.
    #[doc(hidden)]
    pub fn from_store_unsorted(store: S) -> Result<Self, BuildError> {
        Self::build(store, false)
    }

    fn build(store: S, locality_sort: bool) -> Result<Self, BuildError> {
        if store.n_obs() == 0 {
            return Err(BuildError::EmptyObservations);
        }

        let n_obs = store.n_obs();
        let mut factors = Vec::with_capacity(store.n_factors());
        let mut offset = 0;
        for q in 0..store.n_factors() {
            // One pass per column: level count (max + 1) and sortedness.
            let col = store.factor_column(q);
            let mut max = 0;
            let mut sorted = true;
            let mut prev = 0;
            for i in 0..n_obs {
                let v = level_at(&store, col, i, q);
                max = max.max(v);
                sorted &= v >= prev;
                prev = v;
            }
            let n_levels = max + 1;
            factors.push(FactorMeta {
                n_levels,
                offset,
                sorted,
            });
            offset += n_levels;
        }

        // Sort by the highest-cardinality factor (ties resolve to the last):
        // it makes the dominant gather/scatter sequential. The permutation
        // indexes observations as u32 (`obs_perm`); beyond u32::MAX
        // rows it is unrepresentable, so skip the optimization and keep
        // caller order — the solve itself has no such limit.
        let dominant = (0..factors.len()).max_by_key(|&q| factors[q].n_levels);
        let (store, obs_perm) = match dominant {
            Some(d) if locality_sort && !factors[d].sorted && u32::try_from(n_obs).is_ok() => {
                // Argsort key: the contiguous fast path when available, else a
                // gathered copy (strided/virtual columns expose no slice),
                // kept alive so the column gather below reads it back rather
                // than paying a second strided pass through `Store::level`.
                let gathered: Vec<u32>;
                let key: &[u32] = match store.factor_column(d) {
                    Some(col) => col,
                    None => {
                        gathered = (0..n_obs).map(|i| store.level(i, d)).collect();
                        &gathered
                    }
                };
                // Stable argsort, preserving caller order within a level.
                // Must be `sort_by_cached_key`, NOT `sort_by_key`: the latter
                // re-gathers `key[i]` O(n log n) times — cache-miss-bound once
                // the column spills out of cache, and it dominated setup at
                // tens of millions of rows. The guard above proved n_obs
                // fits u32.
                let mut perm: Vec<u32> = (0..n_obs as u32).collect();
                perm.sort_by_cached_key(|&i| key[i as usize]);
                // Gather every column into owned factor-major storage,
                // tracking sortedness in the new order within the same pass.
                // The dominant column comes out sorted by construction, and
                // factors nested in — or duplicating — it come out sorted
                // too, keeping the coalesced scatter for them.
                let factor_levels: Vec<Vec<u32>> = (0..factors.len())
                    .map(|q| {
                        // The dominant factor always reads through `key`: it
                        // is either the store's contiguous column or the copy
                        // gathered for the argsort above.
                        let src = if q == d {
                            Some(key)
                        } else {
                            store.factor_column(q)
                        };
                        let mut col = Vec::with_capacity(n_obs);
                        let mut sorted = true;
                        let mut prev = 0;
                        for &k in &perm {
                            let v = match src {
                                Some(s) => s[k as usize],
                                None => store.level(k as usize, q),
                            };
                            sorted &= v >= prev;
                            prev = v;
                            col.push(v);
                        }
                        factors[q].sorted = sorted;
                        col
                    })
                    .collect();
                let sorted_store = FactorMajorStore {
                    factor_levels,
                    n_obs,
                };
                (InternalStore::Sorted(sorted_store), Some(perm))
            }
            _ => (InternalStore::AsProvided(store), None),
        };

        Ok(Design {
            store,
            factors,
            n_obs,
            n_dofs: offset,
            obs_perm,
        })
    }

    /// Translate a per-observation input from caller order into internal
    /// order: `out[k] = v[obs_perm[k]]`. Borrows unchanged when not permuted.
    pub(crate) fn permute_obs_in<'v>(&self, v: &'v [f64]) -> std::borrow::Cow<'v, [f64]> {
        debug_assert_eq!(v.len(), self.n_obs);
        match &self.obs_perm {
            None => std::borrow::Cow::Borrowed(v),
            Some(perm) => std::borrow::Cow::Owned(perm.iter().map(|&i| v[i as usize]).collect()),
        }
    }

    /// Translate a per-observation result from internal order back into caller
    /// order: `out[obs_perm[k]] = v[k]`. Returns `v` unchanged when not permuted.
    pub(crate) fn permute_obs_out(&self, v: Vec<f64>) -> Vec<f64> {
        debug_assert_eq!(v.len(), self.n_obs);
        match &self.obs_perm {
            None => v,
            Some(perm) => {
                let mut out = vec![0.0; v.len()];
                for (k, &orig) in perm.iter().enumerate() {
                    out[orig as usize] = v[k];
                }
                out
            }
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observation::{ArrayStore, FactorMajorStore};

    #[test]
    fn from_store_sorts_owned_unsorted_dominant() {
        // Factor 0 (3 levels) dominates and is unsorted; factor 1 starts sorted.
        let store = FactorMajorStore::new(vec![vec![2, 0, 1, 0], vec![0, 0, 1, 1]], 4).unwrap();
        let design = Design::from_store(store).unwrap();

        // Stable argsort of [2,0,1,0] → original indices [1,3,2,0].
        assert_eq!(design.obs_perm.as_deref(), Some(&[1u32, 3, 2, 0][..]));
        assert!(design.factors[0].sorted);
        // Rescanned after the sort: factor 1's permuted column [0,1,1,0] is
        // no longer non-decreasing.
        assert!(!design.factors[1].sorted);

        let col = |q| {
            (0..4)
                .map(|i| design.store.level(i, q))
                .collect::<Vec<u32>>()
        };
        assert_eq!(col(0), [0, 0, 1, 2]);
        assert_eq!(col(1), [0, 1, 1, 0]);
    }

    #[test]
    fn rescan_marks_nested_factor_sorted_after_permutation() {
        // Factor 1 is nested in the dominant factor 0 (level = col0 / 2), so
        // sorting by factor 0 also sorts factor 1; the post-sort rescan must
        // detect that instead of conservatively flagging it unsorted.
        let col0 = vec![3u32, 0, 2, 1];
        let col1: Vec<u32> = col0.iter().map(|&v| v / 2).collect();
        let store = FactorMajorStore::new(vec![col0, col1], 4).unwrap();
        let design = Design::from_store(store).unwrap();
        assert!(design.obs_perm.is_some());
        assert!(design.factors[0].sorted);
        assert!(design.factors[1].sorted);
    }

    #[test]
    fn from_store_keeps_sorted_input() {
        let store = FactorMajorStore::new(vec![vec![0, 0, 1, 2], vec![1, 0, 1, 0]], 4).unwrap();
        let design = Design::from_store(store).unwrap();
        assert!(design.obs_perm.is_none());
        assert!(design.factors[0].sorted);
        assert!(!design.factors[1].sorted);
    }

    #[test]
    fn array_store_sorts_unsorted_dominant() {
        // C-order two-column array: `factor_column` is `None` for every
        // column, so both the argsort key and the column gather take the
        // virtual per-element fallback. The borrowed view is never mutated;
        // from_store copies the columns once into an owned sorted internal
        // store.
        let arr = ndarray::Array2::from_shape_vec((4, 2), vec![2u32, 0, 0, 0, 1, 1, 0, 1]).unwrap();
        let store = ArrayStore::new(arr.view()).unwrap();
        assert!(store.factor_column(0).is_none(), "C-order has no fast path");
        let design = Design::from_store(store).unwrap();
        let perm = design.obs_perm.as_ref().expect("permutation applied");
        assert_eq!(perm, &[1, 3, 2, 0]);
        let col = |q| {
            (0..4)
                .map(|i| design.store.level(i, q))
                .collect::<Vec<u32>>()
        };
        assert_eq!(col(0), [0, 0, 1, 2]);
        assert_eq!(col(1), [0, 1, 1, 0]);
    }
}
