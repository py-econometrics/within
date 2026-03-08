pub(crate) mod factor_pairs;

pub(crate) use factor_pairs::build_local_domains;
pub(crate) use factor_pairs::{build_domains_and_gramian_blocks, PairBlockData};

// Re-exports from schwarz-precond
pub use schwarz_precond::PartitionWeights;
pub use schwarz_precond::SubdomainCore;

/// A local subdomain corresponding to a pair of factors.
#[derive(Clone)]
pub struct Subdomain {
    pub factor_pair: (usize, usize),
    pub core: SubdomainCore,
}

impl std::fmt::Debug for Subdomain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Subdomain")
            .field("factor_pair", &self.factor_pair)
            .field("n_dofs", &self.core.global_indices.len())
            .finish()
    }
}

// ===========================================================================
// Weighted design matrix (formerly schema.rs)
// ===========================================================================

// Weighted design matrix: `WeightedDesign<S>` generic over `ObservationStore`.
//
// Stores per-factor metadata via `FactorMeta` and delegates observation
// data access to the pluggable store backend `S`. Provides design matrix
// operations (D·x, D^T·r, D^T·W·r, gramian diagonal) as methods.
//
// `FixedEffectsDesign` is a type alias for the unweighted `FactorMajorStore`
// backend, preserving backward compatibility.
use std::sync::atomic::Ordering;

use portable_atomic::AtomicF64;
use rayon::prelude::*;

use crate::observation::{FactorMajorStore, FactorMeta, ObservationStore};
use crate::{WithinError, WithinResult};

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
            for uid in 0..store.n_obs() {
                let level = store.level(uid, factor);
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

    /// Weight for an observation, respecting the store's weighting mode.
    #[inline]
    pub fn uid_weight(&self, uid: usize) -> f64 {
        if self.store.is_unweighted() {
            1.0
        } else {
            self.store.weight(uid)
        }
    }

    #[inline]
    pub fn n_factors(&self) -> usize {
        self.factors.len()
    }

    /// Pre-compute factor column slices for all factors.
    ///
    /// Returns a vec where entry `q` is the store's contiguous column for factor `q`,
    /// or `None` if the store doesn't support direct column access.
    fn factor_columns(&self) -> Vec<Option<&[u32]>> {
        self.factors
            .iter()
            .enumerate()
            .map(|(q, _)| self.store.factor_column(q))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Design matrix operations (D·x, D^T·r, D^T·W·r, gramian diagonal)
// ---------------------------------------------------------------------------

/// Minimum number of rows before scatter/gather loops are parallelized.
const PAR_THRESHOLD: usize = 10_000;

/// Factor-level threshold for choosing between fold and atomic scatter-add.
///
/// Factors with fewer than this many levels use thread-local fold/reduce
/// (O(n_levels * n_threads) memory). Larger factors use atomic CAS instead,
/// which has low contention when bins vastly outnumber threads.
/// 100K levels * 8 bytes * ~24 Rayon tasks ~ 19 MB — fits comfortably.
const SCATTER_LOCAL_THRESHOLD: usize = 100_000;

/// Strategy for a single factor's scatter-add loop.
enum ScatterStrategy {
    /// Plain sequential loop — used when n_rows is below `PAR_THRESHOLD`.
    Sequential,
    /// Parallel fold/reduce with thread-local accumulators — for small factors.
    Fold,
    /// Parallel atomic CAS — for large factors with low contention.
    Atomic,
}

#[inline]
fn level_from_column_or_store<S: ObservationStore>(
    store: &S,
    levels: Option<&[u32]>,
    row: usize,
    factor: usize,
) -> usize {
    match levels {
        Some(col) => col[row] as usize,
        None => store.level(row, factor) as usize,
    }
}

/// Scatter-add for a single factor, dispatched by strategy.
///
/// Accumulates `value_fn(i)` into `slice[level(i, q)]` for all rows, using the
/// requested parallelization strategy. The `atomic_buf` is reused across calls
/// to avoid repeated allocation in the `Atomic` path.
#[allow(clippy::too_many_arguments)]
fn scatter_add_single_factor<S: ObservationStore>(
    slice: &mut [f64],
    n_rows: usize,
    n_levels: usize,
    store: &S,
    levels: Option<&[u32]>,
    q: usize,
    value_fn: &(impl Fn(usize) -> f64 + Sync),
    strategy: ScatterStrategy,
    atomic_buf: &mut Vec<AtomicF64>,
) {
    match strategy {
        ScatterStrategy::Sequential => {
            for i in 0..n_rows {
                let level = level_from_column_or_store(store, levels, i, q);
                slice[level] += value_fn(i);
            }
        }
        ScatterStrategy::Fold => {
            let min_len = (n_rows / rayon::current_num_threads().max(1)).max(1024);
            let result: Vec<f64> = (0..n_rows)
                .into_par_iter()
                .with_min_len(min_len)
                .fold(
                    || vec![0.0f64; n_levels],
                    |mut acc, i| {
                        let level = level_from_column_or_store(store, levels, i, q);
                        acc[level] += value_fn(i);
                        acc
                    },
                )
                .reduce(
                    || vec![0.0f64; n_levels],
                    |mut a, b| {
                        for (ai, bi) in a.iter_mut().zip(b.iter()) {
                            *ai += *bi;
                        }
                        a
                    },
                );
            for (d, r) in slice.iter_mut().zip(result.iter()) {
                *d += *r;
            }
        }
        ScatterStrategy::Atomic => {
            atomic_buf.clear();
            atomic_buf.extend(slice.iter().map(|&v| AtomicF64::new(v)));
            (0..n_rows).into_par_iter().for_each(|i| {
                let level = level_from_column_or_store(store, levels, i, q);
                atomic_buf[level].fetch_add(value_fn(i), Ordering::Relaxed);
            });
            for (d, a) in slice.iter_mut().zip(atomic_buf.iter()) {
                *d = a.load(Ordering::Relaxed);
            }
        }
    }
}

impl<S: ObservationStore> WeightedDesign<S> {
    /// Gather-add: `dst[i] += src[offset_q + level(i, q)]` for each factor `q` and row `i`.
    ///
    /// This is the core loop of `y = D·x`. Loop order is chosen based on the
    /// store's preferred iteration pattern for cache locality.
    ///
    /// For large problems (n_rows > 10 000), rows are partitioned into chunks
    /// and processed in parallel via Rayon `par_chunks_mut`.
    #[inline]
    fn gather_add(&self, src: &[f64], dst: &mut [f64]) {
        const CHUNK_SIZE: usize = 4096;
        let factor_columns = self.factor_columns();

        if self.n_rows > PAR_THRESHOLD {
            // Parallel path: each chunk processes its own row range.
            // The inner loop iterates factors inside each chunk, which is optimal for the
            // common case (2-3 factors) where all factor data fits in L1 cache. For many
            // factors (10+) a layout with factors in the outer loop might help, but
            // econometric models typically have 2-5 factors so this isn't worth optimizing.
            dst.par_chunks_mut(CHUNK_SIZE)
                .enumerate()
                .for_each(|(chunk_idx, chunk)| {
                    let row_start = chunk_idx * CHUNK_SIZE;
                    for (q, f) in self.factors.iter().enumerate() {
                        let levels = factor_columns[q];
                        for (local, dst_val) in chunk.iter_mut().enumerate() {
                            let i = row_start + local;
                            let level = level_from_column_or_store(&self.store, levels, i, q);
                            *dst_val += src[f.offset + level];
                        }
                    }
                });
        } else {
            // Sequential factor-major: outer loop on factors, inner on observations.
            for (q, f) in self.factors.iter().enumerate() {
                let levels = factor_columns[q];
                for (i, dst_i) in dst.iter_mut().enumerate().take(self.n_rows) {
                    let level = level_from_column_or_store(&self.store, levels, i, q);
                    *dst_i += src[f.offset + level];
                }
            }
        }
    }

    /// Scatter-add: `dst[offset_q + level(i, q)] += value_fn(i)` for each factor `q` and row `i`.
    ///
    /// This is the core loop of `x = D^T · r` (and weighted variant `D^T · W · r`).
    /// The `value_fn` closure computes the per-row contribution:
    /// - unweighted: `|i| r[i]`
    /// - weighted:   `|i| w[i] * r[i]`
    ///
    /// For large problems, each factor's row loop is parallelized:
    /// - Small factors (< 100K levels): thread-local fold/reduce (avoids CAS contention)
    /// - Large factors: atomic CAS scatter (low contention on millions of bins)
    ///   Factors are processed sequentially so each gets the full thread pool.
    #[inline]
    fn scatter_add(&self, dst: &mut [f64], value_fn: impl Fn(usize) -> f64 + Sync) {
        let factor_columns = self.factor_columns();
        let parallel = self.n_rows > PAR_THRESHOLD;
        let max_levels = self.factors.iter().map(|f| f.n_levels).max().unwrap_or(0);
        let mut atomic_buf: Vec<AtomicF64> = Vec::with_capacity(max_levels);

        for (q, f) in self.factors.iter().enumerate() {
            let slice = &mut dst[f.offset..f.offset + f.n_levels];
            let levels = factor_columns[q];
            let strategy = if !parallel {
                ScatterStrategy::Sequential
            } else if f.n_levels < SCATTER_LOCAL_THRESHOLD {
                ScatterStrategy::Fold
            } else {
                ScatterStrategy::Atomic
            };
            scatter_add_single_factor(
                slice,
                self.n_rows,
                f.n_levels,
                &self.store,
                levels,
                q,
                &value_fn,
                strategy,
                &mut atomic_buf,
            );
        }
    }

    /// y = D·x  (gather-add, no weights)
    pub fn matvec_d(&self, x: &[f64], y: &mut [f64]) {
        debug_assert_eq!(x.len(), self.n_dofs);
        debug_assert_eq!(y.len(), self.n_rows);
        y.fill(0.0);
        self.gather_add(x, y);
    }

    /// x = D^T·r  (scatter-add, no weights)
    pub fn rmatvec_dt(&self, r: &[f64], x: &mut [f64]) {
        debug_assert_eq!(r.len(), self.n_rows);
        debug_assert_eq!(x.len(), self.n_dofs);
        x.fill(0.0);
        self.scatter_add(x, |i| r[i]);
    }

    /// x = D^T·W·r  (weighted scatter-add)
    ///
    /// For unweighted stores, this is identical to `rmatvec_dt`.
    /// The branch is outside the inner loop.
    pub fn rmatvec_wdt(&self, r: &[f64], x: &mut [f64]) {
        debug_assert_eq!(r.len(), self.n_rows);
        debug_assert_eq!(x.len(), self.n_dofs);
        if self.store.is_unweighted() {
            return self.rmatvec_dt(r, x);
        }
        x.fill(0.0);
        self.scatter_add(x, |i| self.store.weight(i) * r[i]);
    }

    /// Diagonal of D^T·W·D (weighted level counts).
    ///
    /// Entry `offset_q + j` = sum of weights for observations with level `j` in factor `q`.
    pub fn gramian_diagonal(&self) -> Vec<f64> {
        let mut diag = vec![0.0f64; self.n_dofs];
        let n_obs = self.store.n_obs();
        for (q, f) in self.factors.iter().enumerate() {
            for uid in 0..n_obs {
                diag[f.offset + self.store.level(uid, q) as usize] += self.uid_weight(uid);
            }
        }
        diag
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observation::{FactorMajorStore, ObservationWeights};

    fn make_test_design() -> FixedEffectsDesign {
        let categories = vec![vec![0, 1, 2, 0, 1], vec![0, 1, 2, 3, 0]];
        let n_levels = vec![3, 4];
        let store = FactorMajorStore::new(categories, ObservationWeights::Unit, 5)
            .expect("valid factor-major store");
        FixedEffectsDesign::from_store(store, &n_levels).expect("valid test design")
    }

    fn make_weighted_design(weights: Vec<f64>) -> FixedEffectsDesign {
        let categories = vec![vec![0, 1, 0, 1], vec![0, 0, 1, 1]];
        let n_levels = vec![2, 2];
        let store = FactorMajorStore::new(categories, ObservationWeights::Dense(weights), 4)
            .expect("valid weighted factor-major store");
        FixedEffectsDesign::from_store(store, &n_levels).expect("valid weighted design")
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

    #[test]
    fn test_matvec_d() {
        let dm = make_test_design();
        let x = vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 40.0];
        let mut y = vec![0.0; 5];
        dm.matvec_d(&x, &mut y);
        assert_eq!(y, vec![11.0, 22.0, 33.0, 41.0, 12.0]);
    }

    #[test]
    fn test_rmatvec_dt() {
        let dm = make_test_design();
        let r = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut x = vec![0.0; 7];
        dm.rmatvec_dt(&r, &mut x);
        assert_eq!(x, vec![5.0, 7.0, 3.0, 6.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_rmatvec_wdt_unweighted() {
        let dm = make_test_design();
        let r = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut x_dt = vec![0.0; 7];
        let mut x_wdt = vec![0.0; 7];
        dm.rmatvec_dt(&r, &mut x_dt);
        dm.rmatvec_wdt(&r, &mut x_wdt);
        assert_eq!(x_dt, x_wdt);
    }

    #[test]
    fn test_rmatvec_wdt_weighted() {
        let dm = make_weighted_design(vec![1.0, 2.0, 3.0, 4.0]);
        let r = vec![1.0, 1.0, 1.0, 1.0];
        let mut x = vec![0.0; 4];
        dm.rmatvec_wdt(&r, &mut x);
        // factor 0: level 0 has obs 0(w=1)+2(w=3)=4, level 1 has obs 1(w=2)+3(w=4)=6
        // factor 1: level 0 has obs 0(w=1)+1(w=2)=3, level 1 has obs 2(w=3)+3(w=4)=7
        assert_eq!(x, vec![4.0, 6.0, 3.0, 7.0]);
    }

    #[test]
    fn test_gramian_diagonal_unweighted() {
        let dm = make_test_design();
        let diag = dm.gramian_diagonal();
        // factor 0: levels [0,1,2,0,1] -> counts [2,2,1]
        // factor 1: levels [0,1,2,3,0] -> counts [2,1,1,1]
        assert_eq!(diag, vec![2.0, 2.0, 1.0, 2.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_gramian_diagonal_weighted() {
        let dm = make_weighted_design(vec![1.0, 2.0, 3.0, 4.0]);
        let diag = dm.gramian_diagonal();
        // factor 0: level 0 -> w=1+3=4, level 1 -> w=2+4=6
        // factor 1: level 0 -> w=1+2=3, level 1 -> w=3+4=7
        assert_eq!(diag, vec![4.0, 6.0, 3.0, 7.0]);
    }
}
