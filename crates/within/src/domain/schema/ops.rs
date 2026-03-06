use std::sync::atomic::Ordering;

use portable_atomic::AtomicF64;
use rayon::prelude::*;

use crate::observation::ObservationStore;

use super::WeightedDesign;

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
        const PAR_THRESHOLD: usize = 10_000;
        const CHUNK_SIZE: usize = 4096;
        let factor_columns: Vec<Option<&[u32]>> = self
            .factors
            .iter()
            .enumerate()
            .map(|(q, _)| self.store.factor_column(q))
            .collect();

        if self.n_rows > PAR_THRESHOLD {
            // Parallel path: each chunk processes its own row range.
            dst.par_chunks_mut(CHUNK_SIZE)
                .enumerate()
                .for_each(|(chunk_idx, chunk)| {
                    let row_start = chunk_idx * CHUNK_SIZE;
                    if self.store.prefers_row_major_iteration() {
                        for (local, dst_val) in chunk.iter_mut().enumerate() {
                            let i = row_start + local;
                            for (q, f) in self.factors.iter().enumerate() {
                                *dst_val += src[f.offset + self.store.level(i, q) as usize];
                            }
                        }
                    } else {
                        for (q, f) in self.factors.iter().enumerate() {
                            let levels = factor_columns[q];
                            for (local, dst_val) in chunk.iter_mut().enumerate() {
                                let i = row_start + local;
                                let level = level_from_column_or_store(&self.store, levels, i, q);
                                *dst_val += src[f.offset + level];
                            }
                        }
                    }
                });
        } else if self.store.prefers_row_major_iteration() {
            // Sequential row-major: outer loop on observations, inner on factors.
            for i in 0..self.n_rows {
                for (q, f) in self.factors.iter().enumerate() {
                    dst[i] += src[f.offset + self.store.level(i, q) as usize];
                }
            }
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
        const PAR_THRESHOLD: usize = 10_000;
        /// Factors below this use thread-local fold/reduce; above use atomic CAS.
        /// 100K levels × 8 bytes × ~24 Rayon tasks ≈ 19 MB — fits comfortably.
        const SCATTER_LOCAL_THRESHOLD: usize = 100_000;
        let factor_columns: Vec<Option<&[u32]>> = self
            .factors
            .iter()
            .enumerate()
            .map(|(q, _)| self.store.factor_column(q))
            .collect();

        if self.store.prefers_row_major_iteration() {
            // Row-major: sequential (write conflicts across factors)
            for i in 0..self.n_rows {
                let v = value_fn(i);
                for (q, f) in self.factors.iter().enumerate() {
                    dst[f.offset + self.store.level(i, q) as usize] += v;
                }
            }
        } else if self.n_rows > PAR_THRESHOLD {
            // Factor-major with row-level parallelism.
            // Process factors sequentially so each gets the full thread pool
            // (12 threads vs 3 with the old factor-only parallelism).
            let n_rows = self.n_rows;
            let store = &self.store;
            let max_levels = self.factors.iter().map(|f| f.n_levels).max().unwrap_or(0);
            let mut atomic_buf: Vec<AtomicF64> = Vec::with_capacity(max_levels);
            for (q, f) in self.factors.iter().enumerate() {
                let slice = &mut dst[f.offset..f.offset + f.n_levels];
                let n_levels = f.n_levels;
                let levels = factor_columns[q];

                if n_levels < SCATTER_LOCAL_THRESHOLD {
                    // Thread-local accumulators: each Rayon task folds into
                    // its own Vec, then reduce merges. Safe for small n_levels.
                    let result: Vec<f64> = (0..n_rows)
                        .into_par_iter()
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
                } else {
                    // Atomic scatter: low contention for large factor dimensions
                    // (e.g. 32M individuals with ~10 obs each → <0.001% collision rate).
                    // Uses native atomic float ops where available (e.g. AArch64 ldadd).
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
        } else {
            // Factor-major sequential
            for (q, f) in self.factors.iter().enumerate() {
                let levels = factor_columns[q];
                for i in 0..self.n_rows {
                    let level = level_from_column_or_store(&self.store, levels, i, q);
                    dst[f.offset + level] += value_fn(i);
                }
            }
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

    /// Form the normal-equation RHS `D^T W y`.
    pub(crate) fn normal_equation_rhs(&self, y: &[f64]) -> Vec<f64> {
        let mut rhs = vec![0.0; self.n_dofs];
        self.rmatvec_wdt(y, &mut rhs);
        rhs
    }

    /// Diagonal of D^T·W·D (weighted level counts).
    ///
    /// Uses compressed iteration for backends that deduplicate observations.
    /// Entry `offset_q + j` = sum of weights for observations with level `j` in factor `q`.
    pub fn gramian_diagonal(&self) -> Vec<f64> {
        let mut diag = vec![0.0f64; self.n_dofs];
        let n_unique = self.store.n_unique();
        for (q, f) in self.factors.iter().enumerate() {
            for uid in 0..n_unique {
                diag[f.offset + self.store.unique_level(uid, q) as usize] += self.uid_weight(uid);
            }
        }
        diag
    }
}
