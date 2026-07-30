//! Scatter kernel: observation space → coefficient space (`Dᵀ x`), one
//! strategy per term picked by block size and level-column sortedness.

use std::sync::atomic::Ordering;

use portable_atomic::AtomicF64;
use rayon::prelude::*;

use super::PAR_THRESHOLD;
use crate::domain::Loading;
use crate::domain::{Design, TermMeta};

/// Adjoint scatter over all terms; `base(i)` is the row value each column scales by its loading.
pub(super) fn scatter_apply(
    design: &Design<'_>,
    scratch: &[AtomicF64],
    dst: &mut [f64],
    base: &(impl Fn(usize) -> f64 + Sync),
) {
    debug_assert_eq!(dst.len(), design.n_dofs);
    let parallel = design.n_obs > PAR_THRESHOLD;
    let frame = &design.frame;

    for (q, t) in design.terms.iter().enumerate() {
        let levels = frame.level_column(q);
        let block = &mut dst[t.offset..t.offset + t.n_dofs()];
        match &*t.columns {
            [Loading::Constant] => {
                scatter_term::<1>(block, t, levels, parallel, scratch, |i| [base(i)])
            }
            [Loading::Constant, Loading::Covariate(c0)] => {
                let z0 = frame.loading_column(*c0 as usize);
                scatter_term::<2>(block, t, levels, parallel, scratch, |i| {
                    let b = base(i);
                    [b, z0[i] * b]
                })
            }
            [Loading::Constant, Loading::Covariate(c0), Loading::Covariate(c1)] => {
                let z0 = frame.loading_column(*c0 as usize);
                let z1 = frame.loading_column(*c1 as usize);
                scatter_term::<3>(block, t, levels, parallel, scratch, |i| {
                    let b = base(i);
                    [b, z0[i] * b, z1[i] * b]
                })
            }
            [Loading::Covariate(c0), Loading::Covariate(c1)] => {
                let z0 = frame.loading_column(*c0 as usize);
                let z1 = frame.loading_column(*c1 as usize);
                scatter_term::<2>(block, t, levels, parallel, scratch, |i| {
                    let b = base(i);
                    [z0[i] * b, z1[i] * b]
                })
            }
            columns => {
                for (c, loading) in columns.iter().enumerate() {
                    let start = c * t.n_levels;
                    let slot = &mut block[start..start + t.n_levels];
                    match loading {
                        Loading::Constant => {
                            scatter_term::<1>(slot, t, levels, parallel, scratch, |i| [base(i)]);
                        }
                        Loading::Covariate(k) => {
                            let z = frame.loading_column(*k as usize);
                            scatter_term::<1>(slot, t, levels, parallel, scratch, move |i| {
                                [z[i] * base(i)]
                            });
                        }
                    }
                }
            }
        }
    }
}

/// Scatter one term's coefficient block: `block[c·L + level(i)] += values(i)[c]`.
fn scatter_term<const C: usize>(
    block: &mut [f64],
    meta: &TermMeta,
    levels: &[u32],
    parallel: bool,
    scratch: &[AtomicF64],
    values: impl Fn(usize) -> [f64; C] + Sync,
) {
    let n_levels = meta.n_levels;
    debug_assert_eq!(block.len(), C * n_levels);
    match ScatterStrategy::pick(parallel, C * n_levels, meta.sorted) {
        ScatterStrategy::Sequential => scatter_sequential::<C>(block, n_levels, levels, &values),
        ScatterStrategy::Fold => scatter_fold::<C>(block, n_levels, levels, &values),
        ScatterStrategy::Atomic => scatter_atomic::<C>(block, n_levels, levels, &values, scratch),
        ScatterStrategy::SortedCoalesced => {
            scatter_sorted_coalesced::<C>(block, n_levels, levels, &values, scratch)
        }
    }
}

/// Fold vs atomic scatter-add: below it fold costs O(block · n_threads) memory, above CAS wins.
const SCATTER_LOCAL_THRESHOLD: usize = 100_000;

/// Strategy for a single term's scatter-add loop.
enum ScatterStrategy {
    /// Plain sequential loop — used when n_rows is below `PAR_THRESHOLD`.
    Sequential,
    /// Parallel fold/reduce with thread-local accumulators — for small blocks.
    Fold,
    /// Parallel atomic CAS — for large blocks with low contention.
    Atomic,
    /// Equal-level runs coalesce into one atomic add per level per chunk, avoiding a CAS storm.
    SortedCoalesced,
}

impl ScatterStrategy {
    /// `block` is the coefficient count the kernel writes, `sorted` the level-column sortedness.
    fn pick(parallel: bool, block: usize, sorted: bool) -> Self {
        match (parallel, block < SCATTER_LOCAL_THRESHOLD, sorted) {
            (false, _, _) => ScatterStrategy::Sequential,
            (true, true, _) => ScatterStrategy::Fold,
            (true, false, true) => ScatterStrategy::SortedCoalesced,
            (true, false, false) => ScatterStrategy::Atomic,
        }
    }
}

/// Sequential scatter-add: `block[c·L + levels[i]] += values(i)[c]`.
fn scatter_sequential<const C: usize>(
    block: &mut [f64],
    n_levels: usize,
    levels: &[u32],
    values: &(impl Fn(usize) -> [f64; C] + Sync),
) {
    for (i, &lev) in levels.iter().enumerate() {
        let vals = values(i);
        for (c, v) in vals.into_iter().enumerate() {
            block[c * n_levels + lev as usize] += v;
        }
    }
}

/// Parallel scatter-add via thread-local fold/reduce, best when the block is small.
fn scatter_fold<const C: usize>(
    block: &mut [f64],
    n_levels: usize,
    levels: &[u32],
    values: &(impl Fn(usize) -> [f64; C] + Sync),
) {
    let min_len = (levels.len() / rayon::current_num_threads().max(1)).max(1024);
    let identity = || vec![0.0f64; C * n_levels];
    let fold = |mut acc: Vec<f64>, (i, &lev): (usize, &u32)| {
        let vals = values(i);
        for (c, v) in vals.into_iter().enumerate() {
            acc[c * n_levels + lev as usize] += v;
        }
        acc
    };
    let reduction = |mut a: Vec<f64>, b: Vec<f64>| {
        for (ai, bi) in a.iter_mut().zip(b.iter()) {
            *ai += *bi;
        }
        a
    };
    let result: Vec<f64> = levels
        .par_iter()
        .enumerate()
        .with_min_len(min_len)
        .fold(identity, fold)
        .reduce(identity, reduction);
    for (d, r) in block.iter_mut().zip(result.iter()) {
        *d += *r;
    }
}

/// Seed the atomic scratch with `block`'s contents, returning the trimmed accumulation view.
fn seed_scatter_scratch<'b>(atomic_buf: &'b [AtomicF64], block: &[f64]) -> &'b [AtomicF64] {
    debug_assert!(atomic_buf.len() >= block.len());
    let buf = &atomic_buf[..block.len()];
    for (a, &v) in buf.iter().zip(block.iter()) {
        a.store(v, Ordering::Relaxed);
    }
    buf
}

/// Copy accumulated scratch values back into `block`.
fn writeback_scatter_scratch(block: &mut [f64], buf: &[AtomicF64]) {
    for (d, a) in block.iter_mut().zip(buf.iter()) {
        *d = a.load(Ordering::Relaxed);
    }
}

/// Parallel scatter-add via atomic CAS, best when the block is large; the scratch is reused.
fn scatter_atomic<const C: usize>(
    block: &mut [f64],
    n_levels: usize,
    levels: &[u32],
    values: &(impl Fn(usize) -> [f64; C] + Sync),
    atomic_buf: &[AtomicF64],
) {
    let buf = seed_scatter_scratch(atomic_buf, block);
    levels.par_iter().enumerate().for_each(|(i, &lev)| {
        let vals = values(i);
        for (c, v) in vals.into_iter().enumerate() {
            buf[c * n_levels + lev as usize].fetch_add(v, Ordering::Relaxed);
        }
    });
    writeback_scatter_scratch(block, buf);
}

fn scatter_sorted_coalesced<const C: usize>(
    block: &mut [f64],
    n_levels: usize,
    levels: &[u32],
    values: &(impl Fn(usize) -> [f64; C] + Sync),
    atomic_buf: &[AtomicF64],
) {
    let buf = seed_scatter_scratch(atomic_buf, block);
    // A run split across a chunk boundary is committed by both chunks — additive, so correct.
    const CHUNK: usize = 65_536;
    levels
        .par_chunks(CHUNK)
        .enumerate()
        .for_each(|(c_idx, chunk)| {
            let start = c_idx * CHUNK;
            let mut level = chunk[0] as usize;
            let mut sums = values(start);
            for (i, &li) in (start + 1..).zip(&chunk[1..]) {
                let li = li as usize;
                if li != level {
                    for (c, s) in sums.into_iter().enumerate() {
                        buf[c * n_levels + level].fetch_add(s, Ordering::Relaxed);
                    }
                    level = li;
                    sums = [0.0; C];
                }
                let vals = values(i);
                for (c, v) in vals.into_iter().enumerate() {
                    sums[c] += v;
                }
            }
            for (c, s) in sums.into_iter().enumerate() {
                buf[c * n_levels + level].fetch_add(s, Ordering::Relaxed);
            }
        });
    writeback_scatter_scratch(block, buf);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Must match a naive per-row scatter-add, including a run straddling the
    /// chunk boundary (committed by two chunks via additive atomics) and the
    /// multi-column fused commit. Gated on large sorted terms, so integration
    /// tests never reach it — exercised directly here.
    #[test]
    fn coalesced_scatter_matches_naive() {
        let n = 70_000usize;
        let n_levels = 1_000usize;
        let levels: Vec<u32> = (0..n).map(|i| (i * n_levels / n) as u32).collect();
        let x: Vec<f64> = (0..n).map(|i| (i % 13) as f64 - 6.0).collect();
        let z: Vec<f64> = (0..n).map(|i| ((i * 7) % 11) as f64 / 11.0 - 0.5).collect();

        let buf: Vec<AtomicF64> = (0..2 * n_levels).map(|_| AtomicF64::new(0.0)).collect();
        let mut got = vec![0.0f64; 2 * n_levels];
        scatter_sorted_coalesced::<2>(&mut got, n_levels, &levels, &|i| [x[i], z[i] * x[i]], &buf);

        let mut expect = vec![0.0f64; 2 * n_levels];
        for (i, &l) in levels.iter().enumerate() {
            expect[l as usize] += x[i];
            expect[n_levels + l as usize] += z[i] * x[i];
        }
        for (g, e) in got.iter().zip(expect.iter()) {
            assert!((g - e).abs() < 1e-9, "{g} vs {e}");
        }
    }
}
