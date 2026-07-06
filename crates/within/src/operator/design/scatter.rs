//! Scatter-add kernels: observation space → coefficient space (`Dᵀ x`), one
//! strategy per term picked by block size and level-column sortedness.

use std::sync::atomic::Ordering;

use portable_atomic::AtomicF64;
use rayon::prelude::*;

/// Coefficient-block threshold for choosing between fold and atomic scatter-add.
///
/// Blocks (a term's `n_columns * n_levels` coefficients) smaller than this use
/// thread-local fold/reduce (O(block * n_threads) memory). Larger blocks use
/// atomic CAS instead, which has low contention when bins vastly outnumber
/// threads.
const SCATTER_LOCAL_THRESHOLD: usize = 100_000;

/// Strategy for a single term's scatter-add loop.
pub(super) enum ScatterStrategy {
    /// Plain sequential loop — used when n_rows is below `PAR_THRESHOLD`.
    Sequential,
    /// Parallel fold/reduce with thread-local accumulators — for small blocks.
    Fold,
    /// Parallel atomic CAS — for large blocks with low contention.
    Atomic,
    /// Atomic path for a large sorted term: equal-level runs coalesce into
    /// one atomic add per distinct level per chunk instead of one per row,
    /// avoiding the atomic-CAS storm.
    SortedCoalesced,
}

impl ScatterStrategy {
    /// Pick the scatter strategy for one term; `block` is the coefficient
    /// count written by the kernel call, `sorted` the term's level-column
    /// sortedness (`TermMeta::sorted`).
    pub(super) fn pick(parallel: bool, block: usize, sorted: bool) -> Self {
        match (parallel, block < SCATTER_LOCAL_THRESHOLD, sorted) {
            (false, _, _) => ScatterStrategy::Sequential,
            (true, true, _) => ScatterStrategy::Fold,
            (true, false, true) => ScatterStrategy::SortedCoalesced,
            (true, false, false) => ScatterStrategy::Atomic,
        }
    }
}

/// Sequential scatter-add: `block[c·L + levels[i]] += values(i)[c]`.
pub(super) fn scatter_sequential<const C: usize>(
    block: &mut [f64],
    n_levels: usize,
    levels: &[u32],
    values: &(impl Fn(usize) -> [f64; C] + Sync),
) {
    for (i, &l) in levels.iter().enumerate() {
        let vals = values(i);
        for (c, v) in vals.into_iter().enumerate() {
            block[c * n_levels + l as usize] += v;
        }
    }
}

/// Parallel scatter-add via thread-local fold/reduce — best when the block
/// (the term's coefficient count) is small relative to thread count.
pub(super) fn scatter_fold<const C: usize>(
    block: &mut [f64],
    n_levels: usize,
    levels: &[u32],
    values: &(impl Fn(usize) -> [f64; C] + Sync),
) {
    let min_len = (levels.len() / rayon::current_num_threads().max(1)).max(1024);
    let identity = || vec![0.0f64; C * n_levels];
    let fold = |mut acc: Vec<f64>, (i, &l): (usize, &u32)| {
        let vals = values(i);
        for (c, v) in vals.into_iter().enumerate() {
            acc[c * n_levels + l as usize] += v;
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

/// Seed the operator's atomic scratch with `block`'s current contents,
/// returning the trimmed view the scatter accumulates into.
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

/// Parallel scatter-add via atomic CAS — best when the block is large
/// relative to thread count (low contention). `atomic_buf` is the operator's
/// reusable scratch (sized to the largest term's block); we use its first
/// `block.len()` slots, re-seeding them via `store` so no allocation occurs.
pub(super) fn scatter_atomic<const C: usize>(
    block: &mut [f64],
    n_levels: usize,
    levels: &[u32],
    values: &(impl Fn(usize) -> [f64; C] + Sync),
    atomic_buf: &[AtomicF64],
) {
    let buf = seed_scatter_scratch(atomic_buf, block);
    levels.par_iter().enumerate().for_each(|(i, &l)| {
        let vals = values(i);
        for (c, v) in vals.into_iter().enumerate() {
            buf[c * n_levels + l as usize].fetch_add(v, Ordering::Relaxed);
        }
    });
    writeback_scatter_scratch(block, buf);
}

pub(super) fn scatter_sorted_coalesced<const C: usize>(
    block: &mut [f64],
    n_levels: usize,
    levels: &[u32],
    values: &(impl Fn(usize) -> [f64; C] + Sync),
    atomic_buf: &[AtomicF64],
) {
    let buf = seed_scatter_scratch(atomic_buf, block);
    // Each row-chunk coalesces its equal-level runs locally and commits one
    // atomic add per distinct level per column. A run split across a chunk
    // boundary is committed by both chunks — additive, so still correct —
    // keeping chunks independent without a carry/fixup pass.
    const CHUNK: usize = 65_536;
    levels
        .par_chunks(CHUNK)
        .enumerate()
        .for_each(|(c_idx, chunk)| {
            let start = c_idx * CHUNK;
            // Single flat pass, one level load per row: accumulate the current
            // run's per-column sums and commit them whenever the level changes.
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
