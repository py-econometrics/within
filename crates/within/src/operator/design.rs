use std::borrow::Cow;
use std::sync::atomic::Ordering;

use portable_atomic::AtomicF64;
use rayon::prelude::*;
use schwarz_precond::Operator;

use crate::domain::Design;

// ===========================================================================
// Iteration kernels — module-private, shared between apply / apply_adjoint
// ===========================================================================

/// Minimum number of rows before scatter/gather loops are parallelized.
const PAR_THRESHOLD: usize = 10_000;

/// Factor-level threshold for choosing between fold and atomic scatter-add.
///
/// Factors with fewer than this many levels use thread-local fold/reduce
/// (O(n_levels * n_threads) memory). Larger terms use atomic CAS instead,
/// which has low contention when bins vastly outnumber threads.
const SCATTER_LOCAL_THRESHOLD: usize = 100_000;

/// Strategy for a single factor's scatter-add loop.
enum ScatterStrategy {
    /// Plain sequential loop — used when n_rows is below `PAR_THRESHOLD`.
    Sequential,
    /// Parallel fold/reduce with thread-local accumulators — for small terms.
    Fold,
    /// Parallel atomic CAS — for large terms with low contention.
    Atomic,
    /// Atomic path for a large sorted factor: equal-level runs coalesce into
    /// one atomic add per distinct level per chunk instead of one per row,
    /// avoiding the atomic-CAS storm.
    SortedCoalesced,
}

impl ScatterStrategy {
    /// Pick the scatter strategy for one factor; `sorted` is the factor's
    /// level-column sortedness (`TermMeta::sorted`).
    fn pick(parallel: bool, n_levels: usize, sorted: bool) -> Self {
        match (parallel, n_levels < SCATTER_LOCAL_THRESHOLD, sorted) {
            (false, _, _) => ScatterStrategy::Sequential,
            (true, true, _) => ScatterStrategy::Fold,
            (true, false, true) => ScatterStrategy::SortedCoalesced,
            (true, false, false) => ScatterStrategy::Atomic,
        }
    }
}

/// Gather-apply: `dst[i] = Σ_q src[off_q + level(i, q)]`, times `scale[i]` if given.
///
/// One sweep over `dst` per factor, plus a scale sweep only when given.
pub(crate) fn gather_apply(
    design: &Design<'_>,
    src: &[f64],
    dst: &mut [f64],
    scale: Option<&[f64]>,
) {
    debug_assert!(scale.is_none_or(|s| s.len() == design.n_obs));
    debug_assert_eq!(src.len(), design.n_dofs);
    debug_assert_eq!(dst.len(), design.n_obs);

    dst.fill(0.0);

    let terms = &design.terms;
    let columns: Vec<&[u32]> = (0..terms.len())
        .map(|q| design.frame.level_column(q))
        .collect();

    let kernel = |chunk: &mut [f64], row_start: usize| {
        // `&levels` copies the slice ref out of `columns`; binding `&&[u32]`
        // leaves a non-hoisted double deref in the inner loop (~5% measured).
        for (f, &levels) in terms.iter().zip(columns.iter()) {
            for (local, dst_val) in chunk.iter_mut().enumerate() {
                let i = row_start + local;
                *dst_val += src[f.offset + levels[i] as usize];
            }
        }

        if let Some(scale) = scale {
            for (s, dst_val) in scale[row_start..].iter().zip(chunk.iter_mut()) {
                *dst_val *= s;
            }
        }
    };

    if design.n_obs > PAR_THRESHOLD {
        const CHUNK_SIZE: usize = 4096;

        dst.par_chunks_mut(CHUNK_SIZE)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                kernel(chunk, chunk_idx * CHUNK_SIZE);
            });
    } else {
        kernel(dst, 0);
    }
}

/// Sequential scatter-add: `slice[levels[i]] += value_fn(i)`.
fn scatter_sequential(
    slice: &mut [f64],
    levels: &[u32],
    value_fn: &(impl Fn(usize) -> f64 + Sync),
) {
    for (i, &l) in levels.iter().enumerate() {
        slice[l as usize] += value_fn(i);
    }
}

/// Parallel scatter-add via thread-local fold/reduce — best when `slice.len()`
/// (the factor's level count) is small relative to thread count.
fn scatter_fold(slice: &mut [f64], levels: &[u32], value_fn: &(impl Fn(usize) -> f64 + Sync)) {
    let n_levels = slice.len();
    let min_len = (levels.len() / rayon::current_num_threads().max(1)).max(1024);
    let identity = || vec![0.0f64; n_levels];
    let fold = |mut acc: Vec<f64>, (i, &l): (usize, &u32)| {
        acc[l as usize] += value_fn(i);
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
    for (d, r) in slice.iter_mut().zip(result.iter()) {
        *d += *r;
    }
}

/// Seed the operator's atomic scratch with `slice`'s current contents,
/// returning the trimmed view the scatter accumulates into.
fn seed_scatter_scratch<'b>(atomic_buf: &'b [AtomicF64], slice: &[f64]) -> &'b [AtomicF64] {
    debug_assert!(atomic_buf.len() >= slice.len());
    let buf = &atomic_buf[..slice.len()];
    for (a, &v) in buf.iter().zip(slice.iter()) {
        a.store(v, Ordering::Relaxed);
    }
    buf
}

/// Copy accumulated scratch values back into `slice`.
fn writeback_scatter_scratch(slice: &mut [f64], buf: &[AtomicF64]) {
    for (d, a) in slice.iter_mut().zip(buf.iter()) {
        *d = a.load(Ordering::Relaxed);
    }
}

/// Parallel scatter-add via atomic CAS — best when `slice.len()` is large
/// relative to thread count (low contention). `atomic_buf` is the operator's
/// reusable scratch (sized to the largest factor); we use its first
/// `slice.len()` slots, re-seeding them via `store` so no allocation occurs.
fn scatter_atomic(
    slice: &mut [f64],
    levels: &[u32],
    value_fn: &(impl Fn(usize) -> f64 + Sync),
    atomic_buf: &[AtomicF64],
) {
    let buf = seed_scatter_scratch(atomic_buf, slice);
    levels.par_iter().enumerate().for_each(|(i, &l)| {
        buf[l as usize].fetch_add(value_fn(i), Ordering::Relaxed);
    });
    writeback_scatter_scratch(slice, buf);
}

fn scatter_sorted_coalesced(
    slice: &mut [f64],
    levels: &[u32],
    value_fn: &(impl Fn(usize) -> f64 + Sync),
    atomic_buf: &[AtomicF64],
) {
    let buf = seed_scatter_scratch(atomic_buf, slice);
    // Each row-chunk coalesces its equal-level runs locally and commits one
    // atomic add per distinct level. A run split across a chunk boundary is
    // committed by both chunks — additive, so still correct — keeping chunks
    // independent without a carry/fixup pass.
    const CHUNK: usize = 65_536;
    levels.par_chunks(CHUNK).enumerate().for_each(|(c, chunk)| {
        let start = c * CHUNK;
        // Single flat pass, one level load per row: accumulate the current
        // run's sum and commit it whenever the level changes.
        let mut level = chunk[0] as usize;
        let mut sum = value_fn(start);
        for (i, &li) in (start + 1..).zip(&chunk[1..]) {
            let li = li as usize;
            if li != level {
                buf[level].fetch_add(sum, Ordering::Relaxed);
                level = li;
                sum = 0.0;
            }
            sum += value_fn(i);
        }
        buf[level].fetch_add(sum, Ordering::Relaxed);
    });
    writeback_scatter_scratch(slice, buf);
}

// ===========================================================================
// DesignOperator — D, optionally rescaled by W^{1/2}
// ===========================================================================

/// Rectangular design operator: `D` (unweighted) or `W^{1/2} D` (weighted).
///
/// `apply` = `D x` / `W^{1/2} D x` (gather), `apply_adjoint` = `D^T x` /
/// `D^T W^{1/2} x` (scatter). For the weighted variant, the normal equations
/// `A^T A = D^T W D = G` recover the Gramian, so the same Schwarz
/// preconditioner approximating `G^{-1}` applies. Pass `None` to
/// [`DesignOperator::new`] for `D`, or `Some(&w)` for `W^{1/2} D`. The branch
/// on weights is hoisted outside the per-row loop — the weighted gather applies
/// `W^{1/2}` in a trailing per-chunk sweep, and the adjoint multiplies inline
/// through a closure, so there is no per-row scratch buffer.
pub(crate) struct DesignOperator<'a> {
    design: &'a Design<'a>,
    sqrt_weights: Option<Vec<f64>>,
    /// Reusable atomic-scatter scratch, sized once to the largest factor's
    /// level count and reused across terms and `apply_adjoint` calls so it is
    /// allocated once per operator rather than once per LSMR iteration.
    /// `apply_adjoint` takes `&self`, but `AtomicF64`'s load/store/fetch_add are
    /// `&self` operations, so a plain `Vec<AtomicF64>` (already `Sync`) needs no
    /// lock: each call re-seeds the buffer via `store` instead of resizing it.
    scatter_scratch: Vec<AtomicF64>,
}

impl<'a> DesignOperator<'a> {
    /// Wrap a design matrix as a linear operator.
    ///
    /// Pass `None` for `D`, `Some(&w)` for `W^{1/2} D` (then `w.len()` must
    /// equal `design.n_obs`). Precomputes and stores `sqrt(W)` when weights
    /// are present.
    ///
    /// # Panics
    ///
    /// Panics when `weights.is_some()` and `weights.unwrap().len()` does not
    /// equal `design.n_obs`. The `Solver` entry points perform fallible
    /// validation against `BuildError::WeightCountMismatch` before
    /// construction, so callers that go through `Solver::new` or
    /// `solve()` never trigger this panic.
    pub(crate) fn new(design: &'a Design<'a>, weights: Option<&[f64]>) -> Self {
        let sqrt_weights = weights.map(|w| {
            assert_eq!(
                w.len(),
                design.n_obs,
                "weights length {} does not match design.n_obs {}",
                w.len(),
                design.n_obs
            );
            w.iter().map(|wi| wi.sqrt()).collect()
        });
        let max_levels = design.terms.iter().map(|f| f.n_levels).max().unwrap_or(0);
        Self {
            design,
            sqrt_weights,
            scatter_scratch: (0..max_levels).map(|_| AtomicF64::new(0.0)).collect(),
        }
    }

    /// Compute the observation-space RHS `b = W^{1/2} y`.
    ///
    /// For unweighted designs, borrows `y` (no allocation); the weighted
    /// variant returns an owned scaled copy.
    pub(crate) fn weighted_rhs<'y>(&self, y: &'y [f64]) -> Cow<'y, [f64]> {
        match &self.sqrt_weights {
            None => Cow::Borrowed(y),
            Some(sw) => Cow::Owned(y.iter().zip(sw).map(|(&yi, &swi)| swi * yi).collect()),
        }
    }

    fn scatter_apply<F>(&self, dst: &mut [f64], value_fn: F)
    where
        F: Fn(usize) -> f64 + Sync,
    {
        let design = self.design;
        debug_assert_eq!(dst.len(), design.n_dofs);
        let parallel = design.n_obs > PAR_THRESHOLD;

        for (q, f) in design.terms.iter().enumerate() {
            let slice = &mut dst[f.offset..f.offset + f.n_levels];
            let levels = design.frame.level_column(q);

            match ScatterStrategy::pick(parallel, f.n_levels, f.sorted) {
                ScatterStrategy::Sequential => scatter_sequential(slice, levels, &value_fn),
                ScatterStrategy::Fold => scatter_fold(slice, levels, &value_fn),
                ScatterStrategy::Atomic => {
                    scatter_atomic(slice, levels, &value_fn, &self.scatter_scratch)
                }
                ScatterStrategy::SortedCoalesced => {
                    scatter_sorted_coalesced(slice, levels, &value_fn, &self.scatter_scratch)
                }
            }
        }
    }
}

impl Operator for DesignOperator<'_> {
    fn nrows(&self) -> usize {
        self.design.n_obs
    }

    fn ncols(&self) -> usize {
        self.design.n_dofs
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), schwarz_precond::SolveError> {
        gather_apply(self.design, x, y, self.sqrt_weights.as_deref());
        Ok(())
    }

    fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) -> Result<(), schwarz_precond::SolveError> {
        debug_assert_eq!(x.len(), self.design.n_obs);
        debug_assert_eq!(y.len(), self.design.n_dofs);
        y.fill(0.0);
        // Reuse the per-operator atomic-scatter scratch across iterations. No
        // lock needed: `AtomicF64` mutates through `&self`, and a single
        // operator's `apply_adjoint` calls are sequential (`solve_batch` builds
        // a distinct operator per RHS), so the shared buffer is never raced.
        match &self.sqrt_weights {
            Some(sw) => self.scatter_apply(y, |i| sw[i] * x[i]),
            None => self.scatter_apply(y, |i| x[i]),
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Must match a naive per-row scatter-add, including a run straddling the
    /// chunk boundary (committed by two chunks via additive atomics). Gated on
    /// large sorted terms, so integration tests never reach it — exercised
    /// directly here.
    #[test]
    fn coalesced_scatter_matches_naive() {
        // > 65_536 rows so the level-0 run crosses the chunk boundary, exercising
        // the two-chunk additive commit for a single level.
        let n_rows = 100_000usize;
        let col: Vec<u32> = (0..n_rows).map(|i| u32::from(i >= 70_000)).collect();
        let n_levels = 2usize;
        let values: Vec<f64> = (0..n_rows).map(|i| (i % 7) as f64 - 3.0).collect();

        let mut expected = vec![0.0f64; n_levels];
        for (i, &c) in col.iter().enumerate() {
            expected[c as usize] += values[i];
        }

        let buf: Vec<AtomicF64> = (0..n_levels).map(|_| AtomicF64::new(0.0)).collect();
        let mut got = vec![0.0f64; n_levels];
        scatter_sorted_coalesced(&mut got, &col, &|i| values[i], &buf);

        for (g, e) in got.iter().zip(&expected) {
            assert!((g - e).abs() < 1e-6, "got {g}, expected {e}");
        }
    }
}
