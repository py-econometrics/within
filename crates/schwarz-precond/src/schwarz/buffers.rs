//! Scratch-buffer types for the additive Schwarz executor, one layout per
//! reduction strategy ([`SchwarzBuffers::Atomic`], [`SchwarzBuffers::Reduction`]).
//! [`BufferPool`] hands them to the executor and reclaims them after each apply
//! so the steady state allocates nothing.

use std::cell::RefCell;
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Mutex};

use rayon::prelude::*;
use thread_local::ThreadLocal;

use crate::error::SolveError;

use super::planning::ResolvedReductionStrategy;

// ============================================================================
// Buffer pooling
// ============================================================================

pub(super) struct BufferPool {
    n_dofs: usize,
    max_scratch_size: usize,
    inner: Arc<Mutex<Vec<SchwarzBuffers>>>,
}

impl BufferPool {
    const MAX_POOL_SIZE: usize = 4;

    pub(super) fn new(n_dofs: usize, max_scratch_size: usize) -> Self {
        Self {
            n_dofs,
            max_scratch_size,
            inner: Arc::default(),
        }
    }

    pub(super) fn n_dofs(&self) -> usize {
        self.n_dofs
    }

    pub(super) fn max_scratch_size(&self) -> usize {
        self.max_scratch_size
    }

    pub(super) fn take(
        &self,
        strategy: ResolvedReductionStrategy,
    ) -> Result<SchwarzBuffers, SolveError> {
        let mut pool = self.inner.lock().map_err(|_| SolveError::Synchronization {
            context: "additive.buf_pool.lock.pop",
        })?;
        if let Some(idx) = pool.iter().position(|bufs| bufs.strategy() == strategy) {
            return Ok(pool.swap_remove(idx));
        }
        Ok(SchwarzBuffers::new(
            strategy,
            self.n_dofs,
            self.max_scratch_size,
        ))
    }

    /// Return a buffer to the pool. Infallible by design: pool bookkeeping
    /// must never mask the caller's real `apply_result`. On the error path the
    /// buffer is dropped (see below); on a poisoned pool lock the buffer is
    /// likewise dropped rather than surfaced as a `Synchronization` error.
    pub(super) fn put(&self, bufs: SchwarzBuffers, apply_result: &Result<(), SolveError>) {
        // On error, the atomic backend's swap-zero readout pass is skipped,
        // leaving stale partial-write values in the AtomicU64 vec. Drop the
        // buffer rather than pooling it for the next caller to inherit dirty
        // state.
        if apply_result.is_err() {
            return;
        }
        // A poisoned lock means a worker panicked; just drop the buffer (the
        // pool lazily re-allocates on the next `take`) instead of erroring.
        if let Ok(mut pool) = self.inner.lock() {
            if pool.len() < Self::MAX_POOL_SIZE {
                pool.push(bufs);
            }
        }
    }
}

impl Clone for BufferPool {
    fn clone(&self) -> Self {
        Self {
            n_dofs: self.n_dofs,
            max_scratch_size: self.max_scratch_size,
            inner: Arc::clone(&self.inner),
        }
    }
}

pub(super) struct LocalSolveScratch {
    pub(super) r_scratch: Vec<f64>,
    pub(super) z_scratch: Vec<f64>,
}

impl LocalSolveScratch {
    #[inline]
    pub(super) fn new(max_scratch_size: usize) -> Self {
        Self {
            r_scratch: vec![0.0f64; max_scratch_size],
            z_scratch: vec![0.0f64; max_scratch_size],
        }
    }
}

/// Task-local scratch for the parallel-reduction path.
pub(super) struct AdditiveSweepBuffers {
    pub(super) global_accum: Vec<f64>,
    pub(super) scratch: LocalSolveScratch,
}

impl AdditiveSweepBuffers {
    fn new(n_dofs: usize, max_scratch_size: usize) -> Self {
        Self {
            global_accum: vec![0.0f64; n_dofs],
            scratch: LocalSolveScratch::new(max_scratch_size),
        }
    }
}

/// Pooled buffers that vary by reduction strategy.
pub(super) enum SchwarzBuffers {
    /// Shared atomic accumulator plus a pool of per-worker local-solve scratch.
    Atomic {
        accum: Vec<AtomicU64>,
        scratch_pool: Vec<LocalSolveScratch>,
    },
    /// Reusable task-local buffers for parallel reduction.
    Reduction { pool: Vec<AdditiveSweepBuffers> },
}

impl SchwarzBuffers {
    fn new(strategy: ResolvedReductionStrategy, n_dofs: usize, max_scratch_size: usize) -> Self {
        match strategy {
            ResolvedReductionStrategy::AtomicScatter => Self::Atomic {
                accum: (0..n_dofs).map(|_| AtomicU64::new(0)).collect(),
                scratch_pool: Vec::new(),
            },
            ResolvedReductionStrategy::ParallelReduction => Self::Reduction {
                pool: vec![AdditiveSweepBuffers::new(n_dofs, max_scratch_size)],
            },
        }
    }

    fn strategy(&self) -> ResolvedReductionStrategy {
        match self {
            Self::Atomic { .. } => ResolvedReductionStrategy::AtomicScatter,
            Self::Reduction { .. } => ResolvedReductionStrategy::ParallelReduction,
        }
    }
}

/// Thread-local stack of reusable per-worker buffers backed by a shared pool.
///
/// Each Rayon worker reuses its own buffers across sequential outer tasks via a
/// `ThreadLocal` stack, with no cross-thread synchronization in the hot loop.
/// Nested re-entry on the same worker allocates an extra buffer only when
/// needed, so the number of retained buffers tracks re-entry depth rather than
/// Rayon task splitting. At round end [`into_pool`](Self::into_pool) gathers the
/// shared pool and every worker stack back into one vec for the next round.
pub(super) struct WorkerBufferStack<T: Send> {
    shared_pool: Mutex<Vec<T>>,
    worker_stacks: ThreadLocal<RefCell<Vec<T>>>,
    alloc: Box<dyn Fn() -> T + Send + Sync>,
}

impl<T: Send> WorkerBufferStack<T> {
    pub(super) fn new(pool: Vec<T>, alloc: impl Fn() -> T + Send + Sync + 'static) -> Self {
        Self {
            shared_pool: Mutex::new(pool),
            worker_stacks: ThreadLocal::with_capacity(rayon::current_num_threads().max(1)),
            alloc: Box::new(alloc),
        }
    }

    pub(super) fn with_buffer<R>(&self, f: impl FnOnce(&mut T) -> R) -> R {
        let mut buffer = match self
            .worker_stacks
            .get_or(|| RefCell::new(Vec::new()))
            .borrow_mut()
            .pop()
        {
            Some(buffer) => buffer,
            None => self.take_or_alloc(),
        };
        let result = f(&mut buffer);
        self.worker_stacks
            .get_or(|| RefCell::new(Vec::new()))
            .borrow_mut()
            .push(buffer);
        result
    }

    fn take_or_alloc(&self) -> T {
        self.shared_pool
            .lock()
            .ok()
            .and_then(|mut pool| pool.pop())
            .unwrap_or_else(|| (self.alloc)())
    }

    /// Gather the shared pool and all worker stacks back into one vec so it can
    /// be returned to its [`SchwarzBuffers`] home for the next round. `ctx`
    /// labels the synchronization error if the pool lock was poisoned.
    pub(super) fn into_pool(mut self, ctx: &'static str) -> Result<Vec<T>, SolveError> {
        let mut pool = self
            .shared_pool
            .into_inner()
            .map_err(|_| SolveError::Synchronization { context: ctx })?;
        for worker_stack in self.worker_stacks.iter_mut() {
            pool.append(worker_stack.get_mut());
        }
        Ok(pool)
    }
}

/// Worker-local buffers for the parallel-reduction path: a [`WorkerBufferStack`]
/// of per-worker accumulators plus the reduce-into-`z` step that sums them.
pub(super) struct WorkerReductionBuffers {
    pub(super) stack: WorkerBufferStack<AdditiveSweepBuffers>,
}

impl WorkerReductionBuffers {
    pub(super) fn new(
        pool: Vec<AdditiveSweepBuffers>,
        n_dofs: usize,
        max_scratch_size: usize,
    ) -> Self {
        Self {
            stack: WorkerBufferStack::new(pool, move || {
                AdditiveSweepBuffers::new(n_dofs, max_scratch_size)
            }),
        }
    }

    pub(super) fn finish_round(
        self,
        z: &mut [f64],
        apply_result: &Result<(), SolveError>,
    ) -> Result<Vec<AdditiveSweepBuffers>, SolveError> {
        // Always leave `z` fully written so a failed apply never exposes a
        // partial accumulation. On the apply-error path zero `z` up front —
        // there is nothing to reduce, and this also covers the case where the
        // subsequent pool recovery fails.
        if apply_result.is_err() {
            z.fill(0.0);
        }
        let mut buffers = self.stack.into_pool("additive.reduction.pool.into_inner")?;
        // On success, `reduce_into` zeroes-then-sums into `z`.
        if apply_result.is_ok() {
            Self::reduce_into(z, &buffers);
        }
        // Re-zero each worker's accumulator for the next round. This is a
        // `P × n_dofs` pass run on every apply, so spread it across workers
        // (rayon already owns the thread pool) rather than zeroing serially.
        buffers
            .par_iter_mut()
            .for_each(|b| b.global_accum.fill(0.0));
        Ok(buffers)
    }

    fn reduce_into(z: &mut [f64], buffers: &[AdditiveSweepBuffers]) {
        if buffers.is_empty() {
            z.fill(0.0);
            return;
        }

        const REDUCE_CHUNK: usize = 4096;
        z.par_chunks_mut(REDUCE_CHUNK)
            .enumerate()
            .for_each(|(ci, chunk)| {
                let offset = ci * REDUCE_CHUNK;
                chunk.fill(0.0);
                for buffers in buffers {
                    let accum = &buffers.global_accum[offset..offset + chunk.len()];
                    for (zi, &ai) in chunk.iter_mut().zip(accum) {
                        *zi += ai;
                    }
                }
            });
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::Ordering;

    use super::*;

    /// On `put` with `Err`, the pool must drop the buffer so the next caller
    /// gets a freshly-zeroed allocation rather than inheriting stale atomic
    /// state from a partially-completed atomic-scatter pass.
    #[test]
    fn buffer_pool_drops_dirty_buffer_on_error() {
        let pool = BufferPool::new(8, 4);

        let mut bufs = pool
            .take(ResolvedReductionStrategy::AtomicScatter)
            .expect("first take");
        match &mut bufs {
            SchwarzBuffers::Atomic { accum, .. } => {
                for slot in accum {
                    slot.store(0xdead_beef_dead_beef, Ordering::Relaxed);
                }
            }
            _ => panic!("expected atomic buffer"),
        }

        pool.put(
            bufs,
            &Err(SolveError::Synchronization {
                context: "test.simulated_failure",
            }),
        );

        let fresh = pool
            .take(ResolvedReductionStrategy::AtomicScatter)
            .expect("second take");
        match &fresh {
            SchwarzBuffers::Atomic { accum, .. } => {
                for (i, slot) in accum.iter().enumerate() {
                    assert_eq!(
                        slot.load(Ordering::Relaxed),
                        0,
                        "atomic accumulator slot {i} should be freshly zero, not stale dirty bits"
                    );
                }
            }
            _ => panic!("expected atomic buffer"),
        }
    }

    /// Companion: on `put` with `Ok`, the pool retains the buffer and a
    /// subsequent `take` of the same strategy returns the pooled instance.
    #[test]
    fn buffer_pool_retains_clean_buffer_on_success() {
        let pool = BufferPool::new(8, 4);

        let bufs = pool
            .take(ResolvedReductionStrategy::AtomicScatter)
            .expect("first take");
        pool.put(bufs, &Ok(()));

        let _ = pool
            .take(ResolvedReductionStrategy::AtomicScatter)
            .expect("second take");
        let pool_after = pool.inner.lock().expect("pool lock");
        assert!(
            pool_after.is_empty(),
            "second take should have drained the pooled buffer"
        );
    }
}
