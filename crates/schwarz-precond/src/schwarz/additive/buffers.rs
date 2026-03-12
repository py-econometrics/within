use std::cell::RefCell;
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Mutex};

use thread_local::ThreadLocal;

use crate::error::ApplyError;

use super::planning::ResolvedReductionStrategy;

#[derive(Default, Clone)]
pub(super) struct BufferPool {
    inner: Arc<Mutex<Vec<SchwarzBuffers>>>,
}

impl BufferPool {
    const MAX_POOL_SIZE: usize = 4;

    pub(super) fn take(
        &self,
        strategy: ResolvedReductionStrategy,
        n_dofs: usize,
        max_scratch_size: usize,
    ) -> Result<SchwarzBuffers, ApplyError> {
        let mut pool = self.inner.lock().map_err(|_| ApplyError::Synchronization {
            context: "additive.buf_pool.lock.pop",
        })?;
        if let Some(idx) = pool.iter().position(|bufs| bufs.strategy() == strategy) {
            return Ok(pool.swap_remove(idx));
        }
        Ok(SchwarzBuffers::new(strategy, n_dofs, max_scratch_size))
    }

    pub(super) fn put(
        &self,
        bufs: SchwarzBuffers,
        apply_result: &Result<(), ApplyError>,
    ) -> Result<(), ApplyError> {
        if let Ok(mut pool) = self.inner.lock() {
            if pool.len() < Self::MAX_POOL_SIZE {
                pool.push(bufs);
            }
            Ok(())
        } else if apply_result.is_ok() {
            Err(ApplyError::Synchronization {
                context: "additive.buf_pool.lock.push",
            })
        } else {
            Ok(())
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
    pub(super) fn new(n_dofs: usize, max_scratch_size: usize) -> Self {
        Self {
            global_accum: vec![0.0f64; n_dofs],
            scratch: LocalSolveScratch::new(max_scratch_size),
        }
    }
}

/// Pooled buffers that vary by reduction strategy.
pub(super) enum SchwarzBuffers {
    /// Shared atomic accumulator.
    Atomic { accum: Vec<AtomicU64> },
    /// Reusable task-local buffers for parallel reduction.
    Reduction { pool: Vec<AdditiveSweepBuffers> },
}

impl SchwarzBuffers {
    pub(super) fn new(
        strategy: ResolvedReductionStrategy,
        n_dofs: usize,
        max_scratch_size: usize,
    ) -> Self {
        match strategy {
            ResolvedReductionStrategy::AtomicScatter => Self::Atomic {
                accum: (0..n_dofs).map(|_| AtomicU64::new(0)).collect(),
            },
            ResolvedReductionStrategy::ParallelReduction => Self::Reduction {
                pool: vec![AdditiveSweepBuffers::new(n_dofs, max_scratch_size)],
            },
        }
    }

    pub(super) fn strategy(&self) -> ResolvedReductionStrategy {
        match self {
            Self::Atomic { .. } => ResolvedReductionStrategy::AtomicScatter,
            Self::Reduction { .. } => ResolvedReductionStrategy::ParallelReduction,
        }
    }
}

/// Worker-local buffer stacks for additive parallel reduction.
///
/// Each Rayon worker reuses its own accumulator buffers across sequential outer
/// tasks. Nested re-entry on the same worker allocates a second buffer only when
/// needed, so the number of retained full-length accumulators tracks re-entry
/// depth rather than Rayon task splitting.
pub(super) struct WorkerReductionBuffers {
    shared_pool: Mutex<Vec<AdditiveSweepBuffers>>,
    worker_stacks: ThreadLocal<RefCell<Vec<AdditiveSweepBuffers>>>,
    n_dofs: usize,
    max_scratch_size: usize,
}

impl WorkerReductionBuffers {
    pub(super) fn new(
        pool: Vec<AdditiveSweepBuffers>,
        n_dofs: usize,
        max_scratch_size: usize,
    ) -> Self {
        Self {
            shared_pool: Mutex::new(pool),
            worker_stacks: ThreadLocal::with_capacity(rayon::current_num_threads().max(1)),
            n_dofs,
            max_scratch_size,
        }
    }

    pub(super) fn with_buffer<T>(&self, f: impl FnOnce(&mut AdditiveSweepBuffers) -> T) -> T {
        let worker_stack = self.worker_stacks.get_or(|| RefCell::new(Vec::new()));
        let mut buffers = if let Some(buffers) = worker_stack.borrow_mut().pop() {
            buffers
        } else {
            take_reduction_buffer(&self.shared_pool, self.n_dofs, self.max_scratch_size)
        };

        let result = f(&mut buffers);
        self.worker_stacks
            .get_or(|| RefCell::new(Vec::new()))
            .borrow_mut()
            .push(buffers);
        result
    }

    pub(super) fn into_buffers(mut self) -> Result<Vec<AdditiveSweepBuffers>, ApplyError> {
        let mut buffers =
            self.shared_pool
                .into_inner()
                .map_err(|_| ApplyError::Synchronization {
                    context: "additive.reduction.pool.into_inner",
                })?;
        for worker_stack in self.worker_stacks.iter_mut() {
            buffers.append(worker_stack.get_mut());
        }
        Ok(buffers)
    }
}

pub(super) fn reduce_additive_buffers_into(z: &mut [f64], buffers: &[AdditiveSweepBuffers]) {
    if buffers.is_empty() {
        z.fill(0.0);
        return;
    }

    const REDUCE_CHUNK: usize = 4096;
    use rayon::prelude::*;
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

pub(super) fn clear_additive_buffers(buffers: &mut [AdditiveSweepBuffers]) {
    for buffers in buffers {
        buffers.global_accum.fill(0.0);
    }
}

fn take_reduction_buffer(
    pool: &Mutex<Vec<AdditiveSweepBuffers>>,
    n: usize,
    max_ss: usize,
) -> AdditiveSweepBuffers {
    pool.lock()
        .ok()
        .and_then(|mut pool| pool.pop())
        .unwrap_or_else(|| AdditiveSweepBuffers::new(n, max_ss))
}
