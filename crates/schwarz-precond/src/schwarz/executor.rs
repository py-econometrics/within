//! Additive Schwarz execution engine.
//!
//! [`AdditiveExecutor`] owns the subdomain entries and a [`BufferPool`] that
//! transitively carries the global sizes. Its `apply` method takes a
//! reduction plan from the scheduler and dispatches to either the
//! atomic-scatter or the parallel-reduction backend. Buffers are taken from
//! / returned to the pool so the steady state allocates nothing.
//!
//! The pooled scratch-buffer types live in [`super::buffers`].

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use rayon::prelude::*;

use crate::error::SolveError;
use crate::local_solve::{LocalSolver, SubdomainEntry};

use super::buffers::{
    AdditiveSweepBuffers, BufferPool, LocalSolveScratch, SchwarzBuffers, WorkerBufferStack,
    WorkerReductionBuffers,
};
use super::planning::ReductionPlan;

// ============================================================================
// Additive executor
// ============================================================================

pub(super) struct AdditiveExecutor<S: LocalSolver> {
    subdomains: Arc<Vec<SubdomainEntry<S>>>,
    buf_pool: BufferPool,
}

impl<S: LocalSolver> AdditiveExecutor<S> {
    pub(super) fn new(
        subdomains: Arc<Vec<SubdomainEntry<S>>>,
        n_dofs: usize,
        max_scratch_size: usize,
    ) -> Self {
        Self {
            subdomains,
            buf_pool: BufferPool::new(n_dofs, max_scratch_size),
        }
    }

    pub(super) fn subdomains(&self) -> &[SubdomainEntry<S>] {
        &self.subdomains
    }

    pub(super) fn n_dofs(&self) -> usize {
        self.buf_pool.n_dofs()
    }

    pub(super) fn n_subdomains(&self) -> usize {
        self.subdomains.len()
    }

    /// Dispatch entry point: take a buffer from the pool, run the backend,
    /// return the buffer.
    pub(super) fn apply(
        &self,
        plan: ReductionPlan,
        r: &[f64],
        z: &mut [f64],
    ) -> Result<(), SolveError> {
        let mut bufs = self.buf_pool.take(plan.strategy)?;
        let apply_result = match &mut bufs {
            SchwarzBuffers::Atomic {
                accum,
                scratch_pool,
            } => self.apply_atomic(plan.allow_inner_parallelism, r, z, accum, scratch_pool),
            SchwarzBuffers::Reduction { pool } => {
                self.apply_parallel_reduction(plan.allow_inner_parallelism, r, z, pool)
            }
        };
        // `put` is infallible and never overwrites the real `apply_result`.
        self.buf_pool.put(bufs, &apply_result);
        apply_result
    }

    fn apply_atomic(
        &self,
        allow_inner_parallelism: bool,
        r: &[f64],
        z: &mut [f64],
        accum: &[AtomicU64],
        scratch_pool: &mut Vec<LocalSolveScratch>,
    ) -> Result<(), SolveError> {
        let max_scratch_size = self.buf_pool.max_scratch_size();
        let worker_scratch = WorkerBufferStack::new(std::mem::take(scratch_pool), move || {
            LocalSolveScratch::new(max_scratch_size)
        });
        let apply_result =
            self.subdomains
                .par_iter()
                .enumerate()
                .try_for_each(|(subdomain, entry)| {
                    worker_scratch.with_buffer(|scratch| {
                        entry
                            .apply_weighted_into_atomic(
                                r,
                                accum,
                                &mut scratch.r_scratch,
                                &mut scratch.z_scratch,
                                allow_inner_parallelism,
                            )
                            .map_err(|source| SolveError::LocalSolveFailed { subdomain, source })
                    })
                });

        // Recover the scratch pool for the next round before propagating any
        // apply error. A pool-recovery failure must not mask a real
        // `LocalSolveFailed`, so prefer the original error when it is one.
        match worker_scratch.into_pool("additive.atomic.scratch.into_inner") {
            Ok(recovered) => *scratch_pool = recovered,
            Err(into_err) => return apply_result.and(Err(into_err)),
        }

        // On apply error the swap-zero readout is skipped, leaving `accum`
        // dirty; `BufferPool::put` then drops the whole buffer (matching the
        // pre-pooling behaviour), so the next caller starts from a clean accum.
        apply_result?;

        const READOUT_CHUNK: usize = 4096;
        z.par_chunks_mut(READOUT_CHUNK)
            .enumerate()
            .for_each(|(ci, chunk)| {
                let offset = ci * READOUT_CHUNK;
                for (i, zi) in chunk.iter_mut().enumerate() {
                    let ai = &accum[offset + i];
                    *zi = f64::from_bits(ai.swap(0, Ordering::Relaxed));
                }
            });
        Ok(())
    }

    fn apply_parallel_reduction(
        &self,
        allow_inner_parallelism: bool,
        r: &[f64],
        z: &mut [f64],
        pool: &mut Vec<AdditiveSweepBuffers>,
    ) -> Result<(), SolveError> {
        let worker_buffers = WorkerReductionBuffers::new(
            std::mem::take(pool),
            self.buf_pool.n_dofs(),
            self.buf_pool.max_scratch_size(),
        );
        let apply_result =
            self.subdomains
                .par_iter()
                .enumerate()
                .try_for_each(|(subdomain, entry)| {
                    worker_buffers.stack.with_buffer(|buffers| {
                        entry
                            .apply_weighted_into_with_scratch(
                                r,
                                &mut buffers.global_accum,
                                &mut buffers.scratch.r_scratch,
                                &mut buffers.scratch.z_scratch,
                                allow_inner_parallelism,
                            )
                            .map_err(|source| SolveError::LocalSolveFailed { subdomain, source })
                    })
                });

        // `finish_round` writes `z` (zeroed on the apply-error path) and
        // recovers the pool. A pool-recovery failure must not mask a real
        // `LocalSolveFailed`, so prefer the original error when it is one.
        match worker_buffers.finish_round(z, &apply_result) {
            Ok(recovered) => *pool = recovered,
            Err(finish_err) => return apply_result.and(Err(finish_err)),
        }

        apply_result
    }
}

impl<S: LocalSolver> Clone for AdditiveExecutor<S> {
    fn clone(&self) -> Self {
        Self {
            subdomains: Arc::clone(&self.subdomains),
            buf_pool: self.buf_pool.clone(),
        }
    }
}
