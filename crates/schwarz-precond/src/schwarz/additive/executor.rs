//! Additive Schwarz execution engine.
//!
//! [`AdditiveExecutor`] owns the subdomain entries and dispatches
//! `try_apply` using the reduction plan chosen by the scheduler.
//! It manages the [`BufferPool`](super::buffers::BufferPool) for
//! zero-allocation steady-state operation.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use rayon::prelude::*;

use crate::error::ApplyError;
use crate::local_solve::{LocalSolveInvoker, LocalSolver, SubdomainEntry};

use super::buffers::{
    AdditiveSweepBuffers, BufferPool, LocalSolveScratch, SchwarzBuffers, WorkerReductionBuffers,
};
use super::planning::ReductionPlan;

pub(super) struct AdditiveExecutor<S: LocalSolver, I: LocalSolveInvoker<S>> {
    pub(super) subdomains: Arc<Vec<SubdomainEntry<S>>>,
    pub(super) n_dofs: usize,
    pub(super) max_scratch_size: usize,
    pub(super) invoker: I,
    pub(super) buf_pool: BufferPool,
}

impl<S: LocalSolver, I: LocalSolveInvoker<S>> AdditiveExecutor<S, I> {
    pub(super) fn new(
        entries: Vec<SubdomainEntry<S>>,
        n_dofs: usize,
        max_scratch_size: usize,
        invoker: I,
    ) -> Self {
        Self {
            subdomains: Arc::new(entries),
            n_dofs,
            max_scratch_size,
            invoker,
            buf_pool: BufferPool::default(),
        }
    }

    pub(super) fn subdomains(&self) -> &[SubdomainEntry<S>] {
        &self.subdomains
    }

    pub(super) fn with_fresh_buffers(&self) -> Self {
        Self {
            subdomains: Arc::clone(&self.subdomains),
            n_dofs: self.n_dofs,
            max_scratch_size: self.max_scratch_size,
            invoker: self.invoker.clone(),
            buf_pool: BufferPool::default(),
        }
    }

    pub(super) fn try_apply(
        &self,
        plan: ReductionPlan,
        r: &[f64],
        z: &mut [f64],
    ) -> Result<(), ApplyError> {
        let mut bufs = self
            .buf_pool
            .take(plan.strategy, self.n_dofs, self.max_scratch_size)?;
        let apply_result = match &mut bufs {
            SchwarzBuffers::Atomic { accum } => {
                self.apply_atomic(plan.allow_inner_parallelism, r, z, accum)
            }
            SchwarzBuffers::Reduction { pool } => {
                self.apply_parallel_reduction(plan.allow_inner_parallelism, r, z, pool)
            }
        };
        self.buf_pool.put(bufs, &apply_result)?;
        apply_result
    }

    fn apply_atomic(
        &self,
        allow_inner_parallelism: bool,
        r: &[f64],
        z: &mut [f64],
        accum: &[AtomicU64],
    ) -> Result<(), ApplyError> {
        self.subdomains.par_iter().enumerate().try_for_each_init(
            || LocalSolveScratch::new(self.max_scratch_size),
            |scratch, (subdomain, entry)| {
                entry
                    .apply_weighted_into_atomic_by(
                        r,
                        accum,
                        &mut scratch.r_scratch,
                        &mut scratch.z_scratch,
                        &self.invoker,
                        allow_inner_parallelism,
                    )
                    .map_err(|source| ApplyError::LocalSolveFailed { subdomain, source })
            },
        )?;

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
    ) -> Result<(), ApplyError> {
        let worker_buffers =
            WorkerReductionBuffers::new(std::mem::take(pool), self.n_dofs, self.max_scratch_size);
        let apply_result =
            self.subdomains
                .par_iter()
                .enumerate()
                .try_for_each(|(subdomain, entry)| {
                    worker_buffers.with_buffer(|buffers| {
                        entry
                            .apply_weighted_into_with_scratch_by(
                                r,
                                &mut buffers.global_accum,
                                &mut buffers.scratch.r_scratch,
                                &mut buffers.scratch.z_scratch,
                                &self.invoker,
                                allow_inner_parallelism,
                            )
                            .map_err(|source| ApplyError::LocalSolveFailed { subdomain, source })
                    })
                });

        *pool = worker_buffers.finish_round(z, &apply_result)?;

        apply_result
    }
}

impl<S: LocalSolver, I: LocalSolveInvoker<S>> Clone for AdditiveExecutor<S, I> {
    fn clone(&self) -> Self {
        Self {
            subdomains: Arc::clone(&self.subdomains),
            n_dofs: self.n_dofs,
            max_scratch_size: self.max_scratch_size,
            invoker: self.invoker.clone(),
            buf_pool: self.buf_pool.clone(),
        }
    }
}
