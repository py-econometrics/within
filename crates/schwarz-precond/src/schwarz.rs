//! Schwarz preconditioner: Operator impl + construction.
//!
//! One-level additive Schwarz: M^{-1} = Σ R_i^T D_i B_i^{-1} D_i R_i.
//!
//! Parallel per-domain apply with Rayon.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use rayon::prelude::*;
use thread_local::ThreadLocal;

use crate::error::{validate_entries, ApplyError, PreconditionerBuildError};
use crate::local_solve::{LocalSolver, SubdomainEntry};
use crate::Operator;

// ---------------------------------------------------------------------------
// ReductionStrategy
// ---------------------------------------------------------------------------

/// Strategy for combining per-subdomain results in additive Schwarz apply.
#[derive(Debug, Clone, Copy, Default)]
pub enum ReductionStrategy {
    /// Each subdomain atomically scatters into a shared accumulator.
    /// Memory: O(n_dofs) shared + O(P * max_scratch) thread-local.
    #[default]
    AtomicScatter,
    /// Each thread accumulates into a private buffer, then a parallel
    /// chunk-based reduction combines them.
    /// Memory: O(P * n_dofs) for thread buffers.
    ParallelReduction,
}

// ---------------------------------------------------------------------------
// Buffer types
// ---------------------------------------------------------------------------

/// Per-thread scratch buffers for the parallel-reduction path.
struct AdditiveSweepBuffers {
    global_accum: Vec<f64>,
    r_scratch: Vec<f64>,
    z_scratch: Vec<f64>,
}

/// Thread-local scratch for the atomic scatter path (no global_accum needed).
struct AtomicScratch {
    r_scratch: Vec<f64>,
    z_scratch: Vec<f64>,
}

impl AtomicScratch {
    #[inline]
    fn new(max_scratch_size: usize) -> Self {
        Self {
            r_scratch: vec![0.0f64; max_scratch_size],
            z_scratch: vec![0.0f64; max_scratch_size],
        }
    }
}

/// Pooled buffers that vary by reduction strategy.
enum SchwarzBuffers {
    /// Shared atomic accumulator; scratch is per-task via Rayon init.
    Atomic { accum: Vec<AtomicU64> },
    /// Per-thread full buffers (same as old `AdditiveSweepBuffers`).
    PerThread(ThreadLocal<Mutex<AdditiveSweepBuffers>>),
}

// SAFETY: Vec<AtomicU64> is Send+Sync.
// ThreadLocal<Mutex<T>> is Send+Sync in the per-thread reduction path (T: Send).
// AtomicScratch and AdditiveSweepBuffers contain only Vec<f64>, which is Send.
unsafe impl Send for SchwarzBuffers {}
unsafe impl Sync for SchwarzBuffers {}

/// Wrapper around `*const f64` that is `Send + Sync`.
///
/// SAFETY: The caller must ensure that the pointer is valid for the lifetime
/// of reads and that no mutable aliases exist during the read window.
#[derive(Clone, Copy)]
struct SendSyncPtr(*const f64);
unsafe impl Send for SendSyncPtr {}
unsafe impl Sync for SendSyncPtr {}

// ---------------------------------------------------------------------------
// SchwarzPreconditioner<S>
// ---------------------------------------------------------------------------

/// One-level additive Schwarz preconditioner, generic over the local solver.
///
/// Subdomains (factored matrices) are stored behind `Arc` so that cloning
/// shares the heavy subdomain data. A pool of per-thread buffer sets enables
/// safe concurrent `apply()` calls on the same instance — each caller grabs
/// an independent buffer set from the pool for the duration of the call.
pub struct SchwarzPreconditioner<S: LocalSolver> {
    pub(crate) subdomains: Arc<Vec<SubdomainEntry<S>>>,
    n_dofs: usize,
    /// Maximum scratch size across all subdomains, for buffer sizing.
    max_scratch_size: usize,
    /// Strategy for combining per-subdomain results.
    reduction_strategy: ReductionStrategy,
    /// Pool of reusable buffer sets.
    /// Each concurrent `apply()` call pops one; returns it when done.
    buf_pool: Arc<Mutex<Vec<SchwarzBuffers>>>,
}

impl<S: LocalSolver> SchwarzPreconditioner<S> {
    /// Construct from pre-built subdomain entries using the default strategy.
    pub fn new(
        entries: Vec<SubdomainEntry<S>>,
        n_dofs: usize,
    ) -> Result<Self, PreconditionerBuildError> {
        Self::with_strategy(entries, n_dofs, ReductionStrategy::default())
    }

    /// Construct from pre-built subdomain entries with an explicit reduction strategy.
    pub fn with_strategy(
        entries: Vec<SubdomainEntry<S>>,
        n_dofs: usize,
        strategy: ReductionStrategy,
    ) -> Result<Self, PreconditionerBuildError> {
        validate_entries(&entries, n_dofs)?;
        let max_scratch_size = entries.iter().map(|e| e.scratch_size()).max().unwrap_or(0);
        Ok(Self {
            n_dofs,
            subdomains: Arc::new(entries),
            max_scratch_size,
            reduction_strategy: strategy,
            buf_pool: Arc::new(Mutex::new(Vec::new())),
        })
    }

    /// Access the underlying subdomain entries.
    pub fn subdomains(&self) -> &[SubdomainEntry<S>] {
        &self.subdomains
    }

    /// Return a copy that uses a different reduction strategy.
    ///
    /// Shares the subdomain data via `Arc` (O(1)), but creates a fresh
    /// buffer pool since buffers are strategy-specific.
    pub fn with_reduction_strategy(&self, strategy: ReductionStrategy) -> Self {
        Self {
            subdomains: Arc::clone(&self.subdomains),
            n_dofs: self.n_dofs,
            max_scratch_size: self.max_scratch_size,
            reduction_strategy: strategy,
            buf_pool: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Fallible operator apply that propagates local-solver failures.
    pub fn try_apply(&self, r: &[f64], z: &mut [f64]) -> Result<(), ApplyError> {
        let n = self.n_dofs;
        let max_ss = self.max_scratch_size;

        let bufs = self
            .buf_pool
            .lock()
            .map_err(|_| ApplyError::Synchronization {
                context: "additive.buf_pool.lock.pop",
            })?
            .pop()
            .unwrap_or_else(|| match self.reduction_strategy {
                ReductionStrategy::AtomicScatter => SchwarzBuffers::Atomic {
                    accum: (0..n).map(|_| AtomicU64::new(0)).collect(),
                },
                ReductionStrategy::ParallelReduction => {
                    SchwarzBuffers::PerThread(ThreadLocal::new())
                }
            });

        let apply_result = match &bufs {
            SchwarzBuffers::Atomic { accum } => {
                self.subdomains.par_iter().enumerate().try_for_each_init(
                    || AtomicScratch::new(max_ss),
                    |local, (subdomain, entry)| {
                        entry
                            .apply_weighted_into_atomic(
                                r,
                                accum,
                                &mut local.r_scratch,
                                &mut local.z_scratch,
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
            SchwarzBuffers::PerThread(thread_bufs) => {
                self.subdomains
                    .par_iter()
                    .enumerate()
                    .try_for_each(|(subdomain, entry)| {
                        let mutex = thread_bufs.get_or(|| {
                            Mutex::new(AdditiveSweepBuffers {
                                global_accum: vec![0.0f64; n],
                                r_scratch: vec![0.0f64; max_ss],
                                z_scratch: vec![0.0f64; max_ss],
                            })
                        });
                        let mut bufs = mutex.lock().map_err(|_| ApplyError::Synchronization {
                            context: "additive.per_thread.lock.apply",
                        })?;
                        let AdditiveSweepBuffers {
                            ref mut global_accum,
                            ref mut r_scratch,
                            ref mut z_scratch,
                        } = *bufs;
                        entry
                            .apply_weighted_into_with_scratch(r, global_accum, r_scratch, z_scratch)
                            .map_err(|source| ApplyError::LocalSolveFailed { subdomain, source })
                    })?;

                const REDUCE_CHUNK: usize = 4096;
                let guards: Vec<_> = thread_bufs
                    .iter()
                    .map(|m| {
                        m.lock().map_err(|_| ApplyError::Synchronization {
                            context: "additive.per_thread.lock.reduce",
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let ptrs: Vec<SendSyncPtr> = guards
                    .iter()
                    .map(|g| SendSyncPtr(g.global_accum.as_ptr()))
                    .collect();

                z.par_chunks_mut(REDUCE_CHUNK)
                    .enumerate()
                    .for_each(|(ci, chunk)| {
                        let offset = ci * REDUCE_CHUNK;
                        chunk.fill(0.0);
                        for &SendSyncPtr(ptr) in &ptrs {
                            for (i, zi) in chunk.iter_mut().enumerate() {
                                // SAFETY: all mutex guards are held (exclusive access),
                                // pointers are valid for guards' lifetime,
                                // rayon threads read non-overlapping accum ranges +
                                // write non-overlapping z chunks.
                                unsafe {
                                    *zi += *ptr.add(offset + i);
                                }
                            }
                        }
                    });

                for mut guard in guards {
                    guard.global_accum.fill(0.0);
                }
                Ok(())
            }
        };

        if let Ok(mut pool) = self.buf_pool.lock() {
            pool.push(bufs);
        } else if apply_result.is_ok() {
            return Err(ApplyError::Synchronization {
                context: "additive.buf_pool.lock.push",
            });
        }

        apply_result
    }
}

impl<S: LocalSolver> Clone for SchwarzPreconditioner<S> {
    /// Clone shares both the subdomain data and the buffer pool via `Arc`.
    /// This is O(1) and the clone is fully interchangeable with the original.
    fn clone(&self) -> Self {
        Self {
            subdomains: Arc::clone(&self.subdomains),
            n_dofs: self.n_dofs,
            max_scratch_size: self.max_scratch_size,
            reduction_strategy: self.reduction_strategy,
            buf_pool: Arc::clone(&self.buf_pool),
        }
    }
}

// ---------------------------------------------------------------------------
// Operator impl (symmetric)
// ---------------------------------------------------------------------------

impl<S: LocalSolver> Operator for SchwarzPreconditioner<S> {
    fn nrows(&self) -> usize {
        self.n_dofs
    }

    fn ncols(&self) -> usize {
        self.n_dofs
    }

    fn apply(&self, r: &[f64], z: &mut [f64]) {
        if self.try_apply(r, z).is_err() {
            z.fill(f64::NAN);
        }
    }

    fn apply_adjoint(&self, r: &[f64], z: &mut [f64]) {
        self.apply(r, z); // symmetric
    }

    fn try_apply(&self, r: &[f64], z: &mut [f64]) -> Result<(), ApplyError> {
        SchwarzPreconditioner::try_apply(self, r, z)
    }

    fn try_apply_adjoint(&self, r: &[f64], z: &mut [f64]) -> Result<(), ApplyError> {
        SchwarzPreconditioner::try_apply(self, r, z)
    }
}
