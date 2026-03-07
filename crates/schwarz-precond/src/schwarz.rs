//! Schwarz preconditioner: Operator impl + construction.
//!
//! One-level additive Schwarz: M^{-1} = Σ R_i^T D_i B_i^{-1} D_i R_i.
//!
//! Parallel per-domain apply with Rayon.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use rayon::prelude::*;

use crate::error::{validate_entries, ApplyError, PreconditionerBuildError};
use crate::local_solve::{LocalSolver, SubdomainEntry};
use crate::Operator;

// ---------------------------------------------------------------------------
// Buffer types
// ---------------------------------------------------------------------------

/// Thread-local scratch for the atomic scatter path.
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

/// Pooled buffers: shared atomic accumulator with per-task scratch via Rayon init.
struct SchwarzBuffers {
    accum: Vec<AtomicU64>,
}

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
    /// Pool of reusable buffer sets.
    /// Each concurrent `apply()` call pops one; returns it when done.
    buf_pool: Arc<Mutex<Vec<SchwarzBuffers>>>,
}

impl<S: LocalSolver> SchwarzPreconditioner<S> {
    /// Construct from pre-built subdomain entries.
    pub fn new(
        entries: Vec<SubdomainEntry<S>>,
        n_dofs: usize,
    ) -> Result<Self, PreconditionerBuildError> {
        validate_entries(&entries, n_dofs)?;
        let max_scratch_size = entries.iter().map(|e| e.scratch_size()).max().unwrap_or(0);
        Ok(Self {
            n_dofs,
            subdomains: Arc::new(entries),
            max_scratch_size,
            buf_pool: Arc::new(Mutex::new(Vec::new())),
        })
    }

    /// Access the underlying subdomain entries.
    pub fn subdomains(&self) -> &[SubdomainEntry<S>] {
        &self.subdomains
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
            .unwrap_or_else(|| SchwarzBuffers {
                accum: (0..n).map(|_| AtomicU64::new(0)).collect(),
            });

        let apply_result = {
            let accum = &bufs.accum;
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
