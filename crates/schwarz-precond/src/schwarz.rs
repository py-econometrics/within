//! Schwarz preconditioners: additive and multiplicative variants.
//!
//! **Additive:** M^{-1} = Σ R_i^T D_i B_i^{-1} D_i R_i — parallel per-domain apply with Rayon.
//!
//! **Multiplicative:** Sequential forward/backward sweep with residual updates between subdomains.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use rayon::prelude::*;

use crate::domain::PartitionWeights;
use crate::error::{validate_entries, ApplyError, LocalSolveError, PreconditionerBuildError};
use crate::local_solve::{LocalSolver, SubdomainEntry};
use crate::Operator;

// ============================================================================
// Additive Schwarz
// ============================================================================

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

// ============================================================================
// Multiplicative Schwarz
// ============================================================================

/// Scratch buffers for a multiplicative sweep (single-threaded).
struct SweepBuffers {
    /// Working copy of the residual, updated after each subdomain solve.
    r_work: Vec<f64>,
    /// Local RHS scratch (length = max_scratch_size).
    r_scratch: Vec<f64>,
    /// Local solution scratch (length = max_scratch_size).
    z_scratch: Vec<f64>,
    /// Weighted correction for residual update (length = max local DOFs).
    correction: Vec<f64>,
}

impl SweepBuffers {
    fn new(n_dofs: usize, max_scratch_size: usize, max_local_dofs: usize) -> Self {
        Self {
            r_work: vec![0.0; n_dofs],
            r_scratch: vec![0.0; max_scratch_size],
            z_scratch: vec![0.0; max_scratch_size],
            correction: vec![0.0; max_local_dofs],
        }
    }
}

/// Process a single subdomain: restrict, local solve, prolongate, update residual.
fn apply_subdomain<S: LocalSolver, U: ResidualUpdater>(
    entry: &SubdomainEntry<S>,
    r_work: &mut [f64],
    z: &mut [f64],
    r_scratch: &mut [f64],
    z_scratch: &mut [f64],
    correction: &mut [f64],
    updater: &mut U,
) -> Result<(), LocalSolveError> {
    if entry.core.global_indices.is_empty() {
        return Ok(());
    }

    let n_local = entry.core.global_indices.len();

    entry.core.restrict_weighted(r_work, r_scratch);
    entry.solver.solve_local(r_scratch, z_scratch)?;
    entry.core.prolongate_weighted_add(z_scratch, z);

    match &entry.core.partition_weights {
        PartitionWeights::Uniform(_) => {
            updater.update(&entry.core.global_indices, &z_scratch[..n_local], r_work);
        }
        PartitionWeights::NonUniform(w) => {
            for k in 0..n_local {
                correction[k] = w[k] * z_scratch[k];
            }
            updater.update(&entry.core.global_indices, &correction[..n_local], r_work);
        }
    }
    Ok(())
}

/// Compute subdomain sizing metadata.
fn compute_sizes<S: LocalSolver>(entries: &[SubdomainEntry<S>]) -> (usize, usize) {
    let max_scratch_size = entries.iter().map(|e| e.scratch_size()).max().unwrap_or(0);
    let max_local_dofs = entries
        .iter()
        .map(|e| e.core.global_indices.len())
        .max()
        .unwrap_or(0);
    (max_scratch_size, max_local_dofs)
}

/// Multiplicative Schwarz preconditioner, generic over local solver and residual updater.
pub struct MultiplicativeSchwarzPreconditioner<S: LocalSolver, U: ResidualUpdater> {
    subdomains: Vec<SubdomainEntry<S>>,
    updater: Mutex<U>,
    n_dofs: usize,
    symmetric: bool,
    scratch: Mutex<SweepBuffers>,
}

impl<S: LocalSolver, U: ResidualUpdater> MultiplicativeSchwarzPreconditioner<S, U> {
    /// Construct a multiplicative Schwarz preconditioner.
    pub fn new(
        entries: Vec<SubdomainEntry<S>>,
        updater: U,
        n_dofs: usize,
        symmetric: bool,
    ) -> Result<Self, PreconditionerBuildError> {
        validate_entries(&entries, n_dofs)?;
        let (max_scratch_size, max_local_dofs) = compute_sizes(&entries);
        Ok(Self {
            subdomains: entries,
            updater: Mutex::new(updater),
            n_dofs,
            symmetric,
            scratch: Mutex::new(SweepBuffers::new(n_dofs, max_scratch_size, max_local_dofs)),
        })
    }

    /// Access the underlying subdomain entries.
    pub fn subdomains(&self) -> &[SubdomainEntry<S>] {
        &self.subdomains
    }

    /// Fallible operator apply that propagates local-solver failures.
    pub fn try_apply(&self, r: &[f64], z: &mut [f64]) -> Result<(), ApplyError> {
        let mut bufs = self
            .scratch
            .lock()
            .map_err(|_| ApplyError::Synchronization {
                context: "multiplicative.scratch.lock",
            })?;
        let mut updater = self
            .updater
            .lock()
            .map_err(|_| ApplyError::Synchronization {
                context: "multiplicative.updater.lock",
            })?;
        let SweepBuffers {
            ref mut r_work,
            ref mut r_scratch,
            ref mut z_scratch,
            ref mut correction,
        } = *bufs;

        z.fill(0.0);

        r_work.copy_from_slice(r);
        updater.reset(r);
        for (subdomain, entry) in self.subdomains.iter().enumerate() {
            apply_subdomain(
                entry,
                r_work,
                z,
                r_scratch,
                z_scratch,
                correction,
                &mut *updater,
            )
            .map_err(|source| ApplyError::LocalSolveFailed { subdomain, source })?;
        }

        if self.symmetric {
            updater.reset(r_work);
            for (rev_idx, entry) in self.subdomains.iter().rev().enumerate() {
                let subdomain = self.subdomains.len().saturating_sub(1) - rev_idx;
                apply_subdomain(
                    entry,
                    r_work,
                    z,
                    r_scratch,
                    z_scratch,
                    correction,
                    &mut *updater,
                )
                .map_err(|source| ApplyError::LocalSolveFailed { subdomain, source })?;
            }
        }
        Ok(())
    }
}

impl<S: LocalSolver, U: ResidualUpdater> Operator for MultiplicativeSchwarzPreconditioner<S, U> {
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
        self.apply(r, z);
    }

    fn try_apply(&self, r: &[f64], z: &mut [f64]) -> Result<(), ApplyError> {
        MultiplicativeSchwarzPreconditioner::try_apply(self, r, z)
    }

    fn try_apply_adjoint(&self, r: &[f64], z: &mut [f64]) -> Result<(), ApplyError> {
        MultiplicativeSchwarzPreconditioner::try_apply(self, r, z)
    }
}

// ============================================================================
// Residual updater
// ============================================================================

/// Trait for updating the global residual after a single subdomain correction
/// in a multiplicative Schwarz sweep.
///
/// The multiplicative Schwarz preconditioner needs to update r_work after each
/// subdomain solve. Different strategies exist:
/// - `OperatorResidualUpdater`: naive full recompute via r = b - A*y_accum (O(n) per update)
/// - Observation-space updater (in `within` crate): exploits FE structure for sparse updates
pub trait ResidualUpdater: Send + Sync {
    /// Update the working residual after a subdomain correction.
    ///
    /// `global_indices`: the DOF indices touched by this subdomain
    /// `weighted_correction`: the PoU-weighted local correction (same length as `global_indices`)
    /// `r_work`: the full global residual vector to update in-place
    fn update(&mut self, global_indices: &[u32], weighted_correction: &[f64], r_work: &mut [f64]);

    /// Reset before a new sweep. Called at the start of each forward/backward sweep.
    fn reset(&mut self, r_original: &[f64]);
}

/// Naive residual updater that recomputes r = b - A * y_accum after each subdomain.
///
/// Maintains an internal accumulator y_accum. On each `update()`:
/// 1. Scatter `weighted_correction` into y_accum at `global_indices`
/// 2. Recompute `r_work = r_original - A * y_accum`
///
/// This is O(n_dofs) per update — correct but slow. Useful as a testing baseline.
pub struct OperatorResidualUpdater<'a, A: Operator> {
    operator: &'a A,
    /// Accumulator: sum of all corrections applied so far
    y_accum: Vec<f64>,
    /// Original residual (r before the sweep started)
    r_original: Vec<f64>,
    /// Scratch buffer for A * y_accum
    a_y: Vec<f64>,
}

impl<'a, A: Operator> OperatorResidualUpdater<'a, A> {
    /// Create a new updater.
    pub fn new(operator: &'a A, n_dofs: usize) -> Self {
        Self {
            operator,
            y_accum: vec![0.0; n_dofs],
            r_original: vec![0.0; n_dofs],
            a_y: vec![0.0; n_dofs],
        }
    }
}

impl<A: Operator> ResidualUpdater for OperatorResidualUpdater<'_, A> {
    fn update(&mut self, global_indices: &[u32], weighted_correction: &[f64], r_work: &mut [f64]) {
        // 1. Scatter correction into accumulator
        for (k, &gi) in global_indices.iter().enumerate() {
            self.y_accum[gi as usize] += weighted_correction[k];
        }

        // 2. Recompute r_work = r_original - A * y_accum
        self.operator.apply(&self.y_accum, &mut self.a_y);
        for (r, (&ro, &ay)) in r_work
            .iter_mut()
            .zip(self.r_original.iter().zip(self.a_y.iter()))
        {
            *r = ro - ay;
        }
    }

    fn reset(&mut self, r_original: &[f64]) {
        self.r_original.copy_from_slice(r_original);
        self.y_accum.iter_mut().for_each(|v| *v = 0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple diagonal operator for testing: A = diag(values)
    struct DiagOperator {
        values: Vec<f64>,
    }

    impl DiagOperator {
        fn new(values: Vec<f64>) -> Self {
            Self { values }
        }
    }

    impl Operator for DiagOperator {
        fn nrows(&self) -> usize {
            self.values.len()
        }
        fn ncols(&self) -> usize {
            self.values.len()
        }
        fn apply(&self, x: &[f64], y: &mut [f64]) {
            for i in 0..self.values.len() {
                y[i] = self.values[i] * x[i];
            }
        }
        fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) {
            self.apply(x, y);
        }
    }

    #[test]
    fn test_operator_residual_updater_basic() {
        // A = diag(2, 3, 1, 4)
        let a = DiagOperator::new(vec![2.0, 3.0, 1.0, 4.0]);
        let mut updater = OperatorResidualUpdater::new(&a, 4);

        // r_original = [10, 12, 5, 8]
        let r_original = [10.0, 12.0, 5.0, 8.0];
        updater.reset(&r_original);

        let mut r_work = r_original.to_vec();

        // First correction: indices [0, 2], correction [1.0, 2.0]
        // y_accum becomes [1, 0, 2, 0]
        // A * y_accum = [2, 0, 2, 0]
        // r_work = [10-2, 12-0, 5-2, 8-0] = [8, 12, 3, 8]
        updater.update(&[0, 2], &[1.0, 2.0], &mut r_work);
        assert!((r_work[0] - 8.0).abs() < 1e-12);
        assert!((r_work[1] - 12.0).abs() < 1e-12);
        assert!((r_work[2] - 3.0).abs() < 1e-12);
        assert!((r_work[3] - 8.0).abs() < 1e-12);

        // Second correction: indices [1, 3], correction [0.5, 1.0]
        // y_accum becomes [1, 0.5, 2, 1]
        // A * y_accum = [2, 1.5, 2, 4]
        // r_work = [10-2, 12-1.5, 5-2, 8-4] = [8, 10.5, 3, 4]
        updater.update(&[1, 3], &[0.5, 1.0], &mut r_work);
        assert!((r_work[0] - 8.0).abs() < 1e-12);
        assert!((r_work[1] - 10.5).abs() < 1e-12);
        assert!((r_work[2] - 3.0).abs() < 1e-12);
        assert!((r_work[3] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn test_operator_residual_updater_reset() {
        let a = DiagOperator::new(vec![1.0, 1.0]);
        let mut updater = OperatorResidualUpdater::new(&a, 2);

        // First sweep
        updater.reset(&[5.0, 3.0]);
        let mut r = vec![5.0, 3.0];
        updater.update(&[0], &[2.0], &mut r);
        assert!((r[0] - 3.0).abs() < 1e-12);

        // Reset for second sweep — accumulator should be cleared
        updater.reset(&[10.0, 7.0]);
        let mut r = vec![10.0, 7.0];
        updater.update(&[1], &[1.0], &mut r);
        // y_accum = [0, 1], A*y = [0, 1], r = [10-0, 7-1] = [10, 6]
        assert!((r[0] - 10.0).abs() < 1e-12);
        assert!((r[1] - 6.0).abs() < 1e-12);
    }
}
