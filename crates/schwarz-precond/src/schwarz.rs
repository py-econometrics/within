//! Schwarz preconditioners: additive and multiplicative variants.
//!
//! **Additive:** M^{-1} = Σ R_i^T D_i B_i^{-1} D_i R_i — parallel per-domain apply with Rayon.
//!
//! **Multiplicative:** Sequential forward/backward sweep with residual updates between subdomains.

use std::cell::{Cell, RefCell};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use rayon::prelude::*;
use thread_local::ThreadLocal;

use crate::domain::PartitionWeights;
use crate::error::{validate_entries, ApplyError, LocalSolveError, PreconditionerBuildError};
use crate::local_solve::{LocalSolver, SubdomainEntry};
use crate::Operator;

// ============================================================================
// Reduction strategy
// ============================================================================

/// Strategy for combining per-subdomain results in additive Schwarz apply.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ReductionStrategy {
    /// Choose a backend from build-time metrics and the current Rayon width.
    #[default]
    Auto,
    /// Each subdomain atomically scatters into a shared accumulator.
    /// Memory: O(n_dofs) shared + O(P * max_scratch) thread-local.
    AtomicScatter,
    /// Each task accumulates into a private buffer, then a parallel
    /// chunk-based reduction combines them.
    /// Memory: O(P * n_dofs) for task buffers.
    ParallelReduction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResolvedReductionStrategy {
    AtomicScatter,
    ParallelReduction,
}

impl ResolvedReductionStrategy {
    fn as_public(self) -> ReductionStrategy {
        match self {
            Self::AtomicScatter => ReductionStrategy::AtomicScatter,
            Self::ParallelReduction => ReductionStrategy::ParallelReduction,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct ReductionPlan {
    strategy: ResolvedReductionStrategy,
    allow_inner_parallelism: bool,
}

/// Build-time metrics that describe additive Schwarz scheduling pressure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdditiveSchwarzDiagnostics {
    n_subdomains: usize,
    n_dofs: usize,
    total_inner_parallel_work: usize,
    max_inner_parallel_work: usize,
    total_scatter_dofs: usize,
}

impl AdditiveSchwarzDiagnostics {
    const MIN_INNER_PARALLEL_WORK: usize = 200_000;
    const OUTER_CAPACITY_TARGET: f64 = 0.75;
    const AUTO_REDUCTION_SWEEP_FACTOR: f64 = 1.1;
    const AUTO_INNER_REDUCTION_SWEEP_FACTOR: f64 = 6.0;
    const AUTO_OVERLAP_FOR_REDUCTION: f64 = 4.0;

    fn from_entries<S: LocalSolver>(entries: &[SubdomainEntry<S>], n_dofs: usize) -> Self {
        let (total_inner_parallel_work, max_inner_parallel_work, total_scatter_dofs) =
            entries.iter().fold(
                (0usize, 0usize, 0usize),
                |(total_work, max_work, total_scatter), entry| {
                    let work = entry.solver.inner_parallelism_work_estimate();
                    (
                        total_work.saturating_add(work),
                        max_work.max(work),
                        total_scatter.saturating_add(entry.core.global_indices.len()),
                    )
                },
            );
        Self {
            n_subdomains: entries.len(),
            n_dofs,
            total_inner_parallel_work,
            max_inner_parallel_work,
            total_scatter_dofs,
        }
    }

    /// Number of additive Schwarz subdomains.
    pub fn n_subdomains(&self) -> usize {
        self.n_subdomains
    }

    /// Global number of degrees of freedom.
    pub fn n_dofs(&self) -> usize {
        self.n_dofs
    }

    /// Sum of per-subdomain work estimates that can benefit from nested Rayon.
    pub fn total_inner_parallel_work(&self) -> usize {
        self.total_inner_parallel_work
    }

    /// Largest single-subdomain inner-parallel work estimate.
    pub fn max_inner_parallel_work(&self) -> usize {
        self.max_inner_parallel_work
    }

    /// Sum of local scatter entries across all subdomains.
    pub fn total_scatter_dofs(&self) -> usize {
        self.total_scatter_dofs
    }

    fn reduction_plan(&self, configured: ReductionStrategy, threads: usize) -> ReductionPlan {
        let allow_inner_parallelism = self.allow_inner_parallelism(threads);
        let strategy = self.resolve_strategy(configured, threads, allow_inner_parallelism);
        ReductionPlan {
            strategy,
            allow_inner_parallelism,
        }
    }

    fn allow_inner_parallelism(&self, threads: usize) -> bool {
        if self.max_inner_parallel_work < Self::MIN_INNER_PARALLEL_WORK {
            return false;
        }

        self.outer_parallel_capacity() < (threads as f64 * Self::OUTER_CAPACITY_TARGET)
    }

    fn resolve_strategy(
        &self,
        configured: ReductionStrategy,
        threads: usize,
        allow_inner_parallelism: bool,
    ) -> ResolvedReductionStrategy {
        match configured {
            ReductionStrategy::AtomicScatter => ResolvedReductionStrategy::AtomicScatter,
            ReductionStrategy::ParallelReduction => ResolvedReductionStrategy::ParallelReduction,
            ReductionStrategy::Auto => self.pick_auto_strategy(threads, allow_inner_parallelism),
        }
    }

    fn pick_auto_strategy(
        &self,
        threads: usize,
        allow_inner_parallelism: bool,
    ) -> ResolvedReductionStrategy {
        let overlap = self.scatter_overlap();
        let reduction_to_scatter = self.reduction_sweep_to_scatter(threads);

        if reduction_to_scatter <= Self::AUTO_REDUCTION_SWEEP_FACTOR {
            return ResolvedReductionStrategy::ParallelReduction;
        }

        if allow_inner_parallelism
            && reduction_to_scatter <= Self::AUTO_INNER_REDUCTION_SWEEP_FACTOR
        {
            return ResolvedReductionStrategy::ParallelReduction;
        }

        if overlap >= Self::AUTO_OVERLAP_FOR_REDUCTION {
            return ResolvedReductionStrategy::ParallelReduction;
        }

        ResolvedReductionStrategy::AtomicScatter
    }

    /// Estimated outer parallel capacity: roughly how many heavy subdomains exist.
    pub fn outer_parallel_capacity(&self) -> f64 {
        if self.max_inner_parallel_work == 0 {
            return 0.0;
        }
        self.total_inner_parallel_work as f64 / self.max_inner_parallel_work as f64
    }

    /// Average overlap multiplicity of the additive scatter.
    pub fn scatter_overlap(&self) -> f64 {
        self.total_scatter_dofs as f64 / self.n_dofs.max(1) as f64
    }

    fn reduction_sweep_to_scatter(&self, threads: usize) -> f64 {
        let active_buffers = threads.min(self.n_subdomains).max(1);
        let reduction_sweep = active_buffers.saturating_mul(self.n_dofs);
        let scatter_work = self.total_scatter_dofs.max(1);
        reduction_sweep as f64 / scatter_work as f64
    }
}

std::thread_local! {
    static LOCAL_SOLVER_INNER_PARALLELISM: Cell<bool> = const { Cell::new(true) };
}

/// Returns whether local solvers may spawn nested Rayon work on this thread.
pub fn local_solver_inner_parallelism_enabled() -> bool {
    LOCAL_SOLVER_INNER_PARALLELISM.with(Cell::get)
}

/// Runs `f` with the local-solver nested-parallelism flag set to `enabled`.
pub fn with_local_solver_inner_parallelism<T>(enabled: bool, f: impl FnOnce() -> T) -> T {
    LOCAL_SOLVER_INNER_PARALLELISM.with(|flag| {
        let previous = flag.replace(enabled);
        let result = f();
        flag.set(previous);
        result
    })
}

// ============================================================================
// Buffer types
// ============================================================================

/// Task-local scratch for the parallel-reduction path.
struct AdditiveSweepBuffers {
    global_accum: Vec<f64>,
    r_scratch: Vec<f64>,
    z_scratch: Vec<f64>,
}

impl AdditiveSweepBuffers {
    fn new(n_dofs: usize, max_scratch_size: usize) -> Self {
        Self {
            global_accum: vec![0.0f64; n_dofs],
            r_scratch: vec![0.0f64; max_scratch_size],
            z_scratch: vec![0.0f64; max_scratch_size],
        }
    }
}

/// Thread-local scratch for the atomic scatter path (no per-thread accumulator).
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
    /// Shared atomic accumulator.
    Atomic { accum: Vec<AtomicU64> },
    /// Reusable task-local buffers for parallel reduction.
    Reduction { pool: Vec<AdditiveSweepBuffers> },
}

impl SchwarzBuffers {
    fn strategy(&self) -> ResolvedReductionStrategy {
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
struct WorkerReductionBuffers {
    shared_pool: Mutex<Vec<AdditiveSweepBuffers>>,
    worker_stacks: ThreadLocal<RefCell<Vec<AdditiveSweepBuffers>>>,
    n_dofs: usize,
    max_scratch_size: usize,
}

impl WorkerReductionBuffers {
    fn new(pool: Vec<AdditiveSweepBuffers>, n_dofs: usize, max_scratch_size: usize) -> Self {
        Self {
            shared_pool: Mutex::new(pool),
            worker_stacks: ThreadLocal::with_capacity(rayon::current_num_threads().max(1)),
            n_dofs,
            max_scratch_size,
        }
    }

    fn with_buffer<T>(&self, f: impl FnOnce(&mut AdditiveSweepBuffers) -> T) -> T {
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

    fn into_buffers(mut self) -> Result<Vec<AdditiveSweepBuffers>, ApplyError> {
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

// ============================================================================
// Additive Schwarz
// ============================================================================

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
    /// Build-time metrics used by the adaptive additive scheduler.
    diagnostics: AdditiveSchwarzDiagnostics,
    /// Pool of reusable buffer sets.
    /// Each concurrent `apply()` call pops one; returns it when done.
    buf_pool: Arc<Mutex<Vec<SchwarzBuffers>>>,
}

impl<S: LocalSolver> SchwarzPreconditioner<S> {
    const MAX_POOL_SIZE: usize = 4;

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
        let diagnostics = AdditiveSchwarzDiagnostics::from_entries(&entries, n_dofs);
        Ok(Self {
            n_dofs,
            subdomains: Arc::new(entries),
            max_scratch_size,
            reduction_strategy: strategy,
            diagnostics,
            buf_pool: Arc::new(Mutex::new(Vec::new())),
        })
    }

    /// Access the underlying subdomain entries.
    pub fn subdomains(&self) -> &[SubdomainEntry<S>] {
        &self.subdomains
    }

    /// Configured additive reduction strategy.
    pub fn reduction_strategy(&self) -> ReductionStrategy {
        self.reduction_strategy
    }

    /// Concrete backend selected for the current Rayon thread-pool width.
    pub fn resolved_reduction_strategy(&self) -> ReductionStrategy {
        self.reduction_plan().strategy.as_public()
    }

    /// Build-time metrics used by the additive scheduler.
    pub fn diagnostics(&self) -> AdditiveSchwarzDiagnostics {
        self.diagnostics
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
            diagnostics: self.diagnostics,
            buf_pool: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Fallible operator apply that propagates local-solver failures.
    pub fn try_apply(&self, r: &[f64], z: &mut [f64]) -> Result<(), ApplyError> {
        let n = self.n_dofs;
        let max_ss = self.max_scratch_size;
        let plan = self.reduction_plan();
        let mut bufs = self.take_buffers(plan.strategy, n, max_ss)?;

        let apply_result = match &mut bufs {
            SchwarzBuffers::Atomic { accum } => {
                self.apply_atomic(r, z, accum, max_ss, plan.allow_inner_parallelism)
            }
            SchwarzBuffers::Reduction { pool } => {
                self.apply_parallel_reduction(r, z, pool, n, max_ss, plan.allow_inner_parallelism)
            }
        };

        self.return_buffers(bufs, &apply_result)?;
        apply_result
    }

    fn reduction_plan(&self) -> ReductionPlan {
        let threads = rayon::current_num_threads().max(1);
        self.diagnostics
            .reduction_plan(self.reduction_strategy, threads)
    }

    fn take_buffers(
        &self,
        strategy: ResolvedReductionStrategy,
        n: usize,
        max_ss: usize,
    ) -> Result<SchwarzBuffers, ApplyError> {
        let mut pool = self
            .buf_pool
            .lock()
            .map_err(|_| ApplyError::Synchronization {
                context: "additive.buf_pool.lock.pop",
            })?;
        if let Some(idx) = pool.iter().position(|bufs| bufs.strategy() == strategy) {
            return Ok(pool.swap_remove(idx));
        }
        Ok(match strategy {
            ResolvedReductionStrategy::AtomicScatter => SchwarzBuffers::Atomic {
                accum: (0..n).map(|_| AtomicU64::new(0)).collect(),
            },
            ResolvedReductionStrategy::ParallelReduction => SchwarzBuffers::Reduction {
                pool: vec![AdditiveSweepBuffers::new(n, max_ss)],
            },
        })
    }

    fn return_buffers(
        &self,
        bufs: SchwarzBuffers,
        apply_result: &Result<(), ApplyError>,
    ) -> Result<(), ApplyError> {
        if let Ok(mut pool) = self.buf_pool.lock() {
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

    fn apply_atomic(
        &self,
        r: &[f64],
        z: &mut [f64],
        accum: &[AtomicU64],
        max_ss: usize,
        allow_inner: bool,
    ) -> Result<(), ApplyError> {
        self.subdomains.par_iter().enumerate().try_for_each_init(
            || AtomicScratch::new(max_ss),
            |scratch, (subdomain, entry)| {
                with_local_solver_inner_parallelism(allow_inner, || {
                    entry.apply_weighted_into_atomic(
                        r,
                        accum,
                        &mut scratch.r_scratch,
                        &mut scratch.z_scratch,
                    )
                })
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
                    *zi = f64::from_bits(ai.load(Ordering::Relaxed));
                    ai.store(0, Ordering::Relaxed);
                }
            });
        Ok(())
    }

    fn apply_parallel_reduction(
        &self,
        r: &[f64],
        z: &mut [f64],
        pool: &mut Vec<AdditiveSweepBuffers>,
        n: usize,
        max_ss: usize,
        allow_inner: bool,
    ) -> Result<(), ApplyError> {
        let worker_buffers = WorkerReductionBuffers::new(std::mem::take(pool), n, max_ss);
        let apply_result =
            self.subdomains
                .par_iter()
                .enumerate()
                .try_for_each(|(subdomain, entry)| {
                    worker_buffers.with_buffer(|buffers| {
                        let AdditiveSweepBuffers {
                            ref mut global_accum,
                            ref mut r_scratch,
                            ref mut z_scratch,
                        } = buffers;
                        with_local_solver_inner_parallelism(allow_inner, || {
                            entry.apply_weighted_into_with_scratch(
                                r,
                                global_accum,
                                r_scratch,
                                z_scratch,
                            )
                        })
                        .map_err(|source| ApplyError::LocalSolveFailed { subdomain, source })
                    })
                });

        let mut buffers = worker_buffers.into_buffers()?;
        if apply_result.is_ok() {
            reduce_additive_buffers_into(z, &buffers);
        }
        clear_additive_buffers(&mut buffers);
        *pool = buffers;

        apply_result
    }
}

fn reduce_additive_buffers_into(z: &mut [f64], buffers: &[AdditiveSweepBuffers]) {
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

fn clear_additive_buffers(buffers: &mut [AdditiveSweepBuffers]) {
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

impl<S: LocalSolver> Clone for SchwarzPreconditioner<S> {
    /// Clone shares both the subdomain data and the buffer pool via `Arc`.
    /// This is O(1) and the clone is fully interchangeable with the original.
    fn clone(&self) -> Self {
        Self {
            subdomains: Arc::clone(&self.subdomains),
            n_dofs: self.n_dofs,
            max_scratch_size: self.max_scratch_size,
            reduction_strategy: self.reduction_strategy,
            diagnostics: self.diagnostics,
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
    /// Partition-of-unity-weighted local solution (length = max local DOFs).
    weighted_local_sol: Vec<f64>,
}

impl SweepBuffers {
    fn new(n_dofs: usize, max_scratch_size: usize, max_local_dofs: usize) -> Self {
        Self {
            r_work: vec![0.0; n_dofs],
            r_scratch: vec![0.0; max_scratch_size],
            z_scratch: vec![0.0; max_scratch_size],
            weighted_local_sol: vec![0.0; max_local_dofs],
        }
    }
}

/// Process a single subdomain: restrict, local solve, prolongate, update residual.
fn apply_subdomain<S: LocalSolver, U: ResidualUpdater>(
    entry: &SubdomainEntry<S>,
    bufs: &mut SweepBuffers,
    z: &mut [f64],
    updater: &mut U,
) -> Result<(), LocalSolveError> {
    if entry.core.global_indices.is_empty() {
        return Ok(());
    }

    let n_local = entry.core.global_indices.len();

    entry
        .core
        .restrict_weighted(&bufs.r_work, &mut bufs.r_scratch);
    entry
        .solver
        .solve_local(&mut bufs.r_scratch, &mut bufs.z_scratch)?;
    entry.core.prolongate_weighted_add(&bufs.z_scratch, z);

    match &entry.core.partition_weights {
        PartitionWeights::Uniform(_) => {
            updater.update(
                &entry.core.global_indices,
                &bufs.z_scratch[..n_local],
                &mut bufs.r_work,
            );
        }
        PartitionWeights::NonUniform(w) => {
            for (k, wk) in w.iter().enumerate().take(n_local) {
                bufs.weighted_local_sol[k] = wk * bufs.z_scratch[k];
            }
            updater.update(
                &entry.core.global_indices,
                &bufs.weighted_local_sol[..n_local],
                &mut bufs.r_work,
            );
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
        z.fill(0.0);

        bufs.r_work.copy_from_slice(r);
        updater.reset(r);
        for (subdomain, entry) in self.subdomains.iter().enumerate() {
            apply_subdomain(entry, &mut bufs, z, &mut *updater)
                .map_err(|source| ApplyError::LocalSolveFailed { subdomain, source })?;
        }

        if self.symmetric {
            updater.reset(&bufs.r_work);
            for (rev_idx, entry) in self.subdomains.iter().rev().enumerate() {
                let subdomain = self.subdomains.len().saturating_sub(1) - rev_idx;
                apply_subdomain(entry, &mut bufs, z, &mut *updater)
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

// ============================================================================
// Serde support
// ============================================================================

#[cfg(feature = "serde")]
mod serde_impl {
    use super::*;
    use serde::ser::SerializeStruct;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    // --- SchwarzPreconditioner ---

    impl<S: LocalSolver + Serialize> Serialize for SchwarzPreconditioner<S> {
        fn serialize<Ser: Serializer>(&self, serializer: Ser) -> Result<Ser::Ok, Ser::Error> {
            let mut state = serializer.serialize_struct("SchwarzPreconditioner", 3)?;
            state.serialize_field("subdomains", &*self.subdomains)?;
            state.serialize_field("n_dofs", &self.n_dofs)?;
            state.serialize_field("max_scratch_size", &self.max_scratch_size)?;
            state.end()
        }
    }

    impl<'de, S: LocalSolver + serde::de::DeserializeOwned> Deserialize<'de>
        for SchwarzPreconditioner<S>
    {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            #[derive(Deserialize)]
            #[serde(bound(deserialize = "S: serde::de::DeserializeOwned"))]
            struct Helper<S: LocalSolver> {
                subdomains: Vec<SubdomainEntry<S>>,
                n_dofs: usize,
                max_scratch_size: usize,
            }
            let h: Helper<S> = Helper::deserialize(deserializer)?;
            let diagnostics = AdditiveSchwarzDiagnostics::from_entries(&h.subdomains, h.n_dofs);
            Ok(SchwarzPreconditioner {
                subdomains: Arc::new(h.subdomains),
                n_dofs: h.n_dofs,
                max_scratch_size: h.max_scratch_size,
                reduction_strategy: ReductionStrategy::default(),
                diagnostics,
                buf_pool: Arc::new(Mutex::new(Vec::new())),
            })
        }
    }

    // --- MultiplicativeSchwarzPreconditioner ---

    impl<S: LocalSolver + Serialize, U: ResidualUpdater + Serialize> Serialize
        for MultiplicativeSchwarzPreconditioner<S, U>
    {
        fn serialize<Ser: Serializer>(&self, serializer: Ser) -> Result<Ser::Ok, Ser::Error> {
            let mut state =
                serializer.serialize_struct("MultiplicativeSchwarzPreconditioner", 4)?;
            state.serialize_field("subdomains", &self.subdomains)?;
            let updater = self.updater.lock().map_err(serde::ser::Error::custom)?;
            state.serialize_field("updater", &*updater)?;
            state.serialize_field("n_dofs", &self.n_dofs)?;
            state.serialize_field("symmetric", &self.symmetric)?;
            state.end()
        }
    }

    impl<
            'de,
            S: LocalSolver + serde::de::DeserializeOwned,
            U: ResidualUpdater + serde::de::DeserializeOwned,
        > Deserialize<'de> for MultiplicativeSchwarzPreconditioner<S, U>
    {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            #[derive(Deserialize)]
            #[serde(bound(
                deserialize = "S: serde::de::DeserializeOwned, U: serde::de::DeserializeOwned"
            ))]
            struct Helper<S: LocalSolver, U: ResidualUpdater> {
                subdomains: Vec<SubdomainEntry<S>>,
                updater: U,
                n_dofs: usize,
                symmetric: bool,
            }
            let h = Helper::deserialize(deserializer)?;
            let (max_scratch_size, max_local_dofs) = compute_sizes(&h.subdomains);
            Ok(MultiplicativeSchwarzPreconditioner {
                scratch: Mutex::new(SweepBuffers::new(
                    h.n_dofs,
                    max_scratch_size,
                    max_local_dofs,
                )),
                subdomains: h.subdomains,
                updater: Mutex::new(h.updater),
                n_dofs: h.n_dofs,
                symmetric: h.symmetric,
            })
        }
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
