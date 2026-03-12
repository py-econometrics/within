use crate::error::{validate_entries, ApplyError, PreconditionerBuildError};
use crate::local_solve::{
    DefaultLocalSolveInvoker, LocalSolveInvoker, LocalSolver, SubdomainEntry,
};
use crate::Operator;

use super::executor::AdditiveExecutor;
use super::planning::{
    AdditiveScheduler, AdditiveSchwarzDiagnostics, ReductionPlan, ReductionStrategy,
};

/// One-level additive Schwarz preconditioner, generic over the local solver.
///
/// Subdomains (factored matrices) are stored behind `Arc` so that cloning
/// shares the heavy subdomain data. A pool of per-thread buffer sets enables
/// safe concurrent `apply()` calls on the same instance — each caller grabs
/// an independent buffer set from the pool for the duration of the call.
pub struct SchwarzPreconditioner<S: LocalSolver, I: LocalSolveInvoker<S> = DefaultLocalSolveInvoker>
{
    /// Strategy for combining per-subdomain results.
    pub(super) reduction_strategy: ReductionStrategy,
    /// Build-time scheduler state for additive apply.
    pub(super) scheduler: AdditiveScheduler,
    /// Static execution state for additive apply.
    pub(super) executor: AdditiveExecutor<S, I>,
}

impl<S: LocalSolver> SchwarzPreconditioner<S, DefaultLocalSolveInvoker> {
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
        Self::with_strategy_and_invoker(entries, n_dofs, strategy, DefaultLocalSolveInvoker)
    }
}

impl<S: LocalSolver, I: LocalSolveInvoker<S>> SchwarzPreconditioner<S, I> {
    /// Construct from pre-built subdomain entries with an explicit reduction strategy
    /// and a caller-provided local-solve invoker.
    pub fn with_strategy_and_invoker(
        entries: Vec<SubdomainEntry<S>>,
        n_dofs: usize,
        strategy: ReductionStrategy,
        invoker: I,
    ) -> Result<Self, PreconditionerBuildError> {
        validate_entries(&entries, n_dofs)?;
        let max_scratch_size = entries
            .iter()
            .map(SubdomainEntry::scratch_size)
            .max()
            .unwrap_or(0);
        let scheduler = AdditiveScheduler::from_entries(&entries, n_dofs);
        Ok(Self {
            reduction_strategy: strategy,
            scheduler,
            executor: AdditiveExecutor::new(entries, n_dofs, max_scratch_size, invoker),
        })
    }

    /// Access the underlying subdomain entries.
    pub fn subdomains(&self) -> &[SubdomainEntry<S>] {
        self.executor.subdomains()
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
        self.scheduler.diagnostics()
    }

    /// Return a copy that uses a different reduction strategy.
    ///
    /// Shares the subdomain data via `Arc` (O(1)), but creates a fresh
    /// buffer pool since buffers are strategy-specific.
    pub fn with_reduction_strategy(&self, strategy: ReductionStrategy) -> Self {
        Self {
            reduction_strategy: strategy,
            scheduler: self.scheduler,
            executor: self.executor.with_fresh_buffers(),
        }
    }

    /// Fallible operator apply that propagates local-solver failures.
    pub fn try_apply(&self, r: &[f64], z: &mut [f64]) -> Result<(), ApplyError> {
        let plan = self.reduction_plan();
        self.executor.try_apply(plan, r, z)
    }

    fn reduction_plan(&self) -> ReductionPlan {
        let threads = rayon::current_num_threads().max(1);
        self.scheduler
            .reduction_plan(self.reduction_strategy, threads)
    }
}

impl<S: LocalSolver, I: LocalSolveInvoker<S>> Clone for SchwarzPreconditioner<S, I> {
    /// Clone shares both the subdomain data and the buffer pool via `Arc`.
    /// This is O(1) and the clone is fully interchangeable with the original.
    fn clone(&self) -> Self {
        Self {
            reduction_strategy: self.reduction_strategy,
            scheduler: self.scheduler,
            executor: self.executor.clone(),
        }
    }
}

impl<S: LocalSolver, I: LocalSolveInvoker<S>> Operator for SchwarzPreconditioner<S, I> {
    fn nrows(&self) -> usize {
        self.executor.n_dofs
    }

    fn ncols(&self) -> usize {
        self.executor.n_dofs
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
        SchwarzPreconditioner::try_apply(self, r, z)
    }

    fn try_apply_adjoint(&self, r: &[f64], z: &mut [f64]) -> Result<(), ApplyError> {
        SchwarzPreconditioner::try_apply(self, r, z)
    }
}
