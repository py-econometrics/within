//! Public [`SchwarzPreconditioner`] type — the one-level additive Schwarz.
//!
//! Implements [`Operator`](crate::Operator) so it can be passed directly
//! to CG or GMRES. Construction validates subdomain indices, pre-computes
//! scheduling diagnostics, and allocates the executor. `apply` is lock-free
//! in steady state (buffers are borrowed from a pool).

use crate::error::{validate_entries, BuildError, SolveError};
use crate::local_solve::{LocalSolver, SubdomainEntry};
use crate::Operator;

use super::executor::AdditiveExecutor;
use super::planning::{
    AdditiveScheduler, AdditiveSchwarzDiagnostics, ReductionPlan, ReductionStrategy,
};

// ---------------------------------------------------------------------------
// Serde
// ---------------------------------------------------------------------------

/// Only the subdomain entries, DOF count, and scratch size are persisted.
/// The reduction strategy resets to `Auto` on deserialize; buffers are
/// re-allocated fresh.
#[cfg(feature = "serde")]
impl<S> serde::Serialize for SchwarzPreconditioner<S>
where
    S: LocalSolver + serde::Serialize,
{
    fn serialize<Ser: serde::Serializer>(&self, serializer: Ser) -> Result<Ser::Ok, Ser::Error> {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("SchwarzPreconditioner", 3)?;
        state.serialize_field("subdomains", &*self.executor.subdomains)?;
        state.serialize_field("n_dofs", &self.executor.n_dofs)?;
        state.serialize_field("max_scratch_size", &self.executor.max_scratch_size)?;
        state.end()
    }
}

#[cfg(feature = "serde")]
impl<'de, S> serde::Deserialize<'de> for SchwarzPreconditioner<S>
where
    S: LocalSolver + serde::de::DeserializeOwned,
{
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::Deserialize;

        #[derive(Deserialize)]
        #[serde(bound(deserialize = "S: serde::de::DeserializeOwned"))]
        struct Helper<S: LocalSolver> {
            subdomains: Vec<SubdomainEntry<S>>,
            n_dofs: usize,
            max_scratch_size: usize,
        }

        let h: Helper<S> = Helper::deserialize(deserializer)?;
        Ok(SchwarzPreconditioner {
            reduction_strategy: ReductionStrategy::default(),
            scheduler: AdditiveScheduler::from_entries(&h.subdomains, h.n_dofs),
            executor: AdditiveExecutor::new(h.subdomains, h.n_dofs, h.max_scratch_size),
        })
    }
}

/// One-level additive Schwarz preconditioner, generic over the local solver.
///
/// Subdomains (factored matrices) are stored behind `Arc` so that cloning
/// shares the heavy subdomain data. A pool of per-thread buffer sets enables
/// safe concurrent `apply()` calls on the same instance — each caller grabs
/// an independent buffer set from the pool for the duration of the call.
pub struct SchwarzPreconditioner<S: LocalSolver> {
    /// Strategy for combining per-subdomain results.
    pub(super) reduction_strategy: ReductionStrategy,
    /// Build-time scheduler state for additive apply.
    pub(super) scheduler: AdditiveScheduler,
    /// Static execution state for additive apply.
    pub(super) executor: AdditiveExecutor<S>,
}

impl<S: LocalSolver> SchwarzPreconditioner<S> {
    /// Construct from pre-built subdomain entries using the default strategy.
    pub fn new(entries: Vec<SubdomainEntry<S>>, n_dofs: usize) -> Result<Self, BuildError> {
        Self::with_strategy(entries, n_dofs, ReductionStrategy::default())
    }

    /// Construct from pre-built subdomain entries with an explicit reduction strategy.
    pub fn with_strategy(
        entries: Vec<SubdomainEntry<S>>,
        n_dofs: usize,
        strategy: ReductionStrategy,
    ) -> Result<Self, BuildError> {
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
            executor: AdditiveExecutor::new(entries, n_dofs, max_scratch_size),
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

    fn reduction_plan(&self) -> ReductionPlan {
        let threads = rayon::current_num_threads().max(1);
        self.scheduler
            .reduction_plan(self.reduction_strategy, threads)
    }
}

impl<S: LocalSolver> Clone for SchwarzPreconditioner<S> {
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

impl<S: LocalSolver> Operator for SchwarzPreconditioner<S> {
    fn nrows(&self) -> usize {
        self.executor.n_dofs
    }

    fn ncols(&self) -> usize {
        self.executor.n_dofs
    }

    fn apply(&self, r: &[f64], z: &mut [f64]) -> Result<(), SolveError> {
        let plan = self.reduction_plan();
        self.executor.try_apply(plan, r, z)
    }

    fn apply_adjoint(&self, r: &[f64], z: &mut [f64]) -> Result<(), SolveError> {
        self.apply(r, z)
    }
}
