//! Public [`SchwarzPreconditioner`] type — the one-level additive Schwarz.
//!
//! Implements [`Operator`](crate::Operator) so it can be passed directly
//! to an iterative solver as a preconditioner. The constructor walks the
//! subdomain entries once to derive `n_dofs`, `max_scratch_size`, and the
//! scheduling metrics. `apply` is lock-free in steady state (buffers are
//! borrowed from a pool).

use std::sync::Arc;

use crate::error::SolveError;
use crate::local_solve::{LocalSolver, SubdomainEntry};
use crate::Operator;

use super::executor::AdditiveExecutor;
use super::planning::{AdditiveScheduler, ReductionPlan, ReductionStrategy};

// ---------------------------------------------------------------------------
// Serde
// ---------------------------------------------------------------------------

/// Persists only the subdomain entries; `n_dofs` and `max_scratch_size` are
/// re-derived from the entries at deserialize time. The reduction strategy
/// resets to `Auto`; buffers are re-allocated fresh.
#[cfg(feature = "serde")]
impl<S> serde::Serialize for SchwarzPreconditioner<S>
where
    S: LocalSolver + serde::Serialize,
{
    fn serialize<Ser: serde::Serializer>(&self, serializer: Ser) -> Result<Ser::Ok, Ser::Error> {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("SchwarzPreconditioner", 1)?;
        state.serialize_field("subdomains", self.executor.subdomains())?;
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
        }

        let h: Helper<S> = Helper::deserialize(deserializer)?;
        Ok(SchwarzPreconditioner::new(
            h.subdomains,
            ReductionStrategy::default(),
        ))
    }
}

/// One-level additive Schwarz preconditioner, generic over the local solver.
///
/// Subdomains (factored matrices) are stored behind `Arc` so that cloning
/// shares the heavy subdomain data. A pool of per-thread buffer sets enables
/// safe concurrent `apply()` calls on the same instance — each caller grabs
/// an independent buffer set from the pool for the duration of the call.
pub struct SchwarzPreconditioner<S: LocalSolver> {
    reduction_strategy: ReductionStrategy,
    scheduler: AdditiveScheduler,
    executor: AdditiveExecutor<S>,
}

impl<S: LocalSolver> SchwarzPreconditioner<S> {
    /// Construct from pre-built subdomain entries with a reduction strategy.
    ///
    /// `n_dofs` is derived from the maximum global index across entries
    /// (or 0 if `entries` is empty). An empty entry list yields a degenerate
    /// preconditioner; misuse is caught at apply time by the dimension check.
    pub fn new(entries: Vec<SubdomainEntry<S>>, strategy: ReductionStrategy) -> Self {
        let mut n_dofs: usize = 0;
        let mut max_scratch_size: usize = 0;
        let mut total_inner_parallel_work: usize = 0;
        let mut max_inner_parallel_work: usize = 0;
        let mut total_scatter_dofs: usize = 0;

        for entry in &entries {
            let work = entry.solver().inner_parallelism_work_estimate();
            total_inner_parallel_work = total_inner_parallel_work.saturating_add(work);
            max_inner_parallel_work = max_inner_parallel_work.max(work);
            max_scratch_size = max_scratch_size.max(entry.scratch_size());

            let indices = entry.global_indices();
            total_scatter_dofs = total_scatter_dofs.saturating_add(indices.len());
            if let Some(&max_idx) = indices.iter().max() {
                let candidate = max_idx as usize + 1;
                if candidate > n_dofs {
                    n_dofs = candidate;
                }
            }
        }

        let executor = AdditiveExecutor::new(Arc::new(entries), n_dofs, max_scratch_size);
        let scheduler = AdditiveScheduler {
            total_inner_parallel_work,
            max_inner_parallel_work,
            total_scatter_dofs,
        };
        Self {
            reduction_strategy: strategy,
            scheduler,
            executor,
        }
    }

    /// Access the underlying subdomain entries.
    pub fn subdomains(&self) -> &[SubdomainEntry<S>] {
        self.executor.subdomains()
    }

    /// Concrete backend selected for the current Rayon thread-pool width.
    pub fn reduction_strategy(&self) -> ReductionStrategy {
        self.reduction_plan().strategy.as_public()
    }

    /// Operator apply that propagates local-solver failures.
    pub fn apply(&self, r: &[f64], z: &mut [f64]) -> Result<(), SolveError> {
        let n_dofs = self.executor.n_dofs();
        if r.len() != n_dofs {
            return Err(SolveError::InvalidInput {
                context: "SchwarzPreconditioner::apply",
                message: format!("r.len() ({}) != n_dofs ({})", r.len(), n_dofs),
            });
        }
        if z.len() != n_dofs {
            return Err(SolveError::InvalidInput {
                context: "SchwarzPreconditioner::apply",
                message: format!("z.len() ({}) != n_dofs ({})", z.len(), n_dofs),
            });
        }
        let plan = self.reduction_plan();
        self.executor.apply(plan, r, z)
    }

    fn reduction_plan(&self) -> ReductionPlan {
        let threads = rayon::current_num_threads().max(1);
        self.scheduler.reduction_plan(
            self.reduction_strategy,
            threads,
            self.executor.n_subdomains(),
            self.executor.n_dofs(),
        )
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
        self.executor.n_dofs()
    }

    fn ncols(&self) -> usize {
        self.executor.n_dofs()
    }

    fn apply(&self, r: &[f64], z: &mut [f64]) -> Result<(), SolveError> {
        SchwarzPreconditioner::apply(self, r, z)
    }

    fn apply_adjoint(&self, r: &[f64], z: &mut [f64]) -> Result<(), SolveError> {
        SchwarzPreconditioner::apply(self, r, z)
    }
}
