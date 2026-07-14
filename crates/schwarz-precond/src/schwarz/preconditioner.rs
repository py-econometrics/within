//! Public [`SchwarzPreconditioner`] type — the one-level additive Schwarz.
//!
//! Implements [`Operator`](crate::Operator) so it can be passed directly
//! to an iterative solver as a preconditioner. The constructor walks the
//! subdomain entries once to derive `max_scratch_size` and the scheduling
//! metrics (and, unless given explicitly, `n_dofs`). `apply` is lock-free in steady state (buffers are
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

/// Persists the subdomain entries and the global DOF count `n_dofs` — the
/// latter is not recoverable from the entries alone when some operator columns
/// are covered by no subdomain. `max_scratch_size` and the scheduling metrics
/// are re-derived on deserialize; the reduction strategy resets to `Auto`;
/// buffers are re-allocated fresh.
#[cfg(feature = "serde")]
impl<S> serde::Serialize for SchwarzPreconditioner<S>
where
    S: LocalSolver + serde::Serialize,
{
    fn serialize<Ser: serde::Serializer>(&self, serializer: Ser) -> Result<Ser::Ok, Ser::Error> {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("SchwarzPreconditioner", 2)?;
        state.serialize_field("subdomains", self.executor.subdomains())?;
        // Persist the global DOF count: it is not recoverable from the
        // subdomains alone when the operator has structural-null tail DOFs
        // that no subdomain covers.
        state.serialize_field("n_dofs", &self.executor.n_dofs())?;
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
        }

        let h: Helper<S> = Helper::deserialize(deserializer)?;
        Ok(SchwarzPreconditioner::with_n_dofs(
            h.subdomains,
            h.n_dofs,
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
    /// Construct from pre-built subdomain entries, inferring the global DOF
    /// count from the maximum global index across entries (or 0 if empty).
    ///
    /// Suitable only when every DOF is covered by a subdomain. An operator
    /// with columns no subdomain touches (e.g. an unidentified direction kept
    /// for shape) must state its true width via [`Self::with_n_dofs`], or the
    /// inferred count falls short and the mismatch surfaces at apply time.
    pub fn new(entries: Vec<SubdomainEntry<S>>, strategy: ReductionStrategy) -> Self {
        Self::build(entries, None, strategy)
    }

    /// Construct with an explicit global DOF count: the operator's column
    /// count, which may exceed the span of the subdomains' global indices. A
    /// column no subdomain covers stays in the null space, so its apply output
    /// is `0`. A count below the covered span is a caller bug (a subdomain
    /// would scatter out of bounds), caught in debug builds.
    pub fn with_n_dofs(
        entries: Vec<SubdomainEntry<S>>,
        n_dofs: usize,
        strategy: ReductionStrategy,
    ) -> Self {
        Self::build(entries, Some(n_dofs), strategy)
    }

    /// Shared constructor: one pass over `entries` derives `max_scratch_size`,
    /// the scheduling metrics, and the covered index span. `n_dofs` is taken
    /// as given, or inferred from that span when `None`.
    fn build(
        entries: Vec<SubdomainEntry<S>>,
        n_dofs: Option<usize>,
        strategy: ReductionStrategy,
    ) -> Self {
        let mut max_scratch_size: usize = 0;
        let mut total_inner_parallel_work: usize = 0;
        let mut max_inner_parallel_work: usize = 0;
        let mut total_scatter_dofs: usize = 0;
        let mut covered_span: usize = 0;

        for entry in &entries {
            let work = entry.solver().inner_parallelism_work_estimate();
            total_inner_parallel_work = total_inner_parallel_work.saturating_add(work);
            max_inner_parallel_work = max_inner_parallel_work.max(work);
            max_scratch_size = max_scratch_size.max(entry.scratch_size());

            let indices = entry.global_indices();
            total_scatter_dofs = total_scatter_dofs.saturating_add(indices.len());
            if let Some(&max_idx) = indices.iter().max() {
                covered_span = covered_span.max(max_idx as usize + 1);
            }
        }
        let n_dofs = n_dofs.unwrap_or(covered_span);
        debug_assert!(
            n_dofs >= covered_span,
            "n_dofs ({n_dofs}) is below the covered index span ({covered_span})"
        );

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
