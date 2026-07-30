//! Public [`SchwarzPreconditioner`] type — the one-level additive Schwarz.
//!
//! Implements [`Operator`](crate::Operator) so it can be passed directly
//! to an iterative solver as a preconditioner. The constructor walks the
//! subdomain entries once to derive `max_scratch_size` and the scheduling
//! metrics (and, unless given explicitly, `n_dofs`). `apply` reuses pooled
//! scratch buffers — no large per-apply allocation, but a short buffer-pool
//! lock is taken on both borrow and return.

use std::sync::Arc;

use crate::error::SolveError;
use crate::local_solve::{LocalSolver, SubdomainEntry};
use crate::Operator;

use super::executor::AdditiveExecutor;
use super::planning::{AdditiveScheduler, ReductionPlan, ReductionStrategy};

/// Persists `n_dofs`, which entries alone cannot recover when some columns are uncovered.
#[cfg(feature = "serde")]
impl<S> serde::Serialize for SchwarzPreconditioner<S>
where
    S: LocalSolver + serde::Serialize,
{
    fn serialize<Ser: serde::Serializer>(&self, serializer: Ser) -> Result<Ser::Ok, Ser::Error> {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("SchwarzPreconditioner", 2)?;
        state.serialize_field("subdomains", self.executor.subdomains())?;
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
        // An `n_dofs` below the covered span would let a subdomain scatter out of bounds.
        let covered_span = h
            .subdomains
            .iter()
            .flat_map(|e| e.global_indices().iter().copied())
            .max()
            .map_or(0, |m| m as usize + 1);
        if h.n_dofs < covered_span {
            return Err(serde::de::Error::custom(format!(
                "n_dofs ({}) is below the covered subdomain index span ({covered_span})",
                h.n_dofs
            )));
        }
        Ok(SchwarzPreconditioner::with_n_dofs(
            h.subdomains,
            h.n_dofs,
            ReductionStrategy::default(),
        ))
    }
}

/// One-level additive Schwarz preconditioner; pooled per-thread buffers make `apply()` safe.
pub struct SchwarzPreconditioner<S: LocalSolver> {
    reduction_strategy: ReductionStrategy,
    scheduler: AdditiveScheduler,
    executor: AdditiveExecutor<S>,
}

impl<S: LocalSolver> SchwarzPreconditioner<S> {
    /// Infers `n_dofs` from the maximum global index; only valid when every DOF is covered.
    pub fn new(entries: Vec<SubdomainEntry<S>>, strategy: ReductionStrategy) -> Self {
        Self::build(entries, None, strategy)
    }

    /// Explicit global DOF count; an uncovered column stays in the null space and applies to `0`.
    pub fn with_n_dofs(
        entries: Vec<SubdomainEntry<S>>,
        n_dofs: usize,
        strategy: ReductionStrategy,
    ) -> Self {
        Self::build(entries, Some(n_dofs), strategy)
    }

    /// One pass over `entries` derives `max_scratch_size`, the metrics, and the covered span.
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
    /// Clone shares subdomain data and the buffer pool via `Arc`: O(1) and fully interchangeable.
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
