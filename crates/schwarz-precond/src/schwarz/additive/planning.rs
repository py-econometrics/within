//! Reduction strategy selection and build-time diagnostics.
//!
//! [`ReductionStrategy`] is the user-facing enum (`Auto`, `AtomicScatter`,
//! `ParallelReduction`). [`AdditiveScheduler`] resolves `Auto` at apply-time
//! using build-time metrics ([`AdditiveSchwarzDiagnostics`]) and the current
//! Rayon thread-pool width.
//!
//! The heuristic balances two costs:
//! - **Atomic scatter**: contention grows with overlap (DOFs shared across
//!   many subdomains)
//! - **Parallel reduction**: memory and final-reduction cost grow with
//!   `P × n_dofs` where P is the number of active workers

use crate::local_solve::{LocalSolver, SubdomainEntry};

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
pub(super) enum ResolvedReductionStrategy {
    AtomicScatter,
    ParallelReduction,
}

impl ResolvedReductionStrategy {
    pub(super) fn as_public(self) -> ReductionStrategy {
        match self {
            Self::AtomicScatter => ReductionStrategy::AtomicScatter,
            Self::ParallelReduction => ReductionStrategy::ParallelReduction,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct ReductionPlan {
    pub(super) strategy: ResolvedReductionStrategy,
    pub(super) allow_inner_parallelism: bool,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct AdditiveScheduler {
    diagnostics: AdditiveSchwarzDiagnostics,
}

impl AdditiveScheduler {
    pub(super) fn from_entries<S: LocalSolver>(
        entries: &[SubdomainEntry<S>],
        n_dofs: usize,
    ) -> Self {
        Self {
            diagnostics: AdditiveSchwarzDiagnostics::from_entries(entries, n_dofs),
        }
    }

    pub(super) fn diagnostics(self) -> AdditiveSchwarzDiagnostics {
        self.diagnostics
    }

    pub(super) fn reduction_plan(
        self,
        configured: ReductionStrategy,
        threads: usize,
    ) -> ReductionPlan {
        let allow_inner_parallelism = self.allow_inner_parallelism(threads);
        let strategy = self.resolve_strategy(configured, threads, allow_inner_parallelism);
        ReductionPlan {
            strategy,
            allow_inner_parallelism,
        }
    }

    fn allow_inner_parallelism(self, threads: usize) -> bool {
        if self.diagnostics.max_inner_parallel_work < Self::MIN_INNER_PARALLEL_WORK {
            return false;
        }

        self.diagnostics.outer_parallel_capacity() < (threads as f64 * Self::OUTER_CAPACITY_TARGET)
    }

    fn resolve_strategy(
        self,
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
        self,
        threads: usize,
        allow_inner_parallelism: bool,
    ) -> ResolvedReductionStrategy {
        let overlap = self.diagnostics.scatter_overlap();
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

    fn reduction_sweep_to_scatter(self, threads: usize) -> f64 {
        let active_buffers = threads.min(self.diagnostics.n_subdomains).max(1);
        let reduction_sweep = active_buffers.saturating_mul(self.diagnostics.n_dofs);
        let scatter_work = self.diagnostics.total_scatter_dofs.max(1);
        reduction_sweep as f64 / scatter_work as f64
    }
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
    pub(super) fn from_entries<S: LocalSolver>(
        entries: &[SubdomainEntry<S>],
        n_dofs: usize,
    ) -> Self {
        let (total_inner_parallel_work, max_inner_parallel_work, total_scatter_dofs) =
            entries.iter().fold(
                (0usize, 0usize, 0usize),
                |(total_work, max_work, total_scatter), entry| {
                    let work = entry.solver().inner_parallelism_work_estimate();
                    (
                        total_work.saturating_add(work),
                        max_work.max(work),
                        total_scatter.saturating_add(entry.global_indices().len()),
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
}

impl AdditiveScheduler {
    const MIN_INNER_PARALLEL_WORK: usize = 200_000;
    const OUTER_CAPACITY_TARGET: f64 = 0.75;
    const AUTO_REDUCTION_SWEEP_FACTOR: f64 = 1.1;
    const AUTO_INNER_REDUCTION_SWEEP_FACTOR: f64 = 6.0;
    const AUTO_OVERLAP_FOR_REDUCTION: f64 = 4.0;
}
