//! Reduction strategy selection.
//!
//! [`ReductionStrategy`] is the user-facing enum (`Auto`, `AtomicScatter`,
//! `ParallelReduction`). [`AdditiveScheduler`] resolves `Auto` at apply-time
//! using build-time scheduling metrics and the current Rayon thread-pool
//! width.
//!
//! The heuristic balances two costs:
//! - **Atomic scatter**: contention grows with overlap (DOFs shared across
//!   many subdomains)
//! - **Parallel reduction**: memory and final-reduction cost grow with
//!   `P × n_dofs` where P is the number of active workers

/// Strategy for combining per-subdomain results in additive Schwarz apply.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ReductionStrategy {
    /// Choose a backend from build-time metrics and the current Rayon width.
    #[default]
    Auto,
    /// Atomic scatter into a shared accumulator; O(n_dofs) shared plus O(P · max_scratch) local.
    AtomicScatter,
    /// Private per-task buffers combined by a parallel chunk reduction; O(P · n_dofs).
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

/// Build-time scheduling metrics and `Auto` resolution.
#[derive(Debug, Clone, Copy)]
pub(super) struct AdditiveScheduler {
    pub(super) total_inner_parallel_work: usize,
    pub(super) max_inner_parallel_work: usize,
    pub(super) total_scatter_dofs: usize,
}

impl AdditiveScheduler {
    const MIN_INNER_PARALLEL_WORK: usize = 200_000;
    const OUTER_CAPACITY_TARGET: f64 = 0.75;
    const AUTO_REDUCTION_SWEEP_FACTOR: f64 = 1.1;
    const AUTO_INNER_REDUCTION_SWEEP_FACTOR: f64 = 6.0;
    const AUTO_OVERLAP_FOR_REDUCTION: f64 = 4.0;

    pub(super) fn reduction_plan(
        self,
        configured: ReductionStrategy,
        threads: usize,
        n_subdomains: usize,
        n_dofs: usize,
    ) -> ReductionPlan {
        let allow_inner_parallelism = self.allow_inner_parallelism(threads);
        let strategy = self.resolve_strategy(
            configured,
            threads,
            n_subdomains,
            n_dofs,
            allow_inner_parallelism,
        );
        ReductionPlan {
            strategy,
            allow_inner_parallelism,
        }
    }

    fn outer_parallel_capacity(self) -> f64 {
        if self.max_inner_parallel_work == 0 {
            return 0.0;
        }
        self.total_inner_parallel_work as f64 / self.max_inner_parallel_work as f64
    }

    fn scatter_overlap(self, n_dofs: usize) -> f64 {
        self.total_scatter_dofs as f64 / n_dofs.max(1) as f64
    }

    fn allow_inner_parallelism(self, threads: usize) -> bool {
        if self.max_inner_parallel_work < Self::MIN_INNER_PARALLEL_WORK {
            return false;
        }

        self.outer_parallel_capacity() < (threads as f64 * Self::OUTER_CAPACITY_TARGET)
    }

    fn resolve_strategy(
        self,
        configured: ReductionStrategy,
        threads: usize,
        n_subdomains: usize,
        n_dofs: usize,
        allow_inner_parallelism: bool,
    ) -> ResolvedReductionStrategy {
        match configured {
            ReductionStrategy::AtomicScatter => ResolvedReductionStrategy::AtomicScatter,
            ReductionStrategy::ParallelReduction => ResolvedReductionStrategy::ParallelReduction,
            ReductionStrategy::Auto => {
                self.pick_auto_strategy(threads, n_subdomains, n_dofs, allow_inner_parallelism)
            }
        }
    }

    fn pick_auto_strategy(
        self,
        threads: usize,
        n_subdomains: usize,
        n_dofs: usize,
        allow_inner_parallelism: bool,
    ) -> ResolvedReductionStrategy {
        let overlap = self.scatter_overlap(n_dofs);
        let reduction_to_scatter = self.reduction_sweep_to_scatter(threads, n_subdomains, n_dofs);

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

    fn reduction_sweep_to_scatter(self, threads: usize, n_subdomains: usize, n_dofs: usize) -> f64 {
        let active_buffers = threads.min(n_subdomains).max(1);
        let reduction_sweep = active_buffers.saturating_mul(n_dofs);
        let scatter_work = self.total_scatter_dofs.max(1);
        reduction_sweep as f64 / scatter_work as f64
    }
}
