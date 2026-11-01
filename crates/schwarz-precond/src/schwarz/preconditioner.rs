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

/// Persists `n_dofs`, which entries alone cannot recover when some columns are uncovered,
/// plus the configured reduction strategy (including `Auto`, before runtime resolution).
#[cfg(feature = "serde")]
impl<S> serde::Serialize for SchwarzPreconditioner<S>
where
    S: LocalSolver + serde::Serialize,
{
    fn serialize<Ser: serde::Serializer>(&self, serializer: Ser) -> Result<Ser::Ok, Ser::Error> {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("SchwarzPreconditioner", 3)?;
        state.serialize_field("subdomains", self.executor.subdomains())?;
        state.serialize_field("n_dofs", &self.executor.n_dofs())?;
        state.serialize_field("reduction_strategy", &self.reduction_strategy)?;
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
            reduction_strategy: ReductionStrategy,
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
            h.reduction_strategy,
        ))
    }
}

/// One-level additive Schwarz preconditioner; pooled per-thread buffers make `apply()` safe.
pub struct SchwarzPreconditioner<S: LocalSolver> {
    probe: Arc<ProbeCache>,
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
            probe: Arc::default(),
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

    /// Probe: `z = M⁻¹ r` for `k` interleaved columns (`r[g·k + j]`).
    pub fn apply_block(&self, r: &[f64], z: &mut [f64], k: usize) {
        macro_rules! go {
            ($($k:literal)*) => {
                match k {
                    $($k => self.apply_rows::<$k>(r.as_chunks::<$k>().0, z.as_chunks_mut::<$k>().0),)*
                    _ => panic!("k = {k} unsupported"),
                }
            };
        }
        go!(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)
    }

    fn apply_rows<const K: usize>(&self, r: &[[f64; K]], z: &mut [[f64; K]]) {
        use rayon::prelude::*;
        let subs = self.executor.subdomains();
        let inverse = self.probe.inverse.get_or_init(|| Inverse::new(subs, self.executor.n_dofs()));
        let threaded = rayon::current_num_threads() > 1;
        let thread_min = *self.probe.thread_min.get_or_init(|| {
            std::env::var("PROBE_THREAD_MIN").ok().and_then(|v| v.parse().ok()).unwrap_or(4096)
        });
        let mut locals = std::mem::take(&mut *self.probe.locals.lock().unwrap());
        let total = *inverse.offsets.last().unwrap();
        if locals.len() < total * K {
            locals.resize(total * K, 0.0);
        }
        {
            let mut rest = &mut locals[..total * K].as_chunks_mut::<K>().0[..];
            let mut slices = Vec::with_capacity(subs.len());
            for e in subs {
                let (head, tail) = rest.split_at_mut(e.global_indices().len());
                slices.push(head);
                rest = tail;
            }
            subs.par_iter().zip(slices).for_each(|(e, local)| {
                if local.is_empty() {
                    return;
                }
                let (idx, w) = (e.global_indices(), e.partition_weights());
                for (i, (&g, row)) in idx.iter().zip(local.iter_mut()).enumerate() {
                    let (wi, src) = (w.get(i), &r[g as usize]);
                    (0..K).for_each(|l| row[l] = src[l] * wi);
                }
                e.solver().solve_local_rows::<K>(local, threaded && local.len() >= thread_min);
            });
        }
        let rows = &locals[..total * K].as_chunks::<K>().0;
        z.par_iter_mut().enumerate().with_min_len(1024).for_each(|(g, out)| {
            let mut acc = [0.0; K];
            for e in inverse.ptr[g]..inverse.ptr[g + 1] {
                let (p, w) = (inverse.pos[e] as usize, inverse.weight[e]);
                (0..K).for_each(|l| acc[l] += rows[p][l] * w);
            }
            *out = acc;
        });
        *self.probe.locals.lock().unwrap() = locals;
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
            probe: Arc::clone(&self.probe),
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

/// Probe: lazily built gather map and a reusable local buffer for block applies.
#[derive(Default)]
struct ProbeCache {
    inverse: std::sync::OnceLock<Inverse>,
    thread_min: std::sync::OnceLock<usize>,
    locals: std::sync::Mutex<Vec<f64>>,
}

/// Each global dof's `(position in the concatenated locals, weight)` entries.
struct Inverse {
    offsets: Vec<usize>,
    ptr: Vec<usize>,
    pos: Vec<u32>,
    weight: Vec<f64>,
}

impl Inverse {
    fn new<S: LocalSolver>(subs: &[SubdomainEntry<S>], n_dofs: usize) -> Self {
        let mut offsets = vec![0];
        let mut ptr = vec![0usize; n_dofs + 1];
        for e in subs {
            offsets.push(offsets.last().unwrap() + e.global_indices().len());
            e.global_indices().iter().for_each(|&g| ptr[g as usize + 1] += 1);
        }
        (0..n_dofs).for_each(|g| ptr[g + 1] += ptr[g]);
        let mut fill = ptr.clone();
        let total = *offsets.last().unwrap();
        let (mut pos, mut weight) = (vec![0u32; total], vec![0.0; total]);
        for (s, e) in subs.iter().enumerate() {
            let w = e.partition_weights();
            for (i, &g) in e.global_indices().iter().enumerate() {
                let slot = &mut fill[g as usize];
                pos[*slot] = (offsets[s] + i) as u32;
                weight[*slot] = w.get(i);
                *slot += 1;
            }
        }
        Self { offsets, ptr, pos, weight }
    }
}
