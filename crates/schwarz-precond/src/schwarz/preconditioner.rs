//! [`SchwarzPreconditioner`]: each DOF sums its subdomains' local outputs in subdomain order.

use std::mem::MaybeUninit;
use std::ops::Range;
use std::sync::Arc;

use dyn_stack::{MemStack, StackReq};
use rayon::prelude::*;

use crate::error::SolveError;
use crate::local_solve::{LocalSolver, SubdomainEntry};
use crate::prolongation::Prolongation;
use crate::threads::Threads;
use crate::Operator;

/// Persists `n_dofs`, which entries alone cannot recover when some columns are uncovered.
#[cfg(feature = "serde")]
impl<S> serde::Serialize for SchwarzPreconditioner<S>
where
    S: LocalSolver + serde::Serialize,
{
    fn serialize<Ser: serde::Serializer>(&self, serializer: Ser) -> Result<Ser::Ok, Ser::Error> {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("SchwarzPreconditioner", 2)?;
        state.serialize_field("subdomains", &**self.subdomains)?;
        state.serialize_field("n_dofs", &self.n_dofs())?;
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
        // An `n_dofs` below the covered span would put a subdomain DOF outside the prolongation.
        let covered_span = covered_span(&h.subdomains);
        if h.n_dofs < covered_span {
            return Err(serde::de::Error::custom(format!(
                "n_dofs ({}) is below the covered subdomain index span ({covered_span})",
                h.n_dofs
            )));
        }
        Ok(SchwarzPreconditioner::with_n_dofs(h.subdomains, h.n_dofs))
    }
}

/// One-level additive Schwarz preconditioner.
pub struct SchwarzPreconditioner<S: LocalSolver> {
    subdomains: Arc<Vec<SubdomainEntry<S>>>,
    /// Subdomain `i` owns slots `slot_starts[i]..slot_starts[i + 1]` of the stacked local vector.
    slot_starts: Arc<[usize]>,
    /// Subdomain `i`'s local solve owns bytes `workspace_offsets[i]..[i + 1]` of an apply's
    /// workspace, enough at every budget `local_threads` hands out.
    workspace_offsets: Arc<[usize]>,
    /// Every workspace starts on this boundary: a cache line, or the widest a local solve asks.
    workspace_align: usize,
    prolongation: Arc<Prolongation>,
    /// Total over largest subdomain work: below `OUTER_CAPACITY_TARGET` of the pool, the
    /// subdomains cannot fill it, so each local solve gets every thread.
    outer_capacity: f64,
}

impl<S: LocalSolver> SchwarzPreconditioner<S> {
    const OUTER_CAPACITY_TARGET: f64 = 0.75;

    /// Infers `n_dofs` from the maximum global index; only valid when every DOF is covered.
    pub fn new(entries: Vec<SubdomainEntry<S>>) -> Self {
        let n_dofs = covered_span(&entries);
        Self::with_n_dofs(entries, n_dofs)
    }

    /// Explicit global DOF count; an uncovered column stays in the null space and applies to `0`.
    pub fn with_n_dofs(entries: Vec<SubdomainEntry<S>>, n_dofs: usize) -> Self {
        let slot_starts: Arc<[usize]> = std::iter::once(0)
            .chain(entries.iter().scan(0, |end, e| {
                *end += e.global_indices().len();
                Some(*end)
            }))
            .collect();
        let prolongation = Prolongation::new(
            entries
                .iter()
                .flat_map(|e| e.global_indices().iter().copied()),
            n_dofs,
        );
        let workspaces: Vec<StackReq> = entries
            .iter()
            .map(|e| {
                let workspace = Self::LOCAL_BUDGETS
                    .map(|t| e.solver().scratch(t))
                    .into_iter()
                    .fold(StackReq::EMPTY, StackReq::or);
                assert!(
                    workspace.align_bytes() > 0,
                    "a local solve's scratch overflows"
                );
                workspace
            })
            .collect();
        let workspace_align = (workspaces.iter())
            .map(StackReq::align_bytes)
            .fold(64, usize::max);
        let workspace_offsets = std::iter::once(0)
            .chain(workspaces.iter().scan(0, |end, w| {
                *end += w.size_bytes().next_multiple_of(workspace_align);
                Some(*end)
            }))
            .collect();
        Self {
            workspace_offsets,
            workspace_align,
            outer_capacity: Self::outer_capacity(&entries),
            subdomains: Arc::new(entries),
            slot_starts,
            prolongation: Arc::new(prolongation),
        }
    }

    /// Total over largest subdomain work; infinite when no subdomain merits inner parallelism.
    fn outer_capacity(entries: &[SubdomainEntry<S>]) -> f64 {
        let work = || {
            entries
                .iter()
                .map(|e| e.solver().inner_parallelism_work_estimate())
        };
        let largest = work().max().unwrap_or(0);
        if largest == 0 {
            return f64::INFINITY;
        }
        let total = work().fold(0usize, usize::saturating_add);
        total as f64 / largest as f64
    }

    /// Access the underlying subdomain entries.
    pub fn subdomains(&self) -> &[SubdomainEntry<S>] {
        &self.subdomains
    }

    fn n_dofs(&self) -> usize {
        self.prolongation.n_dofs()
    }

    fn n_slots(&self) -> usize {
        self.slot_starts[self.subdomains.len()]
    }

    fn workspace_len(&self) -> usize {
        self.workspace_offsets[self.subdomains.len()]
    }

    /// `z = M⁻¹ r` on `stack`: each subdomain writes only its own outputs and workspace.
    fn apply_on(&self, r: &[f64], z: &mut [f64], stack: &mut MemStack) -> Result<(), SolveError> {
        self.check_len("r", r.len())?;
        self.check_len("z", z.len())?;
        let threads = self.local_threads();
        let (mut outputs, stack) = stack.make_with(self.n_slots(), |_| 0.0);
        let (workspaces, _) =
            stack.make_aligned_uninit::<u8>(self.workspace_len(), self.workspace_align);
        let all = Span {
            ids: 0..self.subdomains.len(),
            outputs: &mut outputs,
            workspaces,
        };
        rayon::iter::split(all, |span| {
            span.halve(&self.slot_starts, &self.workspace_offsets)
        })
        .try_for_each(|span| self.solve_span(span, r, threads))?;
        self.prolongation.apply(&outputs, z);
        Ok(())
    }

    fn solve_span(
        &self,
        mut span: Span<'_>,
        r: &[f64],
        threads: Threads,
    ) -> Result<(), SolveError> {
        for i in span.ids.clone() {
            let (out, workspace) = span.parts_of(i, &self.slot_starts, &self.workspace_offsets);
            self.subdomains[i]
                .apply_weighted_local(r, out, threads, MemStack::new(workspace))
                .map_err(|source| SolveError::LocalSolveFailed {
                    subdomain: i,
                    source,
                })?;
        }
        Ok(())
    }

    fn check_len(&self, name: &str, len: usize) -> Result<(), SolveError> {
        let n_dofs = self.n_dofs();
        if len == n_dofs {
            return Ok(());
        }
        Err(SolveError::InvalidInput {
            context: "SchwarzPreconditioner::apply",
            message: format!("{name}.len() ({len}) != n_dofs ({n_dofs})"),
        })
    }

    /// Every budget `local_threads` hands a local solve, which each workspace is sized for.
    const LOCAL_BUDGETS: [Threads; 2] = [Threads::All, Threads::ONE];

    /// Every thread to each local solve, unless the subdomains alone already fill the pool.
    fn local_threads(&self) -> Threads {
        let [all, one] = Self::LOCAL_BUDGETS;
        let threads = rayon::current_num_threads().max(1);
        if self.outer_capacity < threads as f64 * Self::OUTER_CAPACITY_TARGET {
            all
        } else {
            one
        }
    }
}

/// One past the largest DOF any subdomain covers.
fn covered_span<S: LocalSolver>(entries: &[SubdomainEntry<S>]) -> usize {
    entries
        .iter()
        .flat_map(|e| e.global_indices().iter().copied())
        .max()
        .map_or(0, |m| m as usize + 1)
}

/// Consecutive subdomains `ids` with exactly their slots of the local outputs and their workspaces.
struct Span<'a> {
    ids: Range<usize>,
    outputs: &'a mut [f64],
    workspaces: &'a mut [MaybeUninit<u8>],
}

impl Span<'_> {
    fn halve(self, slots: &[usize], workspaces: &[usize]) -> (Self, Option<Self>) {
        if self.ids.len() < 2 {
            return (self, None);
        }
        let mid = self.ids.start + self.ids.len() / 2;
        let (out_l, out_r) = self
            .outputs
            .split_at_mut(slots[mid] - slots[self.ids.start]);
        let (workspace_l, workspace_r) = self
            .workspaces
            .split_at_mut(workspaces[mid] - workspaces[self.ids.start]);
        let left = Span {
            ids: self.ids.start..mid,
            outputs: out_l,
            workspaces: workspace_l,
        };
        let right = Span {
            ids: mid..self.ids.end,
            outputs: out_r,
            workspaces: workspace_r,
        };
        (left, Some(right))
    }

    /// Subdomain `i`'s slots and workspace, `i` one of `ids`.
    fn parts_of(
        &mut self,
        i: usize,
        slots: &[usize],
        workspaces: &[usize],
    ) -> (&mut [f64], &mut [MaybeUninit<u8>]) {
        let (s0, w0) = (slots[self.ids.start], workspaces[self.ids.start]);
        (
            &mut self.outputs[slots[i] - s0..slots[i + 1] - s0],
            &mut self.workspaces[workspaces[i] - w0..workspaces[i + 1] - w0],
        )
    }
}

impl<S: LocalSolver> Clone for SchwarzPreconditioner<S> {
    /// Clone shares subdomain data via `Arc`: O(1) and fully interchangeable.
    fn clone(&self) -> Self {
        Self {
            subdomains: Arc::clone(&self.subdomains),
            slot_starts: Arc::clone(&self.slot_starts),
            prolongation: Arc::clone(&self.prolongation),
            workspace_offsets: Arc::clone(&self.workspace_offsets),
            workspace_align: self.workspace_align,
            outer_capacity: self.outer_capacity,
        }
    }
}

impl<S: LocalSolver> Operator for SchwarzPreconditioner<S> {
    fn nrows(&self) -> usize {
        self.n_dofs()
    }

    fn ncols(&self) -> usize {
        self.n_dofs()
    }

    /// The stacked local outputs, then every local solve's workspace.
    fn scratch(&self) -> StackReq {
        StackReq::new::<f64>(self.n_slots()).and(StackReq::new_aligned::<u8>(
            self.workspace_len(),
            self.workspace_align,
        ))
    }

    fn apply_with(&self, r: &[f64], z: &mut [f64], stack: &mut MemStack) -> Result<(), SolveError> {
        self.apply_on(r, z, stack)
    }

    fn apply_adjoint_with(
        &self,
        r: &[f64],
        z: &mut [f64],
        stack: &mut MemStack,
    ) -> Result<(), SolveError> {
        self.apply_on(r, z, stack)
    }
}
