//! [`SchwarzPreconditioner`]: each DOF sums its subdomains' local outputs in subdomain order.

use std::sync::Arc;

use dyn_stack::{MemStack, StackReq};

use crate::error::SolveError;
use crate::local_solve::{LocalSolver, SubdomainEntry};
use crate::parallel::{try_for_each_piece, Pieces, Workspaces};
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
    /// Per budget in [`LOCAL_BUDGETS`](Self::LOCAL_BUDGETS): each local solve's workspace at it.
    workspaces: [Workspaces; 2],
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
        let workspaces = Self::LOCAL_BUDGETS
            .map(|t| Workspaces::new(entries.iter().map(|e| e.solver().scratch(t))));
        Self {
            workspaces,
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

    /// Every budget `local_budget` hands a local solve.
    const LOCAL_BUDGETS: [Threads; 2] = [Threads::All, Threads::ONE];

    /// Each local solve's budget in this pool, with the workspaces sized for it: every thread,
    /// unless the subdomains alone already fill the pool.
    fn local_budget(&self) -> (Threads, &Workspaces) {
        let [all, one] = Self::LOCAL_BUDGETS;
        let [at_all, at_one] = &self.workspaces;
        let threads = rayon::current_num_threads().max(1);
        if self.outer_capacity < threads as f64 * Self::OUTER_CAPACITY_TARGET {
            (all, at_all)
        } else {
            (one, at_one)
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

impl<S: LocalSolver> Clone for SchwarzPreconditioner<S> {
    /// Clone shares subdomain data via `Arc`: O(1) and fully interchangeable.
    fn clone(&self) -> Self {
        Self {
            subdomains: Arc::clone(&self.subdomains),
            slot_starts: Arc::clone(&self.slot_starts),
            prolongation: Arc::clone(&self.prolongation),
            workspaces: self.workspaces.clone(),
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

    /// The stacked local outputs, then every local solve's workspace at the budget this pool
    /// gives it; a stack sized in any pool serves, a smaller one runs fewer tasks at once.
    fn scratch(&self) -> StackReq {
        let (_, workspaces) = self.local_budget();
        StackReq::new::<f64>(self.n_slots()).and(workspaces.scratch())
    }

    /// `z = M⁻¹ r`: each subdomain writes only its own outputs and workspace.
    fn apply_with(&self, r: &[f64], z: &mut [f64], stack: &mut MemStack) -> Result<(), SolveError> {
        self.check_len("r", r.len())?;
        self.check_len("z", z.len())?;
        let (mut outputs, stack) = stack.make_with(self.n_slots(), |_| 0.0);
        let (threads, budgeted) = self.local_budget();
        let [_, at_one] = &self.workspaces;
        let workspaces = if stack.can_hold(budgeted.scratch()) {
            budgeted
        } else {
            at_one
        };
        let parts = (
            Pieces::new(&mut outputs, &self.slot_starts),
            workspaces.carve(stack),
        );
        try_for_each_piece(parts, &|i, (out, workspace)| {
            let stack = MemStack::new(workspace.into_items());
            self.subdomains[i]
                .apply_weighted_local(r, out.into_items(), threads, stack)
                .map_err(|source| SolveError::LocalSolveFailed {
                    subdomain: i,
                    source,
                })
        })?;
        self.prolongation.apply(&outputs, z);
        Ok(())
    }

    fn apply_adjoint_with(
        &self,
        r: &[f64],
        z: &mut [f64],
        stack: &mut MemStack,
    ) -> Result<(), SolveError> {
        self.apply_with(r, z, stack)
    }
}
