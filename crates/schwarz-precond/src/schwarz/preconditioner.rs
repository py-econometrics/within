//! [`SchwarzPreconditioner`]: each DOF sums its subdomains' local outputs in subdomain order.

use std::ops::Range;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use dyn_stack::{MemStack, StackReq};

use crate::error::SolveError;
use crate::local_solve::{LocalSolver, SubdomainEntry};
use crate::prolongation::Prolongation;
use crate::threads::Threads;
use crate::Operator;

use super::spread::{split, Cost, Spread, CLAIMS_PER_WORKER};

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
    prolongation: Arc<Prolongation>,
    /// Each subdomain's work and one-thread scratch, by which an apply spreads its budget.
    costs: Arc<[Cost]>,
}

impl<S: LocalSolver> SchwarzPreconditioner<S> {
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
        let costs = entries
            .iter()
            .map(|e| Cost {
                work: e.solver().work_estimate(),
                scratch: e.solver().scratch(1, Threads::ONE),
            })
            .collect();
        Self {
            subdomains: Arc::new(entries),
            slot_starts,
            prolongation: Arc::new(prolongation),
            costs,
        }
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

    /// Writes the local outputs of the subdomains `pick` takes in `span`.
    fn solve_span(
        &self,
        span: &mut Span<'_>,
        r: &[f64],
        threads: Threads,
        stack: &mut MemStack,
        pick: impl Fn(&Cost) -> bool,
    ) -> Result<(), SolveError> {
        for i in span.ids.clone() {
            if !pick(&self.costs[i]) {
                continue;
            }
            self.subdomains[i]
                .apply_weighted_local(r, span.outputs_of(i, &self.slot_starts), threads, stack)
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
}

// A panic in another claim cannot leave the unclaimed subdomains inconsistent.
fn lock<'m, 'a>(rest: &'m Mutex<Span<'a>>) -> MutexGuard<'m, Span<'a>> {
    rest.lock().unwrap_or_else(PoisonError::into_inner)
}

/// One past the largest DOF any subdomain covers.
fn covered_span<S: LocalSolver>(entries: &[SubdomainEntry<S>]) -> usize {
    entries
        .iter()
        .flat_map(|e| e.global_indices().iter().copied())
        .max()
        .map_or(0, |m| m as usize + 1)
}

/// Consecutive subdomains `ids` and `outputs`, exactly their slots of the local-output buffer.
#[derive(Default)]
struct Span<'a> {
    ids: Range<usize>,
    outputs: &'a mut [f64],
}

impl<'a> Span<'a> {
    /// Splits off the first `n` subdomains, or `None` when none are left.
    fn take_front(&mut self, n: usize, starts: &[usize]) -> Option<Span<'a>> {
        if self.ids.is_empty() {
            return None;
        }
        let mid = self.ids.end.min(self.ids.start + n);
        let (front, back) =
            std::mem::take(&mut self.outputs).split_at_mut(starts[mid] - starts[self.ids.start]);
        let front = Span {
            ids: self.ids.start..mid,
            outputs: front,
        };
        *self = Span {
            ids: mid..self.ids.end,
            outputs: back,
        };
        Some(front)
    }

    /// The slots of subdomain `i`, one of `ids`.
    fn outputs_of(&mut self, i: usize, starts: &[usize]) -> &mut [f64] {
        let base = starts[self.ids.start];
        &mut self.outputs[starts[i] - base..starts[i + 1] - base]
    }
}

impl<S: LocalSolver> Clone for SchwarzPreconditioner<S> {
    /// Clone shares subdomain data via `Arc`: O(1) and fully interchangeable.
    fn clone(&self) -> Self {
        Self {
            subdomains: Arc::clone(&self.subdomains),
            slot_starts: Arc::clone(&self.slot_starts),
            prolongation: Arc::clone(&self.prolongation),
            costs: Arc::clone(&self.costs),
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

    /// The local outputs, then the most a subdomain past the share takes on the whole budget or
    /// the regions the others run on at once.
    fn scratch(&self, _columns: usize, threads: Threads) -> StackReq {
        let spread = Spread::of(&self.costs, threads);
        let alone = self
            .subdomains
            .iter()
            .zip(&*self.costs)
            .filter(|(_, cost)| spread.alone(cost))
            .map(|(e, _)| e.solver().scratch(1, threads))
            .fold(StackReq::EMPTY, StackReq::or);
        let within = spread.region.array(spread.regions());
        StackReq::new::<f64>(self.n_slots()).and(alone.or(within))
    }

    /// Each subdomain past the budget's share on the whole budget, one after another, then the
    /// others at once, one thread each, as many as `stack` holds regions for.
    fn apply(
        &self,
        r: &[f64],
        z: &mut [f64],
        threads: Threads,
        stack: &mut MemStack,
    ) -> Result<(), SolveError> {
        self.check_len("r", r.len())?;
        self.check_len("z", z.len())?;
        let (mut outputs, stack) = stack.make_with(self.n_slots(), |_| 0.0);
        let spread = Spread::of(&self.costs, threads);
        let mut all = Span {
            ids: 0..self.subdomains.len(),
            outputs: &mut outputs,
        };
        self.solve_span(&mut all, r, threads, stack, |c| spread.alone(c))?;
        let within = |c: &Cost| !spread.alone(c);
        let workers = spread.workers(stack);
        if workers < 2 {
            self.solve_span(&mut all, r, Threads::ONE, stack, within)?;
        } else {
            let claim = self.subdomains.len().div_ceil(workers * CLAIMS_PER_WORKER);
            let rest = Mutex::new(all);
            split(
                stack,
                spread.region,
                workers,
                &|stack: &mut MemStack| loop {
                    let Some(mut span) = lock(&rest).take_front(claim, &self.slot_starts) else {
                        return Ok(());
                    };
                    let solved = self.solve_span(&mut span, r, Threads::ONE, stack, within);
                    if solved.is_err() {
                        *lock(&rest) = Span::default();
                        return solved;
                    }
                },
            )?;
        }
        self.prolongation.apply(&outputs, z, threads);
        Ok(())
    }

    fn apply_adjoint(
        &self,
        r: &[f64],
        z: &mut [f64],
        threads: Threads,
        stack: &mut MemStack,
    ) -> Result<(), SolveError> {
        Operator::apply(self, r, z, threads, stack)
    }
}
