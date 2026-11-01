//! [`SchwarzPreconditioner`]: each DOF sums its subdomains' local outputs in subdomain order.

use std::ops::Range;
use std::sync::Arc;

use faer::dyn_stack::{MemStack, StackReq};
use faer::matrix_free::LinOp;
use faer::{MatMut, MatRef};

use crate::local_solve::{LocalSolver, SubdomainEntry};
use crate::par::{join_on_stack, Par};
use crate::prolongation::Prolongation;
use crate::with_unit_stride;

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

/// One-level additive Schwarz preconditioner; every apply takes its buffers from the caller's stack.
pub struct SchwarzPreconditioner<S: LocalSolver> {
    subdomains: Arc<Vec<SubdomainEntry<S>>>,
    /// Subdomain `i` owns slots `slot_starts[i]..slot_starts[i + 1]` of the stacked local vector.
    slot_starts: Arc<[usize]>,
    prolongation: Arc<Prolongation>,
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
        Self {
            subdomains: Arc::new(entries),
            slot_starts,
            prolongation: Arc::new(prolongation),
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

    /// One task's local-solve scratch; a task under `par` runs at most one subdomain at a time.
    fn task_scratch(&self, par: Par) -> StackReq {
        self.subdomains.iter().fold(StackReq::EMPTY, |req, e| {
            req.or(e.solver().solve_local_scratch(1, par))
        })
    }

    /// `z = M⁻¹ r` for one column.
    fn apply_column(&self, r: &[f64], z: &mut [f64], par: Par, stack: &mut MemStack) {
        let (mut outputs, stack) = stack.make_with(self.n_slots(), |_| 0.0);
        let all = Span {
            ids: 0..self.subdomains.len(),
            outputs: &mut outputs,
        };
        self.solve_span(all, r, par, self.task_scratch(par), stack);
        self.prolongation.apply(&outputs, z, par);
    }

    /// Halves `span` and `par` together until a part has one subdomain or a degree of one.
    fn solve_span(
        &self,
        mut span: Span<'_>,
        r: &[f64],
        par: Par,
        task: StackReq,
        stack: &mut MemStack,
    ) {
        if span.ids.len() == 1 {
            let i = span.ids.start;
            let out = span.outputs_of(i, &self.slot_starts);
            self.subdomains[i].apply_weighted_local(r, out, par, stack);
            return;
        }
        if par.degree() <= 1 || span.ids.is_empty() {
            for i in span.ids.clone() {
                let out = span.outputs_of(i, &self.slot_starts);
                self.subdomains[i].apply_weighted_local(r, out, Par::Seq, stack);
            }
            return;
        }
        let (left, right) = span.halve(&self.slot_starts);
        join_on_stack(
            par,
            |p| task.array(p.degree()),
            stack,
            |p, s| self.solve_span(left, r, p, task, s),
            |p, s| self.solve_span(right, r, p, task, s),
        );
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

/// Consecutive subdomains `ids` and `outputs`, exactly their slots of the local-output buffer.
struct Span<'a> {
    ids: Range<usize>,
    outputs: &'a mut [f64],
}

impl<'a> Span<'a> {
    /// Two nonempty halves of about equal slot count; requires at least two subdomains.
    fn halve(self, starts: &[usize]) -> (Span<'a>, Span<'a>) {
        let (lo, hi) = (self.ids.start, self.ids.end);
        debug_assert!(hi - lo >= 2);
        let half = starts[lo] + (starts[hi] - starts[lo]) / 2;
        let mid =
            (lo + 1 + starts[lo + 1..hi].partition_point(|&s| s <= half)).clamp(lo + 1, hi - 1);
        let (left, right) = self.outputs.split_at_mut(starts[mid] - starts[lo]);
        (
            Span {
                ids: lo..mid,
                outputs: left,
            },
            Span {
                ids: mid..hi,
                outputs: right,
            },
        )
    }

    /// The slots of subdomain `i`, one of `ids`.
    fn outputs_of(&mut self, i: usize, starts: &[usize]) -> &mut [f64] {
        let base = starts[self.ids.start];
        &mut self.outputs[starts[i] - base..starts[i + 1] - base]
    }
}

impl<S: LocalSolver> Clone for SchwarzPreconditioner<S> {
    /// Shares the subdomain data via `Arc`: O(1).
    fn clone(&self) -> Self {
        Self {
            subdomains: Arc::clone(&self.subdomains),
            slot_starts: Arc::clone(&self.slot_starts),
            prolongation: Arc::clone(&self.prolongation),
        }
    }
}

impl<S: LocalSolver> std::fmt::Debug for SchwarzPreconditioner<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SchwarzPreconditioner")
            .field("n_subdomains", &self.subdomains.len())
            .field("n_dofs", &self.n_dofs())
            .finish()
    }
}

/// `M⁻¹`, column by column; symmetric, so `conj_apply` is `apply`.
impl<S: LocalSolver> LinOp<f64> for SchwarzPreconditioner<S> {
    fn apply_scratch(&self, _rhs_ncols: usize, par: Par) -> StackReq {
        StackReq::new::<f64>(self.n_slots()).and(self.task_scratch(par).array(par.degree()))
    }

    fn nrows(&self) -> usize {
        self.n_dofs()
    }

    fn ncols(&self) -> usize {
        self.n_dofs()
    }

    fn apply(&self, out: MatMut<'_, f64>, rhs: MatRef<'_, f64>, par: Par, stack: &mut MemStack) {
        let n = self.n_dofs();
        assert_eq!(
            (out.shape(), rhs.nrows()),
            ((n, rhs.ncols()), n),
            "M⁻¹ apply shape"
        );
        for (z, r) in out.col_iter_mut().zip(rhs.col_iter()) {
            with_unit_stride(r, z, |r, z| self.apply_column(r, z, par, stack));
        }
    }

    fn conj_apply(
        &self,
        out: MatMut<'_, f64>,
        rhs: MatRef<'_, f64>,
        par: Par,
        stack: &mut MemStack,
    ) {
        self.apply(out, rhs, par, stack);
    }
}
