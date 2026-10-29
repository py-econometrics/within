//! #365 type draft, faer-shaped: a parallelism budget and stack scratch per apply. Signatures only.
#![allow(dead_code, unused_variables, clippy::too_many_arguments)]

use std::num::NonZeroUsize;

pub use dyn_stack::{PodBuffer, PodStack, StackReq};

use crate::error::{LocalSolveError, SolveError};
use crate::{LsmrResult, MlsmrOptions};

/// How many tasks one parallel region may keep busy on the global rayon pool; `SEQ` runs inline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Par(pub NonZeroUsize);

impl Par {
    /// One task: inline on the calling thread, no rayon call.
    pub const SEQ: Self = Self(NonZeroUsize::MIN);

    /// The width of the caller's rayon pool; read only at public entry points.
    pub fn current() -> Self {
        Self(NonZeroUsize::new(rayon::current_num_threads()).unwrap_or(NonZeroUsize::MIN))
    }

    /// `min(parts, tasks)` budgets summing to this one: the one rule for nesting regions.
    pub fn split(self, parts: usize) -> impl ExactSizeIterator<Item = Self> {
        let (tasks, parts) = (self.0.get(), parts.min(self.0.get()));
        (0..parts).map(move |i| {
            let share = tasks / parts + usize::from(i < tasks % parts);
            Self(NonZeroUsize::new(share).unwrap_or(NonZeroUsize::MIN))
        })
    }

    /// `a` and `b` on halves of this budget: inline under `SEQ`, else one `rayon::join`.
    pub fn join<A: Send, B: Send>(
        self,
        a: impl FnOnce(Self) -> A + Send,
        b: impl FnOnce(Self) -> B + Send,
    ) -> (A, B) {
        let mut halves = self.split(2);
        match (halves.next(), halves.next()) {
            (Some(l), Some(r)) => rayon::join(|| a(l), || b(r)),
            _ => (a(Self::SEQ), b(Self::SEQ)),
        }
    }
}

/// An immutable linear operator; any number of applies may share it.
pub trait Operator: Send + Sync {
    /// Number of rows in the operator.
    fn nrows(&self) -> usize;
    /// Number of columns in the operator.
    fn ncols(&self) -> usize;
    /// Scratch one `apply` or `adjoint_apply` under `par` takes from its stack.
    fn scratch_req(&self, par: Par) -> StackReq;
    /// Computes `y = A x`.
    fn apply(
        &self,
        y: &mut [f64],
        x: &[f64],
        par: Par,
        stack: &mut PodStack,
    ) -> Result<(), SolveError>;
    /// Computes `y = Aᵀ x`; symmetric operators delegate to `apply`.
    fn adjoint_apply(
        &self,
        y: &mut [f64],
        x: &[f64],
        par: Par,
        stack: &mut PodStack,
    ) -> Result<(), SolveError>;
}

/// Exact or approximate solver for one subdomain's local system.
pub trait LocalSolver: Send + Sync {
    /// Number of DOFs in the subdomain.
    fn n_local(&self) -> usize;
    /// Scratch one `solve_local` takes from its stack.
    fn scratch_req(&self) -> StackReq;
    /// Solves in place: `x` spans `n_local()`, rhs in, solution out; bits depend on `x` alone.
    fn solve_local(
        &self,
        x: &mut [f64],
        par: Par,
        stack: &mut PodStack,
    ) -> Result<(), LocalSolveError>;
}

/// Unpreconditioned LSMR; grows `scratch` like `mlsmr`.
pub fn lsmr<A: Operator + ?Sized>(
    operator: &A,
    b: &[f64],
    tol: f64,
    maxiter: usize,
    local_size: Option<usize>,
    par: Par,
    scratch: &mut PodBuffer,
) -> Result<LsmrResult, SolveError> {
    todo!()
}

/// Preconditioned LSMR; grows `scratch` to what `operator` and `preconditioner` need under
/// `par`, so one buffer serves every solve it is passed to.
pub fn mlsmr<A: Operator + ?Sized, M: Operator + ?Sized>(
    operator: &A,
    b: &[f64],
    preconditioner: &M,
    tol: f64,
    maxiter: usize,
    options: MlsmrOptions<'_>,
    par: Par,
    scratch: &mut PodBuffer,
) -> Result<LsmrResult, SolveError> {
    todo!()
}
