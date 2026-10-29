//! #365 type draft on faer's operator contract: `Par` and stack scratch per apply. Signatures only.
#![allow(dead_code, unused_variables)]

use std::num::NonZeroUsize;

pub use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
pub use faer::matrix_free::{BiLinOp, BiPrecond, LinOp, Precond};
pub use faer::{MatMut, MatRef, Par};

use crate::error::SolveError;
use crate::{EscalationPolicy, LsmrOptions, LsmrResult};

/// `a` and `b` on rayon, each with half of `par` (`⌈n/2⌉`, `⌊n/2⌋`); inline under `Par::Seq`.
pub fn join<A: Send, B: Send>(
    par: Par,
    a: impl FnOnce(Par) -> A + Send,
    b: impl FnOnce(Par) -> B + Send,
) -> (A, B) {
    let n = par.degree();
    if n <= 1 {
        return (a(Par::Seq), b(Par::Seq));
    }
    let half = |k: usize| NonZeroUsize::new(k).map_or(Par::Seq, Par::Rayon);
    let (l, r) = (half(n.div_ceil(2)), half(n / 2));
    rayon::join(|| a(l), || b(r))
}

/// Exact or approximate solver for one subdomain's local system.
pub trait LocalSolver: Send + Sync {
    /// Number of DOFs in the subdomain.
    fn n_local(&self) -> usize;
    /// Scratch one `solve_local` on `rhs_ncols` columns under `par` takes from its stack.
    fn solve_local_scratch(&self, rhs_ncols: usize, par: Par) -> StackReq;
    /// Solves each column in place: `x` is `n_local() × k`, rhs in, solution out.
    fn solve_local(&self, x: MatMut<'_, f64>, par: Par, stack: &mut MemStack);
}

/// Where a preconditioned run starts and when it hands off to a stronger map.
#[derive(Clone, Copy, Default)]
pub struct MlsmrOptions<'a> {
    /// Residual-correction start; tolerances stay relative to `‖b‖` (`‖b − A x₀‖` if `b = 0`).
    pub warm_start: Option<&'a [f64]>,
    /// Hands off to a stronger preconditioner mid-run; see [`EscalationPolicy`].
    pub escalation: Option<&'a dyn EscalationPolicy>,
}

/// Unpreconditioned LSMR; grows `scratch` like `mlsmr`.
pub fn lsmr<A: BiLinOp<f64> + ?Sized>(
    operator: &A,
    b: &[f64],
    options: &LsmrOptions,
    par: Par,
    scratch: &mut MemBuffer,
) -> Result<LsmrResult, SolveError> {
    todo!()
}

/// LSMR on `M⁻¹ ≈ (AᵀA)⁻¹`, which `preconditioner` applies; grows `scratch` to what `operator` and
/// `preconditioner` need under `par`, so one buffer serves every solve it is passed to.
pub fn mlsmr<A: BiLinOp<f64> + ?Sized, M: BiPrecond<f64> + ?Sized>(
    operator: &A,
    b: &[f64],
    preconditioner: &M,
    lsmr: &LsmrOptions,
    options: MlsmrOptions<'_>,
    par: Par,
    scratch: &mut MemBuffer,
) -> Result<LsmrResult, SolveError> {
    todo!()
}
