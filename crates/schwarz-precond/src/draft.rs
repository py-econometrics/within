//! #365 type draft on faer's operator contract: `Par` and stack scratch per apply. Signatures only.
#![allow(dead_code, unused_variables)]

use std::num::NonZeroUsize;
use std::ops::ControlFlow;
use std::sync::atomic::{AtomicBool, Ordering};

pub use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
pub use faer::matrix_free::{BiLinOp, LinOp};
pub use faer::{MatMut, MatRef, Par};

use crate::error::SolveError;
use crate::{LsmrOptions, LsmrStopReason, Progress, Staleness};

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

/// Ends a run before its tolerance: an escalation to a stronger map, an interrupt, or both.
pub trait StopRule {
    /// `Break` ends the run after this iteration; checked once per iteration.
    fn check(&mut self, progress: Progress) -> ControlFlow<()>;
}

/// Stops when either rule stops; each keeps its own state.
impl<A: StopRule, B: StopRule> StopRule for (A, B) {
    fn check(&mut self, progress: Progress) -> ControlFlow<()> {
        self.0.check(progress)?;
        self.1.check(progress)
    }
}

/// A flag set from outside the solve, shared by every lane.
impl StopRule for &AtomicBool {
    fn check(&mut self, _: Progress) -> ControlFlow<()> {
        match self.load(Ordering::Relaxed) {
            true => ControlFlow::Break(()),
            false => ControlFlow::Continue(()),
        }
    }
}

impl Staleness {
    /// This policy's stall counter for one run.
    pub fn run(&self) -> impl StopRule {
        struct Todo;
        impl StopRule for Todo {
            fn check(&mut self, _: Progress) -> ControlFlow<()> {
                todo!()
            }
        }
        Todo
    }
}

/// What a run reports; the solution is in the caller's `x` and `b − A x` in its `r`.
#[derive(Clone, Copy, Debug)]
pub struct LsmrInfo {
    /// Whether a tolerance stop survived its applicable recomputed-residual and plain-gradient audits.
    pub converged: bool,
    /// Total number of iterations performed.
    pub iterations: usize,
    /// `‖r‖`.
    pub residual_norm: f64,
    /// `‖Âᵀr‖ / ‖Âᵀb‖`, measured in `M`'s metric when preconditioned.
    pub normal_eq_residual: f64,
    /// Reason the solver stopped.
    pub stop_reason: LsmrStopReason,
}

/// Unpreconditioned LSMR from `x` (all zeros: a cold start) into `x`, with `r = b − A x`;
/// tolerances stay relative to `‖b‖`; grows `scratch` like `mlsmr`.
#[allow(clippy::too_many_arguments)]
pub fn lsmr<A: BiLinOp<f64> + ?Sized>(
    operator: &A,
    b: &[f64],
    x: &mut [f64],
    r: &mut [f64],
    options: &LsmrOptions,
    stop: Option<&mut dyn StopRule>,
    par: Par,
    scratch: &mut MemBuffer,
) -> Result<LsmrInfo, SolveError> {
    todo!()
}

/// LSMR on `M⁻¹ ≈ (AᵀA)⁻¹`, which `preconditioner` applies, from `x` into `x` like `lsmr`;
/// `stop` may end it early; grows `scratch` to what `operator` and
/// `preconditioner` need under `par`, so one buffer serves every solve it is passed to.
#[allow(clippy::too_many_arguments)]
pub fn mlsmr<A: BiLinOp<f64> + ?Sized, M: LinOp<f64> + ?Sized>(
    operator: &A,
    b: &[f64],
    preconditioner: &M,
    x: &mut [f64],
    r: &mut [f64],
    lsmr: &LsmrOptions,
    stop: Option<&mut dyn StopRule>,
    par: Par,
    scratch: &mut MemBuffer,
) -> Result<LsmrInfo, SolveError> {
    todo!()
}
