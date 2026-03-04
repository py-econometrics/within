use std::time::Instant;

use schwarz_precond::solve::lsmr::vec_norm;
use schwarz_precond::Operator;

use super::SolveResult;

/// Timing state captured during solve orchestration.
pub(super) struct TimingContext {
    pub(super) t_start: Instant,
    pub(super) time_setup: f64,
    pub(super) t_solve_start: Instant,
}

/// Interpret LSMR's `istop` convergence flag.
///
/// * 0 — initial x is exact solution
/// * 1 — Ax - b is small enough (atol)
/// * 2 — least-squares condition satisfied (atol)
/// * 3 — condition number limit reached (conlim)
/// * 4 — Ax - b small relative to b (atol)
/// * 5 — least-squares small relative to norms (atol)
/// * 6 — condition number exceeds conlim
/// * 7 — iteration limit reached
pub(super) fn interpret_lsmr_istop(istop: i32) -> bool {
    matches!(istop, 0 | 1 | 2 | 4)
}

/// Finalize a solve: compute residual, assemble timings, and return `SolveResult`.
pub(super) fn solve_and_assemble<A: Operator>(
    op: &A,
    x: Vec<f64>,
    converged: bool,
    iterations: usize,
    rhs: &[f64],
    rhs_norm: f64,
    timing: TimingContext,
) -> SolveResult {
    let time_solve = timing.t_solve_start.elapsed().as_secs_f64();
    let mut scratch = vec![0.0; x.len()];
    let final_residual = compute_relative_residual(op, &x, rhs, rhs_norm, &mut scratch);
    SolveResult {
        x,
        converged,
        iterations,
        final_residual,
        time_total: timing.t_start.elapsed().as_secs_f64(),
        time_setup: timing.time_setup,
        time_solve,
    }
}

/// Compute `||A x - b|| / ||b||` using caller-provided scratch.
fn compute_relative_residual<A: Operator>(
    op: &A,
    x: &[f64],
    b: &[f64],
    b_norm: f64,
    scratch: &mut [f64],
) -> f64 {
    op.apply(x, scratch);
    for i in 0..b.len() {
        scratch[i] -= b[i];
    }
    vec_norm(scratch) / b_norm
}
