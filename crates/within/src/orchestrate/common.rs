use std::time::Instant;

use schwarz_precond::solve::vec_norm;
use schwarz_precond::Operator;

use super::SolveResult;

/// Timing state captured during solve orchestration.
pub(super) struct TimingContext {
    pub(super) t_start: Instant,
    pub(super) time_setup: f64,
    pub(super) t_solve_start: Instant,
}

/// Finalize a solve: compute residual, assemble timings, and return `SolveResult`.
pub(super) fn solve_and_assemble<A: Operator + ?Sized>(
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
fn compute_relative_residual<A: Operator + ?Sized>(
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
