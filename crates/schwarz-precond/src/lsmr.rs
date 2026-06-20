//! LSMR for rectangular least-squares (`min ‖b − A x‖₂`).
//!
//! Two entry points:
//! - [`lsmr`] — standard Golub-Kahan bidiagonalization, no preconditioner.
//! - [`mlsmr`] — Modified Golub-Kahan variant preconditioned with `M ≈ AᵀA`;
//!   requires a single `M⁻¹` application per iteration.

mod bidiag;
mod recurrence;
#[cfg(test)]
mod tests;

use crate::{Operator, SolveError};
use bidiag::{BidiagStep, Bidiagonalization, GolubKahan, ModifiedGolubKahan};
use recurrence::{ConvergenceState, LsmrRecurrenceState, RotationStep, SolutionState, Stop};

/// Euclidean norm of a vector.
#[inline]
pub(crate) fn vec_norm(v: &[f64]) -> f64 {
    let mut s = 0.0f64;
    for &x in v {
        s += x * x;
    }
    s.sqrt()
}

/// Result of an LSMR solve.
#[must_use]
pub struct LsmrResult {
    /// Solution vector.
    pub x: Vec<f64>,
    /// Whether the solver converged within the tolerance.
    pub converged: bool,
    /// Total number of iterations performed.
    pub iterations: usize,
    /// Final residual norm estimate `‖b − A x‖`.
    pub residual_norm: f64,
    /// Reason the solver stopped.
    pub stop_reason: LsmrStopReason,
}

/// Reason an LSMR solve stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LsmrStopReason {
    /// The right-hand side was exactly zero.
    ZeroRhs,
    /// The initial normal-equation residual was exactly zero (`Aᵀb = 0`).
    InitialNormalEquationResidualZero,
    /// The least-squares residual estimate met the absolute tolerance.
    ResidualTolerance,
    /// The normal-equation residual estimate met the relative tolerance.
    NormalEquationTolerance,
    /// The bidiagonalization reached a lucky breakdown.
    BidiagonalizationBreakdown,
    /// The iteration budget was exhausted before convergence.
    MaxIterations,
}

/// Unpreconditioned LSMR.
///
/// Solves `min ‖b − A x‖₂` using the standard Golub-Kahan
/// bidiagonalization.
pub fn lsmr<A: Operator + ?Sized>(
    operator: &A,
    b: &[f64],
    tol: f64,
    maxiter: usize,
    local_size: Option<usize>,
) -> Result<LsmrResult, SolveError> {
    validate_lsmr_inputs(operator, b, tol)?;
    let n = operator.ncols();

    let b_norm = vec_norm(b);
    if b_norm == 0.0 {
        return Ok(LsmrResult {
            x: vec![0.0; n],
            converged: true,
            iterations: 0,
            residual_norm: 0.0,
            stop_reason: LsmrStopReason::ZeroRhs,
        });
    }

    let local_size = local_size.unwrap_or(0);
    let (bidiag, step1) = GolubKahan::init(operator, b, local_size)?;
    lsmr_from_bidiag(bidiag, step1, b_norm, tol, maxiter)
}

/// Preconditioned LSMR with `M ≈ AᵀA`.
///
/// Uses the Modified Golub-Kahan variant requiring one `M⁻¹` application per
/// iteration.
pub fn mlsmr<A: Operator + ?Sized, M: Operator + ?Sized>(
    operator: &A,
    b: &[f64],
    preconditioner: &M,
    tol: f64,
    maxiter: usize,
    local_size: Option<usize>,
) -> Result<LsmrResult, SolveError> {
    validate_lsmr_inputs(operator, b, tol)?;
    let n = operator.ncols();
    if preconditioner.nrows() != n || preconditioner.ncols() != n {
        return Err(SolveError::InvalidInput {
            context: "lsmr",
            message: format!(
                "preconditioner shape {}x{} must match operator column count {n}",
                preconditioner.nrows(),
                preconditioner.ncols(),
            ),
        });
    }

    let b_norm = vec_norm(b);
    if b_norm == 0.0 {
        return Ok(LsmrResult {
            x: vec![0.0; n],
            converged: true,
            iterations: 0,
            residual_norm: 0.0,
            stop_reason: LsmrStopReason::ZeroRhs,
        });
    }

    let local_size = local_size.unwrap_or(0);
    let (bidiag, step1) = ModifiedGolubKahan::init(operator, preconditioner, b, local_size)?;
    lsmr_from_bidiag(bidiag, step1, b_norm, tol, maxiter)
}

/// Run the LSMR scalar/vector recurrences over a bidiagonalization stream.
/// Generic over the bidiagonalization, which is the only place the choice
/// of preconditioner enters.
fn lsmr_from_bidiag<B: Bidiagonalization>(
    mut bidiag: B,
    step1: BidiagStep,
    b_norm: f64,
    tol: f64,
    maxiter: usize,
) -> Result<LsmrResult, SolveError> {
    let n = bidiag.v().len();
    if step1.alpha == 0.0 {
        return Ok(LsmrResult {
            x: vec![0.0; n],
            converged: true,
            iterations: 0,
            residual_norm: b_norm,
            stop_reason: LsmrStopReason::InitialNormalEquationResidualZero,
        });
    }

    let mut recurrence = LsmrRecurrenceState::init(step1);
    let mut solution = SolutionState::init(bidiag.v());
    let mut convergence = ConvergenceState::new(b_norm, tol, step1.alpha);
    let mut prev_rot = RotationStep::initial();

    for itn in 1..=maxiter {
        let step = bidiag.step()?;
        convergence.observe(step);
        let curr_rot = recurrence.step(step);
        solution.update(bidiag.v(), curr_rot, prev_rot);

        // Convergence wins over breakdown when both fire on the same step:
        // the user-specified tolerance is the contract, breakdown is an
        // internal property of the bidiagonalization.
        if let Some(stop_reason) = match convergence.check(&recurrence) {
            Stop::Continue => None,
            Stop::ResidualTolerance => Some(LsmrStopReason::ResidualTolerance),
            Stop::NormalEquationTolerance => Some(LsmrStopReason::NormalEquationTolerance),
        } {
            return Ok(LsmrResult {
                x: solution.into_x(),
                converged: true,
                iterations: itn,
                residual_norm: recurrence.residual_estimate(),
                stop_reason,
            });
        }
        if step.alpha == 0.0 {
            return Ok(LsmrResult {
                x: solution.into_x(),
                converged: true,
                iterations: itn,
                residual_norm: recurrence.residual_estimate(),
                stop_reason: LsmrStopReason::BidiagonalizationBreakdown,
            });
        }
        prev_rot = curr_rot;
    }

    Ok(LsmrResult {
        x: solution.into_x(),
        converged: false,
        iterations: maxiter,
        residual_norm: recurrence.residual_estimate(),
        stop_reason: LsmrStopReason::MaxIterations,
    })
}

fn validate_lsmr_inputs<A: Operator + ?Sized>(
    operator: &A,
    b: &[f64],
    tol: f64,
) -> Result<(), SolveError> {
    if b.len() != operator.nrows() {
        return Err(SolveError::InvalidInput {
            context: "lsmr",
            message: format!(
                "rhs length {} does not match operator row count {}",
                b.len(),
                operator.nrows()
            ),
        });
    }
    if !tol.is_finite() || tol < 0.0 {
        return Err(SolveError::InvalidInput {
            context: "lsmr",
            message: format!("tolerance must be finite and nonnegative, got {tol}"),
        });
    }
    if let Some((index, value)) = b.iter().copied().enumerate().find(|(_, v)| !v.is_finite()) {
        return Err(SolveError::InvalidInput {
            context: "lsmr",
            message: format!("rhs entry {index} must be finite, got {value}"),
        });
    }
    Ok(())
}
