//! LSMR for rectangular least-squares (`min ‖b − A x‖₂`).
//!
//! Two entry points:
//! - [`lsmr`] — standard Golub-Kahan bidiagonalization, no preconditioner.
//! - [`mlsmr`] — preconditioned variant with composable warm starts and escalation.

mod bidiag;
#[cfg(test)]
mod fixtures;
mod recurrence;
#[cfg(test)]
mod tests;

use std::borrow::Cow;

use crate::{Operator, SolveError};
use bidiag::{BidiagStep, Bidiagonalization, GolubKahan, ModifiedGolubKahan};
use recurrence::{ConvergenceCriteria, LsmrRecurrenceState, RotationStep, SolutionState, Stop};

/// Euclidean norm of a vector.
///
/// Max-scaled so the squared sum can't overflow for large-magnitude vectors.
#[inline]
pub(crate) fn vec_norm(v: &[f64]) -> f64 {
    let scale = v.iter().fold(0.0f64, |m, &x| m.max(x.abs()));
    if scale == 0.0 {
        // f64::max ignores NaN, so all-{zero,NaN} vectors land here: propagate
        // NaN instead of laundering it into a finite-looking zero norm.
        return if v.iter().any(|x| x.is_nan()) {
            f64::NAN
        } else {
            0.0
        };
    }
    scale * v.iter().map(|&x| (x / scale).powi(2)).sum::<f64>().sqrt()
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
    /// Per-run relative normal-equation residual, measured in `M`'s metric when preconditioned.
    pub normal_eq_residual: f64,
    /// Reason the solver stopped.
    pub stop_reason: LsmrStopReason,
}

/// Reason an LSMR solve stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LsmrStopReason {
    /// The right-hand side was exactly zero.
    ZeroRhs,
    /// The initial normal-equation residual was exactly zero: `Aᵀ(b − A x0) = 0`, cold `Aᵀb = 0`.
    InitialNormalEquationResidualZero,
    /// The least-squares residual estimate met the absolute tolerance.
    ResidualTolerance,
    /// The estimate `‖Aᵀrₖ‖ / (‖A‖ ‖rₖ‖)` met the relative tolerance.
    NormalEquationTolerance,
    /// The warm start already solved the system: `b − A x0` was exactly zero.
    WarmStartExact,
    /// The iteration budget was exhausted before convergence.
    MaxIterations,
    /// The [`EscalationHandler`] requested a handoff to a stronger preconditioner.
    Escalated,
}

/// One completed iteration's progress.
#[derive(Clone, Copy, Debug)]
pub struct Progress {
    /// 1-based index of the iteration just completed.
    pub iteration: usize,
    /// [`LsmrResult::normal_eq_residual`] as of this iteration.
    pub normal_eq_residual: f64,
}

/// Immutable factory for per-run escalation state.
pub trait EscalationPolicy: Send + Sync {
    /// A handler holding this policy's per-run state.
    fn handler(&self) -> Box<dyn EscalationHandler>;
}

/// Mutable escalation state for one solve run.
pub trait EscalationHandler {
    /// Whether to hand off now.
    fn should_escalate(&mut self, progress: Progress) -> bool;
}

/// Escalates after `window` consecutive contraction ratios exceed `threshold`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Staleness {
    window: usize,
    threshold: f64,
}

/// Invalid configuration for [`Staleness`].
#[derive(Clone, Copy, Debug, thiserror::Error)]
#[non_exhaustive]
pub enum StalenessError {
    /// The contraction window was empty.
    #[error("staleness window must be positive")]
    ZeroWindow,
    /// The contraction threshold fell outside `[0, 1)`.
    #[error("staleness threshold must be in [0, 1), got {threshold}")]
    InvalidThreshold {
        /// Rejected threshold value.
        threshold: f64,
    },
}

impl Staleness {
    /// `window` consecutive contractions must all exceed `threshold`, which lies in `[0, 1)`.
    pub fn try_new(window: usize, threshold: f64) -> Result<Self, StalenessError> {
        if window == 0 {
            return Err(StalenessError::ZeroWindow);
        }
        // Ratios are `|s̄ₖ| ≤ 1` by construction, so a threshold of 1 or more never escalates.
        if !(0.0..1.0).contains(&threshold) {
            return Err(StalenessError::InvalidThreshold { threshold });
        }
        Ok(Self { window, threshold })
    }
}

impl EscalationPolicy for Staleness {
    fn handler(&self) -> Box<dyn EscalationHandler> {
        Box::new(StalenessRun {
            policy: *self,
            previous: f64::NAN,
            stalled: 0,
        })
    }
}

struct StalenessRun {
    policy: Staleness,
    previous: f64,
    stalled: usize,
}

impl EscalationHandler for StalenessRun {
    fn should_escalate(&mut self, progress: Progress) -> bool {
        let current = progress.normal_eq_residual;
        if self.previous.is_finite() && self.previous > 0.0 {
            if current / self.previous > self.policy.threshold {
                self.stalled += 1;
            } else {
                self.stalled = 0;
            }
        }
        self.previous = current;
        self.stalled >= self.policy.window
    }
}

/// Optional behaviors for [`mlsmr`].
#[derive(Clone, Copy, Default)]
pub struct MlsmrOptions<'a> {
    /// Initial iterate for a residual correction; tolerances remain relative to the original `‖b‖`.
    pub warm_start: Option<&'a [f64]>,
    /// Hands off to a stronger preconditioner mid-run; see [`EscalationPolicy`].
    pub escalation: Option<&'a dyn EscalationPolicy>,
    /// Local reorthogonalization window; `None` disables it.
    pub local_size: Option<usize>,
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
            normal_eq_residual: 0.0,
            stop_reason: LsmrStopReason::ZeroRhs,
        });
    }

    let local_size = local_size.unwrap_or(0);
    let (bidiag, step1) = GolubKahan::init(operator, b, local_size)?;
    let criteria = ConvergenceCriteria::new(b_norm, tol);
    lsmr_from_bidiag(bidiag, step1, b_norm, criteria, maxiter, None)
}

/// Preconditioned LSMR with `M ≈ AᵀA` and one `M⁻¹` application per iteration.
pub fn mlsmr<A: Operator + ?Sized, M: Operator + ?Sized>(
    operator: &A,
    b: &[f64],
    preconditioner: &M,
    tol: f64,
    maxiter: usize,
    options: MlsmrOptions<'_>,
) -> Result<LsmrResult, SolveError> {
    validate_lsmr_inputs(operator, b, tol)?;
    let n = operator.ncols();
    if preconditioner.nrows() != n || preconditioner.ncols() != n {
        return Err(invalid_input(format!(
            "preconditioner shape {}x{} must match operator column count {n}",
            preconditioner.nrows(),
            preconditioner.ncols(),
        )));
    }
    let MlsmrOptions {
        warm_start,
        escalation,
        local_size,
    } = options;

    if let Some(x0) = warm_start {
        if x0.len() != n {
            return Err(invalid_input(format!(
                "warm-start length {} does not match operator column count {n}",
                x0.len()
            )));
        }
        if let Some((index, value)) = x0.iter().copied().enumerate().find(|(_, v)| !v.is_finite()) {
            return Err(invalid_input(format!(
                "warm-start entry {index} must be finite, got {value}"
            )));
        }
    }

    let b_norm = vec_norm(b);
    let local_size = local_size.unwrap_or(0);

    let rhs: Cow<'_, [f64]> = match warm_start {
        None => Cow::Borrowed(b),
        Some(x0) => {
            let mut residual = vec![0.0; operator.nrows()];
            operator.apply(x0, &mut residual)?;
            for (ri, &bi) in residual.iter_mut().zip(b) {
                *ri = bi - *ri;
            }
            Cow::Owned(residual)
        }
    };
    let rhs_norm = vec_norm(&rhs);
    // Unlike `b`, `rhs` is computed: an ∞ entry norms to NaN, which reads as β₁ = 0 downstream.
    if !rhs_norm.is_finite() {
        return Err(invalid_input(format!(
            "warm-start residual b - A·x0 has non-finite norm {rhs_norm}"
        )));
    }
    if rhs_norm == 0.0 {
        let (x, stop_reason) = match warm_start {
            Some(x0) => (x0.to_vec(), LsmrStopReason::WarmStartExact),
            None => (vec![0.0; n], LsmrStopReason::ZeroRhs),
        };
        return Ok(LsmrResult {
            x,
            converged: true,
            iterations: 0,
            residual_norm: 0.0,
            normal_eq_residual: 0.0,
            stop_reason,
        });
    }

    let (bidiag, step1) = ModifiedGolubKahan::init(operator, preconditioner, &rhs, local_size)?;
    let criteria = ConvergenceCriteria::new(b_norm, tol);
    let mut result = lsmr_from_bidiag(
        bidiag,
        step1,
        rhs_norm,
        criteria,
        maxiter,
        escalation.map(|policy| policy.handler()),
    )?;
    if let Some(x0) = warm_start {
        for (xi, &x0i) in result.x.iter_mut().zip(x0) {
            *xi += x0i;
        }
    }
    Ok(result)
}

/// Runs the LSMR recurrences over a preconditioner-specific bidiagonalization stream.
fn lsmr_from_bidiag<B: Bidiagonalization>(
    mut bidiag: B,
    step1: BidiagStep,
    rhs_norm: f64,
    criteria: ConvergenceCriteria,
    maxiter: usize,
    mut escalation: Option<Box<dyn EscalationHandler>>,
) -> Result<LsmrResult, SolveError> {
    let n = bidiag.v().len();
    if step1.alpha == 0.0 {
        return Ok(LsmrResult {
            x: vec![0.0; n],
            converged: true,
            iterations: 0,
            residual_norm: rhs_norm,
            normal_eq_residual: 0.0,
            stop_reason: LsmrStopReason::InitialNormalEquationResidualZero,
        });
    }

    let mut convergence = criteria.start(step1.alpha);
    let mut recurrence = LsmrRecurrenceState::init(step1);
    let mut solution = SolutionState::init(bidiag.v());
    let mut prev_rot = RotationStep::initial();

    for itn in 1..=maxiter {
        let step = bidiag.step()?;
        convergence.observe(step);
        let curr_rot = recurrence.step(step);
        solution.update(bidiag.v(), curr_rot, prev_rot);

        // The tolerance test catches breakdown when the residual recurrences collapse.
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
                normal_eq_residual: recurrence.relative_normal_eq_residual(),
                stop_reason,
            });
        }
        if let Some(rule) = escalation.as_deref_mut() {
            let progress = Progress {
                iteration: itn,
                normal_eq_residual: recurrence.relative_normal_eq_residual(),
            };
            if rule.should_escalate(progress) {
                return Ok(LsmrResult {
                    x: solution.into_x(),
                    converged: false,
                    iterations: itn,
                    residual_norm: recurrence.residual_estimate(),
                    normal_eq_residual: progress.normal_eq_residual,
                    stop_reason: LsmrStopReason::Escalated,
                });
            }
        }
        prev_rot = curr_rot;
    }

    Ok(LsmrResult {
        x: solution.into_x(),
        converged: false,
        iterations: maxiter,
        residual_norm: recurrence.residual_estimate(),
        normal_eq_residual: recurrence.relative_normal_eq_residual(),
        stop_reason: LsmrStopReason::MaxIterations,
    })
}

fn validate_lsmr_inputs<A: Operator + ?Sized>(
    operator: &A,
    b: &[f64],
    tol: f64,
) -> Result<(), SolveError> {
    if b.len() != operator.nrows() {
        return Err(invalid_input(format!(
            "rhs length {} does not match operator row count {}",
            b.len(),
            operator.nrows()
        )));
    }
    if !tol.is_finite() || tol < 0.0 {
        return Err(invalid_input(format!(
            "tolerance must be finite and nonnegative, got {tol}"
        )));
    }
    if let Some((index, value)) = b.iter().copied().enumerate().find(|(_, v)| !v.is_finite()) {
        return Err(invalid_input(format!(
            "rhs entry {index} must be finite, got {value}"
        )));
    }
    Ok(())
}

fn invalid_input(message: String) -> SolveError {
    SolveError::InvalidInput {
        context: "lsmr",
        message,
    }
}
