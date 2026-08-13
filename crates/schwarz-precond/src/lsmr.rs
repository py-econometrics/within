//! LSMR for rectangular least-squares (`min ‖b − A x‖₂`).
//!
//! Two entry points:
//! - [`lsmr`] — standard Golub-Kahan bidiagonalization, no preconditioner.
//! - [`mlsmr`] — Modified Golub-Kahan variant preconditioned with `M ≈ AᵀA`;
//!   requires a single `M⁻¹` application per iteration. [`MlsmrOptions`] adds a
//!   warm start and an [`EscalationRule`], which compose into a preconditioner
//!   ladder: run a cheap `M`, escalate when it stops paying, warm-start the next.

mod bidiag;
mod recurrence;
#[cfg(test)]
mod tests;

use crate::{Operator, SolveError};
use bidiag::{BidiagStep, Bidiagonalization, GolubKahan, ModifiedGolubKahan};
use recurrence::{ConvergenceState, LsmrRecurrenceState, RotationStep, SolutionState, Stop};

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
    /// Normal-equation residual estimate relative to the START OF THIS RUN,
    /// recovered from the recurrence scalars (`|ζ̄ₖ| / |ζ̄₀|`, Fong & Saunders) at
    /// no extra cost. Cold, that reference is `‖Aᵀb‖`; warm-started from `x0` it
    /// is `‖Aᵀ(b − A x0)‖`, so each rung of a ladder reports its own progress.
    /// Preconditioned, it is measured in `M`'s metric. It is NOT the quantity
    /// `tol` is compared against — see [`LsmrStopReason::NormalEquationTolerance`].
    pub normal_eq_residual: f64,
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
    /// The normal-equation residual estimate met the relative tolerance:
    /// `‖Aᵀrₖ‖ / (‖A‖ ‖rₖ‖) ≤ tol`.
    NormalEquationTolerance,
    /// The warm start already solved the system: `b − A x0` was exactly zero.
    WarmStartExact,
    /// The iteration budget was exhausted before convergence.
    MaxIterations,
    /// The [`EscalationRule`] asked to hand off to a stronger preconditioner.
    /// Distinct from [`Self::MaxIterations`], which means the caller's whole
    /// budget is gone rather than this rung's turn being over.
    Escalated,
}

/// One iteration's numerical progress, handed to an [`EscalationRule`].
/// Wall-clock is absent: a rule that prices a switch holds its own `Instant`.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct Progress {
    /// 1-based index of the iteration just completed.
    pub iteration: usize,
    /// [`LsmrResult::normal_eq_residual`] as of this iteration.
    pub normal_eq_residual: f64,
}

/// Decides when a preconditioner has stopped paying its way.
///
/// Consulted once per iteration, always AFTER the convergence test, so a solve
/// that finishes is never reported as escalated. Returning `true` stops the run
/// with [`LsmrStopReason::Escalated`]; the returned `x` is a valid warm start
/// for the next rung.
pub trait EscalationRule {
    /// Whether to hand off now.
    fn should_escalate(&mut self, progress: Progress) -> bool;
}

/// Escalate once `window` consecutive contractions of
/// [`Progress::normal_eq_residual`] all exceed `threshold`. Reads only the
/// solver's own history, so it needs no cost model for what it escalates to.
pub struct Staleness {
    window: usize,
    threshold: f64,
    previous: f64,
    stalled: usize,
}

impl Staleness {
    /// `window` consecutive contractions must all exceed `threshold` to escalate.
    #[must_use]
    pub fn new(window: usize, threshold: f64) -> Self {
        assert!(window > 0, "staleness window must be positive");
        Self {
            window,
            threshold,
            previous: f64::NAN,
            stalled: 0,
        }
    }
}

impl EscalationRule for Staleness {
    fn should_escalate(&mut self, progress: Progress) -> bool {
        let current = progress.normal_eq_residual;
        if self.previous.is_finite() && self.previous > 0.0 {
            if current / self.previous > self.threshold {
                self.stalled += 1;
            } else {
                self.stalled = 0;
            }
        }
        self.previous = current;
        self.stalled >= self.window
    }
}

/// Optional behaviours of [`mlsmr`]; `..Default::default()` selects a plain
/// cold solve run to convergence.
#[derive(Default)]
pub struct MlsmrOptions<'a> {
    /// Initial iterate. The solve runs on the residual system `min ‖(b − A x0) − A d‖`
    /// and returns `x0 + d`, reaching the same `x*` as a cold solve. Stopping
    /// tolerances stay measured against the original `‖b‖`.
    pub warm_start: Option<&'a [f64]>,
    /// Hands off to a stronger preconditioner mid-run; see [`EscalationRule`].
    pub escalation: Option<&'a mut dyn EscalationRule>,
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
    lsmr_from_bidiag(bidiag, step1, b_norm, b_norm, tol, maxiter, None)
}

/// Preconditioned LSMR with `M ≈ AᵀA`.
///
/// Uses the Modified Golub-Kahan variant requiring one `M⁻¹` application per
/// iteration. `M` is baked into the recurrence and cannot be swapped mid-run;
/// [`MlsmrOptions::warm_start`] is how an iterate survives a change of
/// preconditioner, and combining it with [`MlsmrOptions::escalation`] gives a
/// ladder of arbitrary depth: run each rung with the previous rung's `x` as
/// `warm_start`, stopping at the first that does not report
/// [`LsmrStopReason::Escalated`].
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
        return Err(SolveError::InvalidInput {
            context: "lsmr",
            message: format!(
                "preconditioner shape {}x{} must match operator column count {n}",
                preconditioner.nrows(),
                preconditioner.ncols(),
            ),
        });
    }
    let MlsmrOptions {
        warm_start,
        escalation,
        local_size,
    } = options;

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

    let Some(x0) = warm_start else {
        let (bidiag, step1) = ModifiedGolubKahan::init(operator, preconditioner, b, local_size)?;
        return lsmr_from_bidiag(bidiag, step1, b_norm, b_norm, tol, maxiter, escalation);
    };

    if x0.len() != n {
        return Err(SolveError::InvalidInput {
            context: "lsmr",
            message: format!(
                "warm-start length {} does not match operator column count {n}",
                x0.len()
            ),
        });
    }
    if let Some((index, value)) = x0.iter().copied().enumerate().find(|(_, v)| !v.is_finite()) {
        return Err(SolveError::InvalidInput {
            context: "lsmr",
            message: format!("warm-start entry {index} must be finite, got {value}"),
        });
    }

    let mut r = vec![0.0; operator.nrows()];
    operator.apply(x0, &mut r)?;
    for (ri, &bi) in r.iter_mut().zip(b) {
        *ri = bi - *ri;
    }

    let r_norm = vec_norm(&r);
    if r_norm == 0.0 {
        return Ok(LsmrResult {
            x: x0.to_vec(),
            converged: true,
            iterations: 0,
            residual_norm: 0.0,
            normal_eq_residual: 0.0,
            stop_reason: LsmrStopReason::WarmStartExact,
        });
    }

    let (bidiag, step1) = ModifiedGolubKahan::init(operator, preconditioner, &r, local_size)?;
    let mut result = lsmr_from_bidiag(bidiag, step1, r_norm, b_norm, tol, maxiter, escalation)?;
    for (xi, &x0i) in result.x.iter_mut().zip(x0) {
        *xi += x0i;
    }
    Ok(result)
}

/// Run the LSMR scalar/vector recurrences over a bidiagonalization stream.
/// Generic over the bidiagonalization, which is the only place the choice
/// of preconditioner enters.
/// `tol_ref_norm` is `rhs_norm` except on a warm restart, which keeps the original `‖b‖`.
fn lsmr_from_bidiag<B: Bidiagonalization>(
    mut bidiag: B,
    step1: BidiagStep,
    rhs_norm: f64,
    tol_ref_norm: f64,
    tol: f64,
    maxiter: usize,
    mut escalation: Option<&mut dyn EscalationRule>,
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

    let mut recurrence = LsmrRecurrenceState::init(step1);
    let mut solution = SolutionState::init(bidiag.v());
    let mut convergence = ConvergenceState::new(tol_ref_norm, tol, step1.alpha);
    let mut prev_rot = RotationStep::initial();

    for itn in 1..=maxiter {
        let step = bidiag.step()?;
        convergence.observe(step);
        let curr_rot = recurrence.step(step);
        solution.update(bidiag.v(), curr_rot, prev_rot);

        // A bidiagonalization breakdown needs no explicit handling: `‖Aᵀr‖`
        // and `‖r‖` are proportional to the bidiagonal entries α, β, so they
        // vanish on the same step the recurrence collapses (α or β = 0). The
        // tolerance check therefore always catches a breakdown as a converged
        // solve.
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
