//! LSMR for rectangular least-squares (`min ‖b − A x‖₂`).
//!
//! Two entry points:
//! - [`lsmr`] — standard Golub-Kahan bidiagonalization, no preconditioner.
//! - [`mlsmr`] — preconditioned variant with composable warm starts and escalation.

mod bidiag;
#[cfg(test)]
mod fixtures;
mod magnitude;
mod recurrence;
#[cfg(test)]
mod tests;

use std::borrow::Cow;

use crate::{Operator, SolveError};
use bidiag::{
    axpby, metric_gradient_norm, residual_into, BidiagStep, Bidiagonalization, GolubKahan,
    ModifiedGolubKahan,
};
use magnitude::Magnitude;
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
    /// Whether a tolerance stop matched `‖b − A x‖`, which misses drift within `range(A)`.
    pub converged: bool,
    /// Total number of iterations performed.
    pub iterations: usize,
    /// `‖b − A x‖`: recomputed at a tolerance stop, the recurrence's estimate at any other.
    pub residual_norm: f64,
    /// `‖Âᵀ(b − A x)‖ / ‖Âᵀb‖`, measured in `M`'s metric when preconditioned.
    pub normal_eq_residual: f64,
    /// Reason the solver stopped.
    pub stop_reason: LsmrStopReason,
    /// `b − A x` itself, present only where a tolerance stop recomputed it.
    pub true_residual: Option<Vec<f64>>,
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
    /// A tolerance stop, and each restart from it, that `‖b − A x‖` refuted.
    FalseConvergence,
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
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
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

    /// Consecutive stalled iterations required before escalating.
    pub fn window(&self) -> usize {
        self.window
    }

    /// Contraction ratio above which an iteration counts as stalled.
    pub fn threshold(&self) -> f64 {
        self.threshold
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for Staleness {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct Helper {
            window: usize,
            threshold: f64,
        }

        let helper = Helper::deserialize(deserializer)?;
        Self::try_new(helper.window, helper.threshold).map_err(serde::de::Error::custom)
    }
}

impl Default for Staleness {
    /// Escalates after four consecutive residual reductions below 30% (ratio above 0.7).
    fn default() -> Self {
        Self {
            window: 4,
            threshold: 0.7,
        }
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
    /// Residual-correction start; tolerances stay relative to `‖b‖` (`‖b − A x₀‖` if `b = 0`).
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

    let b_norm = finite(vec_norm(b), "rhs norm")?;
    if b_norm == 0.0 {
        return Ok(LsmrResult {
            x: vec![0.0; n],
            converged: true,
            iterations: 0,
            residual_norm: 0.0,
            normal_eq_residual: 0.0,
            stop_reason: LsmrStopReason::ZeroRhs,
            true_residual: None,
        });
    }

    let local_size = local_size.unwrap_or(0);
    let (bidiag, step1) = GolubKahan::init(operator, b, b_norm, local_size)?;
    let criteria = ConvergenceCriteria::new(b_norm, tol);
    lsmr_from_bidiag(bidiag, step1, b, None, criteria, maxiter, None)
}

/// Preconditioned LSMR with `M ≈ AᵀA`, one `M⁻¹` apply per iteration; `M⁻¹` must be nonsingular.
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

    // `b` is finite entrywise, but its norm sets the tolerance and an ∞ there certifies anything.
    let b_norm = finite(vec_norm(b), "rhs norm")?;
    let local_size = local_size.unwrap_or(0);

    let (rhs, rhs_norm): (Cow<'_, [f64]>, f64) = match warm_start {
        None => (Cow::Borrowed(b), b_norm),
        Some(x0) => {
            let mut residual = vec![0.0; operator.nrows()];
            // Unlike `b`, the residual is computed: an ∞ entry norms to NaN, read as β₁ = 0 downstream.
            let norm = finite(
                residual_into(operator, x0, b, &mut residual)?,
                "warm-start residual norm",
            )?;
            (Cow::Owned(residual), norm)
        }
    };
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
            true_residual: None,
        });
    }

    // `‖Aᵀb‖` must be taken before the stream exists: the stream's own query clobbers `v₁`.
    let metric = match warm_start {
        None => None,
        Some(_) => Some(metric_gradient_norm(
            operator,
            preconditioner,
            b,
            &mut vec![0.0; n],
            &mut vec![0.0; n],
        )?),
    };
    let (bidiag, step1) =
        ModifiedGolubKahan::init(operator, preconditioner, &rhs, rhs_norm, local_size)?;
    let warm_start = warm_start.zip(metric).map(|(x0, metric)| WarmStart {
        x0,
        reference: NormalEqReference::warm(Magnitude::from(metric), step1),
    });
    let reference_norm = if b_norm > 0.0 { b_norm } else { rhs_norm };
    let criteria = ConvergenceCriteria::new(reference_norm, tol);
    lsmr_from_bidiag(bidiag, step1, b, warm_start, criteria, maxiter, escalation)
}

/// Restarts from a refuted tolerance stop before the solve is refused outright.
const MAX_RESTARTS: usize = 2;

/// `‖Aᵀb‖` in the stream's metric, fixed for the solve so every pass reports against it.
#[derive(Clone, Copy)]
struct NormalEqReference(Magnitude);

impl NormalEqReference {
    /// A cold stream's first `(α₁, β₁)` is `‖Aᵀb‖` already, though the product may not be a double.
    fn cold(step1: BidiagStep) -> Self {
        Self(Magnitude::product(step1.alpha, step1.beta))
    }

    /// A reference that carries no information falls back to the cold `α₁β₁`.
    fn warm(metric: Magnitude, step1: BidiagStep) -> Self {
        if metric.is_normal() {
            Self(metric)
        } else {
            Self::cold(step1)
        }
    }

    fn relative(self, estimate: Magnitude) -> f64 {
        debug_assert!(self.0.is_normal(), "a zero α₁ returns before any report");
        (estimate / self.0).to_f64()
    }
}

/// A warm start and its `‖Aᵀb‖`, since the stream's `(α₁, β₁)` measures `b − A x₀` instead.
struct WarmStart<'a> {
    x0: &'a [f64],
    reference: NormalEqReference,
}

/// Runs LSMR on the correction to `x0`, restarting from any tolerance stop `b − A x` refutes.
fn lsmr_from_bidiag<B: Bidiagonalization>(
    mut bidiag: B,
    mut step1: BidiagStep,
    b: &[f64],
    warm_start: Option<WarmStart<'_>>,
    criteria: ConvergenceCriteria,
    maxiter: usize,
    escalation: Option<&dyn EscalationPolicy>,
) -> Result<LsmrResult, SolveError> {
    let n = bidiag.v().len();
    let reference = warm_start
        .as_ref()
        .map_or_else(|| NormalEqReference::cold(step1), |w| w.reference);
    let mut base: Option<Cow<'_, [f64]>> = warm_start.map(|w| Cow::Borrowed(w.x0));
    let mut iterations = 0;
    let mut restarts = 0;
    loop {
        // A metric reporting no gradient at all may be hiding one outside itself.
        if step1.alpha == 0.0 {
            let x = base.map_or_else(|| vec![0.0; n], Cow::into_owned);
            let (converged, normal_eq_residual) = match bidiag.hidden_gradient()? {
                None => (true, 0.0),
                Some(per_unit_residual) => {
                    let normar = Magnitude::product(step1.beta, per_unit_residual);
                    let plain = bidiag.plain_gradient(b)?;
                    let a_norm_below = plain / vec_norm(b).max(f64::MIN_POSITIVE);
                    let informative = plain > 0.0 && plain.is_finite();
                    (
                        criteria.corroborates(step1.beta, normar, a_norm_below),
                        if informative {
                            (normar / Magnitude::from(plain)).to_f64()
                        } else {
                            1.0
                        },
                    )
                }
            };
            return Ok(LsmrResult {
                x,
                converged,
                iterations,
                residual_norm: step1.beta,
                normal_eq_residual,
                stop_reason: if converged {
                    LsmrStopReason::InitialNormalEquationResidualZero
                } else {
                    LsmrStopReason::FalseConvergence
                },
                true_residual: None,
            });
        }

        // A pass reports its drop from its own ζ̄₀, so a restarted one needs a handler that agrees.
        let mut escalation = escalation.map(EscalationPolicy::handler);
        let mut convergence = criteria.start(step1.alpha);
        let mut recurrence = LsmrRecurrenceState::init(step1);
        let mut solution = SolutionState::init(bidiag.v(), step1.beta);
        let mut prev_rot = RotationStep::initial();
        let stop_reason = 'pass: {
            while iterations < maxiter {
                iterations += 1;
                let step = bidiag.step()?;
                convergence.observe(step);
                let curr_rot = recurrence.step(step);
                solution.update(bidiag.v(), curr_rot, prev_rot);

                // The tolerance test catches breakdown when the residual recurrences collapse.
                match convergence.check(&recurrence) {
                    Stop::Continue => {}
                    Stop::ResidualTolerance => break 'pass LsmrStopReason::ResidualTolerance,
                    Stop::NormalEquationTolerance => {
                        break 'pass LsmrStopReason::NormalEquationTolerance
                    }
                }
                if let Some(rule) = escalation.as_deref_mut() {
                    let progress = Progress {
                        iteration: iterations,
                        normal_eq_residual: recurrence.relative_normal_eq_residual(),
                    };
                    if rule.should_escalate(progress) {
                        break 'pass LsmrStopReason::Escalated;
                    }
                }
                prev_rot = curr_rot;
            }
            LsmrStopReason::MaxIterations
        };

        let mut x = solution.into_x();
        if let Some(base) = &base {
            axpby(&mut x, base, 1.0, 1.0);
        }
        // Only tolerance stops measure `x`, and a non-finite entry never recovers.
        if let Some((index, value)) = x.iter().copied().enumerate().find(|(_, v)| !v.is_finite()) {
            return Err(invalid_input(format!(
                "non-finite solution entry {index} ({value})"
            )));
        }
        let mut result = LsmrResult {
            x,
            converged: false,
            iterations,
            residual_norm: recurrence.residual_estimate(),
            normal_eq_residual: reference.relative(recurrence.normal_eq_residual_estimate()),
            stop_reason,
            true_residual: None,
        };
        // Only a tolerance stop claims convergence, so only it is worth a true-residual evaluation.
        if matches!(
            stop_reason,
            LsmrStopReason::ResidualTolerance | LsmrStopReason::NormalEquationTolerance
        ) {
            let residual_norm = bidiag.residual_norm(&result.x, b)?;
            result.converged = criteria.corroborates_residual(residual_norm, result.residual_norm);
            if !result.converged
                && residual_norm.is_finite()
                && restarts < MAX_RESTARTS
                && iterations < maxiter
            {
                restarts += 1;
                step1 = bidiag.restart(residual_norm)?;
                base = Some(Cow::Owned(result.x));
                continue;
            }
            if !result.converged {
                result.stop_reason = LsmrStopReason::FalseConvergence;
                // The estimate is the refuted claim; a seed from the staged residual measures `x`.
                result.normal_eq_residual = if residual_norm.is_finite() {
                    let step = bidiag.restart(residual_norm)?;
                    reference.relative(Magnitude::product(step.alpha, step.beta))
                } else {
                    residual_norm
                };
            }
            result.residual_norm = residual_norm;
            // A refused stop's reseed normalized the staged residual in place.
            if result.converged {
                result.true_residual = Some(bidiag.into_residual());
            }
        }
        return Ok(result);
    }
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

/// Every comparison against a non-finite value is false, so it reads as a breakdown downstream.
fn finite(value: f64, what: &str) -> Result<f64, SolveError> {
    if !value.is_finite() {
        return Err(invalid_input(format!("non-finite {what} ({value})")));
    }
    Ok(value)
}

fn invalid_input(message: String) -> SolveError {
    SolveError::InvalidInput {
        context: "lsmr",
        message,
    }
}
