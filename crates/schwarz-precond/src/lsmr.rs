//! LSMR for `min ‖b − A x‖₂`; [`mlsmr`] with [`IdentityOp`] is unpreconditioned.

mod bidiag;
#[cfg(test)]
mod fixtures;
mod magnitude;
mod recurrence;
#[cfg(test)]
mod tests;

use std::borrow::Cow;

use crate::{MemBuffer, MemStack, Operator, SolveError};
use bidiag::{axpby, normalize, par_norm, BidiagStep, Bidiagonalization, StagedStart};
use magnitude::Magnitude;
use recurrence::{ConvergenceCriteria, LsmrRecurrenceState, RotationStep, SolutionState};

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

impl LsmrResult {
    /// Whether the run stopped certified: a tolerance stop that survived its audits, a zero
    /// right-hand side, an exact start, or a pass that began at a zero normal-equation residual.
    pub fn converged(&self) -> bool {
        matches!(
            self.stop_reason,
            LsmrStopReason::ZeroRhs
                | LsmrStopReason::InitialNormalEquationResidualZero
                | LsmrStopReason::ResidualTolerance
                | LsmrStopReason::NormalEquationTolerance
                | LsmrStopReason::WarmStartExact
        )
    }
}

/// Reason an LSMR solve stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LsmrStopReason {
    /// The right-hand side was exactly zero.
    ZeroRhs,
    /// The start's gradient vanished in the preconditioner's metric (`α₁ = 0`), and the measured
    /// `‖r‖` or `‖Aᵀr‖ / (L‖r‖)` certifies the start itself.
    InitialNormalEquationResidualZero,
    /// A measured `‖rₖ‖ ≤ tol·‖b‖`; for a zero `b`, `tol·‖b − A x₀‖`, as no backward error below
    /// `1/κ(A)` exists there.
    ResidualTolerance,
    /// `‖Aᵀrₖ‖ / (‖A‖₂ ‖rₖ‖)` met the relative tolerance, certified by measured `rₖ` and `Aᵀrₖ`.
    NormalEquationTolerance,
    /// The iteration budget was exhausted before convergence.
    MaxIterations,
    /// The warm start already solved the system: `b − A x0` was exactly zero.
    WarmStartExact,
    /// A tolerance stop, and each restart from it, refuted by its audit: the measured `‖b − A x‖`
    /// and `‖Aᵀ(b − A x)‖ / (L ‖b − A x‖)`, with `L` the [`norm_floor`](crate::Operator::norm_floor),
    /// both missed the tolerance, or `‖b − A x‖` was not finite; or an `α₁ = 0` start whose hidden
    /// gradient misses it.
    FalseConvergence,
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
    /// Hands off to a stronger preconditioner mid-run; see [`EscalationPolicy`].
    pub escalation: Option<&'a dyn EscalationPolicy>,
    /// Local reorthogonalization window; `None` disables it.
    pub local_size: Option<usize>,
}

/// `M = I`: [`mlsmr`] with it is plain LSMR.
#[derive(Clone, Copy, Debug)]
pub struct IdentityOp {
    /// Dimension of the solution.
    pub n: usize,
}

impl Operator for IdentityOp {
    fn nrows(&self) -> usize {
        self.n
    }
    fn ncols(&self) -> usize {
        self.n
    }
    fn apply_with(&self, x: &[f64], y: &mut [f64], _: &mut MemStack) -> Result<(), SolveError> {
        y.copy_from_slice(x);
        Ok(())
    }
    fn apply_adjoint_with(
        &self,
        x: &[f64],
        y: &mut [f64],
        stack: &mut MemStack,
    ) -> Result<(), SolveError> {
        self.apply_with(x, y, stack)
    }
}

/// Preconditioned LSMR with `M ≈ AᵀA`, one `M⁻¹` apply per iteration; `M⁻¹` must be nonsingular.
///
/// The run spends at most `maxiter` iterations. With a `start` x₀ it solves for the correction
/// to x₀; tolerances stay relative to `‖b‖` (`‖b − A x₀‖` if `b = 0`).
pub fn mlsmr<A: Operator + ?Sized, M: Operator + ?Sized>(
    operator: &A,
    b: &[f64],
    preconditioner: &M,
    tol: f64,
    maxiter: usize,
    start: Option<&[f64]>,
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
        escalation,
        local_size,
    } = options;

    if let Some(x0) = start {
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

    // A zero start vector needs no stream; a warm one must be formed in the stream to be measured.
    let staged = match start {
        None => (b_norm > 0.0).then(|| {
            (
                StagedStart::cold(operator, preconditioner, b, local_size),
                b_norm,
            )
        }),
        Some(x0) => {
            let (staged, norm) = StagedStart::warm(operator, preconditioner, x0, b, local_size)?;
            // Unlike `b`, the residual is computed: an ∞ entry norms to NaN, read as β₁ = 0 downstream.
            let norm = finite(norm, "warm-start residual norm")?;
            (norm > 0.0).then_some((staged, norm))
        }
    };
    let Some((mut staged, rhs_norm)) = staged else {
        let (x, stop_reason) = match start {
            Some(x0) => (x0.to_vec(), LsmrStopReason::WarmStartExact),
            None => (vec![0.0; n], LsmrStopReason::ZeroRhs),
        };
        return Ok(LsmrResult {
            x,
            iterations: 0,
            residual_norm: 0.0,
            normal_eq_residual: 0.0,
            stop_reason,
            true_residual: None,
        });
    };

    let metric = match start {
        Some(_) if b_norm > 0.0 => Some(staged.metric_reference(b, b_norm)?),
        _ => None,
    };
    let (bidiag, step1) = staged.restart(rhs_norm)?;
    let reference = NormalEqReference::new(metric, step1);
    let solve_start = SolveStart {
        x0: start,
        reference,
    };
    let reference_norm = if b_norm > 0.0 { b_norm } else { rhs_norm };
    let criteria = ConvergenceCriteria::new(reference_norm, tol);
    lsmr_from_bidiag(bidiag, step1, b, solve_start, criteria, maxiter, escalation)
}

/// Power-iteration steps behind [`Operator::norm_floor`]'s default.
const POWER_STEPS: usize = 8;

/// `max ‖A x̂‖, ‖Aᵀŷ‖` over the unit iterates of power iteration on `AᵀA`: each is `≤ ‖A‖₂`.
pub(crate) fn power_floor<O: Operator + ?Sized>(op: &O) -> Result<f64, SolveError> {
    // Not all ones: a projected operator such as `(I − P_e) A` annihilates exactly that structure.
    let mut x: Vec<f64> = (0..op.ncols() as u64).map(splitmix64_unit).collect();
    let mut y = vec![0.0; op.nrows()];
    let mut buffer = MemBuffer::new(op.scratch());
    let mut floor = 0.0f64;
    let mut x_norm = par_norm(&x);
    for _ in 0..POWER_STEPS {
        // Unit vectors on both sides keep `AᵀA x` from overflowing where `‖A‖₂ > 1e154`.
        if !x_norm.is_normal() {
            break;
        }
        normalize(&mut x, x_norm);
        op.apply_with(&x, &mut y, MemStack::new(&mut buffer))?;
        let y_norm = par_norm(&y);
        if !y_norm.is_normal() {
            break;
        }
        normalize(&mut y, y_norm);
        op.apply_adjoint_with(&y, &mut x, MemStack::new(&mut buffer))?;
        x_norm = par_norm(&x);
        floor = floor.max(y_norm);
        if x_norm.is_finite() {
            floor = floor.max(x_norm);
        }
    }
    Ok(floor)
}

/// Entry `i` of a fixed pseudo-random vector in `[-1, 1)`: the SplitMix64 output for seed `i + 1`
/// (Steele, Lea & Flood 2014), so the start needs no RNG dependency and is the same on every run.
fn splitmix64_unit(i: u64) -> f64 {
    let mut z = (i + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    // The top 53 bits as a fraction of 2⁵³, mapped from [0, 1) to [-1, 1).
    let unit = ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64;
    2.0 * unit - 1.0
}

/// Restarts from a refuted tolerance stop before it ends in [`LsmrStopReason::FalseConvergence`]; a run pays
/// for at most one more audit than this.
const MAX_RESTARTS: usize = 2;

/// Fixed metric `‖Âᵀb‖` reference for reporting.
#[derive(Clone, Copy)]
struct NormalEqReference {
    metric: Magnitude,
}

impl NormalEqReference {
    /// A usable warm `‖Âᵀb‖`, else the cold stream's first `α₁β₁`, which is `‖Âᵀ rhs‖` already.
    fn new(warm: Option<Magnitude>, step1: BidiagStep) -> Self {
        let metric = warm.filter(|m| m.is_normal());
        Self {
            metric: metric.unwrap_or_else(|| Magnitude::product(step1.alpha, step1.beta)),
        }
    }

    fn relative(self, estimate: Magnitude) -> f64 {
        debug_assert!(
            self.metric.is_normal(),
            "a zero α₁ returns before any report"
        );
        (estimate / self.metric).to_f64()
    }
}

/// Initial iterate and fixed references; warm streams measure `b − A x₀`, not `b`.
struct SolveStart<'a> {
    x0: Option<&'a [f64]>,
    reference: NormalEqReference,
}

/// Runs LSMR on the correction to `x0`, restarting from any tolerance stop `b − A x` refutes.
fn lsmr_from_bidiag<B: Bidiagonalization>(
    mut bidiag: B,
    mut step1: BidiagStep,
    b: &[f64],
    start: SolveStart<'_>,
    criteria: ConvergenceCriteria,
    maxiter: usize,
    escalation: Option<&dyn EscalationPolicy>,
) -> Result<LsmrResult, SolveError> {
    let n = bidiag.v().len();
    let reference = start.reference;
    let mut base: Option<Cow<'_, [f64]>> = start.x0.map(Cow::Borrowed);
    let mut iterations = 0;
    let mut restarts = 0;
    loop {
        // A metric reporting no gradient at all may be hiding one outside itself.
        if step1.alpha == 0.0 {
            let x = base.map_or_else(|| vec![0.0; n], Cow::into_owned);
            let b_norm = vec_norm(b);
            let (converged, normal_eq_residual) = match bidiag.hidden_gradient(b, b_norm)? {
                None => (true, 0.0),
                Some((per_unit_residual, plain)) => {
                    let normar = Magnitude::product(step1.beta, per_unit_residual);
                    let converged = criteria
                        .certify(step1.beta, || Ok((normar, bidiag.norm_floor()?)))?
                        .is_some();
                    let relative = if plain.is_normal() {
                        (normar / plain).to_f64()
                    } else {
                        1.0
                    };
                    (converged, relative)
                }
            };
            return Ok(LsmrResult {
                x,
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
        // `None` asks for an audit; only the audit names a tolerance stop.
        let ended = 'pass: {
            while iterations < maxiter {
                iterations += 1;
                let step = bidiag.step()?;
                convergence.observe(step);
                let curr_rot = recurrence.step(step);
                solution.update(bidiag.v(), curr_rot, prev_rot);

                // The tolerance test catches breakdown when the residual recurrences collapse.
                if convergence.triggers_audit(&recurrence) {
                    break 'pass None;
                }
                if let Some(rule) = escalation.as_deref_mut() {
                    let progress = Progress {
                        iteration: iterations,
                        normal_eq_residual: recurrence.relative_normal_eq_residual(),
                    };
                    // A rule firing on the last permitted iteration leaves nothing to hand off.
                    if rule.should_escalate(progress) && iterations < maxiter {
                        break 'pass Some(LsmrStopReason::Escalated);
                    }
                }
                prev_rot = curr_rot;
            }
            Some(LsmrStopReason::MaxIterations)
        };

        let mut x = solution.into_x();
        if let Some(base) = &base {
            axpby(&mut x, base, 1.0, 1.0);
        }
        // Only an audit measures `x`, and a non-finite entry never recovers.
        if let Some((index, value)) = x.iter().copied().enumerate().find(|(_, v)| !v.is_finite()) {
            return Err(invalid_input(format!(
                "non-finite solution entry {index} ({value})"
            )));
        }
        let normal_eq_residual = reference.relative(recurrence.normal_eq_residual_estimate());
        if let Some(stop_reason) = ended {
            return Ok(LsmrResult {
                x,
                iterations,
                residual_norm: recurrence.residual_estimate(),
                normal_eq_residual,
                stop_reason,
                true_residual: None,
            });
        }

        // One `A` apply measures `r`; one `Aᵀ` apply measures `Aᵀr` if `r` alone falls short.
        let residual_norm = bidiag.residual_norm(&x, b)?;
        if !residual_norm.is_finite() {
            return Ok(LsmrResult {
                x,
                iterations,
                residual_norm,
                normal_eq_residual: residual_norm,
                stop_reason: LsmrStopReason::FalseConvergence,
                true_residual: None,
            });
        }
        let certified = criteria.certify(residual_norm, || {
            Ok((bidiag.plain_gradient(residual_norm)?, bidiag.norm_floor()?))
        })?;
        let (stop_reason, normal_eq_residual, true_residual) = match certified {
            Some(stop_reason) => (
                stop_reason,
                normal_eq_residual,
                Some(bidiag.into_residual()),
            ),
            None if restarts < MAX_RESTARTS && iterations < maxiter => {
                restarts += 1;
                step1 = bidiag.restart(residual_norm)?;
                base = Some(Cow::Owned(x));
                continue;
            }
            None => {
                // The estimate is the refuted claim; a seed from the staged residual measures `x`.
                let step = bidiag.restart(residual_norm)?;
                let measured = reference.relative(Magnitude::product(step.alpha, step.beta));
                (LsmrStopReason::FalseConvergence, measured, None)
            }
        };
        return Ok(LsmrResult {
            x,
            iterations,
            residual_norm,
            normal_eq_residual,
            stop_reason,
            true_residual,
        });
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
