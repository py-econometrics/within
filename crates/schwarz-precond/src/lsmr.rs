//! LSMR for rectangular least-squares (`min ‖b − A x‖₂`).
//!
//! [`mlsmr`]: preconditioned LSMR; an [`InitialGuess`] may warm-start, a [`Staleness`] rule may stop.

mod bidiag;
#[cfg(test)]
mod fixtures;
mod magnitude;
mod recurrence;
#[cfg(test)]
mod tests;

use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::matrix_free::{BiLinOp, LinOp};
use faer::reborrow::{Reborrow, ReborrowMut};
use faer::{ColMut, ColRef};

use crate::par::Par;
use crate::SolveError;
use bidiag::{axpby, BidiagStep, Bidiagonalization, Ctx, StagedStart};
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

/// What a run reports; the solution is in the caller's `x`.
#[derive(Clone, Copy, Debug)]
#[must_use]
pub struct LsmrInfo {
    /// Whether a tolerance stop survived its applicable recomputed-residual and plain-gradient audits.
    pub converged: bool,
    /// Total number of iterations performed.
    pub iterations: usize,
    /// `‖b − A x‖`: recomputed at a tolerance stop, the recurrence's estimate at any other.
    pub residual_norm: f64,
    /// `‖Âᵀ(b − A x)‖ / ‖Âᵀb‖`, measured in `M`'s metric when preconditioned.
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
    /// The iteration budget was exhausted before convergence.
    MaxIterations,
    /// The warm start already solved the system: `b − A x0` was exactly zero.
    WarmStartExact,
    /// A tolerance stop, and each restart from it, refuted by a residual or plain-gradient audit.
    FalseConvergence,
    /// The [`Staleness`] rule stopped a stalling run.
    Stalled,
}

/// Only [`Default`] makes one, so a struct literal of an options type needs `..Default::default()`.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NonExhaustive(());

/// What [`mlsmr`] reads from `x` before it starts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InitialGuess {
    /// Ignore `x` and start from zero.
    Zero,
    /// Start from `x`; tolerances stay relative to `‖b‖`.
    Given,
}

/// When LSMR stops on one right-hand side, and its reorthogonalization window.
#[derive(Debug, Clone, Copy)]
pub struct LsmrOptions {
    /// Relative residual convergence tolerance.
    pub tol: f64,
    /// Iterations per right-hand side; a run resumed on a stronger map gets what is left.
    pub maxiter: usize,
    /// Windowed MGS over the last `local_size` `v` and `p̃`, `local_size · ncols` each.
    pub local_size: Option<usize>,
    /// New fields may follow: build with `..Default::default()`.
    #[doc(hidden)]
    pub non_exhaustive: NonExhaustive,
}

impl Default for LsmrOptions {
    fn default() -> Self {
        Self {
            tol: 1e-8,
            maxiter: 1000,
            local_size: None,
            non_exhaustive: NonExhaustive(()),
        }
    }
}

/// Stops after `window` consecutive contraction ratios exceed `threshold`.
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
        // Ratios are `|s̄ₖ| ≤ 1` by construction, so a threshold of 1 or more never stalls.
        if !(0.0..1.0).contains(&threshold) {
            return Err(StalenessError::InvalidThreshold { threshold });
        }
        Ok(Self { window, threshold })
    }

    /// Consecutive stalled iterations required before stopping.
    pub fn window(&self) -> usize {
        self.window
    }

    /// Contraction ratio above which an iteration counts as stalled.
    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    /// This policy's stall counter for one pass.
    fn run(self) -> StalenessRun {
        StalenessRun {
            policy: self,
            previous: f64::NAN,
            stalled: 0,
        }
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
    /// Stops after four consecutive residual reductions below 30% (ratio above 0.7).
    fn default() -> Self {
        Self {
            window: 4,
            threshold: 0.7,
        }
    }
}

/// [`Staleness`]'s count over one pass; a restart re-seeds the drops, so it starts a fresh one.
struct StalenessRun {
    policy: Staleness,
    previous: f64,
    stalled: usize,
}

impl StalenessRun {
    /// Counts this iteration's normal-equation drop; true once `window` in a row stalled.
    fn stalled(&mut self, current: f64) -> bool {
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

/// LSMR on `min ‖b − A x‖₂` preconditioned by `M⁻¹ ≈ (AᵀA)⁻¹`; [`IdentityPrecond`](faer::matrix_free::IdentityPrecond) for none.
#[allow(clippy::too_many_arguments)]
pub fn mlsmr<A: BiLinOp<f64> + ?Sized, M: LinOp<f64> + ?Sized>(
    operator: &A,
    b: &[f64],
    preconditioner: &M,
    x: &mut [f64],
    start: InitialGuess,
    options: &LsmrOptions,
    stall: Option<Staleness>,
    par: Par,
    scratch: &mut MemBuffer,
) -> Result<LsmrInfo, SolveError> {
    let (m, n) = (operator.nrows(), operator.ncols());
    assert_eq!(
        (b.len(), x.len()),
        (m, n),
        "b and x lengths against the operator"
    );
    let shape = (preconditioner.nrows(), preconditioner.ncols());
    assert_eq!(
        shape,
        (n, n),
        "preconditioner shape against the operator columns"
    );
    if start == InitialGuess::Zero {
        x.fill(0.0);
    }
    validate_lsmr_inputs(b, x, options.tol)?;
    grow(
        scratch,
        StackReq::any_of(&[
            operator.apply_scratch(1, par),
            operator.transpose_apply_scratch(1, par),
            preconditioner.apply_scratch(1, par),
        ]),
    );
    let ctx = Ctx {
        par,
        stack: MemStack::new(scratch),
    };

    // `b` is finite entrywise, but its norm sets the tolerance and an ∞ there certifies anything.
    let b_norm = finite(vec_norm(b), "rhs norm")?;
    let local_size = options.local_size.unwrap_or(0);
    let warm = x.iter().any(|&v| v != 0.0);

    // A zero start vector needs no stream; a warm one must be formed in the stream to be measured.
    let (mut staged, rhs_norm) = if warm {
        let (staged, norm) = StagedStart::warm(operator, preconditioner, x, b, local_size, ctx);
        // Unlike `b`, the residual is computed: an ∞ entry norms to NaN, read as β₁ = 0 downstream.
        let norm = finite(norm, "warm-start residual norm")?;
        if norm == 0.0 {
            return Ok(LsmrInfo::exact(LsmrStopReason::WarmStartExact));
        }
        (staged, norm)
    } else {
        if b_norm == 0.0 {
            return Ok(LsmrInfo::exact(LsmrStopReason::ZeroRhs));
        }
        (
            StagedStart::cold(operator, preconditioner, b, local_size, ctx),
            b_norm,
        )
    };

    let references = if warm && b_norm > 0.0 {
        Some(staged.gradient_references(b, b_norm)?)
    } else {
        None
    };
    let (bidiag, step1, cold_plain_reference) = staged.restart(rhs_norm)?;
    let reference = references.map_or_else(
        || NormalEqReference::cold(step1, cold_plain_reference),
        |(metric, plain)| NormalEqReference::warm(metric, plain, step1, cold_plain_reference),
    );
    let reference_norm = if b_norm > 0.0 { b_norm } else { rhs_norm };
    let criteria = ConvergenceCriteria::new(reference_norm, options.tol);
    lsmr_from_bidiag(bidiag, step1, b, x, reference, criteria, options, stall)
}

impl LsmrInfo {
    /// A run that needed no iteration: the start already solved it.
    fn exact(stop_reason: LsmrStopReason) -> Self {
        Self {
            converged: true,
            iterations: 0,
            residual_norm: 0.0,
            normal_eq_residual: 0.0,
            stop_reason,
        }
    }
}

/// Grows `scratch` to hold `req`; never shrinks it.
fn grow(scratch: &mut MemBuffer, req: StackReq) {
    if !MemStack::new(scratch).can_hold(req) {
        *scratch = MemBuffer::new(req);
    }
}

/// `f(x, y)` on unit-stride slices; a strided column goes through a heap copy.
pub fn with_unit_stride<R>(
    x: ColRef<'_, f64>,
    mut y: ColMut<'_, f64>,
    f: impl FnOnce(&[f64], &mut [f64]) -> R,
) -> R {
    let copied: Vec<f64>;
    let x = match x.try_as_col_major() {
        Some(x) => x.as_slice(),
        None => {
            copied = x.iter().copied().collect();
            &copied
        }
    };
    if let Some(y) = y.rb_mut().try_as_col_major_mut() {
        return f(x, y.as_slice_mut());
    }
    let mut staged: Vec<f64> = y.rb().iter().copied().collect();
    let r = f(x, &mut staged);
    y.copy_from(ColRef::from_slice(&staged));
    r
}

/// Restarts from a refuted tolerance stop before the solve is refused outright.
const MAX_RESTARTS: usize = 2;

/// Fixed metric and plain `‖Aᵀb‖` references for reporting and exit auditing.
#[derive(Clone, Copy)]
struct NormalEqReference {
    metric: Magnitude,
    plain: Magnitude,
}

impl NormalEqReference {
    /// A cold stream's first `(α₁, β₁)` is `‖Aᵀb‖` already, though the product may not be a double.
    fn cold(step1: BidiagStep, plain: Magnitude) -> Self {
        Self {
            metric: Magnitude::product(step1.alpha, step1.beta),
            plain,
        }
    }

    /// Each unusable warm reference falls back independently to its cold-stream counterpart.
    fn warm(metric: Magnitude, plain: Magnitude, step1: BidiagStep, cold_plain: Magnitude) -> Self {
        let cold = Self::cold(step1, cold_plain);
        Self {
            metric: if metric.is_normal() {
                metric
            } else {
                cold.metric
            },
            plain: if plain.is_normal() { plain } else { cold.plain },
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

/// Runs LSMR on the correction to `x`, restarting from any tolerance stop `b − A x` refutes.
#[allow(clippy::too_many_arguments)]
fn lsmr_from_bidiag<B: Bidiagonalization>(
    mut bidiag: B,
    mut step1: BidiagStep,
    b: &[f64],
    x: &mut [f64],
    reference: NormalEqReference,
    criteria: ConvergenceCriteria,
    options: &LsmrOptions,
    stall: Option<Staleness>,
) -> Result<LsmrInfo, SolveError> {
    let maxiter = options.maxiter;
    let par = bidiag.par();
    let mut iterations = 0;
    let mut restarts = 0;
    loop {
        // A metric reporting no gradient at all may be hiding one outside itself.
        if step1.alpha == 0.0 {
            let b_norm = vec_norm(b);
            let (converged, normal_eq_residual) = match bidiag.hidden_gradient(b, b_norm) {
                None => (true, 0.0),
                Some((per_unit_residual, b_image)) => {
                    let normar = Magnitude::product(step1.beta, per_unit_residual);
                    let plain = b_image.norm();
                    (
                        criteria.corroborates(step1.beta, normar, b_image.per_unit()),
                        if plain.is_normal() {
                            (normar / plain).to_f64()
                        } else {
                            1.0
                        },
                    )
                }
            };
            return Ok(LsmrInfo {
                converged,
                iterations,
                residual_norm: step1.beta,
                normal_eq_residual,
                stop_reason: if converged {
                    LsmrStopReason::InitialNormalEquationResidualZero
                } else {
                    LsmrStopReason::FalseConvergence
                },
            });
        }

        let mut stall = stall.map(Staleness::run);
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
                solution.update(bidiag.v(), curr_rot, prev_rot, par);

                // The tolerance test catches breakdown when the residual recurrences collapse.
                match convergence.check(&recurrence) {
                    Stop::Continue => {}
                    Stop::ResidualTolerance => break 'pass LsmrStopReason::ResidualTolerance,
                    Stop::NormalEquationTolerance => {
                        break 'pass LsmrStopReason::NormalEquationTolerance
                    }
                }
                let drop = recurrence.relative_normal_eq_residual();
                if stall.as_mut().is_some_and(|s| s.stalled(drop)) {
                    break 'pass LsmrStopReason::Stalled;
                }
                prev_rot = curr_rot;
            }
            LsmrStopReason::MaxIterations
        };

        axpby(x, solution.correction(), 1.0, 1.0, par);
        // Only tolerance stops measure `x`, and a non-finite entry never recovers.
        if let Some((index, value)) = x.iter().copied().enumerate().find(|(_, v)| !v.is_finite()) {
            return Err(invalid_input(format!(
                "non-finite solution entry {index} ({value})"
            )));
        }
        let mut info = LsmrInfo {
            converged: false,
            iterations,
            residual_norm: recurrence.residual_estimate(),
            normal_eq_residual: reference.relative(recurrence.normal_eq_residual_estimate()),
            stop_reason,
        };
        // Only a tolerance stop claims convergence, so only it is worth a true-residual evaluation.
        if matches!(
            stop_reason,
            LsmrStopReason::ResidualTolerance | LsmrStopReason::NormalEquationTolerance
        ) {
            let residual_norm = bidiag.residual_norm(x, b);
            info.converged = criteria.corroborates_residual(residual_norm, info.residual_norm);
            if info.converged && stop_reason == LsmrStopReason::NormalEquationTolerance {
                info.converged = criteria.corroborates_normal_eq(
                    residual_norm,
                    bidiag.plain_gradient(residual_norm),
                    reference.plain,
                    info.normal_eq_residual,
                );
            }
            if !info.converged
                && residual_norm.is_finite()
                && restarts < MAX_RESTARTS
                && iterations < maxiter
            {
                restarts += 1;
                step1 = bidiag.restart(residual_norm)?;
                continue;
            }
            if !info.converged {
                info.stop_reason = LsmrStopReason::FalseConvergence;
                // The estimate is the refuted claim; a seed from the staged residual measures `x`.
                info.normal_eq_residual = if residual_norm.is_finite() {
                    let step = bidiag.restart(residual_norm)?;
                    reference.relative(Magnitude::product(step.alpha, step.beta))
                } else {
                    residual_norm
                };
            }
            info.residual_norm = residual_norm;
        }
        return Ok(info);
    }
}

fn validate_lsmr_inputs(b: &[f64], x: &[f64], tol: f64) -> Result<(), SolveError> {
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
    if let Some((index, value)) = x.iter().copied().enumerate().find(|(_, v)| !v.is_finite()) {
        return Err(invalid_input(format!(
            "start entry {index} must be finite, got {value}"
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
