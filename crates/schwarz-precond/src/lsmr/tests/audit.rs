//! The true-residual check of tolerance stops, and the restarts it triggers.
use rstest::rstest;

use super::super::bidiag::{BidiagStep, Bidiagonalization};
use super::super::magnitude::Magnitude;
use super::super::recurrence::ConvergenceCriteria;
use super::super::{
    lsmr_from_bidiag, mlsmr, EscalationHandler, EscalationPolicy, LsmrResult, LsmrStopReason,
    MlsmrOptions, NormalEqReference, SolveStart,
};
use crate::lsmr::fixtures::{lsmr, DenseOp, DiagOp, FixedIterations, OverdeterminedOp};
use crate::lsmr::IdentityOp;
use crate::{MemStack, Operator, SolveError};

/// Triggers an audit at every pass's first step; `true_residuals` scripts each check, and
/// every audit that reaches for it measures `‖Aᵀr‖ = gradient` against `‖A‖₂ ≥ 1`.
struct ScriptedStream<'a> {
    v: Vec<f64>,
    true_residuals: std::iter::Copied<std::slice::Iter<'a, f64>>,
    gradient: f64,
}

impl Bidiagonalization for ScriptedStream<'_> {
    fn step(&mut self) -> Result<BidiagStep, SolveError> {
        // β = 0 zeroes φ̄, so the recurrence claims a converged residual immediately.
        Ok(BidiagStep {
            alpha: 0.0,
            beta: 0.0,
        })
    }
    fn v(&self) -> &[f64] {
        &self.v
    }
    fn residual_norm(&mut self, _x: &[f64], _rhs: &[f64]) -> Result<f64, SolveError> {
        Ok(self
            .true_residuals
            .next()
            .expect("a scripted residual per pass"))
    }
    fn plain_gradient(&mut self, _residual_norm: f64) -> Result<Magnitude, SolveError> {
        Ok(Magnitude::from(self.gradient))
    }
    fn norm_floor(&mut self) -> Result<f64, SolveError> {
        Ok(1.0)
    }
    fn restart(&mut self, _beta: f64) -> Result<BidiagStep, SolveError> {
        Ok(BidiagStep {
            alpha: 1.0,
            beta: 1.0,
        })
    }
    fn into_residual(self) -> Vec<f64> {
        Vec::new()
    }
}

/// Counts the handlers a run asks for, one per pass.
#[derive(Default)]
struct CountingPolicy(std::sync::atomic::AtomicUsize);

impl EscalationPolicy for CountingPolicy {
    fn handler(&self) -> Box<dyn EscalationHandler> {
        self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Box::new(FixedIterations(usize::MAX))
    }
}

/// `tol = 1e-10` against `‖b‖ = 1`; an audit's gradient leg sees `‖Aᵀr‖ = 1`, which never certifies.
fn scripted_run(
    true_residuals: &[f64],
    maxiter: usize,
    escalation: Option<&dyn EscalationPolicy>,
) -> LsmrResult {
    scripted_run_with_gradient(true_residuals, 1.0, maxiter, escalation)
}

fn scripted_run_with_gradient(
    true_residuals: &[f64],
    gradient: f64,
    maxiter: usize,
    escalation: Option<&dyn EscalationPolicy>,
) -> LsmrResult {
    let stream = ScriptedStream {
        v: vec![1.0, 2.0],
        true_residuals: true_residuals.iter().copied(),
        gradient,
    };
    let step1 = BidiagStep {
        alpha: 1.0,
        beta: 1.0,
    };
    let criteria = ConvergenceCriteria::new(1.0, 1e-10);
    lsmr_from_bidiag(
        stream,
        step1,
        &[1.0, 1.0],
        SolveStart {
            x0: None,
            reference: NormalEqReference::new(None, step1),
        },
        criteria,
        maxiter,
        escalation,
    )
    .expect("scripted run")
}

#[test]
fn a_stop_the_true_residual_confirms_is_certified() {
    let r = scripted_run(&[1e-12], 5, None);
    assert!(r.converged());
    assert_eq!(r.stop_reason, LsmrStopReason::ResidualTolerance);
    assert_eq!(r.iterations, 1);
    assert_eq!(r.residual_norm, 1e-12);
}

/// A measured `‖Aᵀr‖ / (L‖r‖) = 1e-11` certifies a residual stop that `‖r‖ = 0.5` alone does not, and
/// the result names the certificate that held.
#[test]
fn a_residual_stop_the_gradient_certifies_reports_the_normal_equation() {
    let r = scripted_run_with_gradient(&[0.5], 0.5e-11, 5, None);
    assert!(r.converged());
    assert_eq!(r.stop_reason, LsmrStopReason::NormalEquationTolerance);
    assert_eq!(r.iterations, 1);
    assert_eq!(r.residual_norm, 0.5);
}

/// `v` is constant, so each pass adds the same correction on top of the refuted iterate. A
/// residual 10× over the tolerance is refuted, however close it sits to the stream's estimate.
#[rstest]
fn a_refuted_stop_restarts_from_its_iterate(#[values(1.0, 1e-9)] refuted: f64) {
    let r = scripted_run(&[refuted, 1e-12], 5, None);
    assert!(r.converged());
    assert_eq!(r.stop_reason, LsmrStopReason::ResidualTolerance);
    assert_eq!(r.iterations, 2);
    assert_eq!(r.residual_norm, 1e-12);
    assert_eq!(r.x, vec![2.0, 4.0]);
}

#[test]
fn restarts_are_capped_before_the_stop_is_refused() {
    let r = scripted_run(&[1.0, 1.0, 1.0], 5, None);
    assert!(!r.converged());
    assert_eq!(r.stop_reason, LsmrStopReason::FalseConvergence);
    assert_eq!(r.iterations, 3);
    assert_eq!(r.residual_norm, 1.0);
    assert_eq!(r.normal_eq_residual, 1.0);
    assert_eq!(r.x, vec![3.0, 6.0]);
    assert!(r.true_residual.is_none());
}

/// A spent budget or a non-numeric residual leaves nothing to restart with.
#[rstest]
#[case::budget_exhausted(1.0, 1)]
#[case::non_finite_residual(f64::NAN, 5)]
fn a_refuted_stop_without_a_restart_is_refused(#[case] true_residual: f64, #[case] maxiter: usize) {
    let r = scripted_run(&[true_residual], maxiter, None);
    assert!(!r.converged());
    assert_eq!(r.stop_reason, LsmrStopReason::FalseConvergence);
    assert_eq!(r.iterations, 1);
    // The scripted restart seeds `(1, 1)`: the refuted iterate's `‖Âᵀr‖`, not the claimed 0.
    let expected = if true_residual.is_finite() {
        1.0
    } else {
        true_residual
    };
    assert_eq!(r.normal_eq_residual.to_bits(), expected.to_bits());
    assert!(r.true_residual.is_none());
}

/// `x₀ + Δx` cancels to `x = 0` on the only step allowed, so the stop is refused unrestarted.
#[rstest]
fn a_refused_stop_reports_its_iterates_normal_equation_residual(#[values(1.0, 4.0)] m_inv: f64) {
    let op = DenseOp {
        rows: 1,
        cols: 1,
        data: vec![1.0],
    };
    let r = mlsmr(
        &op,
        &[1.0],
        &DiagOp(vec![m_inv]),
        1e-12,
        1,
        Some(&[1e20]),
        MlsmrOptions::default(),
    )
    .expect("solve");
    assert_eq!(r.stop_reason, LsmrStopReason::FalseConvergence);
    assert_eq!(r.x, vec![0.0]);
    assert_eq!(r.normal_eq_residual, 1.0);
}

/// An unusable `‖Âᵀb‖` falls back to `ζ̄₀ = 2` from `step1 = (2, 1)`, reporting `1.0 / 2`.
#[rstest]
#[case::usable(4.0, 0.25)]
#[case::zero(0.0, 0.5)]
#[case::overflowed(f64::INFINITY, 0.5)]
#[case::nan(f64::NAN, 0.5)]
fn a_report_only_moves_to_a_usable_reference(#[case] metric: f64, #[case] expected: f64) {
    let step1 = BidiagStep {
        alpha: 2.0,
        beta: 1.0,
    };
    assert_eq!(
        NormalEqReference::new(Some(Magnitude::from(metric)), step1).relative(Magnitude::from(1.0)),
        expected
    );
}

/// `tol = 1e-10` against `‖b‖ = 1`: `‖r‖` certifies alone, or `‖Aᵀr‖ / (L‖r‖)` does. A floor below
/// `‖A‖₂` only refuses more, and nothing unmeasurable certifies.
#[rstest]
#[case::residual_alone(1e-11, 1.0, 1.0, true)]
#[case::gradient_meets(1.0, 1e-10, 1.0, true)]
#[case::gradient_misses(1.0, 1e-9, 1.0, false)]
#[case::looser_floor_refuses(1.0, 1e-10, 0.5, false)]
#[case::nan_gradient(1.0, f64::NAN, 1.0, false)]
#[case::zero_floor(1.0, 0.0, 0.0, false)]
#[case::nan_residual(f64::NAN, 0.0, 1.0, false)]
fn each_certificate_leg_is_explicit(
    #[case] residual: f64,
    #[case] normar: f64,
    #[case] floor: f64,
    #[case] expected: bool,
) {
    let criteria = ConvergenceCriteria::new(1.0, 1e-10);
    let certifies = criteria.residual_certifies(residual)
        || criteria.gradient_certifies(residual, Magnitude::from(normar), floor);
    assert_eq!(certifies, expected);
}

/// `M⁻¹ = diag(1, 1e-20)` cannot certify the solution direction it hides.
#[test]
fn a_deflating_preconditioner_cannot_certify_the_direction_it_hides() {
    let op = DenseOp {
        rows: 3,
        cols: 2,
        data: vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    };
    let result = mlsmr(
        &op,
        &[1.0, 1.0, 1.0],
        &DiagOp(vec![1.0, 1e-20]),
        1e-8,
        20,
        None,
        MlsmrOptions::default(),
    )
    .expect("deflated solve");

    assert!(
        !result.converged() || (result.x[1] - 1.0).abs() <= 1e-8,
        "a converged solve left the deflated coordinate at {} after {} iterations",
        result.x[1],
        result.iterations,
    );
}

/// The warm-start certificate keeps the original right-hand side as its reference.
#[test]
fn a_warm_start_near_the_solution_keeps_its_certificate() {
    let op = DenseOp {
        rows: 4,
        cols: 2,
        data: vec![1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, -1.0],
    };
    let rhs = [1.0, 2.0, 3.0, 5.0];
    let solution = lsmr(&op, &rhs, 1e-14, 100, None).expect("cold solve").x;
    let warm_start = [solution[0] + 1e-11, solution[1] - 2e-11];

    let result = mlsmr(
        &op,
        &rhs,
        &IdentityOp { n: 2 },
        1e-8,
        20,
        Some(&warm_start),
        MlsmrOptions::default(),
    )
    .expect("warm solve");

    assert!(result.converged(), "{:?}", result.stop_reason);
}

/// `‖Aᵀr‖ / (‖A‖‖r‖)` at the ends of the float range, where the obvious spellings certify an
/// unsolved stop: the product overflows, dividing by the larger factor first underflows to a zero
/// ratio, clamping a subnormal `‖A‖` up deflates it, and `f64::min` would launder a NaN residual.
#[rstest]
#[case::ordinary(1.0, 4.0, 0.5, 0.5)]
#[case::product_overflows(1e308, 2.0, 1e308, 0.5)]
#[case::double_rounding(f64::from_bits(1), 0.75, 2.0, f64::from_bits(1))]
#[case::representable_product(1e308, 0.5, 1e308, 2.0)]
#[case::quotient_underflows(1e-316, 1e10, 1e-320, 1e-6)]
#[case::subnormal_norm(5e-324, 1.647e-316, 3e-8, 0.99993)]
#[case::vanished_norm(1.0, 0.0, 1.0, f64::INFINITY)]
#[case::overflowed_norm(1.0, f64::INFINITY, 1.0, f64::INFINITY)]
#[case::nan_norm(1.0, f64::NAN, 1.0, f64::INFINITY)]
#[case::nan_residual(1.0, 1.0, f64::NAN, f64::INFINITY)]
#[case::infinite_residual(1.0, 1.0, f64::INFINITY, f64::INFINITY)]
#[case::vanished_residual(1.0, 1.0, 0.0, f64::INFINITY)]
fn the_backward_error_survives_the_ends_of_the_range(
    #[case] normar: f64,
    #[case] a_norm: f64,
    #[case] residual: f64,
    #[case] expected: f64,
) {
    let ratio = super::super::recurrence::backward_error(Magnitude::from(normar), a_norm, residual);
    if expected.is_infinite() {
        assert_eq!(ratio, expected);
    } else if expected < f64::MIN_POSITIVE {
        assert_eq!(ratio.to_bits(), expected.to_bits());
    } else {
        assert!(
            (ratio - expected).abs() <= 1e-3 * expected.max(f64::MIN_POSITIVE),
            "{ratio:e} vs {expected:e}"
        );
    }
}

/// A restart re-seeds `ζ̄₀`, so its drops need a fresh handler, not the refuted pass's `previous`.
#[test]
fn a_restarted_pass_gets_its_own_escalation_handler() {
    let policy = CountingPolicy::default();
    let r = scripted_run(&[1.0, 1e-12], 5, Some(&policy));

    assert!(r.converged());
    assert_eq!(r.iterations, 2);
    assert_eq!(policy.0.load(std::sync::atomic::Ordering::Relaxed), 2);
}

/// Delegates to `inner` but reports `floor` as its norm floor.
struct Floored<'a, A: Operator> {
    inner: &'a A,
    floor: f64,
}

impl<A: Operator> Operator for Floored<'_, A> {
    fn nrows(&self) -> usize {
        self.inner.nrows()
    }
    fn ncols(&self) -> usize {
        self.inner.ncols()
    }
    fn apply_with(&self, x: &[f64], y: &mut [f64], s: &mut MemStack) -> Result<(), SolveError> {
        self.inner.apply_with(x, y, s)
    }
    fn apply_adjoint_with(
        &self,
        x: &[f64],
        y: &mut [f64],
        s: &mut MemStack,
    ) -> Result<(), SolveError> {
        self.inner.apply_adjoint_with(x, y, s)
    }
    fn norm_floor(&self) -> Result<f64, SolveError> {
        Ok(self.floor)
    }
}

/// An inconsistent system stops on its normal equation, which only the operator's floor certifies:
/// a floor of 0 certifies nothing, so the same solve is refused after its restarts.
#[rstest]
#[case::exact_floor(3f64.sqrt(), true)]
#[case::zero_floor(0.0, false)]
fn the_operators_floor_certifies_a_normal_equation_stop(
    #[case] floor: f64,
    #[case] certifies: bool,
) {
    let op = Floored {
        inner: &OverdeterminedOp,
        floor,
    };
    let r = lsmr(&op, &[1.0, 2.0, 3.0, 5.0], 1e-10, 100, None).expect("solve");
    assert_eq!(r.converged(), certifies, "{:?}", r.stop_reason);
    if !certifies {
        assert_eq!(r.stop_reason, LsmrStopReason::FalseConvergence);
    }
}

/// The default floor stays below `‖A‖₂ = max |dᵢ|` up to the rounding of its norms, comes within 1%
/// of it on a clear gap, stays finite where `‖A‖₂²` would overflow, and is 0 with nothing to apply.
#[rstest]
#[case::gap(&[1.0, 2.0, 3.0, 10.0], 10.0)]
#[case::overflowing_square(&[1e200, 1.0], 1e200)]
#[case::tiny(&[1e-200, 3e-200], 3e-200)]
#[case::zero_operator(&[0.0, 0.0], 0.0)]
#[case::no_columns(&[], 0.0)]
fn the_default_floor_approaches_the_norm_from_below(#[case] d: &[f64], #[case] norm: f64) {
    let floor = DiagOp(d.to_vec()).norm_floor().expect("floor");
    assert!(floor <= norm * (1.0 + 1e-15), "{floor:e} above {norm:e}");
    assert!(floor >= 0.99 * norm, "{floor:e} far below {norm:e}");
}
