//! The true-residual check of tolerance stops, and the restarts it triggers.
use rstest::rstest;

use super::super::bidiag::{BidiagStep, Bidiagonalization};
use super::super::recurrence::ConvergenceCriteria;
use super::super::{
    lsmr_from_bidiag, mlsmr, EscalationHandler, EscalationPolicy, LsmrResult, LsmrStopReason,
    MlsmrOptions, NormalEqReference,
};
use crate::lsmr::fixtures::{DenseOp, DiagOp, FixedIterations};
use crate::SolveError;

/// Stops on ResidualTolerance at every pass's first step; `true_residuals` scripts each check.
struct ScriptedStream<'a> {
    v: Vec<f64>,
    true_residuals: std::iter::Copied<std::slice::Iter<'a, f64>>,
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
    fn restart(&mut self, _beta: f64) -> Result<BidiagStep, SolveError> {
        Ok(BidiagStep {
            alpha: 1.0,
            beta: 1.0,
        })
    }
    fn plain_gradient(&mut self, _rhs: &[f64]) -> Result<f64, SolveError> {
        Ok(1.0)
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

/// `tol = 1e-10` against `‖b‖ = 1`, so the gap an honest stop may show is `1e-8`.
fn scripted_run(
    true_residuals: &[f64],
    maxiter: usize,
    escalation: Option<&dyn EscalationPolicy>,
) -> LsmrResult {
    let stream = ScriptedStream {
        v: vec![1.0, 2.0],
        true_residuals: true_residuals.iter().copied(),
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
        None,
        criteria,
        maxiter,
        escalation,
    )
    .expect("scripted run")
}

#[rstest]
#[case::exact(1e-12)]
#[case::inside_the_gap(1e-9)]
fn a_stop_the_true_residual_confirms_is_certified(#[case] true_residual: f64) {
    let r = scripted_run(&[true_residual], 5, None);
    assert!(r.converged);
    assert_eq!(r.stop_reason, LsmrStopReason::ResidualTolerance);
    assert_eq!(r.iterations, 1);
    assert_eq!(r.residual_norm, true_residual);
}

/// `v` is constant, so each pass adds the same correction on top of the refuted iterate.
#[test]
fn a_refuted_stop_restarts_from_its_iterate() {
    let r = scripted_run(&[1.0, 1e-12], 5, None);
    assert!(r.converged);
    assert_eq!(r.stop_reason, LsmrStopReason::ResidualTolerance);
    assert_eq!(r.iterations, 2);
    assert_eq!(r.residual_norm, 1e-12);
    assert_eq!(r.x, vec![2.0, 4.0]);
}

#[test]
fn restarts_are_capped_before_the_stop_is_refused() {
    let r = scripted_run(&[1.0, 1.0, 1.0], 5, None);
    assert!(!r.converged);
    assert_eq!(r.stop_reason, LsmrStopReason::FalseConvergence);
    assert_eq!(r.iterations, 3);
    assert_eq!(r.residual_norm, 1.0);
    assert_eq!(r.normal_eq_residual, 1.0);
    assert_eq!(r.x, vec![3.0, 6.0]);
}

/// A spent budget or a non-numeric residual leaves nothing to restart with.
#[rstest]
#[case::budget_exhausted(1.0, 1)]
#[case::non_finite_residual(f64::NAN, 5)]
fn a_refuted_stop_without_a_restart_is_refused(#[case] true_residual: f64, #[case] maxiter: usize) {
    let r = scripted_run(&[true_residual], maxiter, None);
    assert!(!r.converged);
    assert_eq!(r.stop_reason, LsmrStopReason::FalseConvergence);
    assert_eq!(r.iterations, 1);
    // The scripted restart seeds `(1, 1)`: the refuted iterate's `‖Âᵀr‖`, not the claimed 0.
    let expected = if true_residual.is_finite() {
        1.0
    } else {
        true_residual
    };
    assert_eq!(r.normal_eq_residual.to_bits(), expected.to_bits());
}

/// `x₀ + Δx` cancels to `x = 0` on the only step allowed, so the stop is refused unrestarted.
#[rstest]
fn a_refused_stop_reports_its_iterates_normal_equation_residual(#[values(1.0, 4.0)] m_inv: f64) {
    let op = DenseOp {
        rows: 1,
        cols: 1,
        data: vec![1.0],
    };
    let options = MlsmrOptions {
        warm_start: Some(&[1e20]),
        ..Default::default()
    };
    let r = mlsmr(&op, &[1.0], &DiagOp(vec![m_inv]), 1e-12, 1, options).expect("solve");
    assert_eq!(r.stop_reason, LsmrStopReason::FalseConvergence);
    assert_eq!(r.x, vec![0.0]);
    assert_eq!(r.normal_eq_residual, 1.0);
}

/// An unusable `‖Aᵀb‖` falls back to `ζ̄₀ = 2` from `step1 = (2, 1)`, reporting `1.0 / 2`.
#[rstest]
#[case::usable(4.0, 0.25)]
#[case::zero(0.0, 0.5)]
#[case::negative(-1.0, 0.5)]
#[case::overflowed(f64::INFINITY, 0.5)]
fn a_report_only_moves_to_a_usable_reference(#[case] metric: f64, #[case] expected: f64) {
    let step1 = BidiagStep {
        alpha: 2.0,
        beta: 1.0,
    };
    assert_eq!(
        NormalEqReference::warm(metric, step1).relative(1.0),
        expected
    );
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
    let ratio = super::super::recurrence::backward_error(normar, a_norm, residual);
    if expected.is_infinite() {
        assert_eq!(ratio, expected);
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

    assert!(r.converged);
    assert_eq!(r.iterations, 2);
    assert_eq!(policy.0.load(std::sync::atomic::Ordering::Relaxed), 2);
}
