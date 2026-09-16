//! True-residual audit of tolerance stops.
use rstest::rstest;

use super::super::bidiag::{BidiagStep, Bidiagonalization, Certificate, NormalEquationResidual};
use super::super::recurrence::ConvergenceCriteria;
use super::super::{lsmr_from_bidiag, LsmrStopReason, MlsmrOptions};
use crate::SolveError;

/// Stream whose first step stops on ResidualTolerance and whose audit is scripted.
struct ScriptedStream {
    v: Vec<f64>,
    normr: f64,
    normar: f64,
    normar_raw: Option<f64>,
}

impl Bidiagonalization for ScriptedStream {
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
    fn certify(&mut self, _x: &[f64], _rhs: &[f64]) -> Result<Certificate, SolveError> {
        Ok(Certificate {
            normr: self.normr,
            normar: NormalEquationResidual {
                norm: self.normar,
                reference: 1.0,
            },
            normar_raw: self.normar_raw.map(|norm| NormalEquationResidual {
                norm,
                reference: 1.0,
            }),
        })
    }
}

fn scripted_run(normr: f64, normar: f64, normar_raw: Option<f64>) -> super::super::LsmrResult {
    let stream = ScriptedStream {
        v: vec![0.0; 2],
        normr,
        normar,
        normar_raw,
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
        criteria,
        5,
        MlsmrOptions::default(),
    )
    .expect("scripted run")
}

#[test]
fn collapsed_stop_is_refused_by_the_audit() {
    let r = scripted_run(1.0, 1.0, None);
    assert!(!r.converged);
    assert_eq!(r.stop_reason, LsmrStopReason::FalseConvergence);
    assert_eq!(r.residual_norm, 1.0);
    assert_eq!(r.normal_eq_residual, 1.0);
}

#[test]
fn honest_stop_passes_the_audit() {
    let r = scripted_run(1e-12, 1e-12, None);
    assert!(r.converged);
    assert_eq!(r.stop_reason, LsmrStopReason::ResidualTolerance);
}

#[test]
fn near_consistent_stop_certifies_via_the_initial_ne_drop() {
    // Ratio leg would refuse (1e-12/1e-6 ≫ 100·tol); the drop vs ζ̄₀ = 1 certifies.
    let r = scripted_run(1e-6, 1e-12, None);
    assert!(r.converged);
    assert_eq!(r.stop_reason, LsmrStopReason::ResidualTolerance);
}

/// A residual already inside the tolerance certifies on its own: here both gradient legs refuse,
/// the drop being 1e-3 against a reference of 1 and the ratio 1e6.
#[test]
fn a_residual_inside_the_tolerance_certifies_without_the_gradient() {
    let r = scripted_run(1e-9, 1e-3, None);
    assert!(r.converged);
    assert_eq!(r.stop_reason, LsmrStopReason::ResidualTolerance);
}

/// A metric that annihilates part of `Aᵀr` reports it as zero, so the plain norm has to refuse.
#[test]
fn a_stop_the_metric_cannot_see_is_refused() {
    let r = scripted_run(1e-6, 1e-12, Some(1e-6));
    assert!(!r.converged);
    assert_eq!(r.stop_reason, LsmrStopReason::FalseConvergence);
}

/// A zero or overflowed cold reference cannot replace the stream's own, and never divides.
#[test]
fn a_reference_only_moves_to_a_usable_cold_value() {
    let certificate = |norm: f64, reference: f64| Certificate {
        normr: 0.0,
        normar: NormalEquationResidual { norm, reference },
        normar_raw: None,
    };
    for cold in [0.0, -1.0, f64::INFINITY] {
        let mut cert = certificate(1.0, 4.0);
        cert.rebase(&certificate(cold, 0.0));
        assert_eq!(cert.normar.reference, 4.0, "cold reference {cold:e}");
    }
    let mut cert = certificate(1.0, 4.0);
    cert.rebase(&certificate(9.0, 0.0));
    assert_eq!(cert.normar.reference, 9.0);

    assert!(NormalEquationResidual {
        norm: 1.0,
        reference: 0.0
    }
    .relative()
    .is_finite());
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
