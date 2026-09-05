//! True-residual audit of tolerance stops.

use super::super::bidiag::{BidiagStep, Bidiagonalization, Certificate};
use super::super::recurrence::ConvergenceCriteria;
use super::super::{lsmr_from_bidiag, LsmrStopReason};
use crate::SolveError;

/// Stream whose first step stops on ResidualTolerance and whose audit is scripted.
struct ScriptedStream {
    v: Vec<f64>,
    normr: f64,
    normar: f64,
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
            normar: self.normar,
        })
    }
}

fn scripted_run(normr: f64, normar: f64) -> super::super::LsmrResult {
    let stream = ScriptedStream {
        v: vec![0.0; 2],
        normr,
        normar,
    };
    let step1 = BidiagStep {
        alpha: 1.0,
        beta: 1.0,
    };
    let criteria = ConvergenceCriteria::new(1.0, 1e-10);
    lsmr_from_bidiag(stream, step1, &[1.0, 1.0], 1.0, criteria, 5, None).expect("scripted run")
}

#[test]
fn collapsed_stop_is_refused_by_the_audit() {
    let r = scripted_run(1.0, 1.0);
    assert!(!r.converged);
    assert_eq!(r.stop_reason, LsmrStopReason::FalseConvergence);
    assert_eq!(r.residual_norm, 1.0);
    assert_eq!(r.normal_eq_residual, 1.0);
}

#[test]
fn honest_stop_passes_the_audit() {
    let r = scripted_run(1e-12, 1e-12);
    assert!(r.converged);
    assert_eq!(r.stop_reason, LsmrStopReason::ResidualTolerance);
}

#[test]
fn near_consistent_stop_certifies_at_the_roundoff_floor() {
    // Ratio leg refuses (1e-15/1e-6 ≫ 100·tol); `‖Aᵀr‖` at `ε‖A‖‖b‖` with ‖A‖ = ‖b‖ = 1 certifies.
    let r = scripted_run(1e-6, 1e-15);
    assert!(r.converged);
    assert_eq!(r.stop_reason, LsmrStopReason::ResidualTolerance);
}

#[test]
fn a_relative_drop_alone_no_longer_certifies() {
    // Formerly accepted as `normar ≤ 100·tol·ζ̄₀`; 1e-12 is 4e3 floors above what f64 resolves.
    let r = scripted_run(1e-6, 1e-12);
    assert!(!r.converged);
    assert_eq!(r.stop_reason, LsmrStopReason::FalseConvergence);
}
