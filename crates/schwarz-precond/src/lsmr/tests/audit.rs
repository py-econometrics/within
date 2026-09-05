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
    /// `‖Âᵀb‖` reported when audited at `x = 0`; `None` scripts a cold run.
    normar_of_b: Option<f64>,
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
    fn certify(&mut self, x: &[f64], _rhs: &[f64]) -> Result<Certificate, SolveError> {
        let normar = match self.normar_of_b {
            Some(of_b) if x.iter().all(|&xi| xi == 0.0) => of_b,
            _ => self.normar,
        };
        Ok(Certificate {
            normr: self.normr,
            normar,
        })
    }
}

fn scripted_run(normr: f64, normar: f64) -> super::super::LsmrResult {
    let stream = ScriptedStream {
        v: vec![0.0; 2],
        normr,
        normar,
        normar_of_b: None,
    };
    let step1 = BidiagStep {
        alpha: 1.0,
        beta: 1.0,
    };
    let criteria = ConvergenceCriteria::new(1.0, 1e-10);
    lsmr_from_bidiag(stream, step1, &[1.0, 1.0], None, criteria, 5, None).expect("scripted run")
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
fn near_consistent_stop_certifies_via_the_initial_ne_drop() {
    // Ratio leg would refuse (1e-12/1e-6 ≫ 100·tol); the drop vs ζ̄₀ = 1 certifies.
    let r = scripted_run(1e-6, 1e-12);
    assert!(r.converged);
    assert_eq!(r.stop_reason, LsmrStopReason::ResidualTolerance);
}

#[test]
fn warm_start_audit_is_anchored_to_the_cold_scale() {
    // Restart ζ̄₀ = α₁β₁ = 1e-6 would refuse 1e-12 ≤ 100·tol·ζ̄₀; the cold ‖Âᵀb‖ = 1 certifies.
    let run = |x0: Option<&[f64]>, normar_of_b| {
        let stream = ScriptedStream {
            v: vec![0.0; 2],
            normr: 1e-6,
            normar: 1e-12,
            normar_of_b,
        };
        let step1 = BidiagStep {
            alpha: 1e-3,
            beta: 1e-3,
        };
        let criteria = ConvergenceCriteria::new(1.0, 1e-10);
        lsmr_from_bidiag(stream, step1, &[1.0, 1.0], x0, criteria, 5, None).expect("run")
    };

    let cold = run(None, None);
    assert_eq!(cold.stop_reason, LsmrStopReason::FalseConvergence);

    let x0 = [0.5, 0.5];
    let warm = run(Some(&x0), Some(1.0));
    assert!(warm.converged);
    assert_eq!(warm.x, x0);
    assert_eq!(warm.normal_eq_residual, 1e-12);
}
