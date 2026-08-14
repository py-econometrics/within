//! Breakdown, early-exit and input-validation paths.

use super::super::*;
use crate::lsmr::fixtures::*;
use crate::{Operator, SolveError};

#[test]
fn test_mlsmr_mid_stream_beta_zero_breakdown() {
    // Consistent rank-1 system: b lies in A's range, so A v_1 - alpha_1 u_1
    // collapses and beta_2 == 0, driving the mid-stream beta == 0 branch in
    // both bidiagonalizations (ModifiedGolubKahan also zeroes the paired p̃).
    // The residual estimate is then exactly zero — reported as a converged
    // ResidualTolerance solve, not a distinct breakdown reason.
    let b = vec![5.0, 0.0];
    let identity = IdentityOp { n: 2 };
    for result in [
        lsmr(&ZeroSecondRow, &b, 1e-12, 100, None).expect("Golub-Kahan beta=0"),
        mlsmr(
            &ZeroSecondRow,
            &b,
            &identity,
            1e-12,
            100,
            MlsmrOptions::default(),
        )
        .expect("modified Golub-Kahan beta=0"),
    ] {
        assert!(result.converged);
        assert_eq!(result.iterations, 1);
        assert_eq!(result.stop_reason, LsmrStopReason::ResidualTolerance);
        assert!((result.x[0] - 5.0).abs() < 1e-12);
        assert!(result.x[1].abs() < 1e-12);
    }
}

/// `Aᵀb = 0` with `b ≠ 0` triggers the `step1.alpha == 0` early-exit:
/// the solver immediately returns `x = 0` and reports the trivial residual
/// `‖b‖`. Guards the early-exit branch in `mlsmr` / `lsmr_from_bidiag`.
#[test]
fn test_mlsmr_step1_alpha_zero_early_exit() {
    /// `A = [[1.0], [0.0]]` — column vector e_1.
    /// `Ax = [x, 0]`, `Aᵀy = [y_0]`, so `Aᵀb = 0` whenever `b_0 = 0`.
    struct ColE1;
    impl Operator for ColE1 {
        fn nrows(&self) -> usize {
            2
        }
        fn ncols(&self) -> usize {
            1
        }
        fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
            y[0] = x[0];
            y[1] = 0.0;
            Ok(())
        }
        fn apply_adjoint(&self, u: &[f64], x: &mut [f64]) -> Result<(), SolveError> {
            x[0] = u[0];
            Ok(())
        }
    }

    let b = vec![0.0, 1.0];
    let result = lsmr(&ColE1, &b, 1e-12, 100, None).expect("lsmr alpha=0 early exit");
    assert!(result.converged);
    assert_eq!(result.iterations, 0);
    assert_eq!(result.x, vec![0.0; 1]);
    assert_eq!(
        result.stop_reason,
        LsmrStopReason::InitialNormalEquationResidualZero
    );
    assert!((result.residual_norm - vec_norm(&b)).abs() < 1e-15);
}

/// A mid-stream bidiagonalization breakdown surfaces as a converged solve, not
/// a distinct stop reason. Whenever a step returns `alpha == 0`, that same
/// rotation step drives `zeta_bar` (the ‖Aᵀr‖ estimate) to exactly zero, so
/// the convergence check fires. On an *inconsistent* breakdown (nonzero
/// residual) it fires as NormalEquationTolerance.
///
/// `ZeroSecondRow` with `b = [5, 3]` reaches `alpha_2 = 0` while leaving a
/// residual of 3, exercising exactly this path on both bidiagonalizations.
#[test]
fn test_mid_stream_breakdown_reports_convergence() {
    let b = vec![5.0, 3.0];
    let identity = IdentityOp { n: 2 };
    for result in [
        lsmr(&ZeroSecondRow, &b, 1e-12, 100, None).expect("Golub-Kahan breakdown"),
        mlsmr(
            &ZeroSecondRow,
            &b,
            &identity,
            1e-12,
            100,
            MlsmrOptions::default(),
        )
        .expect("modified Golub-Kahan breakdown"),
    ] {
        assert!(result.converged);
        assert_eq!(result.stop_reason, LsmrStopReason::NormalEquationTolerance);
        // x_0 = 5 fits row 0; row 1 is unmatchable, leaving residual 3.
        assert!((result.x[0] - 5.0).abs() < 1e-10);
        assert!((result.residual_norm - 3.0).abs() < 1e-10);
    }
}

#[test]
fn test_mlsmr_zero_rhs_stop_reason() {
    let b = vec![0.0; 4];
    let result = lsmr(&OverdeterminedOp, &b, 1e-12, 100, None).expect("zero-rhs solve");
    assert!(result.converged);
    assert_eq!(result.iterations, 0);
    assert_eq!(result.stop_reason, LsmrStopReason::ZeroRhs);
    assert_eq!(result.residual_norm, 0.0);
}

/// Near a lucky breakdown `p̃ ≈ 0`, `vp = ⟨v, p̃⟩` (mathematically `≥ 0`) can
/// come out slightly negative from rounding. A value well inside the relative
/// tolerance must be treated as an `α = 0` breakdown, not rejected as an
/// indefinite preconditioner. Regression for the old strict `vp < 0.0` abort.
#[test]
fn test_mlsmr_near_breakdown_vp_clamps_to_zero() {
    // M⁻¹ maps p̃ to (perp(p̃) − 1e-10·p̃): the perpendicular part cancels in
    // ⟨v, p̃⟩, leaving vp ≈ −1e-10·‖p̃‖² — negative, but ~8 orders inside the
    // √ε·‖v‖‖p̃‖ floor, so it must clamp rather than raise.
    struct NearBreakdownPrecond;
    impl Operator for NearBreakdownPrecond {
        fn nrows(&self) -> usize {
            2
        }
        fn ncols(&self) -> usize {
            2
        }
        fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
            y[0] = -x[1] - 1e-10 * x[0];
            y[1] = x[0] - 1e-10 * x[1];
            Ok(())
        }
        fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
            self.apply(x, y)
        }
    }

    let b = vec![3.0, 4.0];
    let op = IdentityOp { n: 2 };
    let result = mlsmr(
        &op,
        &b,
        &NearBreakdownPrecond,
        1e-10,
        100,
        MlsmrOptions::default(),
    );
    assert!(
        result.is_ok(),
        "rounding-scale negative vp must clamp to breakdown, got err {:?}",
        result.err()
    );
}

#[test]
fn test_mlsmr_rejects_invalid_inputs() {
    let bad_len = lsmr(&OverdeterminedOp, &[1.0, 2.0], 1e-10, 100, None);
    assert!(matches!(bad_len, Err(SolveError::InvalidInput { .. })));

    let bad_tol = lsmr(
        &OverdeterminedOp,
        &[1.0, 2.0, 3.0, 4.0],
        f64::NAN,
        100,
        None,
    );
    assert!(matches!(bad_tol, Err(SolveError::InvalidInput { .. })));

    let bad_rhs = lsmr(
        &OverdeterminedOp,
        &[1.0, f64::INFINITY, 3.0, 4.0],
        1e-10,
        100,
        None,
    );
    assert!(matches!(bad_rhs, Err(SolveError::InvalidInput { .. })));
}

#[test]
fn test_mlsmr_rejects_bad_preconditioner_shape() {
    struct BadPrecond;
    impl Operator for BadPrecond {
        fn nrows(&self) -> usize {
            2
        }
        fn ncols(&self) -> usize {
            2
        }
        fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
            y.copy_from_slice(x);
            Ok(())
        }
        fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
            self.apply(x, y)
        }
    }

    let b = vec![1.0, 2.0, 3.0, 3.0];
    let result = mlsmr(
        &OverdeterminedOp,
        &b,
        &BadPrecond,
        1e-10,
        100,
        MlsmrOptions::default(),
    );
    assert!(matches!(result, Err(SolveError::InvalidInput { .. })));
}

/// A genuinely indefinite preconditioner (`M⁻¹ = −I` ⇒ `vp = −‖p̃‖²`, relative
/// magnitude 1) must still be rejected — the clamp only ever absorbs a
/// negligible negative direction, never a real loss of positive-definiteness.
#[test]
fn test_mlsmr_rejects_indefinite_preconditioner() {
    struct NegIdentity;
    impl Operator for NegIdentity {
        fn nrows(&self) -> usize {
            2
        }
        fn ncols(&self) -> usize {
            2
        }
        fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
            for (yi, &xi) in y.iter_mut().zip(x) {
                *yi = -xi;
            }
            Ok(())
        }
        fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
            self.apply(x, y)
        }
    }

    let b = vec![3.0, 4.0];
    let op = IdentityOp { n: 2 };
    let result = mlsmr(&op, &b, &NegIdentity, 1e-10, 100, MlsmrOptions::default());
    assert!(matches!(result, Err(SolveError::InvalidInput { .. })));
}
