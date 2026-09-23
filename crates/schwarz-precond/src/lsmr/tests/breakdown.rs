//! Breakdown, early-exit and input-validation paths.

use rstest::rstest;

use super::super::*;
use crate::lsmr::fixtures::*;
use crate::{Operator, SolveError};

/// `ZeroSecondRow` under either bidiagonalization; the modified one also zeroes the paired p̃.
fn zero_second_row_solve(b: &[f64], modified: bool) -> LsmrResult {
    if modified {
        let identity = IdentityOp { n: 2 };
        mlsmr(
            &ZeroSecondRow,
            b,
            &identity,
            1e-12,
            100,
            MlsmrOptions::default(),
        )
        .expect("modified Golub-Kahan")
    } else {
        lsmr(&ZeroSecondRow, b, 1e-12, 100, None).expect("Golub-Kahan")
    }
}

/// b lies in A's range, so `A v₁ − α₁ u₁` collapses and β₂ = 0; the residual estimate is then
/// exactly zero — a converged ResidualTolerance solve, not a distinct breakdown reason.
#[rstest]
#[case::golub_kahan(false)]
#[case::modified_golub_kahan(true)]
fn test_mlsmr_mid_stream_beta_zero_breakdown(#[case] modified: bool) {
    let result = zero_second_row_solve(&[5.0, 0.0], modified);
    assert!(result.converged);
    assert_eq!(result.iterations, 1);
    assert_eq!(result.stop_reason, LsmrStopReason::ResidualTolerance);
    assert!((result.x[0] - 5.0).abs() < 1e-12);
    assert!(result.x[1].abs() < 1e-12);
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

/// Every case below reaches the `α₁ = 0` exit: `diag(m)` annihilates `Aᵀ rhs`, or squaring an
/// already-tiny gradient underflows. `None` runs the unpreconditioned stream instead.
fn zero_initial_gradient(
    a: &[f64],
    m: Option<&[f64]>,
    b: &[f64],
    x0: Option<&[f64]>,
) -> LsmrResult {
    let a = DiagOp(a.to_vec());
    match m {
        Some(m) => mlsmr(
            &a,
            b,
            &DiagOp(m.to_vec()),
            1e-10,
            100,
            MlsmrOptions {
                warm_start: x0,
                ..Default::default()
            },
        ),
        None => lsmr(&a, b, 1e-10, 100, None),
    }
    .expect("α₁ = 0 exit")
}

/// `α₁ = √(p̃ᵀ M⁻¹ p̃)` vanishes whenever `Aᵀb` lies in `ker(M⁻¹)`, however large it is there, so
/// the exit must be audited outside the metric — from a warm start too, where a zero correction
/// does not mean a zero `x`. The last three are ways to flatter the backward-error leg into
/// certifying anyway: a bound that overflowed although every entry of `Aᵀb` is finite, a
/// denominator that overflows although both factors are finite, and a subnormal bound a clamp
/// would raise by eight orders. `scaled_design` separates `‖Aᵀb‖ / ‖b‖` from its reciprocal,
/// which certifies it: only a bound that understates `‖A‖` overstates the error.
#[rstest]
#[case::kernel_cold(&[1.0, 1.0], &[0.0, 1.0], None, 1.0)]
#[case::kernel_warm(&[1.0, 1.0], &[1.0, 1.0], Some(&[1.0, 0.0][..]), 1.0)]
#[case::scaled_design(&[1e-6, 1e-6], &[1.0, 1.0], Some(&[1e6, 0.0][..]), 1e-6)]
#[case::overflowed_bound(&[2.0, 2.0], &[8e307, 8e307], Some(&[4e307, 0.0][..]), 1.6e308)]
#[case::overflowed_denominator(&[2.0, 1.0], &[5e307, 0.0], Some(&[2.5e307, -1e308][..]), 1e308)]
#[case::subnormal_bound(&[1.8e-316, 1.8e-316], &[0.0, 3e-8], None, 5.4e-324)]
fn a_zero_initial_gradient_the_metric_cannot_see_is_refused(
    #[case] a: &[f64],
    #[case] b: &[f64],
    #[case] x0: Option<&[f64]>,
    #[case] unsolved: f64,
) {
    let result = zero_initial_gradient(a, Some(&[1.0, 0.0]), b, x0);
    let residual = normal_equation_residual(&DiagOp(a.to_vec()), &result.x, b);

    assert!(!result.converged, "{:?}", result.stop_reason);
    assert_eq!(result.stop_reason, LsmrStopReason::FalseConvergence);
    assert_eq!(result.iterations, 0);
    assert!((residual / unsolved - 1.0).abs() < 1e-3, "{residual:e}");
}

/// An `α₁` whose product form underflows is a scale to recover, not a breakdown at `x = 0`.
#[rstest]
#[case::no_metric(None)]
#[case::identity_metric(Some(&[1.0, 1.0][..]))]
#[case::scaled_metric(Some(&[1e-10, 1e-10][..]))]
fn an_underflowed_alpha_is_a_scale_not_a_breakdown(#[case] m: Option<&[f64]>) {
    let a = &[0.0, 1.0];
    let b = &[1e100, 1e-100];
    let result = zero_initial_gradient(a, m, b, None);

    assert!(result.converged, "{:?}", result.stop_reason);
    assert_eq!(result.iterations, 1);
    assert!((result.x[1] / 1e-100 - 1.0).abs() < 1e-9, "{:?}", result.x);
}

/// The exit must not refuse what it cannot measure a drop for. `inside_residual_tolerance` is
/// carried by `‖b‖`, which the correction does not touch, and `‖b‖ = 1e6` separates that leg from
/// the relative tolerance. `meets_normal_equation_tolerance` has only the `‖A‖` bound to go on,
/// and `exact_warm_start` fails outright if the audit reaches for a cold certificate it cannot
/// square.
#[rstest]
#[case::inside_residual_tolerance(&[1.0, 1.0], Some(&[1.0, 0.0][..]), &[1e6, 1e-5], Some(&[1e6, 0.0][..]))]
#[case::meets_normal_equation_tolerance(&[1.0, 1e-12], Some(&[1.0, 0.0][..]), &[1.0, 1e-4], Some(&[1.0, 0.0][..]))]
#[case::exact_warm_start(&[1.0, 0.0], Some(&[1.0, 1.0][..]), &[1e200, 1.0], Some(&[1e200, 0.0][..]))]
fn a_zero_initial_gradient_nothing_contradicts_certifies(
    #[case] a: &[f64],
    #[case] m: Option<&[f64]>,
    #[case] b: &[f64],
    #[case] x0: Option<&[f64]>,
) {
    let result = zero_initial_gradient(a, m, b, x0);

    assert!(result.converged, "{:?}", result.stop_reason);
    assert_eq!(result.iterations, 0);
}

/// A mid-stream bidiagonalization breakdown surfaces as a converged solve, not
/// a distinct stop reason. Whenever a step returns `alpha == 0`, that same
/// rotation step drives `zeta_bar` (the ‖Aᵀr‖ estimate) to exactly zero, so
/// the convergence check fires. On an *inconsistent* breakdown (nonzero
/// residual) it fires as NormalEquationTolerance.
///
/// `ZeroSecondRow` with `b = [5, 3]` reaches `alpha_2 = 0` while leaving a
/// residual of 3, exercising exactly this path on both bidiagonalizations.
#[rstest]
#[case::golub_kahan(false)]
#[case::modified_golub_kahan(true)]
fn test_mid_stream_breakdown_reports_convergence(#[case] modified: bool) {
    let result = zero_second_row_solve(&[5.0, 3.0], modified);
    assert!(result.converged);
    assert_eq!(result.stop_reason, LsmrStopReason::NormalEquationTolerance);
    // x_0 = 5 fits row 0; row 1 is unmatchable, leaving residual 3.
    assert!((result.x[0] - 5.0).abs() < 1e-10);
    assert!((result.residual_norm - 3.0).abs() < 1e-10);
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

    // Finite entrywise, but `‖b‖` overflows and would scale u₁ to zero, reading as α₁ = 0.
    let overflowing_rhs = lsmr(&OverdeterminedOp, &[f64::MAX; 4], 1e-10, 100, None);
    assert!(matches!(
        overflowing_rhs,
        Err(SolveError::InvalidInput { .. })
    ));
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
