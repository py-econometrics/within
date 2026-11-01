//! Warm starts and escalation: the preconditioner ladder.

use rstest::rstest;

use super::super::*;
use crate::lsmr::fixtures::*;
use crate::SolveError;

#[test]
fn test_mlsmr_warm_tolerance_is_relative_to_original_rhs() {
    let op = vandermonde(30, 12);
    let x_true: Vec<f64> = (0..op.ncols()).map(|j| 1.0 / (1.0 + j as f64)).collect();
    let b = times(&op, &x_true);
    let x0: Vec<f64> = x_true.iter().map(|v| v * 0.9999).collect();
    let m = jacobi(&op);
    let options = LsmrOptions {
        local_size: Some(12),
        ..opts(1e-6, 200)
    };
    let (_, warm) = run(&op, &b, &m, Some(&x0), &options, None).expect("warm solve");
    assert_eq!(warm.stop_reason, LsmrStopReason::ResidualTolerance);

    let r: Vec<f64> = b
        .iter()
        .zip(times(&op, &x0))
        .map(|(bi, ai)| bi - ai)
        .collect();
    let (_, naive) = run(&op, &r, &m, None, &options, None).expect("naive restart");
    assert!(
        warm.iterations < naive.iterations,
        "warm took {} iters, naive restart {} — the tolerance basis is not `‖b‖`",
        warm.iterations,
        naive.iterations
    );
}

/// A zero RHS must not shadow warm-start validation: the same bad `x0` has to be
/// rejected whether or not `b` short-circuits the solve.
#[rstest]
#[case::nan_x0(&[1.0, 2.0, 3.0], &[0.0, f64::NAN, 0.0])]
#[case::nan_x0_zero_rhs(&[0.0, 0.0, 0.0], &[0.0, f64::NAN, 0.0])]
fn test_mlsmr_rejects_bad_warm_start_for_any_rhs(#[case] b: &[f64], #[case] x0: &[f64]) {
    let op = IdentityPrecond { dim: 3 };
    assert!(matches!(
        run(&op, b, &op, Some(x0), &opts(1e-8, 10), None),
        Err(SolveError::InvalidInput { .. })
    ));
}

/// A short warm start panics like any length mismatch, whether or not `b` is zero.
#[rstest]
#[should_panic(expected = "b and x lengths")]
fn a_short_warm_start_panics(#[values(&[1.0, 2.0, 3.0], &[0.0, 0.0, 0.0])] b: &[f64]) {
    let op = IdentityPrecond { dim: 3 };
    let _ = run(&op, b, &op, Some(&[0.0, 0.0]), &opts(1e-8, 10), None);
}

/// Regression: `A·x₀` overflowing made `b - A·x₀` norm to NaN, which `init` read
/// as β₁ = 0 and reported as a converged solve with `x = x₀` and a NaN residual.
#[test]
fn test_mlsmr_warm_rejects_overflowing_warm_start_residual() {
    let op = diag(&[1e200; 3]);
    let m = IdentityPrecond { dim: 3 };
    let x0 = [1e200; 3];
    assert!(matches!(
        run(&op, &[1.0, 2.0, 3.0], &m, Some(&x0), &opts(1e-8, 10), None),
        Err(SolveError::InvalidInput { .. })
    ));
}

/// `‖b‖` sets the tolerance even when the warm-start residual it is measured against is finite.
#[test]
fn test_mlsmr_warm_rejects_overflowing_rhs_norm() {
    let (op, m) = (diag(&[1.0, 1.1]), diag(&[1e-310; 2]));
    let x0 = [1.2e308, 1.1e308];
    let err = run(&op, &[1.3e308; 2], &m, Some(&x0), &opts(1e-10, 10), None)
        .expect_err("an infinite ‖b‖ certified the warm start");
    assert!(
        matches!(&err, SolveError::InvalidInput { message, .. } if message.contains("rhs norm")),
        "{err}"
    );
}

#[test]
fn test_mlsmr_zero_rhs_corrects_non_exact_warm_start() {
    let op = IdentityPrecond { dim: 3 };
    let x0 = [1.0, 2.0, 3.0];
    let (x, result) =
        run(&op, &[0.0; 3], &op, Some(&x0), &opts(1e-10, 10), None).expect("warm correction");

    assert!(result.converged);
    // `b = 0` leaves `‖b − A x₀‖` as the only scale, so the residual leg is satisfiable.
    assert_eq!(result.stop_reason, LsmrStopReason::ResidualTolerance);
    assert!(vec_norm(&x) < 1e-12);
    // `‖Aᵀb‖ = 0` here, so the audit's reference falls back to the stream's own.
    assert!(result.normal_eq_residual.is_finite());
    assert!(result.iterations > 0);
}

#[rstest]
#[case::identity(Mat::identity(3, 3), &[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0])]
#[case::null_direction(zero_second_row(), &[0.0, 0.0], &[0.0, 7.0])]
fn test_mlsmr_exact_warm_start_is_returned_untouched(
    #[case] op: Mat<f64>,
    #[case] b: &[f64],
    #[case] x0: &[f64],
) {
    let m = IdentityPrecond { dim: op.ncols() };
    let (x, result) = run(&op, b, &m, Some(x0), &opts(1e-10, 50), None).expect("exact warm start");
    assert_eq!(result.stop_reason, LsmrStopReason::WarmStartExact);
    assert!(result.converged);
    assert_eq!(x, x0);
    assert_eq!(result.iterations, 0);
}

#[test]
fn test_mlsmr_options_are_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}

    assert_send_sync::<(LsmrOptions, Staleness)>();
}

#[test]
fn test_mlsmr_converged_solve_is_never_escalated() {
    let m = IdentityPrecond { dim: 3 };
    let (_, result) = run(
        &m,
        &[1.0, 2.0, 3.0],
        &m,
        None,
        &opts(1e-10, 50),
        Some(stop_at(2)),
    )
    .expect("solve");
    assert!(result.converged);
    assert_ne!(result.stop_reason, LsmrStopReason::Stalled);
}

/// An escalation is a stop like any other: a warm-started one reports its residuals against the
/// original `b`, not against the restart's own initial residual.
#[test]
fn test_warm_escalated_stop_is_measured_against_the_original_rhs() {
    let (op, b) = vandermonde_ls();
    let m = jacobi(&op);
    let (cold, _) = run(&op, &b, &m, None, &opts(1e-12, 4), None).expect("cold rung");

    let options = LsmrOptions {
        local_size: Some(12),
        ..opts(1e-12, 3)
    };
    let warm = |stall| {
        run(&op, &b, &m, Some(&cold), &options, stall)
            .expect("warm rung")
            .1
    };
    // Same three iterations either way; only the exit taken differs.
    let escalated = warm(Some(stop_at(3)));
    let exhausted = warm(None);

    assert_eq!(escalated.stop_reason, LsmrStopReason::Stalled);
    assert_eq!(exhausted.stop_reason, LsmrStopReason::MaxIterations);
    assert_eq!(escalated.iterations, exhausted.iterations);
    assert!(
        (escalated.normal_eq_residual - exhausted.normal_eq_residual).abs()
            <= 1e-12 * exhausted.normal_eq_residual,
        "escalated stop reported {} against the restart's base, not {}",
        escalated.normal_eq_residual,
        exhausted.normal_eq_residual
    );
}

#[test]
fn test_mlsmr_ladder_warm_starts_and_escalates() {
    let (op, b) = vandermonde_ls();
    let weak = IdentityPrecond { dim: op.ncols() };
    let strong = jacobi(&op);
    let options = LsmrOptions {
        local_size: Some(12),
        ..opts(1e-9, 200)
    };
    let stall = Some(stop_at(3));

    let (first, info) = run(&op, &b, &weak, None, &options, stall).expect("first rung");
    assert_eq!(info.stop_reason, LsmrStopReason::Stalled);

    let (middle, info) = run(&op, &b, &weak, Some(&first), &options, stall).expect("middle rung");
    assert_eq!(info.stop_reason, LsmrStopReason::Stalled);

    let (last, info) = run(&op, &b, &strong, Some(&middle), &options, None).expect("last rung");
    assert!(info.converged);

    let (cold, _) = run(&op, &b, &strong, None, &options, None).expect("cold solve");
    let ladder = normal_equation_residual(&op, &last, &b);
    assert!(
        ladder <= 10.0 * normal_equation_residual(&op, &cold, &b).max(1e-14),
        "ladder landed short of a cold solve: {ladder}"
    );
}

#[test]
fn test_staleness_escalates_only_the_stalling_preconditioner() {
    let rule = Staleness::try_new(4, 0.7).expect("valid staleness policy");
    let mut stalled = rule.run();
    for residual in [1.0, 0.8, 0.64, 0.512] {
        assert!(!stalled.stalled(residual));
    }
    assert!(stalled.stalled(0.4096));

    let mut contracting = rule.run();
    for residual in [1.0, 0.5, 0.25, 0.125, 0.0625] {
        assert!(!contracting.stalled(residual));
    }
}

#[test]
fn test_staleness_default_exposes_its_fields() {
    let default = Staleness::default();
    assert_eq!(default.window(), 4);
    assert_eq!(default.threshold(), 0.7);
}

#[cfg(feature = "serde")]
#[test]
fn test_staleness_serde_round_trips() {
    let default = Staleness::default();
    let bytes = postcard::to_stdvec(&default).expect("serialize");
    assert_eq!(
        postcard::from_bytes::<Staleness>(&bytes).expect("deserialize"),
        default
    );
}

#[cfg(feature = "serde")]
#[rstest]
#[case::zero_window(0, 0.7)]
#[case::threshold_one(4, 1.0)]
fn test_staleness_serde_validates(#[case] window: usize, #[case] threshold: f64) {
    let bytes = postcard::to_stdvec(&(window, threshold)).expect("serialize");
    assert!(postcard::from_bytes::<Staleness>(&bytes).is_err());
}

#[test]
fn test_staleness_rejects_a_zero_window() {
    assert!(matches!(
        Staleness::try_new(0, 0.7),
        Err(StalenessError::ZeroWindow)
    ));
}

#[rstest]
fn test_staleness_rejects_an_invalid_threshold(
    #[values(f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.1, 1.0, 1.5)] threshold: f64,
) {
    assert!(matches!(
        Staleness::try_new(4, threshold),
        Err(StalenessError::InvalidThreshold { .. })
    ));
}

/// A budget-exhausted warm start reports against `‖Aᵀb‖` like every other exit. `x₀ = 0.9 x*`
/// makes `rhs = 0.1 b` exactly, so the un-rebased stream reference is 10× too small.
#[test]
fn test_mlsmr_maxiter_residuals_are_rebased_onto_the_original_rhs() {
    let op = vandermonde(30, 12);
    let x_true: Vec<f64> = (0..op.ncols()).map(|j| 1.0 / (1.0 + j as f64)).collect();
    let b = times(&op, &x_true);
    let x0: Vec<f64> = x_true.iter().map(|v| v * 0.9).collect();

    let m = IdentityPrecond { dim: op.ncols() };
    let (x, result) = run(&op, &b, &m, Some(&x0), &opts(1e-15, 3), None).expect("warm solve");

    assert_eq!(result.stop_reason, LsmrStopReason::MaxIterations);
    assert!(!result.converged);

    let expected = normal_equation_residual(&op, &x, &b)
        / normal_equation_residual(&op, &vec![0.0; op.ncols()], &b);
    assert!(
        (result.normal_eq_residual - expected).abs() <= 1e-9 * expected,
        "reported {} vs true relative {expected}",
        result.normal_eq_residual,
    );

    let r: Vec<f64> = b
        .iter()
        .zip(times(&op, &x))
        .map(|(bi, ai)| bi - ai)
        .collect();
    let true_normr = vec_norm(&r);
    assert!((result.residual_norm - true_normr).abs() <= 1e-9 * true_normr);
}

/// The cold audit's metric product `⟨Aᵀb, M⁻¹Aᵀb⟩` overflows at `‖b‖ ≳ 1e154` unless the
/// gradient is scaled first, which failed every warm-started solve at that scale as soon as it
/// converged.
#[test]
fn test_mlsmr_warm_start_audits_at_extreme_magnitude() {
    let b = [1e160, 5e159];
    let x0 = [9e159, 4.5e159];
    let id = IdentityPrecond { dim: 2 };
    let (_, result) = run(&id, &b, &id, Some(&x0), &opts(1e-15, 5), None)
        .expect("the audit must stay representable");

    assert!(result.converged, "{:?}", result.stop_reason);
}
