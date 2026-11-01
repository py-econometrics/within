//! Breakdown, early-exit and input-validation paths.

use rstest::rstest;

use super::super::*;
use crate::lsmr::fixtures::*;
use crate::SolveError;
use faer::{MatMut, MatRef};

/// `zero_second_row` under either metric; the dense identity also zeroes the paired p̃.
fn zero_second_row_solve(b: &[f64], modified: bool) -> (Vec<f64>, LsmrInfo) {
    if modified {
        let identity = Mat::<f64>::identity(2, 2);
        run(
            &zero_second_row(),
            b,
            &identity,
            None,
            &opts(1e-12, 100),
            None,
        )
        .expect("modified Golub-Kahan")
    } else {
        lsmr(&zero_second_row(), b, 1e-12, 100, None).expect("Golub-Kahan")
    }
}

/// b lies in A's range, so `A v₁ − α₁ u₁` collapses and β₂ = 0; the residual estimate is then
/// exactly zero — a converged ResidualTolerance solve, not a distinct breakdown reason.
#[rstest]
#[case::golub_kahan(false)]
#[case::modified_golub_kahan(true)]
fn test_mlsmr_mid_stream_beta_zero_breakdown(#[case] modified: bool) {
    let (x, info) = zero_second_row_solve(&[5.0, 0.0], modified);
    assert!(info.converged);
    assert_eq!(info.iterations, 1);
    assert_eq!(info.stop_reason, LsmrStopReason::ResidualTolerance);
    assert!((x[0] - 5.0).abs() < 1e-12);
    assert!(x[1].abs() < 1e-12);
}

/// `Aᵀb = 0` with `b ≠ 0` triggers the `step1.alpha == 0` early-exit:
/// the solver immediately returns `x = 0` and reports the trivial residual
/// `‖b‖`. Guards the early-exit branch in `mlsmr` / `lsmr_from_bidiag`.
#[test]
fn test_mlsmr_step1_alpha_zero_early_exit() {
    // `A = e₁` as a column, so `Aᵀb = 0` whenever `b₀ = 0`.
    let col_e1 = dense(2, 1, &[1.0, 0.0]);
    let b = vec![0.0, 1.0];
    let (x, info) = lsmr(&col_e1, &b, 1e-12, 100, None).expect("lsmr alpha=0 early exit");
    assert!(info.converged);
    assert_eq!(info.iterations, 0);
    assert_eq!(x, vec![0.0; 1]);
    assert_eq!(
        info.stop_reason,
        LsmrStopReason::InitialNormalEquationResidualZero
    );
    assert!((info.residual_norm - vec_norm(&b)).abs() < 1e-15);
}

/// Every case below reaches the `α₁ = 0` exit: `diag(m)` annihilates `Aᵀ rhs`, or squaring an
/// already-tiny gradient underflows. `None` runs the unpreconditioned stream instead.
fn zero_initial_gradient(
    a: &[f64],
    m: Option<&[f64]>,
    b: &[f64],
    x0: Option<&[f64]>,
) -> (Vec<f64>, LsmrInfo) {
    let a = diag(a);
    match m {
        Some(m) => run(&a, b, &diag(m), x0, &opts(1e-10, 100), None),
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
    let (x, info) = zero_initial_gradient(a, Some(&[1.0, 0.0]), b, x0);
    let residual = normal_equation_residual(&diag(a), &x, b);

    assert!(!info.converged, "{:?}", info.stop_reason);
    assert_eq!(info.stop_reason, LsmrStopReason::FalseConvergence);
    assert_eq!(info.iterations, 0);
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
    let (x, info) = zero_initial_gradient(a, m, b, None);

    assert!(info.converged, "{:?}", info.stop_reason);
    assert_eq!(info.iterations, 1);
    assert!((x[1] / 1e-100 - 1.0).abs() < 1e-9, "{x:?}");
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
    let (_, info) = zero_initial_gradient(a, m, b, x0);

    assert!(info.converged, "{:?}", info.stop_reason);
    assert_eq!(info.iterations, 0);
}

/// A mid-stream bidiagonalization breakdown surfaces as a converged solve, not
/// a distinct stop reason. Whenever a step returns `alpha == 0`, that same
/// rotation step drives `zeta_bar` (the ‖Aᵀr‖ estimate) to exactly zero, so
/// the convergence check fires. On an *inconsistent* breakdown (nonzero
/// residual) it fires as NormalEquationTolerance.
///
/// `zero_second_row` with `b = [5, 3]` reaches `alpha_2 = 0` while leaving a
/// residual of 3, exercising exactly this path on both bidiagonalizations.
#[rstest]
#[case::golub_kahan(false)]
#[case::modified_golub_kahan(true)]
fn test_mid_stream_breakdown_reports_convergence(#[case] modified: bool) {
    let (x, info) = zero_second_row_solve(&[5.0, 3.0], modified);
    assert!(info.converged);
    assert_eq!(info.stop_reason, LsmrStopReason::NormalEquationTolerance);
    // x_0 = 5 fits row 0; row 1 is unmatchable, leaving residual 3.
    assert!((x[0] - 5.0).abs() < 1e-10);
    assert!((info.residual_norm - 3.0).abs() < 1e-10);
}

#[test]
fn test_mlsmr_zero_rhs_stop_reason() {
    let b = vec![0.0; 4];
    let (_, info) = lsmr(&overdetermined(), &b, 1e-12, 100, None).expect("zero-rhs solve");
    assert!(info.converged);
    assert_eq!(info.iterations, 0);
    assert_eq!(info.stop_reason, LsmrStopReason::ZeroRhs);
    assert_eq!(info.residual_norm, 0.0);
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
    let near_breakdown = dense(2, 2, &[-1e-10, -1.0, 1.0, -1e-10]);
    let b = vec![3.0, 4.0];
    let op = IdentityPrecond { dim: 2 };
    let result = run(&op, &b, &near_breakdown, None, &opts(1e-10, 100), None);
    assert!(
        result.is_ok(),
        "rounding-scale negative vp must clamp to breakdown, got err {:?}",
        result.err()
    );
}

/// A length mismatch is a caller bug, so it panics rather than returning an error.
#[test]
#[should_panic(expected = "b and x lengths")]
fn a_short_rhs_panics() {
    let _ = lsmr(&overdetermined(), &[1.0, 2.0], 1e-10, 100, None);
}

#[test]
fn test_mlsmr_rejects_invalid_inputs() {
    let bad_tol = lsmr(
        &overdetermined(),
        &[1.0, 2.0, 3.0, 4.0],
        f64::NAN,
        100,
        None,
    );
    assert!(matches!(bad_tol, Err(SolveError::InvalidInput { .. })));

    let bad_rhs = lsmr(
        &overdetermined(),
        &[1.0, f64::INFINITY, 3.0, 4.0],
        1e-10,
        100,
        None,
    );
    assert!(matches!(bad_rhs, Err(SolveError::InvalidInput { .. })));

    // Finite entrywise, but `‖b‖` overflows and would scale u₁ to zero, reading as α₁ = 0.
    let overflowing_rhs = lsmr(&overdetermined(), &[f64::MAX; 4], 1e-10, 100, None);
    assert!(matches!(
        overflowing_rhs,
        Err(SolveError::InvalidInput { .. })
    ));
}

#[test]
#[should_panic(expected = "preconditioner shape")]
fn test_mlsmr_rejects_bad_preconditioner_shape() {
    let bad_precond = IdentityPrecond { dim: 2 };
    let b = vec![1.0, 2.0, 3.0, 3.0];
    let _ = run(
        &overdetermined(),
        &b,
        &bad_precond,
        None,
        &opts(1e-10, 100),
        None,
    );
}

/// A genuinely indefinite preconditioner (`M⁻¹ = −I` ⇒ `vp = −‖p̃‖²`, relative
/// magnitude 1) must still be rejected — the clamp only ever absorbs a
/// negligible negative direction, never a real loss of positive-definiteness.
#[test]
fn test_mlsmr_rejects_indefinite_preconditioner() {
    let neg_identity = diag(&[-1.0, -1.0]);
    let b = vec![3.0, 4.0];
    let op = IdentityPrecond { dim: 2 };
    let result = run(&op, &b, &neg_identity, None, &opts(1e-10, 100), None);
    assert!(matches!(result, Err(SolveError::InvalidInput { .. })));
}

/// Reorthogonalization cancels the last Krylov direction to noise whose `⟨v, p̃⟩` rounds negative.
#[rstest]
#[case::jacobi(
    dense(
        4,
        3,
        &[
            -0.16929096530790788, -0.4984562277653519, 0.13072317043406312,
            0.13009772651294826, 0.05618700961392087, -0.2772395553399575,
            -0.03358570159357288, 0.3721462377814392, -0.06825699516078332,
            0.0945669171117478, -0.3058591259616409, -0.037184894934893964,
        ],
    ),
    &[0.040795682335399985, 0.4534654228284515, -0.07445456668775552, 0.22621989036785384],
    // Jacobi: `1 / ‖a_j‖²`.
    dense(
        3,
        3,
        &[17.967595266556337, 0.0, 0.0, 0.0, 2.0675757429778487, 0.0, 0.0, 0.0, 10.000794849860734],
    ),
    10,
)]
#[case::ill_conditioned_dense(
    dense(
        5,
        2,
        &[
            -0.13472791693601827, -0.4850321710037593,
            0.8182735477289922, -0.8988020854355594,
            -0.5527756975109148, 0.6831246540331621,
            -0.5254968366620636, 0.7268238608110611,
            0.8935718535267778, -0.11936140914510718,
        ],
    ),
    &[
        0.12656882841955985, 0.22982262397321862, 0.01944542907009761,
        0.05913717014335362, -0.32647238073075635,
    ],
    dense(2, 2, &[1.0, 1.0, 1.0, 1.00000001]),
    2,
)]
fn a_reorthogonalized_breakdown_is_not_an_indefinite_metric(
    #[case] a: Mat<f64>,
    #[case] b: &[f64],
    #[case] metric: Mat<f64>,
    #[case] local_size: usize,
) {
    let options = LsmrOptions {
        local_size: Some(local_size),
        ..opts(1e-10, 50)
    };
    let (x, r) = run(&a, b, &metric, None, &options, None).expect("SPD metric");

    assert!(r.converged, "{:?}", r.stop_reason);
    let zero = vec![0.0; a.ncols()];
    let ratio = normal_equation_residual(&a, &x, b) / normal_equation_residual(&a, &zero, b);
    assert!(ratio <= 1e-12, "{ratio:e}");
}

/// `A` in unfused row order, the rounding these cases' `vp` signs were found under.
#[derive(Debug)]
struct RowOrder(Mat<f64>);

impl LinOp<f64> for RowOrder {
    fn apply_scratch(&self, _: usize, _: Par) -> StackReq {
        StackReq::EMPTY
    }
    fn nrows(&self) -> usize {
        self.0.nrows()
    }
    fn ncols(&self) -> usize {
        self.0.ncols()
    }
    fn apply(&self, mut out: MatMut<'_, f64>, x: MatRef<'_, f64>, _: Par, _: &mut MemStack) {
        for i in 0..self.0.nrows() {
            out[(i, 0)] = (0..self.0.ncols())
                .map(|j| self.0[(i, j)] * x[(j, 0)])
                .sum();
        }
    }
    fn conj_apply(&self, out: MatMut<'_, f64>, x: MatRef<'_, f64>, p: Par, s: &mut MemStack) {
        self.apply(out, x, p, s)
    }
}

impl BiLinOp<f64> for RowOrder {
    fn transpose_apply_scratch(&self, _: usize, _: Par) -> StackReq {
        StackReq::EMPTY
    }
    fn transpose_apply(
        &self,
        mut out: MatMut<'_, f64>,
        u: MatRef<'_, f64>,
        _: Par,
        _: &mut MemStack,
    ) {
        for j in 0..self.0.ncols() {
            out[(j, 0)] = (0..self.0.nrows()).fold(0.0, |s, i| s + self.0[(i, j)] * u[(i, 0)]);
        }
    }
    fn adjoint_apply(&self, out: MatMut<'_, f64>, u: MatRef<'_, f64>, p: Par, s: &mut MemStack) {
        self.transpose_apply(out, u, p, s)
    }
}

/// Recomputing `v = M⁻¹ p̃` keeps a negative direction negative: the refresh never absorbs it.
#[rstest]
#[case::one_negative_entry(
    dense(
        4,
        3,
        &[1.0, 0.3, 0.0, 0.2, 1.0, 0.1, 0.0, 0.4, 1.0, 0.5, 0.5, 0.5],
    ),
    &[1.0, 2.0, -1.0, 0.5],
    &[1.0, 1.0, -0.5],
)]
// The window cancels the positive direction, leaving a remainder of `vp ≈ −1.6e-49` to refresh.
#[case::negative_remainder(
    dense(
        3,
        2,
        &[
            -1.2991112911427987, 1e-24,
            -0.6591885378662519, -1e-24,
            0.8763697896531903, 0.0,
        ],
    ),
    &[1.0, 1.0, 1.0],
    &[1.0, -1.0],
)]
fn an_indefinite_metric_is_refused_after_reorthogonalization(
    #[case] a: Mat<f64>,
    #[case] b: &[f64],
    #[case] m: &[f64],
    #[values(None, Some(1), Some(2), Some(10))] local_size: Option<usize>,
) {
    let options = LsmrOptions {
        local_size,
        ..opts(1e-12, 50)
    };
    let r = run(&RowOrder(a), b, &diag(m), None, &options, None);
    assert!(matches!(r, Err(SolveError::InvalidInput { .. })), "{m:?}");
}
