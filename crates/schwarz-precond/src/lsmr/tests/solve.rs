//! Solve behaviors: preconditioning, rank, and local reorthogonalization.

use rstest::rstest;

use super::super::*;
use crate::lsmr::fixtures::*;

#[test]
fn vec_norm_is_overflow_safe_and_propagates_nan() {
    assert_eq!(vec_norm(&[]), 0.0);
    assert_eq!(vec_norm(&[3.0, 4.0]), 5.0);
    // Scaling keeps the squared sum finite where a naive Σx² would overflow.
    assert!(vec_norm(&[1e300, 1e300]).is_finite());
    // f64::max ignores NaN, so an all-{zero,NaN} vector must still report NaN.
    assert!(vec_norm(&[f64::NAN]).is_nan());
    assert!(vec_norm(&[0.0, f64::NAN]).is_nan());
}

/// Regression: a finite but large-magnitude RHS must not overflow β₁ = ‖b‖ to
/// ∞ inside `init` (unscaled Σb²), which zeroed u₁ and α₁ and returned a silent
/// converged x = 0.
#[test]
fn test_lsmr_large_magnitude_rhs_not_silently_zero() {
    // Entries ~1e155: the unscaled Σb² = 1e310 overflows f64, but the
    // max-scaled ‖b‖ stays finite. A = I, so the exact solution is x = b.
    let b = vec![1e155, 2e155, 3e155];
    let (x, info) = lsmr(&IdentityPrecond { dim: 3 }, &b, 1e-10, 100, None).expect("lsmr solve");

    let all_zero = x.iter().all(|&xi| xi == 0.0);
    assert!(
        !(all_zero && info.converged),
        "large-magnitude b silently returned the all-zero solution as converged",
    );

    let diff: Vec<f64> = x.iter().zip(&b).map(|(x, bi)| x - bi).collect();
    let rel_err = vec_norm(&diff) / vec_norm(&b);
    assert!(rel_err < 1e-6, "relative solution error: {rel_err}");
}

/// Companion for the preconditioned path: within's default solve runs `mlsmr`
/// → `ModifiedGolubKahan::restart`, which carries the same β₁ = ‖b‖ fix. With
/// A = I and M = I the exact solution is again x = b.
#[test]
fn test_mlsmr_large_magnitude_rhs_not_silently_zero() {
    let b = vec![1e155, 2e155, 3e155];
    let op = Mat::<f64>::identity(3, 3);
    let (x, info) =
        run(&op, &b, &op, None, &opts(1e-10, 100), None).expect("preconditioned mlsmr solve");

    let all_zero = x.iter().all(|&xi| xi == 0.0);
    assert!(
        !(all_zero && info.converged),
        "large-magnitude b silently returned the all-zero solution as converged",
    );

    let diff: Vec<f64> = x.iter().zip(&b).map(|(x, bi)| x - bi).collect();
    let rel_err = vec_norm(&diff) / vec_norm(&b);
    assert!(rel_err < 1e-6, "relative solution error: {rel_err}");
}

#[test]
fn test_mlsmr_unpreconditioned() {
    let b = vec![1.0, 2.0, 3.0, 3.0];
    let (x, info) = lsmr(&overdetermined(), &b, 1e-10, 100, None).expect("lsmr solve");
    assert!(info.converged, "MLSMR did not converge");
    let err: f64 = x
        .iter()
        .zip([1.0, 2.0, 3.0].iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();
    assert!(err < 1e-6, "MLSMR solution error: {err}");
}

#[test]
fn test_mlsmr_preconditioned() {
    let b = vec![1.0, 2.0, 3.0, 3.0];
    let preconditioner = diag(&[0.5, 0.5, 1.0]);
    let (x, info) = run(
        &overdetermined(),
        &b,
        &preconditioner,
        None,
        &opts(1e-10, 100),
        None,
    )
    .expect("preconditioned mlsmr solve");
    assert!(info.converged, "Preconditioned MLSMR did not converge");
    let err: f64 = x
        .iter()
        .zip([1.0, 2.0, 3.0].iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();
    assert!(err < 1e-6, "Preconditioned MLSMR solution error: {err}");
}

#[test]
fn test_mlsmr_inconsistent_system() {
    let b = vec![1.0, 2.0, 3.0, 0.0];
    let (x, info) = lsmr(&overdetermined(), &b, 1e-10, 100, None).expect("lsmr solve");
    assert!(
        info.converged,
        "MLSMR did not converge on inconsistent system"
    );
    let normal_resid = normal_equation_residual(&overdetermined(), &x, &b);
    assert!(
        normal_resid < 1e-6,
        "Normal equation residual too large: {normal_resid}"
    );
}

#[test]
fn test_mlsmr_underdetermined_system() {
    let under = dense(2, 3, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    let b = vec![1.0, 2.0];
    let (x, info) = lsmr(&under, &b, 1e-12, 100, None).expect("underdetermined solve");
    assert!(info.converged);
    assert!((x[0] - 1.0).abs() < 1e-10);
    assert!((x[1] - 2.0).abs() < 1e-10);
    assert!(x[2].abs() < 1e-10);
}

#[test]
fn test_mlsmr_rank_deficient_system() {
    let rank_deficient = dense(2, 2, &[1.0, 1.0, 2.0, 2.0]);
    let b = vec![3.0, 6.0];
    let (x, info) = lsmr(&rank_deficient, &b, 1e-12, 100, None).expect("rank-deficient solve");
    assert!(info.converged);
    assert!(((x[0] + x[1]) - 3.0).abs() < 1e-10);
    assert!(normal_equation_residual(&rank_deficient, &x, &b) < 1e-10);
}

/// A preconditioner with a null direction cannot certify the direction it removed.
#[rstest]
#[case(&[1.0, 0.0])]
#[case(&[1.0, 1.0, 0.0])]
#[case(&[1e20, 0.0])]
fn a_singular_preconditioner_cannot_certify_the_direction_it_annihilates(#[case] m: &[f64]) {
    let a = IdentityPrecond { dim: m.len() };
    let b = vec![1.0; m.len()];
    let (x, info) = run(&a, &b, &diag(m), None, &opts(1e-10, 100), None)
        .expect("singular-preconditioner solve");

    assert!(!info.converged, "{:?}", info.stop_reason);
    assert_eq!(info.stop_reason, LsmrStopReason::FalseConvergence);
    let residual = normal_equation_residual(&a, &x, &b);
    assert!(
        (residual - 1.0).abs() < 1e-9,
        "normal-equation residual: {residual}"
    );
}

#[test]
fn test_mlsmr_zero_column_and_zero_row() {
    let b = vec![2.0, 3.0];
    let (x, info) = lsmr(&zero_second_row(), &b, 1e-12, 100, None).expect("degenerate solve");
    assert!(info.converged);
    assert!((x[0] - 2.0).abs() < 1e-10);
    assert!(x[1].abs() < 1e-10);
    assert!(normal_equation_residual(&zero_second_row(), &x, &b) < 1e-10);
}

#[test]
fn test_mlsmr_maxiter_exhaustion() {
    let b = vec![1.0, 2.0, 3.0, 3.0];
    let (_, info) = lsmr(&overdetermined(), &b, 1e-15, 1, None).expect("lsmr solve");
    assert!(
        !info.converged,
        "should not converge in 1 iteration at 1e-15 tol"
    );
    assert_eq!(info.iterations, 1);
    assert_eq!(info.stop_reason, LsmrStopReason::MaxIterations);
}

/// The identity metric as `IdentityPrecond` and as a dense identity are the same algorithm, so
/// they must agree on the solution and the iteration count.
#[test]
fn test_mlsmr_none_matches_identity_precond() {
    let b = vec![1.0, 2.0, 3.0, 3.0];
    let id = Mat::<f64>::identity(3, 3);

    let (none_x, none) = lsmr(&overdetermined(), &b, 1e-12, 100, None).expect("lsmr solve");
    let (id_x, id_info) = run(&overdetermined(), &b, &id, None, &opts(1e-12, 100), None)
        .expect("preconditioned Identity solve");

    assert!(none.converged && id_info.converged);
    assert!(
        (none.iterations as isize - id_info.iterations as isize).abs() <= 1,
        "iteration counts disagree: {} vs {}",
        none.iterations,
        id_info.iterations,
    );

    let diff: f64 = none_x
        .iter()
        .zip(id_x.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();
    assert!(
        diff < 1e-10,
        "GolubKahan vs ModifiedGolubKahan-with-identity solutions disagree: {diff}"
    );
    assert!(
        (none.residual_norm - id_info.residual_norm).abs() < 1e-10,
        "residual norm estimates disagree: {} vs {}",
        none.residual_norm,
        id_info.residual_norm
    );
}

/// Same equivalence guarantee as above but with windowed reorthogonalization
/// active. The M-weighted MGS path uses dot products against `p̃ = M v` and
/// scales `p̃` by `1/α`; with `M = I` this must reduce to the Euclidean
/// MGS used by the unpreconditioned path. Guards the windowed scaling
/// logic in `WindowRing<2>::push` against drift.
#[test]
fn test_mlsmr_none_matches_identity_precond_windowed() {
    // 30×12 Vandermonde, cond(A) ≈ 1e10 — chosen to stress the windowed reorth
    // path (see test_mlsmr_local_reorth_unpreconditioned for rationale).
    let (op, b) = vandermonde_ls();
    let id = Mat::<f64>::identity(op.ncols(), op.ncols());
    let local = Some(10);

    // Tight tolerance with headroom in maxiter: drives both paths to the
    // same minimum so the comparison isn't governed by rounding noise in
    // the convergence test.
    let (none_x, none) = lsmr(&op, &b, 1e-12, 50, local).expect("lsmr windowed solve");
    let options = LsmrOptions {
        local_size: local,
        ..opts(1e-12, 50)
    };
    let (id_x, id_info) =
        run(&op, &b, &id, None, &options, None).expect("preconditioned Identity windowed solve");

    assert!(none.converged && id_info.converged);
    // The two paths do the same algebra differently (par_dot on `v` vs on
    // `p̃ = M v`), so rounding can shift the convergence test by one step.
    // The solve must still land on the same answer.
    assert!(
        (none.iterations as isize - id_info.iterations as isize).abs() <= 1,
        "iteration counts disagree: {} vs {}",
        none.iterations,
        id_info.iterations,
    );

    // Agreement bound is governed by what each path is converging to —
    // the windowed Vandermonde test asserts a normal-equation residual of
    // 1e-6, so 1e-7 here cleanly catches scaling drift in the M-weighted
    // reorth without flagging algebra-order rounding.
    let diff: f64 = none_x
        .iter()
        .zip(id_x.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();
    assert!(
        diff < 1e-7,
        "windowed GolubKahan vs ModifiedGolubKahan-with-identity disagree: {diff}"
    );
    assert!(
        (none.residual_norm - id_info.residual_norm).abs() < 1e-7,
        "windowed residual norm estimates disagree: {} vs {}",
        none.residual_norm,
        id_info.residual_norm
    );
}

/// `local_size = 0` is the no-op fast path — it must match repeated runs
/// bit-for-bit and reproduce the same answer the unwindowed test gets.
/// Guards against the `is_empty()` early return ever drifting.
#[test]
fn test_mlsmr_local_reorth_zero_is_identity() {
    let b = vec![1.0, 2.0, 3.0, 3.0];
    let (x1, r1) = lsmr(&overdetermined(), &b, 1e-10, 100, None).expect("unwindowed solve");
    let (x2, r2) = lsmr(&overdetermined(), &b, 1e-10, 100, Some(0)).expect("zero-window solve");
    assert_eq!(r1.iterations, r2.iterations);
    // Tight tolerance, not exact bit-for-bit, so determinism remains testable
    // if a future refactor adds parallel reductions.
    for (a, b) in x1.iter().zip(&x2) {
        assert!((a - b).abs() < 1e-15, "determinism: {a} vs {b}");
    }
    assert!((r1.residual_norm - r2.residual_norm).abs() < 1e-15);
    assert!(r1.converged);
}

/// Ill-conditioned overdetermined system where the standard short
/// recurrence loses v-orthogonality. Windowed reorthogonalization
/// recovers convergence within the iteration budget.
#[test]
fn test_mlsmr_local_reorth_unpreconditioned() {
    // RHS sampled from a smooth function — well-approximable by the
    // polynomial basis, so the least-squares residual is near zero.
    let (op, b) = vandermonde_ls();
    let tol = 1e-9;
    let maxiter = 30;

    let (_, r0) = lsmr(&op, &b, tol, maxiter, None).expect("no-reorth solve");
    let (x10, r10) = lsmr(&op, &b, tol, maxiter, Some(10)).expect("windowed solve");

    // The windowed solve should reach the tolerance; the unwindowed one
    // typically stalls or overshoots maxiter on this matrix.
    assert!(
        r10.converged,
        "windowed LSMR failed to converge (iters = {})",
        r10.iterations
    );
    assert!(
        !r0.converged || r10.iterations <= r0.iterations,
        "windowed solve must not be slower than unwindowed: r0={} r10={}",
        r0.iterations,
        r10.iterations
    );

    // Verify the windowed solution actually solves the normal equations.
    let normal_resid = normal_equation_residual(&op, &x10, &b);
    assert!(
        normal_resid < 1e-6,
        "normal equation residual: {normal_resid}"
    );
}

/// Same shape but preconditioned: the M-weighted MGS path needs to stay
/// numerically consistent and not lose convergence vs the no-reorth case.
#[test]
fn test_mlsmr_local_reorth_preconditioned() {
    let (op, b) = vandermonde_ls();
    let m = jacobi(&op);
    let options = LsmrOptions {
        local_size: Some(10),
        ..opts(1e-9, 30)
    };
    let (x10, r10) = run(&op, &b, &m, None, &options, None).expect("windowed preconditioned solve");
    assert!(
        r10.converged,
        "windowed preconditioned LSMR failed to converge (iters = {})",
        r10.iterations
    );

    let normal_resid = normal_equation_residual(&op, &x10, &b);
    assert!(
        normal_resid < 1e-6,
        "normal equation residual: {normal_resid}"
    );
}

/// Window sizes at the boundaries of useful values; each must converge to a small
/// normal-equation residual.
#[rstest]
#[case::ring_of_one(1)]
#[case::cols(12)]
#[case::cols_plus_one(13)]
fn test_mlsmr_local_reorth_window_boundary_sizes(#[case] window_size: usize) {
    let (op, b) = vandermonde_ls();

    // 200 iterations lets a ring of one (no real reorthogonalization) converge at cond ≈ 1e10.
    let (x, info) =
        lsmr(&op, &b, 1e-9, 200, Some(window_size)).expect("lsmr boundary-window solve");
    assert!(info.converged);
    assert!(
        normal_equation_residual(&op, &x, &b) < 1e-6,
        "normal-eq residual too large with window {window_size}",
    );
}

/// `α² + β²` overflows for an operator this large, and an infinite `‖A‖` estimate divides the
/// normal-equation ratio to zero, which certifies any residual.
#[test]
fn an_overflowing_operator_norm_estimate_does_not_certify_a_stop() {
    let a = diag(&[1.2e154, 1e154]);
    let (_, info) = lsmr(&a, &[1.0, 1.0], 1e-10, 100, None).expect("extreme-scale solve");

    assert!(info.converged);
    // The refused stop sat at 0.254 after one iteration.
    assert!(
        info.residual_norm < 1e-10,
        "residual norm: {}",
        info.residual_norm
    );
}

/// `‖Aᵀb‖` far below `‖b‖` leaves only the backward-error leg; the plain audit must use it too.
#[test]
fn a_response_nearly_orthogonal_to_the_design_certifies_under_an_identity_preconditioner() {
    let op = dense(3, 2, &[1.0, 0.0, 0.0, 1e-2, 0.0, 0.0]);
    let b = vec![1e-12, 1e-12, 1.0];
    let (_, plain) = lsmr(&op, &b, 1e-10, 50, None).expect("lsmr solve");
    let id = Mat::<f64>::identity(2, 2);
    let (_, id) =
        run(&op, &b, &id, None, &opts(1e-10, 50), None).expect("identity-preconditioned solve");
    assert!(plain.converged);
    assert_eq!(id.converged, plain.converged);
    assert_eq!(id.stop_reason, plain.stop_reason);
}

/// The plain audit's reference is `‖Aᵀb‖`; a metric one rescales it and refuses an honest stop.
#[test]
fn a_rescaling_preconditioner_does_not_deflate_the_plain_audit_reference() {
    let op = dense(3, 2, &[1e6, 0.0, 0.0, 1e-3, 0.0, 0.0]);
    let m = diag(&[1e-12, 1e-16]);
    let (_, r) =
        run(&op, &[1.0, 1.0, 1.0], &m, None, &opts(1e-10, 50), None).expect("rescaled solve");
    assert!(r.converged, "stop_reason: {:?}", r.stop_reason);
}

/// An absolute `ε` floor on `ρρ̄` would drop the whole update on a design with `‖A‖ ≈ 1e-9`.
#[test]
fn a_small_normed_design_still_gets_its_solution_update() {
    let op = dense(2, 1, &[1e-9, 0.0]);
    let (x, r) = lsmr(&op, &[1e-5, 1.0], 1e-10, 1, None).expect("small-normed solve");
    assert!(r.converged, "stop_reason: {:?}", r.stop_reason);
    assert!((x[0] / 1e4 - 1.0).abs() < 1e-9, "{x:?}");
}

/// `ζ/(ρρ̄)` rounds to zero below `2^-1074` although `β₁` scales it back to a normal `x`.
#[rstest]
#[case::overflowing_rhs(960, 1023)]
#[case::dominant_miss(23, 80)]
fn the_solution_update_survives_its_unit_rhs_underflowing(
    #[case] fit_exp: i32,
    #[case] miss_exp: i32,
) {
    let op = dense(2, 1, &[2f64.powi(1023), 0.0]);
    let b = [2f64.powi(fit_exp), 2f64.powi(miss_exp)];
    let (x, r) = lsmr(&op, &b, 1e-10, 10, None).expect("extreme solve");
    assert!(r.converged, "stop_reason: {:?}", r.stop_reason);
    let want = 2f64.powi(fit_exp - 1023);
    assert!((x[0] / want - 1.0).abs() < 1e-12, "{:e} vs {want:e}", x[0]);
}

/// A budget stop's `‖r_k‖` is the returned iterate's own, not LSQR's `|φ̄_k|`.
#[rstest]
fn the_residual_estimate_is_the_iterates_own(#[values(2, 4, 8)] maxiter: usize) {
    let (op, b) = vandermonde_ls();
    let (x, r) = lsmr(&op, &b, 1e-14, maxiter, None).expect("budget solve");
    assert_eq!(r.stop_reason, LsmrStopReason::MaxIterations);

    let residual: Vec<f64> = b
        .iter()
        .zip(times(&op, &x))
        .map(|(bi, ai)| bi - ai)
        .collect();
    let recomputed = vec_norm(&residual);
    assert!(
        (r.residual_norm / recomputed - 1.0).abs() < 1e-8,
        "estimate {:e} vs recomputed {recomputed:e}",
        r.residual_norm
    );
}

/// Power-of-two scaling is exact, so an extreme `‖A‖` or `‖b‖` must reproduce the unit-scale solve.
#[rstest]
#[case::small_design(-560, 0)]
#[case::small_product(-560, -470)]
#[case::large_design(530, 0)]
#[case::large_product(530, 500)]
fn a_power_of_two_scaling_reproduces_the_unit_scale_solve(
    #[case] design_exp: i32,
    #[case] rhs_exp: i32,
    #[values(false, true)] metric: bool,
) {
    let solve = |op: &Mat<f64>, b: &[f64]| {
        let identity = Mat::<f64>::identity(op.ncols(), op.ncols());
        match metric {
            false => lsmr(op, b, 1e-4, 100, None),
            true => run(op, b, &identity, None, &opts(1e-4, 100), None),
        }
        .expect("scaled solve")
    };
    // Conditioned well enough that one rounding apart stays far below the checked agreement.
    let op = vandermonde(30, 6);
    // A sawtooth no low-degree polynomial fits keeps the residual far above the tolerance.
    let b: Vec<f64> = (0..op.nrows())
        .map(|i| (1.0 + i as f64 / (op.nrows() - 1) as f64).ln() + 0.1 * (-1f64).powi(i as i32))
        .collect();
    let (unit_x, unit) = solve(&op, &b);
    assert_eq!(unit.stop_reason, LsmrStopReason::NormalEquationTolerance);

    let (s, t) = (2f64.powi(design_exp), 2f64.powi(rhs_exp));
    let scaled_op = Mat::from_fn(op.nrows(), op.ncols(), |i, j| op[(i, j)] * s);
    let scaled_b: Vec<f64> = b.iter().map(|bi| bi * t).collect();
    let (x, r) = solve(&scaled_op, &scaled_b);

    assert_eq!(
        (r.stop_reason, r.iterations),
        (unit.stop_reason, unit.iterations)
    );
    assert!(r.converged);
    assert!((r.residual_norm / t / unit.residual_norm - 1.0).abs() < 1e-10);
    assert!(
        (r.normal_eq_residual / unit.normal_eq_residual - 1.0).abs() < 1e-6,
        "{:e} vs {:e}",
        r.normal_eq_residual,
        unit.normal_eq_residual
    );
    let x_max = unit_x.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    for (xi, ui) in x.iter().zip(&unit_x) {
        assert!((xi * s / t - ui).abs() < 1e-8 * x_max, "{x:?}");
    }
}
