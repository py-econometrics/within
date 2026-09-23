//! Solve behaviors: preconditioning, rank, and local reorthogonalization.

use rstest::rstest;

use super::super::*;
use crate::lsmr::fixtures::*;
use crate::{Operator, SolveError};

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
    let result = lsmr(&IdentityOp { n: 3 }, &b, 1e-10, 100, None).expect("lsmr solve");

    let all_zero = result.x.iter().all(|&xi| xi == 0.0);
    assert!(
        !(all_zero && result.converged),
        "large-magnitude b silently returned the all-zero solution as converged",
    );

    let diff: Vec<f64> = result.x.iter().zip(&b).map(|(x, bi)| x - bi).collect();
    let rel_err = vec_norm(&diff) / vec_norm(&b);
    assert!(rel_err < 1e-6, "relative solution error: {rel_err}");
}

/// Companion for the preconditioned path: within's default solve runs `mlsmr`
/// → `ModifiedGolubKahan::init`, which carries the same β₁ = ‖b‖ fix. With
/// A = I and M = I the exact solution is again x = b.
#[test]
fn test_mlsmr_large_magnitude_rhs_not_silently_zero() {
    let b = vec![1e155, 2e155, 3e155];
    let op = IdentityOp { n: 3 };
    let result = mlsmr(&op, &b, &op, 1e-10, 100, MlsmrOptions::default())
        .expect("preconditioned mlsmr solve");

    let all_zero = result.x.iter().all(|&xi| xi == 0.0);
    assert!(
        !(all_zero && result.converged),
        "large-magnitude b silently returned the all-zero solution as converged",
    );

    let diff: Vec<f64> = result.x.iter().zip(&b).map(|(x, bi)| x - bi).collect();
    let rel_err = vec_norm(&diff) / vec_norm(&b);
    assert!(rel_err < 1e-6, "relative solution error: {rel_err}");
}

#[test]
fn test_mlsmr_unpreconditioned() {
    let b = vec![1.0, 2.0, 3.0, 3.0];
    let result = lsmr(&OverdeterminedOp, &b, 1e-10, 100, None).expect("lsmr solve");
    assert!(result.converged, "MLSMR did not converge");
    let err: f64 = result
        .x
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
    let preconditioner = DiagOp(vec![0.5, 0.5, 1.0]);
    let result = mlsmr(
        &OverdeterminedOp,
        &b,
        &preconditioner,
        1e-10,
        100,
        MlsmrOptions::default(),
    )
    .expect("preconditioned mlsmr solve");
    assert!(result.converged, "Preconditioned MLSMR did not converge");
    let err: f64 = result
        .x
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
    let result = lsmr(&OverdeterminedOp, &b, 1e-10, 100, None).expect("lsmr solve");
    assert!(
        result.converged,
        "MLSMR did not converge on inconsistent system"
    );
    let normal_resid = normal_equation_residual(&OverdeterminedOp, &result.x, &b);
    assert!(
        normal_resid < 1e-6,
        "Normal equation residual too large: {normal_resid}"
    );
}

#[test]
fn test_mlsmr_underdetermined_system() {
    struct UnderOp;
    impl Operator for UnderOp {
        fn nrows(&self) -> usize {
            2
        }
        fn ncols(&self) -> usize {
            3
        }
        fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
            y[0] = x[0];
            y[1] = x[1];
            Ok(())
        }
        fn apply_adjoint(&self, u: &[f64], x: &mut [f64]) -> Result<(), SolveError> {
            x[0] = u[0];
            x[1] = u[1];
            x[2] = 0.0;
            Ok(())
        }
    }

    let b = vec![1.0, 2.0];
    let result = lsmr(&UnderOp, &b, 1e-12, 100, None).expect("underdetermined solve");
    assert!(result.converged);
    assert!((result.x[0] - 1.0).abs() < 1e-10);
    assert!((result.x[1] - 2.0).abs() < 1e-10);
    assert!(result.x[2].abs() < 1e-10);
}

#[test]
fn test_mlsmr_rank_deficient_system() {
    struct RankDeficientOp;
    impl Operator for RankDeficientOp {
        fn nrows(&self) -> usize {
            2
        }
        fn ncols(&self) -> usize {
            2
        }
        fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
            let s = x[0] + x[1];
            y[0] = s;
            y[1] = 2.0 * s;
            Ok(())
        }
        fn apply_adjoint(&self, u: &[f64], x: &mut [f64]) -> Result<(), SolveError> {
            let s = u[0] + 2.0 * u[1];
            x[0] = s;
            x[1] = s;
            Ok(())
        }
    }

    let b = vec![3.0, 6.0];
    let result = lsmr(&RankDeficientOp, &b, 1e-12, 100, None).expect("rank-deficient solve");
    assert!(result.converged);
    assert!(((result.x[0] + result.x[1]) - 3.0).abs() < 1e-10);
    assert!(normal_equation_residual(&RankDeficientOp, &result.x, &b) < 1e-10);
}

/// A preconditioner with a null direction reports `Aᵀr` as zero along it, so the metric audit
/// alone certifies a stop that still carries a full unit of normal-equation residual. Scaling
/// `M⁻¹` must not buy certification either: the plain leg has to be free of that scale.
#[rstest]
#[case::three_dim(&[1.0, 1.0, 0.0])]
#[case::two_dim(&[1.0, 0.0])]
#[case::two_dim_scaled(&[1e20, 0.0])]
fn a_singular_preconditioner_cannot_certify_the_direction_it_annihilates(#[case] m: &[f64]) {
    let a = IdentityOp { n: m.len() };
    let b = vec![1.0; m.len()];
    let result = mlsmr(
        &a,
        &b,
        &DiagOp(m.to_vec()),
        1e-10,
        100,
        MlsmrOptions::default(),
    )
    .expect("singular-preconditioner solve");

    assert!(!result.converged, "{:?}", result.stop_reason);
    assert_eq!(result.stop_reason, LsmrStopReason::FalseConvergence);
    let residual = normal_equation_residual(&a, &result.x, &b);
    assert!(
        (residual - 1.0).abs() < 1e-9,
        "normal-equation residual: {residual}"
    );
}

#[test]
fn test_mlsmr_zero_column_and_zero_row() {
    let b = vec![2.0, 3.0];
    let result = lsmr(&ZeroSecondRow, &b, 1e-12, 100, None).expect("degenerate solve");
    assert!(result.converged);
    assert!((result.x[0] - 2.0).abs() < 1e-10);
    assert!(result.x[1].abs() < 1e-10);
    assert!(normal_equation_residual(&ZeroSecondRow, &result.x, &b) < 1e-10);
}

#[test]
fn test_mlsmr_maxiter_exhaustion() {
    let b = vec![1.0, 2.0, 3.0, 3.0];
    let result = lsmr(&OverdeterminedOp, &b, 1e-15, 1, None).expect("lsmr solve");
    assert!(
        !result.converged,
        "should not converge in 1 iteration at 1e-15 tol"
    );
    assert_eq!(result.iterations, 1);
    assert_eq!(result.stop_reason, LsmrStopReason::MaxIterations);
}

/// `lsmr` (GolubKahan path) and `mlsmr` with `M = I`
/// (ModifiedGolubKahan with the identity) are mathematically the same
/// algorithm.
/// They should produce numerically equivalent solutions and iteration
/// counts; this guards against future drift between the two
/// bidiagonalization implementations.
#[test]
fn test_mlsmr_none_matches_identity_precond() {
    let b = vec![1.0, 2.0, 3.0, 3.0];
    let id = IdentityOp { n: 3 };

    let none_result = lsmr(&OverdeterminedOp, &b, 1e-12, 100, None).expect("lsmr solve");
    let id_result = mlsmr(
        &OverdeterminedOp,
        &b,
        &id,
        1e-12,
        100,
        MlsmrOptions::default(),
    )
    .expect("preconditioned Identity solve");

    assert!(none_result.converged && id_result.converged);
    assert!(
        (none_result.iterations as isize - id_result.iterations as isize).abs() <= 1,
        "iteration counts disagree: {} vs {}",
        none_result.iterations,
        id_result.iterations,
    );

    let diff: f64 = none_result
        .x
        .iter()
        .zip(id_result.x.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();
    assert!(
        diff < 1e-10,
        "GolubKahan vs ModifiedGolubKahan-with-identity solutions disagree: {diff}"
    );
    assert!(
        (none_result.residual_norm - id_result.residual_norm).abs() < 1e-10,
        "residual norm estimates disagree: {} vs {}",
        none_result.residual_norm,
        id_result.residual_norm
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
    let op = DenseOp::vandermonde(30, 12);
    let b: Vec<f64> = (0..op.rows)
        .map(|i| {
            let x = i as f64 / (op.rows - 1) as f64;
            (1.0 + x).ln()
        })
        .collect();
    let id = IdentityOp { n: op.cols };
    let local = Some(10);

    // Tight tolerance with headroom in maxiter: drives both paths to the
    // same minimum so the comparison isn't governed by rounding noise in
    // the convergence test.
    let none_result = lsmr(&op, &b, 1e-12, 50, local).expect("lsmr windowed solve");
    let opts = MlsmrOptions {
        local_size: local,
        ..Default::default()
    };
    let id_result =
        mlsmr(&op, &b, &id, 1e-12, 50, opts).expect("preconditioned Identity windowed solve");

    assert!(none_result.converged && id_result.converged);
    // The two paths do the same algebra differently (par_dot on `v` vs on
    // `p̃ = M v`), so rounding can shift the convergence test by one step.
    // The solve must still land on the same answer.
    assert!(
        (none_result.iterations as isize - id_result.iterations as isize).abs() <= 1,
        "iteration counts disagree: {} vs {}",
        none_result.iterations,
        id_result.iterations,
    );

    // Agreement bound is governed by what each path is converging to —
    // the windowed Vandermonde test asserts a normal-equation residual of
    // 1e-6, so 1e-7 here cleanly catches scaling drift in the M-weighted
    // reorth without flagging algebra-order rounding.
    let diff: f64 = none_result
        .x
        .iter()
        .zip(id_result.x.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();
    assert!(
        diff < 1e-7,
        "windowed GolubKahan vs ModifiedGolubKahan-with-identity disagree: {diff}"
    );
    assert!(
        (none_result.residual_norm - id_result.residual_norm).abs() < 1e-7,
        "windowed residual norm estimates disagree: {} vs {}",
        none_result.residual_norm,
        id_result.residual_norm
    );
}

/// `local_size = 0` is the no-op fast path — it must match repeated runs
/// bit-for-bit and reproduce the same answer the unwindowed test gets.
/// Guards against the `is_empty()` early return ever drifting.
#[test]
fn test_mlsmr_local_reorth_zero_is_identity() {
    let b = vec![1.0, 2.0, 3.0, 3.0];
    let r1 = lsmr(&OverdeterminedOp, &b, 1e-10, 100, None).expect("unwindowed solve");
    let r2 = lsmr(&OverdeterminedOp, &b, 1e-10, 100, Some(0)).expect("zero-window solve");
    assert_eq!(r1.iterations, r2.iterations);
    // Tight tolerance, not exact bit-for-bit, so determinism remains testable
    // if a future refactor adds parallel reductions.
    for (a, b) in r1.x.iter().zip(&r2.x) {
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
    let op = DenseOp::vandermonde(30, 12);
    // RHS sampled from a smooth function — well-approximable by the
    // polynomial basis, so the least-squares residual is near zero.
    let b: Vec<f64> = (0..op.rows)
        .map(|i| {
            let x = i as f64 / (op.rows - 1) as f64;
            (1.0 + x).ln()
        })
        .collect();
    let tol = 1e-9;
    let maxiter = 30;

    let r0 = lsmr(&op, &b, tol, maxiter, None).expect("no-reorth solve");
    let r10 = lsmr(&op, &b, tol, maxiter, Some(10)).expect("windowed solve");

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
    let normal_resid = normal_equation_residual(&op, &r10.x, &b);
    assert!(
        normal_resid < 1e-6,
        "normal equation residual: {normal_resid}"
    );
}

/// Same shape but preconditioned: the M-weighted MGS path needs to stay
/// numerically consistent and not lose convergence vs the no-reorth case.
#[test]
fn test_mlsmr_local_reorth_preconditioned() {
    let op = DenseOp::vandermonde(30, 12);
    let m = jacobi(&op);

    let b: Vec<f64> = (0..op.rows)
        .map(|i| {
            let x = i as f64 / (op.rows - 1) as f64;
            (1.0 + x).ln()
        })
        .collect();
    let tol = 1e-9;
    let maxiter = 30;

    let opts = MlsmrOptions {
        local_size: Some(10),
        ..Default::default()
    };
    let r10 = mlsmr(&op, &b, &m, tol, maxiter, opts).expect("windowed preconditioned solve");
    assert!(
        r10.converged,
        "windowed preconditioned LSMR failed to converge (iters = {})",
        r10.iterations
    );

    let normal_resid = normal_equation_residual(&op, &r10.x, &b);
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
    let op = DenseOp::vandermonde(30, 12);
    let b: Vec<f64> = (0..op.rows)
        .map(|i| {
            let x = i as f64 / (op.rows - 1) as f64;
            (1.0 + x).ln()
        })
        .collect();

    // 200 iterations lets a ring of one (no real reorthogonalization) converge at cond ≈ 1e10.
    let result = lsmr(&op, &b, 1e-9, 200, Some(window_size)).expect("lsmr boundary-window solve");
    assert!(result.converged);
    assert!(
        normal_equation_residual(&op, &result.x, &b) < 1e-6,
        "normal-eq residual too large with window {window_size}",
    );
}

/// `α² + β²` overflows for an operator this large, and an infinite `‖A‖` estimate divides the
/// normal-equation ratio to zero, which certifies any residual.
#[test]
fn an_overflowing_operator_norm_estimate_does_not_certify_a_stop() {
    let a = DiagOp(vec![1.2e154, 1e154]);
    let result = lsmr(&a, &[1.0, 1.0], 1e-10, 100, None).expect("extreme-scale solve");

    assert!(result.converged);
    // The refused stop sat at 0.254 after one iteration.
    assert!(
        result.residual_norm < 1e-10,
        "residual norm: {}",
        result.residual_norm
    );
}

/// `‖Aᵀb‖` far below `‖b‖` leaves only the backward-error leg; the plain audit must use it too.
#[test]
fn a_response_nearly_orthogonal_to_the_design_certifies_under_an_identity_preconditioner() {
    let op = DenseOp {
        rows: 3,
        cols: 2,
        data: vec![1.0, 0.0, 0.0, 1e-2, 0.0, 0.0],
    };
    let b = vec![1e-12, 1e-12, 1.0];
    let plain = lsmr(&op, &b, 1e-10, 50, None).expect("lsmr solve");
    let id = mlsmr(
        &op,
        &b,
        &IdentityOp { n: op.cols },
        1e-10,
        50,
        MlsmrOptions::default(),
    )
    .expect("identity-preconditioned solve");
    assert!(plain.converged);
    assert_eq!(id.converged, plain.converged);
    assert_eq!(id.stop_reason, plain.stop_reason);
}

/// The plain audit's reference is `‖Aᵀb‖`; a metric one rescales it and refuses an honest stop.
#[test]
fn a_rescaling_preconditioner_does_not_deflate_the_plain_audit_reference() {
    let op = DenseOp {
        rows: 3,
        cols: 2,
        data: vec![1e6, 0.0, 0.0, 1e-3, 0.0, 0.0],
    };
    let m = DiagOp(vec![1e-12, 1e-16]);
    let r = mlsmr(
        &op,
        &[1.0, 1.0, 1.0],
        &m,
        1e-10,
        50,
        MlsmrOptions::default(),
    )
    .expect("rescaled solve");
    assert!(r.converged, "stop_reason: {:?}", r.stop_reason);
}

/// An absolute `ε` floor on `ρρ̄` would drop the whole update on a design with `‖A‖ ≈ 1e-9`.
#[test]
fn a_small_normed_design_still_gets_its_solution_update() {
    let op = DenseOp {
        rows: 2,
        cols: 1,
        data: vec![1e-9, 0.0],
    };
    let r = lsmr(&op, &[1e-5, 1.0], 1e-10, 1, None).expect("small-normed solve");
    assert!(r.converged, "stop_reason: {:?}", r.stop_reason);
    assert!((r.x[0] / 1e4 - 1.0).abs() < 1e-9, "{:?}", r.x);
}

/// `ρρ̄` scales with `‖A‖²`, so it leaves the double range long before `‖A‖` itself does.
#[rstest]
#[case::underflowing(1e-163)]
#[case::overflowing(1e160)]
fn an_extreme_scale_design_still_gets_its_solution_update(#[case] s: f64) {
    let r = lsmr(&DiagOp(vec![s, 2.0 * s]), &[1.0, 1.0], 1e-10, 50, None).expect("extreme solve");
    assert!(r.converged, "stop_reason: {:?}", r.stop_reason);
    for (xi, want) in r.x.iter().zip([1.0 / s, 0.5 / s]) {
        assert!((xi / want - 1.0).abs() < 1e-12, "{:?}", r.x);
    }
}

/// A budget stop's `‖r_k‖` is the returned iterate's own, not LSQR's `|φ̄_k|`.
#[rstest]
fn the_residual_estimate_is_the_iterates_own(#[values(2, 4, 8)] maxiter: usize) {
    let (op, b) = vandermonde_ls();
    let r = lsmr(&op, &b, 1e-14, maxiter, None).expect("budget solve");
    assert_eq!(r.stop_reason, LsmrStopReason::MaxIterations);

    let mut residual = vec![0.0; b.len()];
    op.apply(&r.x, &mut residual).expect("apply");
    for (ri, &bi) in residual.iter_mut().zip(&b) {
        *ri = bi - *ri;
    }
    let recomputed = vec_norm(&residual);
    assert!(
        (r.residual_norm / recomputed - 1.0).abs() < 1e-8,
        "estimate {:e} vs recomputed {recomputed:e}",
        r.residual_norm
    );
}
