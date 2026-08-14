//! Warm starts and escalation: the preconditioner ladder.

use super::super::*;
use crate::lsmr::fixtures::*;
use crate::{Operator, SolveError};

#[test]
fn test_mlsmr_warm_tolerance_is_relative_to_original_rhs() {
    let op = DenseOp::vandermonde(30, 12);
    let x_true: Vec<f64> = (0..op.cols).map(|j| 1.0 / (1.0 + j as f64)).collect();
    let mut b = vec![0.0; op.rows];
    op.apply(&x_true, &mut b).expect("apply");
    let x0: Vec<f64> = x_true.iter().map(|v| v * 0.9999).collect();
    let tol = 1e-6;
    let window = Some(12);
    let m = jacobi(&op);
    let base = MlsmrOptions {
        local_size: window,
        ..Default::default()
    };
    let warm = mlsmr(
        &op,
        &b,
        &m,
        tol,
        200,
        MlsmrOptions {
            warm_start: Some(&x0),
            ..base
        },
    )
    .expect("warm solve");
    assert_eq!(warm.stop_reason, LsmrStopReason::ResidualTolerance);

    let mut r = vec![0.0; op.rows];
    op.apply(&x0, &mut r).expect("apply");
    for (ri, bi) in r.iter_mut().zip(&b) {
        *ri = bi - *ri;
    }
    let naive = mlsmr(&op, &r, &m, tol, 200, base).expect("naive restart");
    assert!(
        warm.iterations < naive.iterations,
        "warm took {} iters, naive restart {} — the tolerance basis is not `‖b‖`",
        warm.iterations,
        naive.iterations
    );
}

/// A zero RHS must not shadow warm-start validation: the same bad `x0` has to be
/// rejected whether or not `b` short-circuits the solve.
#[test]
fn test_mlsmr_rejects_bad_warm_start_for_any_rhs() {
    let op = IdentityOp { n: 3 };
    for b in [[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]] {
        for x0 in [&[0.0, 0.0][..], &[0.0, f64::NAN, 0.0]] {
            let options = MlsmrOptions {
                warm_start: Some(x0),
                ..Default::default()
            };
            assert!(matches!(
                mlsmr(&op, &b, &op, 1e-8, 10, options),
                Err(SolveError::InvalidInput { .. })
            ));
        }
    }
}

/// Regression: `A·x₀` overflowing made `b - A·x₀` norm to NaN, which `init` read
/// as β₁ = 0 and reported as a converged solve with `x = x₀` and a NaN residual.
#[test]
fn test_mlsmr_warm_rejects_overflowing_warm_start_residual() {
    let op = DiagOp(vec![1e200; 3]);
    let m = IdentityOp { n: 3 };
    let x0 = [1e200; 3];
    let options = MlsmrOptions {
        warm_start: Some(&x0),
        ..Default::default()
    };
    assert!(matches!(
        mlsmr(&op, &[1.0, 2.0, 3.0], &m, 1e-8, 10, options),
        Err(SolveError::InvalidInput { .. })
    ));
}

#[test]
fn test_mlsmr_zero_rhs_corrects_non_exact_warm_start() {
    let op = IdentityOp { n: 3 };
    let x0 = [1.0, 2.0, 3.0];
    let options = MlsmrOptions {
        warm_start: Some(&x0),
        ..Default::default()
    };
    let result = mlsmr(&op, &[0.0, 0.0, 0.0], &op, 1e-10, 10, options).expect("warm correction");

    assert!(result.converged);
    assert!(vec_norm(&result.x) < 1e-12);
    assert!(result.iterations > 0);
}

#[test]
fn test_mlsmr_exact_warm_start_is_returned_untouched() {
    let cases: [(&dyn Operator, &[f64], &[f64]); 2] = [
        (&IdentityOp { n: 3 }, &[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]),
        (&ZeroSecondRow, &[0.0, 0.0], &[0.0, 7.0]),
    ];
    for (op, b, x0) in cases {
        let options = MlsmrOptions {
            warm_start: Some(x0),
            ..Default::default()
        };
        let m = IdentityOp { n: op.ncols() };
        let result = mlsmr(op, b, &m, 1e-10, 50, options).expect("exact warm start");
        assert_eq!(result.stop_reason, LsmrStopReason::WarmStartExact);
        assert!(result.converged);
        assert_eq!(result.x, x0);
        assert_eq!(result.iterations, 0);
    }
}

#[test]
fn test_mlsmr_options_are_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}

    assert_send_sync::<MlsmrOptions<'static>>();
}

#[test]
fn test_mlsmr_converged_solve_is_never_escalated() {
    let opts = MlsmrOptions {
        escalation: Some(&FixedIterations(1)),
        ..Default::default()
    };
    let m = IdentityOp { n: 3 };
    let result = mlsmr(&IdentityOp { n: 3 }, &[1.0, 2.0, 3.0], &m, 1e-10, 50, opts).expect("solve");
    assert!(result.converged);
    assert_ne!(result.stop_reason, LsmrStopReason::Escalated);
}

#[test]
fn test_mlsmr_ladder_warm_starts_and_escalates() {
    let op = DenseOp::vandermonde(30, 12);
    let b: Vec<f64> = (0..op.rows)
        .map(|i| (1.0 + i as f64 / (op.rows - 1) as f64).ln())
        .collect();
    let (tol, window) = (1e-9, Some(12));
    let weak = IdentityOp { n: op.cols };
    let strong = jacobi(&op);
    let rung = MlsmrOptions {
        escalation: Some(&FixedIterations(3)),
        local_size: window,
        ..Default::default()
    };
    let final_rung = MlsmrOptions {
        local_size: window,
        ..Default::default()
    };

    let first = mlsmr(&op, &b, &weak, tol, 200, rung).expect("first rung");
    assert_eq!(first.stop_reason, LsmrStopReason::Escalated);

    let middle = mlsmr(
        &op,
        &b,
        &weak,
        tol,
        200,
        MlsmrOptions {
            warm_start: Some(&first.x),
            ..rung
        },
    )
    .expect("middle rung");
    assert_eq!(middle.stop_reason, LsmrStopReason::Escalated);

    let last = mlsmr(
        &op,
        &b,
        &strong,
        tol,
        200,
        MlsmrOptions {
            warm_start: Some(&middle.x),
            ..final_rung
        },
    )
    .expect("last rung");
    assert!(last.converged);

    let cold = mlsmr(&op, &b, &strong, tol, 200, final_rung).expect("cold solve");
    let ladder = normal_equation_residual(&op, &last.x, &b);
    assert!(
        ladder <= 10.0 * normal_equation_residual(&op, &cold.x, &b).max(1e-14),
        "ladder landed short of a cold solve: {ladder}"
    );
}

#[test]
fn test_staleness_escalates_only_the_stalling_preconditioner() {
    let rule = Staleness::try_new(4, 0.7).expect("valid staleness policy");
    let progress = |iteration, normal_eq_residual| Progress {
        iteration,
        normal_eq_residual,
    };
    let mut stalled = rule.handler();
    for (iteration, residual) in [1.0, 0.8, 0.64, 0.512].into_iter().enumerate() {
        assert!(!stalled.should_escalate(progress(iteration + 1, residual)));
    }
    assert!(stalled.should_escalate(progress(5, 0.4096)));

    let mut contracting = rule.handler();
    for (iteration, residual) in [1.0, 0.5, 0.25, 0.125, 0.0625].into_iter().enumerate() {
        assert!(!contracting.should_escalate(progress(iteration + 1, residual)));
    }
}

#[test]
fn test_staleness_rejects_invalid_configuration() {
    assert!(matches!(
        Staleness::try_new(0, 0.7),
        Err(StalenessError::ZeroWindow)
    ));
    for threshold in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.1, 1.0, 1.5] {
        assert!(matches!(
            Staleness::try_new(4, threshold),
            Err(StalenessError::InvalidThreshold { .. })
        ));
    }
}
