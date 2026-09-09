//! White-box checks on the bidiagonalization itself.

use super::{alpha_from_vp, dot, Bidiagonalization, GolubKahan};
use crate::lsmr::fixtures::{DenseOp, DiagOp};
use crate::{Operator, SolveError};

/// A NaN `vp` clamped to α = 0 via `f64::max`; a product bound overflowed to ∞ or underflowed to 0.
#[test]
fn alpha_from_vp_rejects_non_finite_and_indefinite_pairs() {
    let rejected = [
        (vec![1.0, f64::NAN], vec![1.0, 1.0]),
        (vec![1.0, f64::INFINITY], vec![1.0, 1.0]),
        (vec![-1e100], vec![1e100]),
        (vec![-1e302, f64::MAX, f64::MAX], vec![1.0, 0.0, 0.0]),
        (vec![-2e300, 1e308], vec![1e-316, 0.0]),
    ];
    for (v, p) in &rejected {
        assert!(
            matches!(alpha_from_vp(v, p), Err(SolveError::InvalidInput { .. })),
            "{v:?}·{p:?} accepted"
        );
    }
    let clamped = [
        (vec![1.0, 1.0], vec![1.0, -1.0 - 1e-12]),
        (vec![1e-100, 1e-100], vec![1e-100, -1.000000000001e-100]),
        (vec![1e-316, 1e-316], vec![1e308, -1.000000000001e308]),
    ];
    for (v, p) in &clamped {
        assert_eq!(alpha_from_vp(v, p).expect("within √ε"), 0.0, "{v:?}·{p:?}");
    }
}

/// `beta == 0.0` and `alpha > 0.0` are both false for NaN; unguarded, an overflow poisons the run.
#[test]
fn a_non_finite_operator_norm_is_an_error() {
    for bad in [f64::NAN, f64::MAX] {
        let result = crate::lsmr::lsmr(&DiagOp(vec![bad, 1.0]), &[1.0, 1.0], 1e-10, 50, None);
        assert!(
            matches!(result, Err(SolveError::InvalidInput { .. })),
            "{bad:e} accepted"
        );
    }
}

/// A finite adjoint keeps `init` clean, so the overflow reaches the `step` guard on β.
#[test]
fn an_overflow_after_initialization_is_an_error() {
    struct OverflowingForward;
    impl Operator for OverflowingForward {
        fn nrows(&self) -> usize {
            2
        }
        fn ncols(&self) -> usize {
            2
        }
        fn apply(&self, _x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
            y.fill(f64::MAX);
            Ok(())
        }
        fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
            y.copy_from_slice(x);
            Ok(())
        }
    }
    let (mut bidiag, first) =
        GolubKahan::init(&OverflowingForward, &[1.0, 1.0], 0).expect("finite init");
    assert!(first.alpha > 0.0);
    let err = bidiag.step().err().expect("an overflowing β was accepted");
    assert!(
        matches!(&err, SolveError::InvalidInput { message, .. } if message.contains("β")),
        "{err}"
    );
}

/// Window smaller than the iteration count: the ring must wrap correctly.
/// We re-run the bidiagonalization manually with the same window and
/// verify the last `local_size` `v` vectors are mutually orthogonal to
/// tighter tolerance than they would be without reorthogonalization.
#[test]
fn local_reorth_keeps_the_window_vectors_orthogonal() {
    let op = DenseOp::vandermonde(30, 12);
    let b: Vec<f64> = (0..op.rows)
        .map(|i| {
            let x = i as f64 / (op.rows - 1) as f64;
            (1.0 + x).ln()
        })
        .collect();

    let local_size = 3;
    let n_iters = 10;

    // Run the bidiagonalization directly so we can capture v_k after each
    // step. Mirrors the body of `lsmr_from_bidiag` minus the recurrence.
    let collect_vs = |window_size: usize| -> Vec<Vec<f64>> {
        let (mut bidiag, _) = GolubKahan::init(&op, &b, window_size).expect("init");
        let mut vs = vec![bidiag.v().to_vec()];
        for _ in 0..n_iters {
            bidiag.step().expect("step");
            vs.push(bidiag.v().to_vec());
        }
        vs
    };

    let vs_no_reorth = collect_vs(0);
    let vs_windowed = collect_vs(local_size);

    // Compare the maximum |⟨v_i, v_j⟩| over the last `local_size` vectors.
    let max_off_diag = |vs: &[Vec<f64>]| -> f64 {
        let n = vs.len();
        let start = n.saturating_sub(local_size);
        let mut worst: f64 = 0.0;
        for i in start..n {
            for j in (i + 1)..n {
                worst = worst.max(dot(&vs[i], &vs[j]).abs());
            }
        }
        worst
    };

    let drift_no = max_off_diag(&vs_no_reorth);
    let drift_yes = max_off_diag(&vs_windowed);
    assert!(
        drift_yes < drift_no,
        "windowed drift ({drift_yes:e}) should be smaller than \
         unwindowed drift ({drift_no:e})"
    );
    assert!(
        drift_yes < 1e-10,
        "last {local_size} v's not mutually orthogonal: {drift_yes:e}"
    );
}
