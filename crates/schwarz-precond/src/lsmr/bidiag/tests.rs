//! White-box checks on the bidiagonalization itself.

use super::{dot, Bidiagonalization, GolubKahan};
use crate::lsmr::fixtures::DenseOp;

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
