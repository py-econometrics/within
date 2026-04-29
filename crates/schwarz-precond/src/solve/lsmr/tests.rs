//! LSMR test suite.

use super::bidiag::{Bidiagonalization, GolubKahan};
use super::*;
use crate::solve::{dot, vec_norm};
use crate::{IdentityOperator, Operator, SolveError};

/// Simple 4×3 overdetermined system.
/// A = [1 0 0; 0 1 0; 0 0 1; 1 1 0]
struct OverdeterminedOp;

impl Operator for OverdeterminedOp {
    fn nrows(&self) -> usize {
        4
    }
    fn ncols(&self) -> usize {
        3
    }
    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        y[0] = x[0];
        y[1] = x[1];
        y[2] = x[2];
        y[3] = x[0] + x[1];
        Ok(())
    }
    fn apply_adjoint(&self, u: &[f64], x: &mut [f64]) -> Result<(), SolveError> {
        x[0] = u[0] + u[3];
        x[1] = u[1] + u[3];
        x[2] = u[2];
        Ok(())
    }
}

/// Diagonal preconditioner: M⁻¹ = diag(1/2, 1/2, 1)
/// Approximates (Aᵀ A)⁻¹ = diag(2, 2, 1)⁻¹
struct DiagPrecond;

impl Operator for DiagPrecond {
    fn nrows(&self) -> usize {
        3
    }
    fn ncols(&self) -> usize {
        3
    }
    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        y[0] = x[0] / 2.0;
        y[1] = x[1] / 2.0;
        y[2] = x[2];
        Ok(())
    }
    fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        self.apply(x, y)
    }
}

/// `‖Aᵀ (b - A x)‖₂` — normal-equation residual, the scale-invariant
/// "did we actually solve the least-squares problem?" check.
fn normal_equation_residual<O: Operator + ?Sized>(op: &O, x: &[f64], b: &[f64]) -> f64 {
    let mut ax = vec![0.0; op.nrows()];
    op.apply(x, &mut ax).expect("apply");
    let resid: Vec<f64> = b.iter().zip(&ax).map(|(bi, ai)| bi - ai).collect();
    let mut atr = vec![0.0; op.ncols()];
    op.apply_adjoint(&resid, &mut atr).expect("apply");
    vec_norm(&atr)
}

fn unpreconditioned_mlsmr<A: Operator + ?Sized>(
    operator: &A,
    b: &[f64],
    tol: f64,
    maxiter: usize,
    local_size: Option<usize>,
) -> Result<LsmrResult, SolveError> {
    mlsmr::<A, IdentityOperator>(operator, b, None, tol, maxiter, local_size)
}

fn preconditioned_mlsmr<A: Operator + ?Sized, M: Operator + ?Sized>(
    operator: &A,
    b: &[f64],
    preconditioner: &M,
    tol: f64,
    maxiter: usize,
    local_size: Option<usize>,
) -> Result<LsmrResult, SolveError> {
    mlsmr(operator, b, Some(preconditioner), tol, maxiter, local_size)
}

#[test]
fn test_mlsmr_unpreconditioned() {
    let b = vec![1.0, 2.0, 3.0, 3.0];
    let result =
        unpreconditioned_mlsmr(&OverdeterminedOp, &b, 1e-10, 100, None).expect("lsmr solve");
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
    let result = preconditioned_mlsmr(&OverdeterminedOp, &b, &DiagPrecond, 1e-10, 100, None)
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
    let result =
        unpreconditioned_mlsmr(&OverdeterminedOp, &b, 1e-10, 100, None).expect("lsmr solve");
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
    let result =
        unpreconditioned_mlsmr(&UnderOp, &b, 1e-12, 100, None).expect("underdetermined solve");
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
    let result = unpreconditioned_mlsmr(&RankDeficientOp, &b, 1e-12, 100, None)
        .expect("rank-deficient solve");
    assert!(result.converged);
    assert!(((result.x[0] + result.x[1]) - 3.0).abs() < 1e-10);
    assert!(normal_equation_residual(&RankDeficientOp, &result.x, &b) < 1e-10);
}

#[test]
fn test_mlsmr_zero_column_and_zero_row() {
    struct DegenerateOp;
    impl Operator for DegenerateOp {
        fn nrows(&self) -> usize {
            2
        }
        fn ncols(&self) -> usize {
            2
        }
        fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
            y[0] = x[0];
            y[1] = 0.0;
            Ok(())
        }
        fn apply_adjoint(&self, u: &[f64], x: &mut [f64]) -> Result<(), SolveError> {
            x[0] = u[0];
            x[1] = 0.0;
            Ok(())
        }
    }

    let b = vec![2.0, 3.0];
    let result =
        unpreconditioned_mlsmr(&DegenerateOp, &b, 1e-12, 100, None).expect("degenerate solve");
    assert!(result.converged);
    assert!((result.x[0] - 2.0).abs() < 1e-10);
    assert!(result.x[1].abs() < 1e-10);
    assert!(normal_equation_residual(&DegenerateOp, &result.x, &b) < 1e-10);
}

#[test]
fn test_mlsmr_mid_stream_beta_zero_breakdown() {
    // Consistent rank-1 system: b lies entirely in A's range. After the
    // first step, A v_1 - alpha_1 u_1 collapses to zero exactly, so beta_2
    // is zero. Exercises the mid-stream beta == 0 branch in
    // GolubKahan::step that zeroes v before the caller's solution.update.
    struct ZeroSecondRow;
    impl Operator for ZeroSecondRow {
        fn nrows(&self) -> usize {
            2
        }
        fn ncols(&self) -> usize {
            2
        }
        fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
            y[0] = x[0];
            y[1] = 0.0;
            Ok(())
        }
        fn apply_adjoint(&self, u: &[f64], x: &mut [f64]) -> Result<(), SolveError> {
            x[0] = u[0];
            x[1] = 0.0;
            Ok(())
        }
    }

    let b = vec![5.0, 0.0];
    let result =
        unpreconditioned_mlsmr(&ZeroSecondRow, &b, 1e-12, 100, None).expect("beta=0 solve");
    assert!(result.converged);
    assert!((result.x[0] - 5.0).abs() < 1e-12);
    assert!(result.x[1].abs() < 1e-12);
}

#[test]
fn test_mlsmr_maxiter_exhaustion() {
    let b = vec![1.0, 2.0, 3.0, 3.0];
    let result = unpreconditioned_mlsmr(&OverdeterminedOp, &b, 1e-15, 1, None).expect("lsmr solve");
    assert!(
        !result.converged,
        "should not converge in 1 iteration at 1e-15 tol"
    );
    assert_eq!(result.iterations, 1);
    assert_eq!(result.stop_reason, LsmrStopReason::MaxIterations);
}

#[test]
fn test_mlsmr_optional_preconditioner_paths() {
    let b = vec![1.0, 2.0, 3.0, 3.0];
    let unpreconditioned = mlsmr(
        &OverdeterminedOp,
        &b,
        None::<&DiagPrecond>,
        1e-10,
        100,
        None,
    )
    .expect("unpreconditioned lsmr");
    let preconditioned = mlsmr(&OverdeterminedOp, &b, Some(&DiagPrecond), 1e-10, 100, None)
        .expect("preconditioned lsmr");

    assert!(unpreconditioned.converged);
    assert!(preconditioned.converged);
    for (actual, expected) in unpreconditioned.x.iter().zip([1.0, 2.0, 3.0]) {
        assert!((*actual - expected).abs() < 1e-6);
    }
    for (actual, expected) in preconditioned.x.iter().zip([1.0, 2.0, 3.0]) {
        assert!((*actual - expected).abs() < 1e-6);
    }
}

/// `None` (GolubKahan path) and `Some(&IdentityOperator)`
/// (ModifiedGolubKahan with M = I) are mathematically the same algorithm.
/// They should produce numerically equivalent solutions and iteration
/// counts; this guards against future drift between the two
/// bidiagonalization implementations.
#[test]
fn test_mlsmr_none_matches_identity_precond() {
    let b = vec![1.0, 2.0, 3.0, 3.0];
    let id = IdentityOperator::new(3);

    let none_result =
        unpreconditioned_mlsmr(&OverdeterminedOp, &b, 1e-12, 100, None).expect("lsmr solve");
    let id_result = preconditioned_mlsmr(&OverdeterminedOp, &b, &id, 1e-12, 100, None)
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
/// logic in `ModifiedLocalReorth::push` against drift.
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
    let id = IdentityOperator::new(op.cols);
    let local = Some(10);

    // Tight tolerance with headroom in maxiter: drives both paths to the
    // same minimum so the comparison isn't governed by rounding noise in
    // the convergence test.
    let none_result =
        unpreconditioned_mlsmr(&op, &b, 1e-12, 50, local).expect("lsmr windowed solve");
    let id_result = preconditioned_mlsmr(&op, &b, &id, 1e-12, 50, local)
        .expect("preconditioned Identity windowed solve");

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

/// Dense row-major test operator. Used by the local-reorth tests to build
/// ill-conditioned least-squares problems (Vandermonde-flavored) that
/// stress the v-orthogonality of the bidiagonalization.
struct DenseOp {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

impl DenseOp {
    /// Vandermonde-like matrix `A[i,j] = (i / (rows-1))^j`.
    ///
    /// `rows = 30`, `cols = 12` gives `cond(A) ≈ 1e10` — well past where
    /// the `v` short-recurrence drifts in floating point and convergence
    /// stalls without reorthogonalization.
    fn vandermonde(rows: usize, cols: usize) -> Self {
        let mut data = vec![0.0; rows * cols];
        for i in 0..rows {
            let x = i as f64 / (rows - 1).max(1) as f64;
            let mut p = 1.0;
            for j in 0..cols {
                data[i * cols + j] = p;
                p *= x;
            }
        }
        Self { rows, cols, data }
    }
}

impl Operator for DenseOp {
    fn nrows(&self) -> usize {
        self.rows
    }
    fn ncols(&self) -> usize {
        self.cols
    }
    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        for (yi, row) in y.iter_mut().zip(self.data.chunks_exact(self.cols)) {
            *yi = row.iter().zip(x).map(|(a, b)| a * b).sum();
        }
        Ok(())
    }
    fn apply_adjoint(&self, u: &[f64], x: &mut [f64]) -> Result<(), SolveError> {
        for (j, xj) in x.iter_mut().enumerate() {
            let mut s = 0.0;
            for (ui, row) in u.iter().zip(self.data.chunks_exact(self.cols)) {
                s += row[j] * ui;
            }
            *xj = s;
        }
        Ok(())
    }
}

/// `local_size = 0` is the no-op fast path — it must match repeated runs
/// bit-for-bit and reproduce the same answer the unwindowed test gets.
/// Guards against the `is_empty()` early return ever drifting.
#[test]
fn test_mlsmr_local_reorth_zero_is_identity() {
    let b = vec![1.0, 2.0, 3.0, 3.0];
    let r1 =
        unpreconditioned_mlsmr(&OverdeterminedOp, &b, 1e-10, 100, None).expect("unwindowed solve");
    let r2 = unpreconditioned_mlsmr(&OverdeterminedOp, &b, 1e-10, 100, Some(0))
        .expect("zero-window solve");
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

    let r0 = unpreconditioned_mlsmr(&op, &b, tol, maxiter, None).expect("no-reorth solve");
    let r10 = unpreconditioned_mlsmr(&op, &b, tol, maxiter, Some(10)).expect("windowed solve");

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
/// Diagonal preconditioner approximating diag(AᵀA)⁻¹.
#[test]
fn test_mlsmr_local_reorth_preconditioned() {
    let op = DenseOp::vandermonde(30, 12);

    // Build M⁻¹ ≈ diag(AᵀA)⁻¹.
    let mut diag_inv = vec![0.0; op.cols];
    for (j, di) in diag_inv.iter_mut().enumerate() {
        let s: f64 = op
            .data
            .chunks_exact(op.cols)
            .map(|row| row[j] * row[j])
            .sum();
        *di = if s > 0.0 { 1.0 / s } else { 1.0 };
    }
    struct DiagOp(Vec<f64>);
    impl Operator for DiagOp {
        fn nrows(&self) -> usize {
            self.0.len()
        }
        fn ncols(&self) -> usize {
            self.0.len()
        }
        fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
            for ((yi, &xi), &di) in y.iter_mut().zip(x).zip(self.0.iter()) {
                *yi = di * xi;
            }
            Ok(())
        }
        fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
            self.apply(x, y)
        }
    }
    let m = DiagOp(diag_inv);

    let b: Vec<f64> = (0..op.rows)
        .map(|i| {
            let x = i as f64 / (op.rows - 1) as f64;
            (1.0 + x).ln()
        })
        .collect();
    let tol = 1e-9;
    let maxiter = 30;

    let r10 = preconditioned_mlsmr(&op, &b, &m, tol, maxiter, Some(10))
        .expect("windowed preconditioned solve");
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

/// Window smaller than the iteration count: the ring must wrap correctly.
/// We re-run the bidiagonalization manually with the same window and
/// verify the last `local_size` `v` vectors are mutually orthogonal to
/// tighter tolerance than they would be without reorthogonalization.
#[test]
fn test_mlsmr_local_reorth_window_smaller_than_iter_count() {
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
    let result =
        unpreconditioned_mlsmr(&ColE1, &b, 1e-12, 100, None).expect("lsmr alpha=0 early exit");
    assert!(result.converged);
    assert_eq!(result.iterations, 0);
    assert_eq!(result.x, vec![0.0; 1]);
    assert_eq!(
        result.stop_reason,
        LsmrStopReason::InitialNormalEquationResidualZero
    );
    assert!((result.residual_norm - vec_norm(&b)).abs() < 1e-15);
}

#[test]
fn test_mlsmr_zero_rhs_stop_reason() {
    let b = vec![0.0; 4];
    let result =
        unpreconditioned_mlsmr(&OverdeterminedOp, &b, 1e-12, 100, None).expect("zero-rhs solve");
    assert!(result.converged);
    assert_eq!(result.iterations, 0);
    assert_eq!(result.stop_reason, LsmrStopReason::ZeroRhs);
    assert_eq!(result.residual_norm, 0.0);
}

/// Window sizes at the boundaries of useful values: `Some(1)` (degenerate
/// ring of one), `Some(12)` (= number of columns), `Some(13)` (= cols + 1).
/// All three must converge and produce a small normal-equation residual.
#[test]
fn test_mlsmr_local_reorth_window_boundary_sizes() {
    let op = DenseOp::vandermonde(30, 12);
    let b: Vec<f64> = (0..op.rows)
        .map(|i| {
            let x = i as f64 / (op.rows - 1) as f64;
            (1.0 + x).ln()
        })
        .collect();

    // Budget of 200 iterations gives `Some(1)` (which degenerates to no real
    // reorthogonalization) enough room to converge on this cond ≈ 1e10 system,
    // while still being a small bounded budget for the larger window sizes.
    for window_size in [Some(1usize), Some(12), Some(13)] {
        let result = unpreconditioned_mlsmr(&op, &b, 1e-9, 200, window_size)
            .expect("lsmr boundary-window solve");
        assert!(
            result.converged,
            "did not converge with window {window_size:?}"
        );
        assert!(
            normal_equation_residual(&op, &result.x, &b) < 1e-6,
            "normal-eq residual too large with window {window_size:?}",
        );
    }
}

#[test]
fn test_mlsmr_rejects_invalid_inputs() {
    let bad_len = unpreconditioned_mlsmr(&OverdeterminedOp, &[1.0, 2.0], 1e-10, 100, None);
    assert!(matches!(bad_len, Err(SolveError::InvalidInput { .. })));

    let bad_tol = unpreconditioned_mlsmr(
        &OverdeterminedOp,
        &[1.0, 2.0, 3.0, 4.0],
        f64::NAN,
        100,
        None,
    );
    assert!(matches!(bad_tol, Err(SolveError::InvalidInput { .. })));

    let bad_rhs = unpreconditioned_mlsmr(
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
    let result = preconditioned_mlsmr(&OverdeterminedOp, &b, &BadPrecond, 1e-10, 100, None);
    assert!(matches!(result, Err(SolveError::InvalidInput { .. })));
}
