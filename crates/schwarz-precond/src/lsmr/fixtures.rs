//! Operators and helpers shared by the LSMR test modules.

use super::*;
use crate::SolveError;
use faer::diag::Diag;
pub(crate) use faer::matrix_free::IdentityPrecond;
pub(crate) use faer::Mat;
use faer::{Col, MatMut, MatRef};

/// [`mlsmr`] on the ambient pool, warm from `x0` or cold.
pub(crate) fn run(
    op: &(impl BiLinOp<f64> + ?Sized),
    b: &[f64],
    m: &(impl LinOp<f64> + ?Sized),
    x0: Option<&[f64]>,
    options: &LsmrOptions,
    stall: Option<Staleness>,
) -> Result<(Vec<f64>, LsmrInfo), SolveError> {
    let mut x = x0.map_or_else(|| vec![0.0; op.ncols()], <[f64]>::to_vec);
    let start = match x0 {
        Some(_) => InitialGuess::Given,
        None => InitialGuess::Zero,
    };
    let mut scratch = MemBuffer::new(StackReq::EMPTY);
    let par = Par::rayon(0);
    let info = mlsmr(op, b, m, &mut x, start, options, stall, par, &mut scratch)?;
    Ok((x, info))
}

/// Unpreconditioned [`run`] from a cold start.
pub(crate) fn lsmr(
    op: &(impl BiLinOp<f64> + ?Sized),
    b: &[f64],
    tol: f64,
    maxiter: usize,
    local_size: Option<usize>,
) -> Result<(Vec<f64>, LsmrInfo), SolveError> {
    let options = LsmrOptions {
        local_size,
        ..opts(tol, maxiter)
    };
    run(
        op,
        b,
        &IdentityPrecond { dim: op.ncols() },
        None,
        &options,
        None,
    )
}

pub(crate) fn opts(tol: f64, maxiter: usize) -> LsmrOptions {
    LsmrOptions {
        tol,
        maxiter,
        ..Default::default()
    }
}

/// `y = A x` on one thread.
pub(crate) fn times(op: &(impl LinOp<f64> + ?Sized), x: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; op.nrows()];
    let mut scratch = MemBuffer::new(op.apply_scratch(1, Par::Seq));
    let (out, rhs) = (column_mut(&mut y), column(x));
    op.apply(out, rhs, Par::Seq, MemStack::new(&mut scratch));
    y
}

/// `y = Aᵀ x` on one thread.
pub(crate) fn times_t(op: &(impl BiLinOp<f64> + ?Sized), x: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; op.ncols()];
    let mut scratch = MemBuffer::new(op.transpose_apply_scratch(1, Par::Seq));
    let (out, rhs) = (column_mut(&mut y), column(x));
    op.transpose_apply(out, rhs, Par::Seq, MemStack::new(&mut scratch));
    y
}

pub(crate) fn column(x: &[f64]) -> MatRef<'_, f64> {
    MatRef::from_column_major_slice(x, x.len(), 1)
}

pub(crate) fn column_mut(x: &mut [f64]) -> MatMut<'_, f64> {
    let n = x.len();
    MatMut::from_column_major_slice_mut(x, n, 1)
}

/// `rows × cols` from row-major `data`.
pub(crate) fn dense(rows: usize, cols: usize, data: &[f64]) -> Mat<f64> {
    Mat::from_fn(rows, cols, |i, j| data[i * cols + j])
}

pub(crate) fn diag(d: &[f64]) -> Diag<f64> {
    Col::from_fn(d.len(), |i| d[i]).into_diagonal()
}

/// `A = [1 0 0; 0 1 0; 0 0 1; 1 1 0]`.
pub(crate) fn overdetermined() -> Mat<f64> {
    dense(
        4,
        3,
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
    )
}

/// `A = [[1, 0], [0, 0]]`: a zero second row and column, for the breakdown tests.
pub(crate) fn zero_second_row() -> Mat<f64> {
    dense(2, 2, &[1.0, 0.0, 0.0, 0.0])
}

/// `‖Aᵀ (b - A x)‖₂`, the scale-invariant check that the least-squares problem is solved.
pub(crate) fn normal_equation_residual(
    op: &(impl BiLinOp<f64> + ?Sized),
    x: &[f64],
    b: &[f64],
) -> f64 {
    let resid: Vec<f64> = b.iter().zip(times(op, x)).map(|(bi, ai)| bi - ai).collect();
    vec_norm(&times_t(op, &resid))
}

/// `M⁻¹ ≈ diag(AᵀA)⁻¹`.
pub(crate) fn jacobi(op: &Mat<f64>) -> Diag<f64> {
    Col::from_fn(op.ncols(), |j| {
        let s: f64 = op.col(j).iter().map(|a| a * a).sum();
        if s > 0.0 {
            1.0 / s
        } else {
            1.0
        }
    })
    .into_diagonal()
}

/// `A[i,j] = (i / (rows-1))^j`; `30 × 12` has `cond(A) ≈ 1e10`, where unreorthogonalized `v` drifts.
pub(crate) fn vandermonde(rows: usize, cols: usize) -> Mat<f64> {
    let mut a = Mat::zeros(rows, cols);
    for i in 0..rows {
        let x = i as f64 / (rows - 1).max(1) as f64;
        let mut p = 1.0;
        for j in 0..cols {
            a[(i, j)] = p;
            p *= x;
        }
    }
    a
}

/// The ill-conditioned least-squares problem the ladder tests run on, with a smooth RHS.
pub(crate) fn vandermonde_ls() -> (Mat<f64>, Vec<f64>) {
    let op = vandermonde(30, 12);
    let b = (0..op.nrows())
        .map(|i| (1.0 + i as f64 / (op.nrows() - 1) as f64).ln())
        .collect();
    (op, b)
}

/// Stops at iteration `n ≥ 2`: a zero threshold counts every positive drop ratio as stalled.
pub(crate) fn stop_at(n: usize) -> Staleness {
    Staleness::try_new(n - 1, 0.0).expect("n ≥ 2")
}
