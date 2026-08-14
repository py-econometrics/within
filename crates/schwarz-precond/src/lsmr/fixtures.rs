//! Operators and helpers shared by the LSMR test modules.

use super::*;
use crate::{Operator, SolveError};

/// Identity operator used by mlsmr equivalence tests.
pub(crate) struct IdentityOp {
    pub(crate) n: usize,
}

impl Operator for IdentityOp {
    fn nrows(&self) -> usize {
        self.n
    }
    fn ncols(&self) -> usize {
        self.n
    }
    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        y.copy_from_slice(x);
        Ok(())
    }
    fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        y.copy_from_slice(x);
        Ok(())
    }
}

/// Simple 4×3 overdetermined system.
/// A = [1 0 0; 0 1 0; 0 0 1; 1 1 0]
pub(crate) struct OverdeterminedOp;

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

/// Degenerate 2×2 operator `A = [[1, 0], [0, 0]]` — zero second row and zero
/// second column. Shared by the zero-row/column and mid-stream `beta == 0`
/// breakdown tests.
pub(crate) struct ZeroSecondRow;

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

/// `‖Aᵀ (b - A x)‖₂` — normal-equation residual, the scale-invariant
/// "did we actually solve the least-squares problem?" check.
pub(crate) fn normal_equation_residual<O: Operator + ?Sized>(op: &O, x: &[f64], b: &[f64]) -> f64 {
    let mut ax = vec![0.0; op.nrows()];
    op.apply(x, &mut ax).expect("apply succeeds");
    let resid: Vec<f64> = b.iter().zip(&ax).map(|(bi, ai)| bi - ai).collect();
    let mut atr = vec![0.0; op.ncols()];
    op.apply_adjoint(&resid, &mut atr)
        .expect("apply_adjoint succeeds");
    vec_norm(&atr)
}

/// Diagonal operator, used as a stand-in preconditioner `M⁻¹`.
pub(crate) struct DiagOp(pub(crate) Vec<f64>);

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

/// `M⁻¹ ≈ diag(AᵀA)⁻¹`.
pub(crate) fn jacobi(op: &DenseOp) -> DiagOp {
    let mut diag_inv = vec![0.0; op.cols];
    for (j, di) in diag_inv.iter_mut().enumerate() {
        let s: f64 = op
            .data
            .chunks_exact(op.cols)
            .map(|row| row[j] * row[j])
            .sum();
        *di = if s > 0.0 { 1.0 / s } else { 1.0 };
    }
    DiagOp(diag_inv)
}

/// Dense row-major test operator. Used by the local-reorth tests to build
/// ill-conditioned least-squares problems (Vandermonde-flavored) that
/// stress the v-orthogonality of the bidiagonalization.
pub(crate) struct DenseOp {
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) data: Vec<f64>,
}

impl DenseOp {
    /// Vandermonde-like matrix `A[i,j] = (i / (rows-1))^j`.
    ///
    /// `rows = 30`, `cols = 12` gives `cond(A) ≈ 1e10` — well past where
    /// the `v` short-recurrence drifts in floating point and convergence
    /// stalls without reorthogonalization.
    pub(crate) fn vandermonde(rows: usize, cols: usize) -> Self {
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

#[derive(Clone, Copy)]
pub(crate) struct FixedIterations(pub(crate) usize);

impl EscalationPolicy for FixedIterations {
    fn handler(&self) -> Box<dyn EscalationHandler> {
        Box::new(*self)
    }
}

impl EscalationHandler for FixedIterations {
    fn should_escalate(&mut self, progress: Progress) -> bool {
        progress.iteration >= self.0
    }
}
