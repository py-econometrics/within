use schwarz_precond::{with_unit_stride, BiLinOp, LinOp, MatMut, MatRef, MemStack, Par, StackReq};

use crate::domain::PreparedDesign;

mod gather;
mod scatter;

#[cfg(test)]
mod tests;

use gather::gather_apply;
use scatter::{scatter_apply, scatter_scratch};

/// Minimum number of rows before scatter/gather loops are parallelized.
const PAR_THRESHOLD: usize = 10_000;

impl PreparedDesign<'_> {
    /// `y − D x`, read off this operator's measured `b − A x` unless a zero weight erased a row.
    pub(crate) fn demeaned(&self, x: &[f64], y: &[f64], par: Par) -> Vec<f64> {
        let mut demeaned = vec![0.0; y.len()];
        gather_apply(self, x, &mut demeaned, None, par);
        for (d, &yi) in demeaned.iter_mut().zip(y) {
            *d = yi - *d;
        }
        demeaned
    }
}

impl std::fmt::Debug for PreparedDesign<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedDesign")
            .field("n_obs", &self.design.n_obs)
            .field("n_dofs", &self.design.n_dofs)
            .field("weighted", &self.sqrt_weights().is_some())
            .finish()
    }
}

/// Design operator `D` or `W^{1/2} D`, whose normal equations `AᵀA = DᵀWD` recover the Gramian.
impl LinOp<f64> for PreparedDesign<'_> {
    fn apply_scratch(&self, _rhs_ncols: usize, _par: Par) -> StackReq {
        StackReq::EMPTY
    }

    fn nrows(&self) -> usize {
        self.design.n_obs
    }

    fn ncols(&self) -> usize {
        self.design.n_dofs
    }

    fn apply(&self, out: MatMut<'_, f64>, rhs: MatRef<'_, f64>, par: Par, _: &mut MemStack) {
        let (rows, cols) = (self.design.n_obs, self.design.n_dofs);
        assert_eq!(
            (out.shape(), rhs.nrows()),
            ((rows, rhs.ncols()), cols),
            "D apply shape"
        );
        for (y, x) in out.col_iter_mut().zip(rhs.col_iter()) {
            with_unit_stride(x, y, |x, y| {
                gather_apply(self, x, y, self.sqrt_weights(), par)
            });
        }
    }

    fn conj_apply(
        &self,
        out: MatMut<'_, f64>,
        rhs: MatRef<'_, f64>,
        par: Par,
        stack: &mut MemStack,
    ) {
        LinOp::apply(self, out, rhs, par, stack)
    }
}

impl BiLinOp<f64> for PreparedDesign<'_> {
    fn transpose_apply_scratch(&self, _rhs_ncols: usize, par: Par) -> StackReq {
        scatter_scratch(self, par)
    }

    fn transpose_apply(
        &self,
        out: MatMut<'_, f64>,
        rhs: MatRef<'_, f64>,
        par: Par,
        stack: &mut MemStack,
    ) {
        let (rows, cols) = (self.design.n_dofs, self.design.n_obs);
        assert_eq!(
            (out.shape(), rhs.nrows()),
            ((rows, rhs.ncols()), cols),
            "Dᵀ apply shape"
        );
        for (y, x) in out.col_iter_mut().zip(rhs.col_iter()) {
            with_unit_stride(x, y, |x, y| match self.sqrt_weights() {
                Some(sw) => scatter_apply(self, y, &|i| sw[i] * x[i], par, stack),
                None => scatter_apply(self, y, &|i| x[i], par, stack),
            });
        }
    }

    fn adjoint_apply(
        &self,
        out: MatMut<'_, f64>,
        rhs: MatRef<'_, f64>,
        par: Par,
        stack: &mut MemStack,
    ) {
        self.transpose_apply(out, rhs, par, stack)
    }
}
