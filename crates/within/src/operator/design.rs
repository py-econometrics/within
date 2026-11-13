use schwarz_precond::Operator;

use crate::domain::{PreparedDesign, PreparedTerm, PAR_THRESHOLD};

mod gather;
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "`System::Reduced` wires the reduced operator in")
)]
mod reduced;
mod scatter;

#[cfg(test)]
mod tests;

use gather::{gather_apply, unscale};
#[cfg_attr(
    not(test),
    expect(
        unused_imports,
        reason = "`System::Reduced` wires the reduced operator in"
    )
)]
pub(crate) use reduced::ReducedDesign;
use scatter::scatter_block;

impl PreparedDesign<'_> {
    /// `y − D x`, read off this operator's measured `b − A x` unless a zero weight erased a row.
    pub(crate) fn demeaned(&self, x: &[f64], y: &[f64], measured: Option<Vec<f64>>) -> Vec<f64> {
        match (measured, self.sqrt_weights()) {
            (Some(residual), None) => residual,
            (Some(mut residual), Some(sw)) if !sw.contains(&0.0) => {
                unscale(&mut residual, sw);
                residual
            }
            (measured, _) => {
                let mut demeaned = measured.unwrap_or_else(|| vec![0.0; y.len()]);
                gather_apply(self, x, &mut demeaned, None);
                for (d, &yi) in demeaned.iter_mut().zip(y) {
                    *d = yi - *d;
                }
                demeaned
            }
        }
    }

    /// `block = A_tᵀ v` for one term `t`.
    fn adjoint_block(&self, t: &PreparedTerm<'_>, v: &[f64], block: &mut [f64]) {
        match self.sqrt_weights() {
            Some(sw) => scatter_block(t, block, &|i| sw[i] * v[i]),
            None => scatter_block(t, block, &|i| v[i]),
        }
    }
}

/// Design operator `D` or `W^{1/2} D`, whose normal equations `AᵀA = DᵀWD` recover the Gramian.
impl Operator for PreparedDesign<'_> {
    fn nrows(&self) -> usize {
        self.design.n_obs
    }

    fn ncols(&self) -> usize {
        self.design.n_dofs
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), schwarz_precond::SolveError> {
        debug_assert_eq!(x.len(), self.design.n_dofs);
        debug_assert_eq!(y.len(), self.design.n_obs);
        gather_apply(self, x, y, self.sqrt_weights());
        Ok(())
    }

    fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) -> Result<(), schwarz_precond::SolveError> {
        debug_assert_eq!(x.len(), self.design.n_obs);
        debug_assert_eq!(y.len(), self.design.n_dofs);
        for t in self.terms() {
            self.adjoint_block(&t, x, &mut y[t.term.dofs()]);
        }
        Ok(())
    }
}
