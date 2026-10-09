use schwarz_precond::Operator;
use std::borrow::Cow;

use crate::domain::PreparedDesign;

mod gather;
mod scatter;

#[cfg(test)]
mod tests;

use gather::{gather_apply, unscale};
use scatter::scatter_apply;

/// Minimum number of rows before scatter/gather loops are parallelized.
const PAR_THRESHOLD: usize = 10_000;

/// Design operator `D` or `W^{1/2} D`, whose normal equations `AᵀA = DᵀWD` recover the Gramian.
pub(crate) struct DesignOperator<'a> {
    prepared: &'a PreparedDesign<'a>,
}

impl<'a> DesignOperator<'a> {
    pub(crate) fn new(prepared: &'a PreparedDesign<'a>) -> Self {
        Self { prepared }
    }

    /// `y − D x`, read off this operator's measured `b − A x` unless a zero weight erased a row.
    pub(crate) fn demeaned(&self, x: &[f64], y: &[f64], measured: Option<Vec<f64>>) -> Vec<f64> {
        match (measured, self.prepared.sqrt_weights()) {
            (Some(residual), None) => residual,
            (Some(mut residual), Some(sw)) if !sw.contains(&0.0) => {
                unscale(&mut residual, sw);
                residual
            }
            (measured, _) => {
                let mut demeaned = measured.unwrap_or_else(|| vec![0.0; y.len()]);
                gather_apply(self.prepared, x, &mut demeaned, None);
                for (d, &yi) in demeaned.iter_mut().zip(y) {
                    *d = yi - *d;
                }
                demeaned
            }
        }
    }

    /// Observation-space RHS `b = W^{1/2} y`; borrows unweighted, owns weighted.
    pub(crate) fn weighted_rhs<'y>(&self, y: &'y [f64]) -> Cow<'y, [f64]> {
        match self.prepared.sqrt_weights() {
            None => Cow::Borrowed(y),
            Some(sw) => Cow::Owned(y.iter().zip(sw).map(|(&yi, &swi)| swi * yi).collect()),
        }
    }
}

impl Operator for DesignOperator<'_> {
    fn nrows(&self) -> usize {
        self.prepared.design.n_obs
    }

    fn ncols(&self) -> usize {
        self.prepared.design.n_dofs
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), schwarz_precond::SolveError> {
        debug_assert_eq!(x.len(), self.prepared.design.n_dofs);
        debug_assert_eq!(y.len(), self.prepared.design.n_obs);
        gather_apply(self.prepared, x, y, self.prepared.sqrt_weights());
        Ok(())
    }

    fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) -> Result<(), schwarz_precond::SolveError> {
        debug_assert_eq!(x.len(), self.prepared.design.n_obs);
        debug_assert_eq!(y.len(), self.prepared.design.n_dofs);
        y.fill(0.0);
        match self.prepared.sqrt_weights() {
            Some(sw) => scatter_apply(self.prepared, y, &|i| sw[i] * x[i]),
            None => scatter_apply(self.prepared, y, &|i| x[i]),
        }
        Ok(())
    }
}
