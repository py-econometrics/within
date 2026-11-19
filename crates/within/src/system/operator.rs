//! `A = W^{1/2} D̃`: the system matrix LSMR iterates on, over `D̃`'s term blocks.

use schwarz_precond::Operator;

use super::OrthogonalDesign;
use crate::design::product::{gather, scatter};
use crate::design::TermBlock;

#[cfg(test)]
mod tests;

/// `A = W^{1/2} D̃`: the operator LSMR iterates on.
pub(crate) struct DesignOperator<'p> {
    blocks: Vec<TermBlock<'p>>,
    sqrt: Option<&'p [f64]>,
    n_obs: usize,
    n_dofs: usize,
}

impl<'p> DesignOperator<'p> {
    pub(crate) fn new(design: &'p OrthogonalDesign<'p>) -> Self {
        Self {
            blocks: design.terms().collect(),
            sqrt: design.weights.sqrt(),
            n_obs: design.n_obs(),
            n_dofs: design.n_dofs(),
        }
    }
}

/// `AᵀA = D̃ᵀWD̃` is the Gramian.
impl Operator for DesignOperator<'_> {
    fn nrows(&self) -> usize {
        self.n_obs
    }

    fn ncols(&self) -> usize {
        self.n_dofs
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), schwarz_precond::SolveError> {
        gather(&self.blocks, x, y, self.sqrt);
        Ok(())
    }

    fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) -> Result<(), schwarz_precond::SolveError> {
        debug_assert_eq!(x.len(), self.n_obs);
        debug_assert_eq!(y.len(), self.n_dofs);
        for t in &self.blocks {
            let block = &mut y[t.layout().dofs()];
            match self.sqrt {
                Some(sw) => scatter(t, block, &|i| sw[i] * x[i]),
                None => scatter(t, block, &|i| x[i]),
            }
        }
        Ok(())
    }
}
