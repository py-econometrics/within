//! `A = W^{1/2} D̃`: the system matrix LSMR iterates on, over `D̃`'s term blocks.

use schwarz_precond::{MemStack, Operator, StackReq};

use super::OrthogonalDesign;
use crate::design::product::{gather, scatter, scatter_scratch};
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
            n_dofs: design.unknowns().len(),
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

    /// The adjoint's by-row partial sums; the forward apply takes nothing.
    fn scratch(&self) -> StackReq {
        scatter_scratch(&self.blocks)
    }

    fn apply_with(
        &self,
        x: &[f64],
        y: &mut [f64],
        _: &mut MemStack,
    ) -> Result<(), schwarz_precond::SolveError> {
        debug_assert_eq!((x.len(), y.len()), (self.n_dofs, self.n_obs));
        gather(&self.blocks, x, y, self.sqrt);
        Ok(())
    }

    fn apply_adjoint_with(
        &self,
        x: &[f64],
        y: &mut [f64],
        stack: &mut MemStack,
    ) -> Result<(), schwarz_precond::SolveError> {
        debug_assert_eq!((x.len(), y.len()), (self.n_obs, self.n_dofs));
        scatter(&self.blocks, x, y, self.sqrt, stack);
        Ok(())
    }
}
