//! `A = W^{1/2} D̃`: the system matrix LSMR iterates on, over `D̃`'s term blocks.

use schwarz_precond::parallel::Workspaces;
use schwarz_precond::{MemStack, Operator, StackReq};

use super::product::{fold_workspaces, gather, scatter};
use super::OrthogonalDesign;

#[cfg(test)]
mod tests;

/// `A = W^{1/2} D̃`: the operator LSMR iterates on.
pub(crate) struct DesignOperator<'p> {
    design: &'p OrthogonalDesign<'p>,
    /// Each term's fold workspace for the adjoint.
    folds: Workspaces,
}

impl<'p> DesignOperator<'p> {
    pub(crate) fn new(design: &'p OrthogonalDesign<'p>) -> Self {
        Self {
            design,
            folds: fold_workspaces(design),
        }
    }
}

/// `AᵀA = D̃ᵀWD̃` is the Gramian.
impl Operator for DesignOperator<'_> {
    fn nrows(&self) -> usize {
        self.design.n_obs()
    }

    fn ncols(&self) -> usize {
        self.design.unknowns().len()
    }

    /// The adjoint's by-row partial sums; the forward apply takes nothing.
    fn scratch(&self) -> StackReq {
        self.folds.scratch()
    }

    fn apply_with(
        &self,
        x: &[f64],
        y: &mut [f64],
        _: &mut MemStack,
    ) -> Result<(), schwarz_precond::SolveError> {
        debug_assert_eq!((x.len(), y.len()), (self.ncols(), self.nrows()));
        gather(self.design, x, y, self.design.weights.sqrt());
        Ok(())
    }

    fn apply_adjoint_with(
        &self,
        x: &[f64],
        y: &mut [f64],
        stack: &mut MemStack,
    ) -> Result<(), schwarz_precond::SolveError> {
        debug_assert_eq!((x.len(), y.len()), (self.nrows(), self.ncols()));
        let sqrt = self.design.weights.sqrt();
        scatter(self.design, &self.folds, x, y, sqrt, stack);
        Ok(())
    }
}
