pub(crate) mod csr_block;
pub mod gramian;
pub(crate) mod local_solver;
pub(crate) mod residual_update;
pub(crate) mod schur_complement;
pub mod schwarz;

#[cfg(test)]
mod tests;

// ---------------------------------------------------------------------------
// DesignOperator — rectangular, D·x / D^T·x (no weights)
// ---------------------------------------------------------------------------

use schwarz_precond::Operator;

use crate::domain::WeightedDesign;
use crate::observation::ObservationStore;

/// Operator wrapper around `&WeightedDesign<S>`.
///
/// `apply` = D·x (gather-add), `apply_adjoint` = D^T·x (scatter-add).
/// This is the raw design matrix — no observation weights.
pub struct DesignOperator<'a, S: ObservationStore> {
    design: &'a WeightedDesign<S>,
}

impl<'a, S: ObservationStore> DesignOperator<'a, S> {
    pub fn new(design: &'a WeightedDesign<S>) -> Self {
        Self { design }
    }
}

impl<S: ObservationStore> Operator for DesignOperator<'_, S> {
    fn nrows(&self) -> usize {
        self.design.n_rows
    }

    fn ncols(&self) -> usize {
        self.design.n_dofs
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) {
        self.design.matvec_d(x, y);
    }

    fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) {
        self.design.rmatvec_dt(x, y);
    }
}
