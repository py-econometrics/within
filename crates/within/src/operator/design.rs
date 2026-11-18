use schwarz_precond::Operator;

use crate::domain::{Design, PreparedDesign, PreparedTerm, RowWeights, PAR_THRESHOLD};

mod gather;
mod scatter;

#[cfg(test)]
mod tests;

use gather::gather_apply;
use scatter::scatter_block;

/// `A = W^{1/2} D̃`: the operator LSMR iterates on.
#[derive(Clone, Copy)]
pub(crate) struct DesignOperator<'p> {
    /// `D̃`: the design in the solve basis.
    pub(crate) design: &'p Design<'p>,
    pub(crate) weights: &'p RowWeights,
}

impl PreparedDesign<'_> {
    /// `A`, read from this design's factors.
    pub(crate) fn operator(&self) -> DesignOperator<'_> {
        DesignOperator {
            design: &self.design,
            weights: &self.weights,
        }
    }

    /// `y − D x`, read off this operator's measured `b − A x` unless a zero weight erased a row.
    pub(crate) fn demeaned(&self, x: &[f64], y: &[f64], measured: Option<Vec<f64>>) -> Vec<f64> {
        match measured {
            Some(mut residual) if !self.weights.erases_rows() => {
                self.weights.unscale(&mut residual);
                residual
            }
            measured => {
                let mut demeaned = measured.unwrap_or_else(|| vec![0.0; y.len()]);
                gather_apply(&self.design, x, &mut demeaned, None);
                for (d, &yi) in demeaned.iter_mut().zip(y) {
                    *d = yi - *d;
                }
                demeaned
            }
        }
    }

    /// Fits the dominant block of `x` to `b = W^½ y` given the other blocks; 0 if unidentified.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "the reduced system recovers with it")
    )]
    pub(crate) fn recover_eliminated(&self, b: &[f64], x: &mut [f64]) {
        let eliminated = self.design.layout.dominant();
        let dofs = self.design.layout.term(eliminated).dofs();
        x[dofs.clone()].fill(0.0);
        let mut residual = vec![0.0; b.len()];
        gather_apply(&self.design, x, &mut residual, self.weights.sqrt());
        for (r, &bi) in residual.iter_mut().zip(b) {
            *r = bi - *r;
        }
        let block = &mut x[dofs];
        self.operator()
            .adjoint_block(&self.design.term(eliminated), &residual, block);
        // Whitening makes each level's block of `A_eᵀA_e` diagonal.
        for (xj, g) in block
            .iter_mut()
            .zip(self.gramian().term_diagonal(eliminated))
        {
            *xj = if g > 0.0 { *xj / g } else { 0.0 };
        }
    }
}

impl DesignOperator<'_> {
    /// `block = A_tᵀ v` for one term `t`.
    fn adjoint_block(&self, t: &PreparedTerm<'_>, v: &[f64], block: &mut [f64]) {
        match self.weights.sqrt() {
            Some(sw) => scatter_block(t, block, &|i| sw[i] * v[i]),
            None => scatter_block(t, block, &|i| v[i]),
        }
    }
}

/// Design operator `D` or `W^{1/2} D`, whose normal equations `AᵀA = DᵀWD` recover the Gramian.
impl Operator for DesignOperator<'_> {
    fn nrows(&self) -> usize {
        self.design.n_obs()
    }

    fn ncols(&self) -> usize {
        self.design.n_dofs()
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), schwarz_precond::SolveError> {
        gather_apply(self.design, x, y, self.weights.sqrt());
        Ok(())
    }

    fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) -> Result<(), schwarz_precond::SolveError> {
        debug_assert_eq!(x.len(), self.design.n_obs());
        debug_assert_eq!(y.len(), self.design.n_dofs());
        for t in self.design.terms() {
            self.adjoint_block(&t, x, &mut y[t.layout.dofs()]);
        }
        Ok(())
    }
}
