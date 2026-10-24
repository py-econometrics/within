use std::sync::Mutex;

use schwarz_precond::{Operator, SolveError};

use crate::domain::elimination::Elimination;

/// A full-coefficient map seen through the kept coefficients: embed, apply, restrict.
pub(crate) struct RestrictedInverse<'a> {
    elimination: &'a Elimination<'a>,
    full: &'a dyn Operator,
    scratch: Mutex<(Vec<f64>, Vec<f64>)>,
}

impl<'a> RestrictedInverse<'a> {
    pub(crate) fn new(elimination: &'a Elimination<'a>, full: &'a dyn Operator) -> Self {
        let n = elimination.prepared().design.n_dofs;
        Self {
            elimination,
            full,
            scratch: Mutex::new((vec![0.0; n], vec![0.0; n])),
        }
    }
}

impl Operator for RestrictedInverse<'_> {
    fn nrows(&self) -> usize {
        self.elimination.n_kept()
    }
    fn ncols(&self) -> usize {
        self.nrows()
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        let mut scratch = self.scratch.lock().expect("restriction scratch poisoned");
        let (rhs, solution) = &mut *scratch;
        self.elimination.embed(x, rhs);
        self.full.apply(rhs, solution)?;
        self.elimination.restrict(solution, y);
        Ok(())
    }

    fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        self.apply(x, y)
    }
}
