use std::sync::Mutex;

use schwarz_precond::Operator;

use crate::domain::WeightedDesign;
use crate::observation::ObservationStore;

/// Implicit weighted Gramian operator: computes `D^T(W·(D·x))`.
///
/// For unweighted designs, this is `D^T(D·x)`.
/// Scratch space avoids per-call allocation.
pub struct GramianOperator<'a, S: ObservationStore> {
    design: &'a WeightedDesign<S>,
    scratch: Mutex<Vec<f64>>,
}

impl<'a, S: ObservationStore> GramianOperator<'a, S> {
    pub fn new(design: &'a WeightedDesign<S>) -> Self {
        Self {
            scratch: Mutex::new(vec![0.0; design.n_rows]),
            design,
        }
    }
}

impl<S: ObservationStore> Operator for GramianOperator<'_, S> {
    fn nrows(&self) -> usize {
        self.design.n_dofs
    }

    fn ncols(&self) -> usize {
        self.design.n_dofs
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) {
        let mut tmp = self.scratch.lock().unwrap();
        tmp.fill(0.0);
        self.design.matvec_d(x, &mut tmp);
        self.design.rmatvec_wdt(&tmp, y);
    }

    fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) {
        self.apply(x, y);
    }
}
