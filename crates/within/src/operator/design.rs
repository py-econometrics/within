//! Design matrix operators wrapping `WeightedDesign<S>`.
//!
//! `DesignOperator` is a thin Operator wrapper for D·x / D^T·x (no weights).

use schwarz_precond::Operator;

use crate::domain::WeightedDesign;
use crate::observation::ObservationStore;

// ---------------------------------------------------------------------------
// DesignOperator — rectangular, D·x / D^T·x (no weights)
// ---------------------------------------------------------------------------

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

    /// y = D @ x  (gather-add)
    pub fn matvec_d(&self, x: &[f64], y: &mut [f64]) {
        self.design.matvec_d(x, y);
    }

    /// x = D^T @ r  (scatter-add)
    pub fn rmatvec_dt(&self, r: &[f64], x: &mut [f64]) {
        self.design.rmatvec_dt(r, x);
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
        self.matvec_d(x, y);
    }

    fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) {
        self.rmatvec_dt(x, y);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::FixedEffectsDesign;

    fn make_test_design() -> FixedEffectsDesign {
        FixedEffectsDesign::new(
            vec![vec![0, 1, 2, 0, 1], vec![0, 1, 2, 3, 0]],
            vec![3, 4],
            5,
        )
        .expect("valid test design")
    }

    #[test]
    fn test_design_operator_dimensions() {
        let schema = make_test_design();
        let op = DesignOperator::new(&schema);
        assert_eq!(op.nrows(), 5);
        assert_eq!(op.ncols(), 7);
    }

    #[test]
    fn test_design_operator_adjoint() {
        let schema = make_test_design();
        let op = DesignOperator::new(&schema);

        let x = vec![1.0, -0.5, 2.0, 0.3, -1.0, 0.7, 1.5];
        let r = vec![0.1, 0.2, -0.3, 0.4, -0.5];

        let mut dx = vec![0.0; 5];
        op.apply(&x, &mut dx);
        let lhs: f64 = dx.iter().zip(r.iter()).map(|(a, b)| a * b).sum();

        let mut dtr = vec![0.0; 7];
        op.apply_adjoint(&r, &mut dtr);
        let rhs: f64 = x.iter().zip(dtr.iter()).map(|(a, b)| a * b).sum();

        assert!((lhs - rhs).abs() < 1e-12);
    }

    #[test]
    fn test_matvec_d() {
        let schema = make_test_design();
        let op = DesignOperator::new(&schema);
        let x = vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 40.0];
        let mut y = vec![0.0; 5];
        op.matvec_d(&x, &mut y);
        assert_eq!(y, vec![11.0, 22.0, 33.0, 41.0, 12.0]);
    }

    #[test]
    fn test_rmatvec_dt() {
        let schema = make_test_design();
        let op = DesignOperator::new(&schema);
        let r = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut x = vec![0.0; 7];
        op.rmatvec_dt(&r, &mut x);
        assert_eq!(x, vec![5.0, 7.0, 3.0, 6.0, 2.0, 3.0, 4.0]);
    }
}
