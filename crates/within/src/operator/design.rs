//! Design matrix operators wrapping `WeightedDesign<S>`.
//!
//! `DesignOperator` is a thin Operator wrapper for D·x / D^T·x (no weights).
//! `PreconditionedDesign` fuses diagonal preconditioning (and optional
//! W^{1/2} weighting for LSMR) into a single scatter-gather pass.

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

// ---------------------------------------------------------------------------
// PreconditionedDesign — fused W^{1/2}·D·diag^{-1} specialization
// ---------------------------------------------------------------------------

/// Fused operator for W^{1/2}·D·diag^{-1}.
///
/// For unweighted designs, this reduces to D·diag^{-1}.
/// `apply` = W^{1/2}·D·(diag^{-1}·x), `apply_adjoint` = diag^{-1}·D^T·W^{1/2}·x.
pub struct PreconditionedDesign<'a, S: ObservationStore> {
    design: &'a WeightedDesign<S>,
    inv_diag: Vec<f64>,
    /// Pre-computed sqrt(weight(i)) for each observation.
    /// Empty when design is unweighted (all weights = 1.0).
    sqrt_weights: Vec<f64>,
}

impl<'a, S: ObservationStore> PreconditionedDesign<'a, S> {
    pub fn new(design: &'a WeightedDesign<S>, inv_diag: Vec<f64>) -> Self {
        debug_assert_eq!(inv_diag.len(), design.n_dofs);
        let sqrt_weights = if design.store.is_unweighted() {
            Vec::new()
        } else {
            (0..design.n_rows)
                .map(|i| design.weight(i).sqrt())
                .collect()
        };
        Self {
            design,
            inv_diag,
            sqrt_weights,
        }
    }

    /// Access the inverse diagonal for post-solve recovery: x = diag^{-1} · z.
    pub fn inv_diag(&self) -> &[f64] {
        &self.inv_diag
    }

    /// y = W^{1/2} · D · (inv_diag .* z)  (fused gather-scale, optional weight)
    #[allow(clippy::needless_range_loop)]
    fn matvec_precond(&self, z: &[f64], y: &mut [f64]) {
        debug_assert_eq!(z.len(), self.design.n_dofs);
        debug_assert_eq!(y.len(), self.design.n_rows);
        y.fill(0.0);
        for (q, f) in self.design.factors.iter().enumerate() {
            for i in 0..self.design.n_rows {
                let c = self.design.level(i, q) as usize;
                let j = f.offset + c;
                y[i] += self.inv_diag[j] * z[j];
            }
        }
        // Apply pre-computed sqrt(w) if weighted — branch outside the inner loop
        if !self.sqrt_weights.is_empty() {
            for (yi, &w) in y.iter_mut().zip(&self.sqrt_weights) {
                *yi *= w;
            }
        }
    }

    /// x = inv_diag .* D^T · W^{1/2} · r  (scatter then scale)
    fn rmatvec_precond_t(&self, r: &[f64], x: &mut [f64]) {
        debug_assert_eq!(r.len(), self.design.n_rows);
        debug_assert_eq!(x.len(), self.design.n_dofs);
        x.fill(0.0);
        if self.sqrt_weights.is_empty() {
            for (q, f) in self.design.factors.iter().enumerate() {
                for i in 0..self.design.n_rows {
                    x[f.offset + self.design.level(i, q) as usize] += r[i];
                }
            }
        } else {
            for (q, f) in self.design.factors.iter().enumerate() {
                for i in 0..self.design.n_rows {
                    x[f.offset + self.design.level(i, q) as usize] += self.sqrt_weights[i] * r[i];
                }
            }
        }
        for (xi, &di) in x.iter_mut().zip(self.inv_diag.iter()) {
            *xi *= di;
        }
    }
}

impl<S: ObservationStore> Operator for PreconditionedDesign<'_, S> {
    fn nrows(&self) -> usize {
        self.design.n_rows
    }

    fn ncols(&self) -> usize {
        self.design.n_dofs
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) {
        self.matvec_precond(x, y);
    }

    fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) {
        self.rmatvec_precond_t(x, y);
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

    #[test]
    fn test_rmatvec_precond_t() {
        let schema = make_test_design();
        let inv_diag = vec![0.5, 0.5, 0.5, 0.25, 0.25, 0.25, 0.25];

        let op = DesignOperator::new(&schema);
        let r = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut dtr = vec![0.0; 7];
        op.rmatvec_dt(&r, &mut dtr);
        let expected: Vec<f64> = dtr
            .iter()
            .zip(inv_diag.iter())
            .map(|(a, b)| a * b)
            .collect();

        let precond = PreconditionedDesign::new(&schema, inv_diag);
        let mut actual = vec![0.0; 7];
        precond.apply_adjoint(&r, &mut actual);

        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!((a - e).abs() < 1e-14);
        }
    }

    #[test]
    fn test_preconditioned_design_matches_manual() {
        let schema = make_test_design();
        let inv_diag = vec![0.5, 0.5, 0.5, 0.25, 0.25, 0.25, 0.25];
        let op = PreconditionedDesign::new(&schema, inv_diag.clone());

        let z = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];

        let design_op = DesignOperator::new(&schema);
        let scaled: Vec<f64> = z.iter().zip(inv_diag.iter()).map(|(a, b)| a * b).collect();
        let mut expected = vec![0.0; 5];
        design_op.apply(&scaled, &mut expected);

        let mut actual = vec![0.0; 5];
        op.apply(&z, &mut actual);

        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!((a - e).abs() < 1e-14);
        }
    }
}
