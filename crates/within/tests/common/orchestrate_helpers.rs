#![allow(dead_code)]

use schwarz_precond::Operator;

use within::operator::gramian::GramianOperator;
use within::{FactorMajorStore, ObservationWeights, SolveResult, WeightedDesign};

pub fn make_test_design() -> WeightedDesign<FactorMajorStore> {
    make_weighted_design(
        vec![vec![0, 1, 0, 1, 2], vec![0, 0, 1, 1, 0]],
        ObservationWeights::Unit,
    )
    .expect("valid test design")
}

pub fn make_weighted_design(
    categories: Vec<Vec<u32>>,
    weights: ObservationWeights,
) -> within::WithinResult<WeightedDesign<FactorMajorStore>> {
    let n_rows = categories.first().map_or(0, Vec::len);
    let store = FactorMajorStore::new(categories, weights, n_rows)?;
    WeightedDesign::from_store(store)
}

/// Compute y = D * 1 so that the true solution of `min ||y - Dx||^2` is x = 1.
pub fn make_y_from_unit_solution(design: &WeightedDesign<FactorMajorStore>) -> Vec<f64> {
    let x_true = vec![1.0; design.n_dofs];
    let mut y = vec![0.0; design.n_rows];
    design.matvec_d(&x_true, &mut y);
    y
}

/// Compute rhs = G * 1 in normal-equation space (for low-level Schwarz tests).
pub fn make_rhs_from_unit_solution(design: &WeightedDesign<FactorMajorStore>) -> Vec<f64> {
    let gramian_op = GramianOperator::new(design);
    let x_true = vec![1.0; design.n_dofs];
    let mut rhs = vec![0.0; design.n_dofs];
    gramian_op.apply(&x_true, &mut rhs);
    rhs
}

pub fn assert_converged_with_small_residual(result: &SolveResult, tol: f64) {
    assert!(result.converged, "solver did not converge");
    assert!(
        result.final_residual < tol,
        "residual too large: {}",
        result.final_residual
    );
}

pub fn assert_solution_finite(result: &SolveResult) {
    assert!(
        result.x.iter().all(|v| v.is_finite()),
        "Non-finite solution"
    );
}
