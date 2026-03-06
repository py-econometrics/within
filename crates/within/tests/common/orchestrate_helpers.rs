#![allow(dead_code)]

use schwarz_precond::Operator;

use within::operator::gramian::GramianOperator;
use within::{
    solve_normal_equations, FixedEffectsDesign, ObservationStore, SolveResult, SolverParams,
};

pub fn make_test_design() -> FixedEffectsDesign {
    FixedEffectsDesign::new(
        vec![vec![0, 1, 0, 1, 2], vec![0, 0, 1, 1, 0]],
        vec![3, 2],
        5,
    )
    .expect("valid test design")
}

pub fn make_rhs_from_unit_solution(design: &FixedEffectsDesign) -> Vec<f64> {
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

pub fn solve_response<S: ObservationStore>(
    design: &within::WeightedDesign<S>,
    y: &[f64],
    params: &SolverParams,
) -> within::WithinResult<SolveResult> {
    let mut rhs = vec![0.0; design.n_dofs];
    design.rmatvec_wdt(y, &mut rhs);
    solve_normal_equations(design, &rhs, params)
}
