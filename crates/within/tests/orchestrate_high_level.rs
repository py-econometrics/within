use within::{
    solve, solve_weighted, OperatorRepr, Preconditioner, SchwarzConfig, SolverMethod, SolverParams,
};

#[path = "common/orchestrate_helpers.rs"]
mod common;

#[test]
fn test_high_level_solve() {
    let categories: Vec<Vec<u32>> = vec![vec![0, 1, 0, 1, 2], vec![0, 0, 1, 1, 0]];
    let n_levels = vec![3, 2];
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let params = SolverParams::default();
    let result = solve(&categories, &n_levels, &y, &params, None).expect("solve");
    common::assert_converged_with_small_residual(&result, 1e-6);
    common::assert_solution_finite(&result);
}

#[test]
fn test_high_level_solve_weighted() {
    let categories: Vec<Vec<u32>> = vec![vec![0, 1, 0, 1, 2], vec![0, 0, 1, 1, 0]];
    let n_levels = vec![3, 2];
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let weights = vec![1.0, 2.0, 1.5, 0.5, 3.0];

    let params = SolverParams::default();
    let result = solve_weighted(&categories, &n_levels, &y, &weights, &params, None)
        .expect("solve weighted");
    common::assert_converged_with_small_residual(&result, 1e-6);
    common::assert_solution_finite(&result);
}

#[test]
fn test_high_level_solve_preconditioned() {
    let categories: Vec<Vec<u32>> = vec![vec![0, 1, 0, 1, 2], vec![0, 0, 1, 1, 0]];
    let n_levels = vec![3, 2];
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let params = SolverParams {
        method: SolverMethod::Cg {
            preconditioner: Preconditioner::Additive(SchwarzConfig::default()),
            operator: OperatorRepr::Implicit,
        },
        tol: 1e-8,
        maxiter: 1000,
    };
    let result = solve(&categories, &n_levels, &y, &params, None).expect("solve preconditioned");
    common::assert_converged_with_small_residual(&result, 1e-6);
    common::assert_solution_finite(&result);
}
