use ndarray::array;
use within::{solve, KrylovMethod, LocalSolverConfig, OperatorRepr, Preconditioner, SolverParams};

#[path = "common/orchestrate_helpers.rs"]
mod common;

#[test]
fn test_high_level_solve() {
    let categories = array![[0u32, 0], [1, 0], [0, 1], [1, 1], [2, 0]];
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let params = SolverParams::default();
    let precond = Preconditioner::Additive(LocalSolverConfig::solver_default());
    let result = solve(categories.view(), &y, None, &params, Some(&precond)).expect("solve");
    common::assert_converged_with_small_residual(&result, 1e-6);
    common::assert_solution_finite(&result);
}

#[test]
fn test_high_level_solve_weighted() {
    let categories = array![[0u32, 0], [1, 0], [0, 1], [1, 1], [2, 0]];
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let weights = vec![1.0, 2.0, 1.5, 0.5, 3.0];

    let params = SolverParams::default();
    let precond = Preconditioner::Additive(LocalSolverConfig::solver_default());
    let result = solve(
        categories.view(),
        &y,
        Some(&weights),
        &params,
        Some(&precond),
    )
    .expect("solve weighted");
    common::assert_converged_with_small_residual(&result, 1e-6);
    common::assert_solution_finite(&result);
}

#[test]
fn test_high_level_solve_preconditioned() {
    let categories = array![[0u32, 0], [1, 0], [0, 1], [1, 1], [2, 0]];
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let params = SolverParams {
        krylov: KrylovMethod::Cg,
        operator: OperatorRepr::Implicit,
        tol: 1e-8,
        maxiter: 1000,
    };
    let precond = Preconditioner::Additive(LocalSolverConfig::solver_default());
    let result =
        solve(categories.view(), &y, None, &params, Some(&precond)).expect("solve preconditioned");
    common::assert_converged_with_small_residual(&result, 1e-6);
    common::assert_solution_finite(&result);
}
