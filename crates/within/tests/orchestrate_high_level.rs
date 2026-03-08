use ndarray::array;
use within::{
    solve, solve_batch, KrylovMethod, LocalSolverConfig, OperatorRepr, Preconditioner, SolverParams,
};

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

#[test]
fn test_solve_batch_matches_individual() {
    let categories = array![[0u32, 0], [1, 0], [0, 1], [1, 1], [2, 0]];
    let y1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y2 = vec![5.0, 4.0, 3.0, 2.0, 1.0];

    let params = SolverParams::default();
    let precond = Preconditioner::Additive(LocalSolverConfig::solver_default());

    let r1 = solve(categories.view(), &y1, None, &params, Some(&precond)).expect("solve y1");
    let r2 = solve(categories.view(), &y2, None, &params, Some(&precond)).expect("solve y2");

    let batch = solve_batch(
        categories.view(),
        &[&y1, &y2],
        None,
        &params,
        Some(&precond),
    )
    .expect("solve batch");

    assert_eq!(batch.n_rhs(), 2);
    for (a, b) in batch.x(0).iter().zip(r1.x.iter()) {
        assert!((a - b).abs() < 1e-12, "batch vs individual x mismatch");
    }
    for (a, b) in batch.x(1).iter().zip(r2.x.iter()) {
        assert!((a - b).abs() < 1e-12, "batch vs individual x mismatch");
    }
}

#[test]
fn test_solve_batch_single_rhs() {
    let categories = array![[0u32, 0], [1, 0], [0, 1], [1, 1], [2, 0]];
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let params = SolverParams::default();
    let precond = Preconditioner::Additive(LocalSolverConfig::solver_default());

    let batch = solve_batch(categories.view(), &[&y[..]], None, &params, Some(&precond))
        .expect("solve batch single");

    assert_eq!(batch.n_rhs(), 1);
    assert!(batch.converged()[0]);
    assert!(batch.x(0).iter().all(|v| v.is_finite()));
}

#[test]
fn test_solve_batch_weighted() {
    let categories = array![[0u32, 0], [1, 0], [0, 1], [1, 1], [2, 0]];
    let y1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y2 = vec![5.0, 4.0, 3.0, 2.0, 1.0];
    let weights = vec![1.0, 2.0, 1.5, 0.5, 3.0];

    let params = SolverParams::default();
    let precond = Preconditioner::Additive(LocalSolverConfig::solver_default());

    let batch = solve_batch(
        categories.view(),
        &[&y1, &y2],
        Some(&weights),
        &params,
        Some(&precond),
    )
    .expect("solve batch weighted");

    assert_eq!(batch.n_rhs(), 2);
    assert!(batch.converged().iter().all(|&c| c));
}
