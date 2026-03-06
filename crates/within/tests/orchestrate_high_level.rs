use within::{
    demean_batch, solve, solve_weighted, GmresPrecond, LocalSolverConfig, OperatorRepr,
    SolverMethod, SolverParams,
};

#[path = "common/orchestrate_helpers.rs"]
mod common;

#[test]
fn test_high_level_solve() {
    let categories: Vec<Vec<u32>> = vec![vec![0, 1, 0, 1, 2], vec![0, 0, 1, 1, 0]];
    let n_levels = vec![3, 2];
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let params = SolverParams::default();
    let result = solve(&categories, &n_levels, &y, &params).expect("solve");
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
    let result =
        solve_weighted(&categories, &n_levels, &y, &weights, &params).expect("solve weighted");
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
            preconditioner: Some(LocalSolverConfig::cg_default()),
            operator: OperatorRepr::Implicit,
        },
        tol: 1e-8,
        maxiter: 1000,
    };
    let result = solve(&categories, &n_levels, &y, &params).expect("solve preconditioned");
    common::assert_converged_with_small_residual(&result, 1e-6);
    common::assert_solution_finite(&result);
}

#[test]
fn test_demean_batch_gmres_multiplicative() {
    let design = common::make_test_design();
    let columns = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5.0, 4.0, 3.0, 2.0, 1.0]];
    let params = SolverParams {
        method: SolverMethod::Gmres {
            preconditioner: Some(GmresPrecond::Multiplicative(
                LocalSolverConfig::gmres_default(),
            )),
            operator: OperatorRepr::Implicit,
            restart: 30,
        },
        tol: 1e-8,
        maxiter: 1000,
    };
    let result = demean_batch(&design, &columns, &params).expect("demean batch");
    assert!(result.all_converged, "batch solve must converge");
    assert_eq!(result.columns.len(), columns.len(), "column count mismatch");
    assert!(
        result
            .columns
            .iter()
            .flatten()
            .all(|value| value.is_finite()),
        "demeaned values must be finite"
    );
}
