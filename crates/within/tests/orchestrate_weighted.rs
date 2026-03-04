use within::{
    solve_least_squares, CgPreconditioner, FixedEffectsDesign, SchwarzConfig, SolverMethod,
    SolverParams,
};

#[path = "common/orchestrate_helpers.rs"]
mod common;

#[test]
fn test_least_squares_weighted_lsmr() {
    let design = FixedEffectsDesign::new_weighted(
        vec![vec![0, 1, 0, 1, 2], vec![0, 0, 1, 1, 0]],
        vec![3, 2],
        5,
        vec![1.0, 2.0, 1.5, 0.5, 3.0],
    )
    .expect("valid weighted design");
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let params = SolverParams {
        method: SolverMethod::Lsmr { conlim: 1e8 },
        tol: 1e-8,
        maxiter: 1000,
    };
    let result = solve_least_squares(&design, &y, None, &params).expect("weighted lsmr solve");
    common::assert_converged_with_small_residual(&result, 1e-6);
    common::assert_solution_finite(&result);
}

#[test]
fn test_least_squares_weighted_cg_preconditioned() {
    let design = FixedEffectsDesign::new_weighted(
        vec![vec![0, 1, 0, 1, 2], vec![0, 0, 1, 1, 0]],
        vec![3, 2],
        5,
        vec![1.0, 2.0, 1.5, 0.5, 3.0],
    )
    .expect("valid weighted design");
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let params = SolverParams {
        method: SolverMethod::Cg {
            preconditioner: CgPreconditioner::OneLevel(SchwarzConfig::default()),
        },
        tol: 1e-8,
        maxiter: 1000,
    };
    let result = solve_least_squares(&design, &y, None, &params).expect("weighted cg solve");
    common::assert_converged_with_small_residual(&result, 1e-6);
    common::assert_solution_finite(&result);
}
