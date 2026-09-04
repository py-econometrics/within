use within::config::{LocalSolverConfig, ReductionStrategy};
use within::{LsmrOptions, PreconditionerConfig, Solver};

#[path = "common/orchestrate_helpers.rs"]
mod common;

#[test]
fn test_lsmr_unpreconditioned() {
    let design = common::make_test_design();
    let y = common::make_deterministic_y(&design);

    let params = LsmrOptions {
        tol: 1e-8,
        maxiter: 1000,
        ..Default::default()
    };
    let solver = Solver::new(design, None, None).expect("build solver");
    let result = solver.solve(&y, &params).expect("solve");
    common::assert_converged_with_small_residual(&result, 1e-6);
    common::assert_normal_equations_satisfied(&common::test_categories(), None, &y, &result, 1e-6);
}

#[test]
fn test_lsmr_preconditioned() {
    let design = common::make_test_design();
    let y = common::make_deterministic_y(&design);

    let params = LsmrOptions {
        tol: 1e-8,
        maxiter: 1000,
        ..Default::default()
    };
    let precond = PreconditionerConfig::Additive {
        local_solver: LocalSolverConfig::default(),
        reduction: ReductionStrategy::Auto,
    };
    let solver = Solver::new(design, None, &precond).expect("build solver");
    let result = solver.solve(&y, &params).expect("solve");
    common::assert_converged_with_small_residual(&result, 1e-6);
    common::assert_normal_equations_satisfied(&common::test_categories(), None, &y, &result, 1e-6);
}

#[test]
fn test_lsmr_diagonal_preconditioned() {
    let design = common::make_test_design();
    let y = common::make_deterministic_y(&design);

    let params = LsmrOptions {
        tol: 1e-8,
        maxiter: 1000,
        ..Default::default()
    };
    let precond = PreconditionerConfig::Diagonal;
    let solver = Solver::new(design, None, &precond).expect("build solver");
    let result = solver.solve(&y, &params).expect("solve");
    common::assert_converged_with_small_residual(&result, 1e-6);
    common::assert_solution_finite(&result);
    common::assert_normal_equations_satisfied(&common::test_categories(), None, &y, &result, 1e-6);
}

#[test]
fn test_lsmr_least_squares() {
    let design = common::make_test_design();
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let params = LsmrOptions {
        tol: 1e-8,
        maxiter: 1000,
        ..Default::default()
    };
    let solver = Solver::new(design, None, None).expect("build solver");
    let result = solver.solve(&y, &params).expect("solve");
    assert!(result.converged, "LSMR LS did not converge");
    common::assert_solution_finite(&result);
    common::assert_normal_equations_satisfied(&common::test_categories(), None, &y, &result, 1e-6);
}

#[test]
fn test_lsmr_least_squares_weighted_preconditioned() {
    let design =
        common::make_design(vec![vec![0, 1, 0, 1, 2], vec![0, 0, 1, 1, 0]]).expect("valid design");
    let weights = vec![1.0, 2.0, 1.5, 0.5, 3.0];
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let params = LsmrOptions {
        tol: 1e-8,
        maxiter: 1000,
        ..Default::default()
    };
    let precond = PreconditionerConfig::default();
    let solver = Solver::new(design, Some(weights.clone()), &precond).expect("build solver");
    let result = solver.solve(&y, &params).expect("solve");
    common::assert_converged_with_small_residual(&result, 1e-6);
    common::assert_solution_finite(&result);
    common::assert_normal_equations_satisfied(
        &common::test_categories(),
        Some(weights.as_slice()),
        &y,
        &result,
        1e-6,
    );
}
