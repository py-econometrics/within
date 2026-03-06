use approx_chol::Config;

use within::{
    solve_least_squares, solve_normal_equations, LocalSolverConfig, OperatorRepr, Preconditioner,
    SchwarzConfig, SolverMethod, SolverParams,
};

#[path = "common/orchestrate_helpers.rs"]
mod common;

#[test]
fn test_normal_equations_lsmr_unpreconditioned() {
    let design = common::make_test_design();
    let rhs = common::make_rhs_from_unit_solution(&design);

    let params = SolverParams {
        method: SolverMethod::Lsmr { conlim: 1e8 },
        tol: 1e-8,
        maxiter: 1000,
    };
    let result =
        solve_normal_equations(&design, &rhs, None, &params).expect("normal equations lsmr");
    common::assert_converged_with_small_residual(&result, 1e-6);
}

#[test]
fn test_normal_equations_cg_unpreconditioned() {
    let design = common::make_test_design();
    let rhs = common::make_rhs_from_unit_solution(&design);

    let params = SolverParams {
        method: SolverMethod::Cg {
            preconditioner: Preconditioner::None,
            operator: OperatorRepr::Implicit,
        },
        tol: 1e-8,
        maxiter: 1000,
    };
    let result = solve_normal_equations(&design, &rhs, None, &params).expect("normal equations cg");
    common::assert_converged_with_small_residual(&result, 1e-6);
}

#[test]
fn test_normal_equations_preconditioned() {
    let design = common::make_test_design();
    let rhs = common::make_rhs_from_unit_solution(&design);

    let params = SolverParams {
        method: SolverMethod::Cg {
            preconditioner: Preconditioner::Additive(SchwarzConfig {
                smoother: Config {
                    seed: 42,
                    ..Default::default()
                },
                local_solver: LocalSolverConfig::default(),
            }),
            operator: OperatorRepr::Implicit,
        },
        tol: 1e-8,
        maxiter: 1000,
    };
    let result = solve_normal_equations(&design, &rhs, None, &params)
        .expect("normal equations preconditioned");
    common::assert_converged_with_small_residual(&result, 1e-6);
    assert!(result.time_setup > 0.0, "Setup time should be positive");
}

#[test]
fn test_least_squares_lsmr() {
    let design = common::make_test_design();
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let params = SolverParams {
        method: SolverMethod::Lsmr { conlim: 1e8 },
        tol: 1e-8,
        maxiter: 1000,
    };
    let result = solve_least_squares(&design, &y, None, &params).expect("least squares lsmr");
    assert!(result.converged, "LSMR LS did not converge");
    common::assert_solution_finite(&result);
}

#[test]
fn test_least_squares_cg() {
    let design = common::make_test_design();
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let params = SolverParams {
        method: SolverMethod::Cg {
            preconditioner: Preconditioner::None,
            operator: OperatorRepr::Implicit,
        },
        tol: 1e-8,
        maxiter: 1000,
    };
    let result = solve_least_squares(&design, &y, None, &params).expect("least squares cg");
    assert!(result.converged, "CG LS did not converge");
    common::assert_solution_finite(&result);
}
