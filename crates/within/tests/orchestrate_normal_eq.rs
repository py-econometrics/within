use within::{solve_normal_equations, LocalSolverConfig, OperatorRepr, SolverMethod, SolverParams};

#[path = "common/orchestrate_helpers.rs"]
mod common;

#[test]
fn test_normal_equations_cg_unpreconditioned() {
    let design = common::make_test_design();
    let rhs = common::make_rhs_from_unit_solution(&design);

    let params = SolverParams {
        method: SolverMethod::Cg {
            preconditioner: None,
            operator: OperatorRepr::Implicit,
        },
        tol: 1e-8,
        maxiter: 1000,
    };
    let result = solve_normal_equations(&design, &rhs, &params).expect("normal equations cg");
    common::assert_converged_with_small_residual(&result, 1e-6);
}

#[test]
fn test_normal_equations_preconditioned() {
    let design = common::make_test_design();
    let rhs = common::make_rhs_from_unit_solution(&design);

    let params = SolverParams {
        method: SolverMethod::Cg {
            preconditioner: Some(LocalSolverConfig::default()),
            operator: OperatorRepr::Implicit,
        },
        tol: 1e-8,
        maxiter: 1000,
    };
    let result =
        solve_normal_equations(&design, &rhs, &params).expect("normal equations preconditioned");
    common::assert_converged_with_small_residual(&result, 1e-6);
    assert!(result.time_setup > 0.0, "Setup time should be positive");
}

#[test]
fn test_least_squares_cg() {
    let design = common::make_test_design();
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let params = SolverParams {
        method: SolverMethod::Cg {
            preconditioner: None,
            operator: OperatorRepr::Implicit,
        },
        tol: 1e-8,
        maxiter: 1000,
    };
    let mut rhs = vec![0.0; design.n_dofs];
    design.rmatvec_wdt(&y, &mut rhs);
    let result = solve_normal_equations(&design, &rhs, &params).expect("least squares cg");
    assert!(result.converged, "CG LS did not converge");
    common::assert_solution_finite(&result);
}

#[test]
fn test_least_squares_weighted_cg_preconditioned() {
    let design = common::make_weighted_design(
        vec![vec![0, 1, 0, 1, 2], vec![0, 0, 1, 1, 0]],
        vec![3, 2],
        within::ObservationWeights::Dense(vec![1.0, 2.0, 1.5, 0.5, 3.0]),
    )
    .expect("valid weighted design");
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let params = SolverParams {
        method: SolverMethod::Cg {
            preconditioner: Some(LocalSolverConfig::cg_default()),
            operator: OperatorRepr::Implicit,
        },
        tol: 1e-8,
        maxiter: 1000,
    };
    let mut rhs = vec![0.0; design.n_dofs];
    design.rmatvec_wdt(&y, &mut rhs);
    let result = solve_normal_equations(&design, &rhs, &params).expect("weighted cg solve");
    common::assert_converged_with_small_residual(&result, 1e-6);
    common::assert_solution_finite(&result);
}
