use within::config::LocalSolverConfig;
use within::{KrylovMethod, OperatorRepr, Preconditioner, ReductionStrategy, Solver, SolverParams};

#[path = "common/orchestrate_helpers.rs"]
mod common;

// ===========================================================================
// Solver-based tests (replacing former solve_normal_equations tests)
// ===========================================================================

#[test]
fn test_cg_unpreconditioned() {
    let design = common::make_test_design();
    let y = common::make_y_from_unit_solution(&design);

    let params = SolverParams {
        krylov: KrylovMethod::Cg,
        operator: OperatorRepr::Implicit,
        tol: 1e-8,
        maxiter: 1000,
        ..Default::default()
    };
    let solver = Solver::from_design(design, None, &params, None).expect("build solver");
    let result = solver.solve(&y).expect("solve");
    common::assert_converged_with_small_residual(&result, 1e-6);
}

#[test]
fn test_cg_preconditioned() {
    let design = common::make_test_design();
    let y = common::make_y_from_unit_solution(&design);

    let params = SolverParams {
        krylov: KrylovMethod::Cg,
        operator: OperatorRepr::Implicit,
        tol: 1e-8,
        maxiter: 1000,
        ..Default::default()
    };
    let precond = Preconditioner::Additive(LocalSolverConfig::default(), ReductionStrategy::Auto);
    let solver = Solver::from_design(design, None, &params, Some(&precond)).expect("build solver");
    let result = solver.solve(&y).expect("solve");
    common::assert_converged_with_small_residual(&result, 1e-6);
}

#[test]
fn test_least_squares_cg() {
    let design = common::make_test_design();
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let params = SolverParams {
        krylov: KrylovMethod::Cg,
        operator: OperatorRepr::Implicit,
        tol: 1e-8,
        maxiter: 1000,
        ..Default::default()
    };
    let solver = Solver::from_design(design, None, &params, None).expect("build solver");
    let result = solver.solve(&y).expect("solve");
    assert!(result.converged(), "CG LS did not converge");
    common::assert_solution_finite(&result);
}

#[test]
fn test_least_squares_weighted_cg_preconditioned() {
    let design = common::make_design(vec![vec![0, 1, 0, 1, 2], vec![0, 0, 1, 1, 0]])
        .expect("valid weighted design");
    let weights = vec![1.0, 2.0, 1.5, 0.5, 3.0];
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let params = SolverParams {
        krylov: KrylovMethod::Cg,
        operator: OperatorRepr::Implicit,
        tol: 1e-8,
        maxiter: 1000,
        ..Default::default()
    };
    let precond =
        Preconditioner::Additive(LocalSolverConfig::solver_default(), ReductionStrategy::Auto);
    let solver =
        Solver::from_design(design, Some(weights), &params, Some(&precond)).expect("build solver");
    let result = solver.solve(&y).expect("solve");
    common::assert_converged_with_small_residual(&result, 1e-6);
    common::assert_solution_finite(&result);
}

// ===========================================================================
// Preconditioner tests (formerly orchestrate_preconditioners.rs)
// ===========================================================================

#[test]
fn test_schwarz_builder_schur_complement_modes_end_to_end() {
    use schwarz_precond::solve::cg::pcg;
    use schwarz_precond::solve::vec_norm;
    use schwarz_precond::Operator;
    use within::config::{ApproxCholConfig, ApproxSchurConfig};
    use within::operator::gramian::GramianOperator;
    use within::operator::preconditioner::build_preconditioner;

    let design = common::make_test_design();
    let rhs = common::make_rhs_from_unit_solution(&design);
    let gramian = GramianOperator::new(&design, None);

    let local_solvers = [
        LocalSolverConfig {
            approx_chol: ApproxCholConfig {
                seed: 11,
                ..Default::default()
            },
            approx_schur: None,
            dense_threshold: within::config::DEFAULT_DENSE_SCHUR_THRESHOLD,
        },
        LocalSolverConfig {
            approx_chol: ApproxCholConfig {
                seed: 13,
                ..Default::default()
            },
            approx_schur: Some(ApproxSchurConfig {
                seed: 42,
                ..Default::default()
            }),
            dense_threshold: within::config::DEFAULT_DENSE_SCHUR_THRESHOLD,
        },
    ];

    for local_solver in local_solvers {
        let schwarz = build_preconditioner(
            &design,
            None,
            None,
            &Preconditioner::Additive(local_solver.clone(), ReductionStrategy::default()),
        )
        .expect("build schwarz preconditioner");
        let result = pcg(&gramian, &rhs, &schwarz, 1e-8, 500).expect("cg solve");
        assert!(
            result.converged,
            "CG with Schur-complement local solver did not converge"
        );

        let mut residual = vec![0.0; design.n_dofs];
        gramian.apply(&result.x, &mut residual).expect("apply");
        for (ri, &bi) in residual.iter_mut().zip(rhs.iter()) {
            *ri -= bi;
        }
        let rel_resid = vec_norm(&residual) / vec_norm(&rhs).max(1e-15);
        assert!(
            rel_resid < 1e-6,
            "Schur-complement integration residual too large: {rel_resid:.2e}"
        );
    }
}

#[test]
#[ignore] // Run with: cargo test --release -- --ignored --nocapture
fn test_compare_factorization_strategies() {
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};
    use schwarz_precond::solve::cg::pcg;
    use schwarz_precond::solve::vec_norm;
    use schwarz_precond::Operator;
    use std::time::Instant;
    use within::config::ApproxCholConfig;
    use within::domain::Design;
    use within::operator::gramian::GramianOperator;
    use within::operator::preconditioner::build_preconditioner;
    use within::operator::DesignOperator;

    let configs: Vec<(ApproxCholConfig, &str)> = vec![
        (
            ApproxCholConfig {
                seed: 42,
                ..Default::default()
            },
            "AC      ",
        ),
        (
            ApproxCholConfig {
                seed: 42,
                split_merge: Some(2),
            },
            "AC2(2,2)",
        ),
    ];

    let problems: Vec<(&str, Vec<usize>, usize, u64)> = vec![
        ("2fe 500x500 50K", vec![500, 500], 50_000, 111),
        ("2fe 1Kx1K 100K", vec![1000, 1000], 100_000, 222),
        ("2fe 5Kx5K 500K", vec![5000, 5000], 500_000, 666),
        ("2fe 10Kx10K 1M", vec![10_000, 10_000], 1_000_000, 777),
        ("3fe 200^3 60K", vec![200, 200, 200], 60_000, 333),
        ("3fe 500^3 250K", vec![500, 500, 500], 250_000, 888),
        ("3fe 1K^3 500K", vec![1000, 1000, 1000], 500_000, 999),
        ("2fe 2Kx2K 10K", vec![2000, 2000], 10_000, 555),
        ("2fe 10Kx10K 50K", vec![10_000, 10_000], 50_000, 1111),
    ];

    for (prob_label, n_lev, n_rows, seed) in &problems {
        let mut rng = SmallRng::seed_from_u64(*seed);
        let cats: Vec<Vec<u32>> = n_lev
            .iter()
            .map(|&nl| {
                (0..(*n_rows))
                    .map(|_| rng.random_range(0..nl as u32))
                    .collect()
            })
            .collect();
        let store = within::FactorMajorStore::new(cats, *n_rows).expect("valid factor-major store");
        let design = Design::from_store(store).expect("valid synthetic design");

        let y: Vec<f64> = (0..*n_rows).map(|_| rng.random::<f64>()).collect();
        let design_op = DesignOperator::new(&design, None);
        let mut rhs = vec![0.0; design.n_dofs];
        design_op.apply_adjoint(&y, &mut rhs).expect("apply");
        let gramian_op = GramianOperator::new(&design, None);

        for (ac_config, label) in &configs {
            let local_solver = LocalSolverConfig {
                approx_chol: *ac_config,
                approx_schur: Some(within::config::ApproxSchurConfig::default()),
                dense_threshold: within::config::DEFAULT_DENSE_SCHUR_THRESHOLD,
            };
            let t0 = Instant::now();
            let schwarz = build_preconditioner(
                &design,
                None,
                None,
                &Preconditioner::Additive(local_solver.clone(), ReductionStrategy::default()),
            )
            .expect("build schwarz preconditioner");
            let _setup_ms = t0.elapsed().as_secs_f64() * 1000.0;

            let t1 = Instant::now();
            let cg_result = pcg(&gramian_op, &rhs, &schwarz, 1e-8, 1000).expect("cg solve");
            let _solve_ms = t1.elapsed().as_secs_f64() * 1000.0;

            let mut gx = vec![0.0; design.n_dofs];
            gramian_op.apply(&cg_result.x, &mut gx).expect("apply");
            for i in 0..design.n_dofs {
                gx[i] -= rhs[i];
            }
            let residual = vec_norm(&gx) / vec_norm(&rhs).max(1e-15);

            assert!(
                cg_result.converged,
                "{} did not converge on {}",
                label, prob_label
            );
            assert!(
                residual < 1e-6,
                "{} residual too large on {}: {:.2e}",
                label,
                prob_label,
                residual
            );
        }
    }
}

#[test]
fn test_gmres_multiplicative_implicit() {
    let design = common::make_test_design();
    let y = common::make_y_from_unit_solution(&design);

    let params = SolverParams {
        krylov: KrylovMethod::Gmres { restart: 30 },
        operator: OperatorRepr::Implicit,
        tol: 1e-8,
        maxiter: 1000,
        ..Default::default()
    };
    let precond = Preconditioner::Multiplicative(LocalSolverConfig::default());
    let solver = Solver::from_design(design, None, &params, Some(&precond)).expect("build solver");
    let result = solver.solve(&y).expect("solve");
    common::assert_converged_with_small_residual(&result, 1e-6);
    common::assert_solution_finite(&result);
}

#[test]
fn test_gmres_multiplicative_explicit() {
    let design = common::make_test_design();
    let y = common::make_y_from_unit_solution(&design);

    let params = SolverParams {
        krylov: KrylovMethod::Gmres { restart: 30 },
        operator: OperatorRepr::Explicit,
        tol: 1e-8,
        maxiter: 1000,
        ..Default::default()
    };
    let precond = Preconditioner::Multiplicative(LocalSolverConfig::default());
    let solver = Solver::from_design(design, None, &params, Some(&precond)).expect("build solver");
    let result = solver.solve(&y).expect("solve");
    common::assert_converged_with_small_residual(&result, 1e-6);
    common::assert_solution_finite(&result);
}

// ===========================================================================
// ===========================================================================
// MLSMR rectangular solver tests
// ===========================================================================
#[test]
fn test_lsmr_weighted() {
    let design = common::make_design(vec![vec![0, 1, 0, 1, 2], vec![0, 0, 1, 1, 0]])
        .expect("valid weighted design");
    let weights = vec![1.0, 2.0, 1.5, 0.5, 3.0];
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let params = SolverParams {
        krylov: KrylovMethod::Lsmr { local_size: None },
        operator: OperatorRepr::Implicit,
        tol: 1e-8,
        maxiter: 1000,
        ..Default::default()
    };
    let precond =
        Preconditioner::Additive(LocalSolverConfig::solver_default(), ReductionStrategy::Auto);
    let solver =
        Solver::from_design(design, Some(weights), &params, Some(&precond)).expect("build solver");
    let result = solver.solve(&y).expect("solve");
    common::assert_converged_with_small_residual(&result, 1e-6);
    common::assert_solution_finite(&result);
}

#[test]
fn test_lsmr_matches_cg_solution() {
    let design = common::make_test_design();
    let y = common::make_y_from_unit_solution(&design);

    let precond = Preconditioner::Additive(LocalSolverConfig::default(), ReductionStrategy::Auto);

    let cg_params = SolverParams {
        krylov: KrylovMethod::Cg,
        tol: 1e-10,
        maxiter: 1000,
        ..Default::default()
    };
    let cg_solver =
        Solver::from_design(common::make_test_design(), None, &cg_params, Some(&precond))
            .expect("build cg solver");
    let cg_result = cg_solver.solve(&y).expect("cg solve");

    let lsmr_params = SolverParams {
        krylov: KrylovMethod::Lsmr { local_size: None },
        tol: 1e-10,
        maxiter: 1000,
        ..Default::default()
    };
    let lsmr_solver =
        Solver::from_design(design, None, &lsmr_params, Some(&precond)).expect("build lsmr solver");
    let lsmr_result = lsmr_solver.solve(&y).expect("lsmr solve");

    // Solutions should match to reasonable precision
    let err: f64 = cg_result
        .x()
        .iter()
        .zip(lsmr_result.x().iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();
    assert!(err < 1e-6, "CG and LSMR solutions differ: {err:.2e}");
}
