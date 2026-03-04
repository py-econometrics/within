use within::{
    solve_normal_equations, CgPreconditioner, GmresPreconditioner, LocalSolverConfig,
    SchwarzConfig, SolverMethod, SolverParams,
};

#[path = "common/orchestrate_helpers.rs"]
mod common;

#[test]
fn test_schwarz_builder_schur_complement_modes_end_to_end() {
    use approx_chol::Config;
    use schwarz_precond::solve::cg::cg_solve_preconditioned;
    use schwarz_precond::solve::lsmr::vec_norm;
    use schwarz_precond::Operator;
    use within::operator::gramian::GramianOperator;
    use within::operator::schwarz::build_schwarz_default;
    use within::ApproxSchurConfig;

    let design = common::make_test_design();
    let rhs = common::make_rhs_from_unit_solution(&design);
    let gramian = GramianOperator::new(&design);
    let ac = Config {
        seed: 7,
        ..Default::default()
    };

    let local_solvers = [
        LocalSolverConfig::SchurComplement {
            approx_chol: Config {
                seed: 11,
                ..Default::default()
            },
            approx_schur: None,
            dense_threshold: within::DEFAULT_DENSE_SCHUR_THRESHOLD,
        },
        LocalSolverConfig::SchurComplement {
            approx_chol: Config {
                seed: 13,
                ..Default::default()
            },
            approx_schur: Some(ApproxSchurConfig { seed: 42 }),
            dense_threshold: within::DEFAULT_DENSE_SCHUR_THRESHOLD,
        },
    ];

    for local_solver in local_solvers {
        let schwarz = build_schwarz_default(&design, &ac, &local_solver)
            .expect("build schwarz preconditioner");
        let result =
            cg_solve_preconditioned(&gramian, &schwarz, &rhs, 1e-8, 500).expect("cg solve");
        assert!(
            result.converged,
            "CG with Schur-complement local solver did not converge"
        );

        let mut residual = vec![0.0; design.n_dofs];
        gramian.apply(&result.x, &mut residual);
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
    use approx_chol::Config;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};
    use schwarz_precond::solve::cg::cg_solve_preconditioned;
    use schwarz_precond::solve::lsmr::vec_norm;
    use schwarz_precond::Operator;
    use std::time::Instant;
    use within::operator::design::DesignOperator;
    use within::operator::gramian::GramianOperator;
    use within::operator::schwarz::build_schwarz_default;
    use within::FixedEffectsDesign;

    // Compare AC (standard) vs AC2 (multi-edge split).
    // Ordering control is available via `low_level::Builder::ordering()`.
    let configs: Vec<(Config, &str)> = vec![
        (
            Config {
                seed: 42,
                ..Default::default()
            },
            "AC      ",
        ),
        (
            Config {
                seed: 42,
                split_merge: Some(2),
                ..Default::default()
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
        let cats: Vec<Vec<i64>> = n_lev
            .iter()
            .map(|&nl| {
                (0..(*n_rows))
                    .map(|_| rng.random_range(0..nl as i64))
                    .collect()
            })
            .collect();
        let design =
            FixedEffectsDesign::new(cats, n_lev.clone(), *n_rows).expect("valid synthetic design");

        let y: Vec<f64> = (0..*n_rows).map(|_| rng.random::<f64>()).collect();
        let design_op = DesignOperator::new(&design);
        let mut rhs = vec![0.0; design.n_dofs];
        design_op.apply_adjoint(&y, &mut rhs);
        let gramian_op = GramianOperator::new(&design);

        for (config, label) in &configs {
            let t0 = Instant::now();
            let schwarz = build_schwarz_default(&design, config, &LocalSolverConfig::default())
                .expect("build schwarz preconditioner");
            let _setup_ms = t0.elapsed().as_secs_f64() * 1000.0;

            let t1 = Instant::now();
            let cg_result =
                cg_solve_preconditioned(&gramian_op, &schwarz, &rhs, 1e-8, 1000).expect("cg solve");
            let _solve_ms = t1.elapsed().as_secs_f64() * 1000.0;

            let mut gx = vec![0.0; design.n_dofs];
            gramian_op.apply(&cg_result.x, &mut gx);
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
fn test_normal_equations_cg_multiplicative_one_level() {
    let design = common::make_test_design();
    let rhs = common::make_rhs_from_unit_solution(&design);

    let params = SolverParams {
        method: SolverMethod::Cg {
            preconditioner: CgPreconditioner::MultiplicativeOneLevel(SchwarzConfig::default()),
        },
        tol: 1e-8,
        maxiter: 1000,
    };
    let result = solve_normal_equations(&design, &rhs, None, &params)
        .expect("cg multiplicative normal equations");
    common::assert_converged_with_small_residual(&result, 1e-6);
    common::assert_solution_finite(&result);
}

#[test]
fn test_normal_equations_gmres_multiplicative_one_level() {
    let design = common::make_test_design();
    let rhs = common::make_rhs_from_unit_solution(&design);

    let params = SolverParams {
        method: SolverMethod::Gmres {
            preconditioner: GmresPreconditioner::MultiplicativeOneLevel(SchwarzConfig::default()),
            restart: 30,
        },
        tol: 1e-8,
        maxiter: 1000,
    };
    let result = solve_normal_equations(&design, &rhs, None, &params)
        .expect("gmres multiplicative normal equations");
    common::assert_converged_with_small_residual(&result, 1e-6);
    common::assert_solution_finite(&result);
}
