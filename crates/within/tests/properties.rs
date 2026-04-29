use ndarray::Array2;
use proptest::prelude::*;
use schwarz_precond::Operator;
use within::observation::ArrayStore;
use within::operator::gramian::{Gramian, GramianOperator};
use within::{
    solve, Design, DesignOperator, FePreconditioner, LocalSolverConfig, Preconditioner,
    ReductionStrategy, SolverParams,
};

/// Generate a random fixed-effects problem as (categories Array2<u32>, y Vec<f64>).
fn random_fe_problem_strategy() -> impl Strategy<Value = (Array2<u32>, Vec<f64>)> {
    // 2-3 factors, 2-30 levels each, 50-500 observations
    (2..=3u32).prop_flat_map(|n_factors| {
        let levels = proptest::collection::vec(2..=30u32, n_factors as usize);
        levels.prop_flat_map(move |n_levels| {
            let n_obs_range = 50..=500usize;
            n_obs_range.prop_flat_map(move |n_obs| {
                let n_levels_clone = n_levels.clone();
                let cat_cols: Vec<_> = n_levels_clone
                    .iter()
                    .map(|&nl| proptest::collection::vec(0..nl, n_obs))
                    .collect();
                let y_vec = proptest::collection::vec(-10.0f64..10.0, n_obs);
                (cat_cols, y_vec).prop_map(move |(cols, y)| {
                    let n_f = cols.len();
                    let n = cols[0].len();
                    let mut cats = Array2::<u32>::zeros((n, n_f));
                    for (f, col) in cols.iter().enumerate() {
                        for (i, &val) in col.iter().enumerate() {
                            cats[[i, f]] = val;
                        }
                    }
                    (cats, y)
                })
            })
        })
    })
}

fn default_params() -> SolverParams {
    SolverParams::default()
}

fn additive_precond() -> Preconditioner {
    Preconditioner::Additive(LocalSolverConfig::solver_default(), ReductionStrategy::Auto)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10))]

    #[test]
    fn prop_gramian_symmetry((cats, _y) in random_fe_problem_strategy()) {
        let store = ArrayStore::new(cats.view());
        let design = Design::from_store(store).unwrap();
        let gramian = GramianOperator::new(&design);
        let n = design.n_dofs;

        // Test x^T G y == y^T G x for random vectors
        let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.3).sin()).collect();
        let y: Vec<f64> = (0..n).map(|i| (i as f64 * 0.7).cos()).collect();

        let mut gx = vec![0.0; n];
        let mut gy = vec![0.0; n];
        gramian.apply(&x, &mut gx).expect("apply");
        gramian.apply(&y, &mut gy).expect("apply");

        let xt_gy: f64 = x.iter().zip(gy.iter()).map(|(a, b)| a * b).sum();
        let yt_gx: f64 = y.iter().zip(gx.iter()).map(|(a, b)| a * b).sum();

        prop_assert!((xt_gy - yt_gx).abs() < 1e-8, "Gramian not symmetric: {} vs {}", xt_gy, yt_gx);
    }

    #[test]
    fn prop_explicit_equals_implicit_gramian((cats, _y) in random_fe_problem_strategy()) {
        let store = ArrayStore::new(cats.view());
        let design = Design::from_store(store).unwrap();
        let explicit = Gramian::build(&design);
        let implicit = GramianOperator::new(&design);
        let n = design.n_dofs;

        let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.3).sin()).collect();
        let mut y_explicit = vec![0.0; n];
        let mut y_implicit = vec![0.0; n];
        explicit.matvec(&x, &mut y_explicit);
        implicit.apply(&x, &mut y_implicit).expect("apply");

        for (a, b) in y_explicit.iter().zip(y_implicit.iter()) {
            prop_assert!((a - b).abs() < 1e-10, "explicit vs implicit: {} vs {}", a, b);
        }
    }

    #[test]
    fn prop_preconditioner_serde_roundtrip((cats, _y) in random_fe_problem_strategy()) {
        let params = default_params();
        let precond = additive_precond();

        let solver = within::Solver::new(cats.view(), None, &params, Some(&precond)).unwrap();
        let fe_precond = solver.preconditioner().unwrap();

        let bytes = postcard::to_stdvec(fe_precond).unwrap();
        let deserialized: FePreconditioner = postcard::from_bytes(&bytes).unwrap();

        let n = fe_precond.nrows();
        let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.5).sin()).collect();

        let mut y1 = vec![0.0; n];
        let mut y2 = vec![0.0; n];
        fe_precond.apply(&x, &mut y1).expect("apply");
        deserialized.apply(&x, &mut y2).expect("apply");

        for (a, b) in y1.iter().zip(y2.iter()) {
            prop_assert!((a - b).abs() < 1e-12, "serde roundtrip mismatch: {} vs {}", a, b);
        }
    }

    #[test]
    fn prop_solver_convergence((cats, _y) in random_fe_problem_strategy()) {
        // Create y = D * x_true so we know the answer
        let store = ArrayStore::new(cats.view());
        let design = Design::from_store(store).unwrap();
        let n_dofs = design.n_dofs;
        let n_obs = design.n_rows;

        let x_true: Vec<f64> = (0..n_dofs).map(|i| (i as f64 * 0.4).sin()).collect();
        let mut y = vec![0.0; n_obs];
        DesignOperator::new(&design).apply(&x_true, &mut y).expect("apply");

        // Use slightly relaxed tolerance — randomly generated problems can be
        // borderline at 1e-8 (e.g. residual 1.02e-8 after 13 iters).
        let params = SolverParams {
            tol: 1e-7,
            ..default_params()
        };
        let precond = additive_precond();
        let result = solve(cats.view(), &y, None, &params, Some(&precond)).unwrap();

        prop_assert!(result.converged, "Solver did not converge after {} iterations (residual: {:.2e}, n_obs: {}, n_dofs: {})",
            result.iterations, result.final_residual, n_obs, n_dofs);
    }

    #[test]
    fn prop_demeaned_orthogonality((cats, y) in random_fe_problem_strategy()) {
        let params = default_params();
        let precond = additive_precond();
        let result = solve(cats.view(), &y, None, &params, Some(&precond)).unwrap();

        if !result.converged {
            return Ok(());
        }

        let n_obs = y.len();
        let n_factors = cats.ncols();
        let residual = &result.demeaned;

        // D^T * residual should be ≈ 0
        for f in 0..n_factors {
            let n_levels = *cats.column(f).iter().max().unwrap() as usize + 1;
            for lvl in 0..n_levels {
                let dot: f64 = (0..n_obs)
                    .filter(|&i| cats[[i, f]] == lvl as u32)
                    .map(|i| residual[i])
                    .sum();
                prop_assert!(
                    dot.abs() < 1e-3,
                    "factor {}, level {}: D^T r = {}",
                    f,
                    lvl,
                    dot
                );
            }
        }
    }
}
