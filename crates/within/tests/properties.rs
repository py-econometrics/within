use proptest::prelude::*;
use within::{solve, LsmrOptions, Preconditioner};

#[path = "common/property_strategies.rs"]
mod strategies;
use strategies::{additive_precond, random_fe_problem_strategy};

fn default_params() -> LsmrOptions {
    LsmrOptions::default()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10))]

#[test]
    fn prop_preconditioner_serde_roundtrip((cats, _y) in random_fe_problem_strategy()) {
        let precond = additive_precond();

        let solver = within::Solver::new(cats.view(), None, &precond).unwrap();
        let fe_precond = solver.preconditioner().unwrap();

        let bytes = postcard::to_stdvec(fe_precond).unwrap();
        let deserialized: Preconditioner = postcard::from_bytes(&bytes).unwrap();

        let n = fe_precond.nrows();
        let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.5).sin()).collect();

        let mut y1 = vec![0.0; n];
        let mut y2 = vec![0.0; n];
        fe_precond.apply(&x, &mut y1).expect("apply succeeds");
        deserialized.apply(&x, &mut y2).expect("deserialized apply succeeds");

        for (a, b) in y1.iter().zip(y2.iter()) {
            prop_assert!((a - b).abs() < 1e-12, "serde roundtrip mismatch: {} vs {}", a, b);
        }
    }

    #[test]
    fn prop_solver_convergence((cats, y) in random_fe_problem_strategy()) {
        // LSMR converges on the least-squares system min ||y - Dx||^2 for any
        // y, so we use the random y directly from the strategy.
        let params = LsmrOptions {
            tol: 1e-7,
            ..default_params()
        };
        let precond = additive_precond();
        let result = solve(cats.view(), &y, None, &params, &precond).unwrap();

        prop_assert!(
            result.converged,
            "Solver did not converge after {} iterations (residual: {:.2e}, n_obs: {})",
            result.iterations, result.residual, y.len(),
        );
    }

    #[test]
    fn prop_demeaned_orthogonality((cats, y) in random_fe_problem_strategy()) {
        let params = default_params();
        let precond = additive_precond();
        let result = solve(cats.view(), &y, None, &params, &precond).unwrap();

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
