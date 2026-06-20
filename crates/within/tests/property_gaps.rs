use ndarray::Array2;
use proptest::prelude::*;
use within::{solve, LsmrOptions, Solver};

#[path = "common/property_strategies.rs"]
mod strategies;
use strategies::{additive_precond, random_fe_problem_strategy};

/// 4-factor problem: 2–10 levels each, 100–500 observations.
fn random_4_factor_problem_strategy() -> impl Strategy<Value = (Array2<u32>, Vec<f64>)> {
    proptest::collection::vec(2..=10u32, 4usize).prop_flat_map(|n_levels| {
        let n_obs_range = 100..=500usize;
        n_obs_range.prop_flat_map(move |n_obs| {
            let n_levels_clone = n_levels.clone();
            let cat_cols: Vec<_> = n_levels_clone
                .iter()
                .map(|&nl| proptest::collection::vec(0..nl, n_obs))
                .collect();
            let y_vec = proptest::collection::vec(-10.0f64..10.0, n_obs);
            (cat_cols, y_vec).prop_map(move |(cols, y)| {
                let n = cols[0].len();
                let mut cats = Array2::<u32>::zeros((n, 4));
                for (f, col) in cols.iter().enumerate() {
                    for (i, &val) in col.iter().enumerate() {
                        cats[[i, f]] = val;
                    }
                }
                (cats, y)
            })
        })
    })
}

/// Single-factor problem: 2–50 levels, 50–300 observations.
fn single_factor_strategy() -> impl Strategy<Value = (Array2<u32>, Vec<f64>)> {
    (2..=50u32).prop_flat_map(|n_levels| {
        (50..=300usize).prop_flat_map(move |n_obs| {
            let cat = proptest::collection::vec(0..n_levels, n_obs);
            let y_vec = proptest::collection::vec(-10.0f64..10.0, n_obs);
            (cat, y_vec).prop_map(move |(col, y)| {
                let n = col.len();
                let mut cats = Array2::<u32>::zeros((n, 1));
                for (i, &val) in col.iter().enumerate() {
                    cats[[i, 0]] = val;
                }
                (cats, y)
            })
        })
    })
}

// ---------------------------------------------------------------------------
// Property tests
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10))]

    /// A 4-factor problem solved with additive Schwarz should converge.
    /// This exercises the partition-of-unity construction over C(4,2)=6 domains.
    #[test]
    fn prop_4_factor_convergence((cats, y) in random_4_factor_problem_strategy()) {
        let params = LsmrOptions {
            tol: 1e-7,
            ..LsmrOptions::default()
        };
        let precond = additive_precond();
        // LSMR converges on the least-squares system min ||y - Dx||^2 for any y.
        let result = solve(cats.view(), &y, None, &params, &precond).unwrap();
        prop_assert!(
            result.converged,
            "4-factor solve did not converge (n_obs={}, residual={:.2e})",
            y.len(),
            result.residual
        );
    }

    /// `solve()` and `Solver::new().solve(, &params)` must produce bit-identical
    /// results given the same design and RHS. Both use the ArrayStore path
    /// (raw category view); the wrappers differ only in timing accounting.
    /// Both should reach the same fixed point.
    #[test]
    fn prop_solve_vs_solver_identical((cats, y) in random_fe_problem_strategy()) {
        let params = LsmrOptions {
            tol: 1e-7,
            ..LsmrOptions::default()
        };
        let precond = additive_precond();

        // Path A: convenience `solve()` (uses ArrayStore internally)
        let result_a = solve(cats.view(), &y, None, &params, &precond).unwrap();

        // Path B: Solver::new() — identical to solve() but without timing wrapper
        let solver_b = Solver::new(cats.view(), None, &precond).unwrap();
        let result_b = solver_b.solve(&y, &params).unwrap();

        prop_assert_eq!(
            result_a.x.len(),
            result_b.x.len(),
            "x length mismatch"
        );
        for (i, (a, b)) in result_a.x.iter().zip(result_b.x.iter()).enumerate() {
            prop_assert!(
                (a - b).abs() < 1e-12,
                "x[{}] mismatch: solve()={} vs Solver::new().solve(, &params)={}",
                i, a, b
            );
        }
    }

    /// Verify demeaned = y - D*x.
    /// After a converged solve, `demeaned[i]` must equal `y[i] - sum_q x[dof(i,q)]`.
    #[test]
    fn prop_demeaned_identity_all_paths((cats, y) in random_fe_problem_strategy()) {
        let params = LsmrOptions {
            tol: 1e-7,
            ..LsmrOptions::default()
        };
        let precond = additive_precond();
        let result = solve(cats.view(), &y, None, &params, &precond).unwrap();

        if !result.converged {
            return Ok(());
        }

        // Manually reconstruct D*x: for each observation, sum the DOF values
        // for each factor's level.
        let n_obs = y.len();
        let n_factors = cats.ncols();

        // Compute factor offsets (same ordering as Design)
        let mut offsets = vec![0usize; n_factors];
        for f in 1..n_factors {
            let n_levels_prev = *cats.column(f - 1).iter().max().unwrap() as usize + 1;
            offsets[f] = offsets[f - 1] + n_levels_prev;
        }

        for i in 0..n_obs {
            let dx_i: f64 = (0..n_factors)
                .map(|f| {
                    let level = cats[[i, f]] as usize;
                    result.x[offsets[f] + level]
                })
                .sum();
            let expected_demeaned = y[i] - dx_i;
            prop_assert!(
                (result.demeaned[i] - expected_demeaned).abs() < 1e-8,
                "demeaned[{}]: got {}, expected {} (y={}, Dx={})",
                i, result.demeaned[i], expected_demeaned, y[i], dx_i
            );
        }
    }

    /// Single-factor problems have a diagonal Gramian. Unpreconditioned LSMR
    /// on a diagonal system converges in at most n_levels iterations (one per
    /// distinct singular value); in practice far fewer are needed.
    #[test]
    fn prop_single_factor_converges((cats, y) in single_factor_strategy()) {
        let n_levels = *cats.column(0).iter().max().unwrap() as usize + 1;
        let params = LsmrOptions {
            tol: 1e-8,
            maxiter: n_levels + 10,
            ..LsmrOptions::default()
        };
        let result = solve(cats.view(), &y, None, &params, None).unwrap();

        prop_assert!(
            result.converged,
            "single-factor LSMR did not converge in {} iterations (residual={:.2e}, n_levels={})",
            result.iterations,
            result.residual,
            n_levels
        );
        prop_assert!(result.x.iter().all(|v| v.is_finite()), "non-finite x");
    }
}
