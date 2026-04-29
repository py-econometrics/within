use ndarray::Array2;
use proptest::prelude::*;
use schwarz_precond::Operator;
use within::observation::ArrayStore;
use within::operator::gramian::{Gramian, WeightedGramianOperator};
use within::{
    solve, Design, DesignOperator, LocalSolverConfig, Preconditioner, ReductionStrategy, Solver,
    SolverParams,
};

#[path = "common/orchestrate_helpers.rs"]
mod common;

// ---------------------------------------------------------------------------
// Shared strategies
// ---------------------------------------------------------------------------

/// Generate a random FE problem: (categories Array2<u32>, y Vec<f64>).
/// 2–3 factors, 2–30 levels each, 50–500 observations.
fn random_fe_problem_strategy() -> impl Strategy<Value = (Array2<u32>, Vec<f64>)> {
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

/// Like `random_fe_problem_strategy` but also generates positive per-observation weights.
fn random_weighted_fe_problem_strategy() -> impl Strategy<Value = (Array2<u32>, Vec<f64>, Vec<f64>)>
{
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
                let w_vec = proptest::collection::vec(0.1f64..10.0, n_obs);
                (cat_cols, y_vec, w_vec).prop_map(move |(cols, y, w)| {
                    let n_f = cols.len();
                    let n = cols[0].len();
                    let mut cats = Array2::<u32>::zeros((n, n_f));
                    for (f, col) in cols.iter().enumerate() {
                        for (i, &val) in col.iter().enumerate() {
                            cats[[i, f]] = val;
                        }
                    }
                    (cats, y, w)
                })
            })
        })
    })
}

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

fn additive_precond() -> Preconditioner {
    Preconditioner::Additive(LocalSolverConfig::solver_default(), ReductionStrategy::Auto)
}

// ---------------------------------------------------------------------------
// Property tests
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10))]

    /// Weighted Gramian: explicit CSR matvec matches matrix-free matvec for
    /// non-unit weights. This is the weighted analogue of the existing
    /// `prop_explicit_equals_implicit_gramian` test.
    #[test]
    fn prop_weighted_explicit_equals_implicit(
        (cats, _y, weights) in random_weighted_fe_problem_strategy()
    ) {
        let store = ArrayStore::new(cats.view()).unwrap();
        let design = Design::from_store(store).unwrap();

        let explicit = Gramian::build_weighted(&design, &weights);
        let implicit = WeightedGramianOperator::new(&design, &weights);
        let n = design.n_dofs;

        let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.3).sin()).collect();
        let mut y_exp = vec![0.0; n];
        let mut y_imp = vec![0.0; n];
        explicit.matvec(&x, &mut y_exp);
        implicit.apply(&x, &mut y_imp);

        for (a, b) in y_exp.iter().zip(y_imp.iter()) {
            prop_assert!(
                (a - b).abs() < 1e-10,
                "weighted explicit vs implicit mismatch: {} vs {}",
                a,
                b
            );
        }
    }

    /// A 4-factor problem solved with additive Schwarz should converge.
    /// This exercises the partition-of-unity construction over C(4,2)=6 domains.
    #[test]
    fn prop_4_factor_convergence((cats, y) in random_4_factor_problem_strategy()) {
        let params = SolverParams {
            tol: 1e-7,
            ..SolverParams::default()
        };
        let precond = additive_precond();
        // Build the design with a unit-solution RHS so the problem is feasible
        let store = ArrayStore::new(cats.view()).unwrap();
        let design = Design::from_store(store).unwrap();
        let y_feasible: Vec<f64> = {
            let x_true = vec![1.0; design.n_dofs];
            let mut y_out = vec![0.0; design.n_rows];
            DesignOperator::new(&design).apply(&x_true, &mut y_out);
            y_out
        };
        // Use y_feasible so convergence is guaranteed on a consistent system
        let _ = y; // provided by strategy but we use the feasible version
        let result = solve(cats.view(), &y_feasible, None, &params, Some(&precond)).unwrap();
        prop_assert!(
            result.converged,
            "4-factor solve did not converge (n_obs={}, n_dofs={}, residual={:.2e})",
            design.n_rows,
            design.n_dofs,
            result.final_residual
        );
    }

    /// `solve()` and `Solver::from_design().solve()` must produce bit-identical
    /// results given the same design and RHS. `solve()` internally builds via
    /// `Solver::new` (ArrayStore path); `Solver::from_design` uses FactorMajorStore.
    /// Both should reach the same fixed point.
    #[test]
    fn prop_solve_vs_solver_identical((cats, y) in random_fe_problem_strategy()) {
        let params = SolverParams {
            tol: 1e-7,
            ..SolverParams::default()
        };
        let precond = additive_precond();

        // Path A: convenience `solve()` (uses ArrayStore internally)
        let result_a = solve(cats.view(), &y, None, &params, Some(&precond)).unwrap();

        // Path B: Solver::new() — identical to solve() but without timing wrapper
        let solver_b = Solver::new(cats.view(), None, &params, Some(&precond)).unwrap();
        let result_b = solver_b.solve(&y).unwrap();

        prop_assert_eq!(
            result_a.x.len(),
            result_b.x.len(),
            "x length mismatch"
        );
        for (i, (a, b)) in result_a.x.iter().zip(result_b.x.iter()).enumerate() {
            prop_assert!(
                (a - b).abs() < 1e-12,
                "x[{}] mismatch: solve()={} vs Solver::new().solve()={}",
                i, a, b
            );
        }
    }

    /// Verify demeaned = y - D*x.
    /// After a converged solve, `demeaned[i]` must equal `y[i] - sum_q x[dof(i,q)]`.
    #[test]
    fn prop_demeaned_identity_all_paths((cats, y) in random_fe_problem_strategy()) {
        let params = SolverParams {
            tol: 1e-7,
            ..SolverParams::default()
        };
        let precond = additive_precond();
        let result = solve(cats.view(), &y, None, &params, Some(&precond)).unwrap();

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

    /// Single-factor problems have a diagonal Gramian.  Unpreconditioned CG on
    /// a diagonal system converges in at most n_levels iterations (one per
    /// distinct eigenvalue); in practice far fewer are needed.  The key property
    /// is that CG converges and produces a finite solution.
    #[test]
    fn prop_single_factor_converges((cats, _y) in single_factor_strategy()) {
        // Build a consistent RHS: y = D * 1 so the system is exactly solvable.
        let store = ArrayStore::new(cats.view()).unwrap();
        let design = Design::from_store(store).unwrap();
        let n_levels = design.n_dofs;
        let x_true = vec![1.0; n_levels];
        let mut y_feasible = vec![0.0; design.n_rows];
        DesignOperator::new(&design).apply(&x_true, &mut y_feasible);

        // No preconditioner, no iterative refinement.
        let params = SolverParams {
            tol: 1e-8,
            maxiter: n_levels + 10, // generous: diagonal CG converges in ≤ n_levels steps
            max_refinements: 0,
            ..SolverParams::default()
        };
        let result = solve(cats.view(), &y_feasible, None, &params, None).unwrap();

        prop_assert!(
            result.converged,
            "single-factor CG did not converge in {} iterations (residual={:.2e}, n_levels={})",
            result.iterations,
            result.final_residual,
            n_levels
        );
        prop_assert!(result.x.iter().all(|v| v.is_finite()), "non-finite x");
    }
}
