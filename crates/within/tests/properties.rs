use proptest::prelude::*;
use within::{
    solve, Channel, CoefficientAddress, Effect, LsmrOptions, Preconditioner, PreconditionerConfig,
    Solver,
};

#[path = "common/property_strategies.rs"]
mod strategies;
use strategies::{additive_precond, random_fe_problem_strategy, random_slopes_problem_strategy};

fn at(term: usize, level: usize, column: usize) -> CoefficientAddress {
    CoefficientAddress {
        channel: Channel { term, column },
        level,
    }
}

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
        // LSMR converges on min ||y - Dx||^2 for any y, so the random y is used directly.
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

        prop_assume!(result.converged);

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

    /// First-order optimality (the definition of a least-squares solution):
    /// the returned coefficients solve `min ||√W (y − D x)||²` iff the residual
    /// is design-orthogonal in the W-metric, `Dᵀ W (y − D x) = 0`. This is a
    /// self-contained truth oracle — it verifies the coefficients against the
    /// problem's own optimality condition, not against another solver. Holds
    /// for rank-deficient designs too: unidentified directions lie in the
    /// column span, so the residual is orthogonal to them as well.
    #[test]
    fn prop_slopes_normal_equations(problem in random_slopes_problem_strategy()) {
        let factors = &problem.factors;
        let weights = &problem.weights;
        let y = &problem.y;
        let n_obs = y.len();

        let effects: Vec<Effect> = factors
            .iter()
            .map(|f| {
                Effect::new(&f.levels, f.intercept, f.slopes.iter().map(Vec::as_slice))
                    .expect("valid effect")
            })
            .collect();

        // Non-converged draws are rejected, so proptest caps how many may be produced.
        let params = LsmrOptions {
            tol: 1e-10,
            maxiter: 2000,
            local_size: Some(10),
        };
        let result = Solver::new(effects, Some(weights.clone()), PreconditionerConfig::default())
            .expect("build solver")
            .solve(y.as_slice(), &params)
            .expect("solve");
        prop_assume!(result.converged);

        // Reconstruct r = y − D x from the coefficients ALONE, sharing no code with within.
        let x = &result.x;
        let layout = &result.layout;
        let mut fitted = vec![0.0f64; n_obs];
        for (t, f) in factors.iter().enumerate() {
            let slope_base = usize::from(f.intercept);
            for i in 0..n_obs {
                let lvl = f.levels[i] as usize;
                if f.intercept {
                    fitted[i] += x[layout.index(at(t, lvl, 0)).unwrap()];
                }
                for (s, col) in f.slopes.iter().enumerate() {
                    fitted[i] += x[layout.index(at(t, lvl, slope_base + s)).unwrap()] * col[i];
                }
            }
        }

        // g = Dᵀ W (y − D x) must vanish relative to g0 = Dᵀ W y.
        let mut g = vec![0.0f64; layout.n_dofs()];
        let mut g0 = vec![0.0f64; layout.n_dofs()];
        for (t, f) in factors.iter().enumerate() {
            let slope_base = usize::from(f.intercept);
            for i in 0..n_obs {
                let lvl = f.levels[i] as usize;
                let wr = weights[i] * (y[i] - fitted[i]);
                let wy = weights[i] * y[i];
                if f.intercept {
                    let k = layout.index(at(t, lvl, 0)).unwrap();
                    g[k] += wr;
                    g0[k] += wy;
                }
                for (s, col) in f.slopes.iter().enumerate() {
                    let k = layout.index(at(t, lvl, slope_base + s)).unwrap();
                    g[k] += wr * col[i];
                    g0[k] += wy * col[i];
                }
            }
        }

        let norm = |v: &[f64]| v.iter().map(|a| a * a).sum::<f64>().sqrt();
        let rel = norm(&g) / norm(&g0).max(1e-12);
        prop_assert!(
            rel <= 1e-6,
            "first-order optimality violated: ||DᵀW(y−Dx)|| / ||DᵀWy|| = {rel:.3e} \
             (n_obs={n_obs}, n_terms={}, iters={})",
            factors.len(),
            result.iterations,
        );
    }
}
