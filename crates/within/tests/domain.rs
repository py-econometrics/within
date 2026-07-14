//! Integration tests for the domain layer: solver convergence through the
//! public `solve` API for designs that exercise partition-of-unity weights
//! and disconnected bipartite structure.

use within::observation::ObservationFrame;
use within::Design;

// Three-factor design: shared DOFs across factor pairs force NonUniform
// partition weights; verified via the public solve API.

#[test]
fn test_three_factor_design_solve_converges() {
    use within::{solve, LsmrOptions, PreconditionerConfig};

    let n_obs = 60;
    let n_lev = 5usize;
    let fa: Vec<u32> = (0..n_obs).map(|i| (i % n_lev) as u32).collect();
    let fb: Vec<u32> = (0..n_obs).map(|i| ((i / n_lev) % n_lev) as u32).collect();
    let fc: Vec<u32> = (0..n_obs).map(|i| ((i * 3) % n_lev) as u32).collect();

    let frame = ObservationFrame::new(vec![fa.into(), fb.into(), fc.into()], Vec::new())
        .expect("valid 3-factor frame");
    let dm = Design::from_frame(frame).expect("valid 3-factor design");

    assert_eq!(dm.n_factors(), 3);

    let y: Vec<f64> = (0..n_obs).map(|i| (i as f64 * 0.31).sin()).collect();

    let n_factors = 3;
    let mut cats = ndarray::Array2::<u32>::zeros((n_obs, n_factors));
    for i in 0..n_obs {
        cats[[i, 0]] = (i % n_lev) as u32;
        cats[[i, 1]] = ((i / n_lev) % n_lev) as u32;
        cats[[i, 2]] = ((i * 3) % n_lev) as u32;
    }

    let params = LsmrOptions {
        tol: 1e-8,
        maxiter: 500,
        ..LsmrOptions::default()
    };
    let precond = PreconditionerConfig::default();
    let result = solve(cats.view(), &y, None, &params, &precond).expect("solve should not error");

    assert!(
        result.converged,
        "3-factor solver did not converge (residual: {:.2e})",
        result.residual
    );
}

// ---------------------------------------------------------------------------
// 2. Disconnected bipartite graph → multiple subdomains per factor pair
// ---------------------------------------------------------------------------

/// Verify that a disconnected bipartite design converges under additive Schwarz.
/// The disconnected structure means `build_local_domains` splits pair (0,1) into
/// 2 subdomains — correctness is validated indirectly through convergence.
#[test]
fn test_disconnected_design_larger_converges() {
    use within::{solve, LsmrOptions, PreconditionerConfig};

    let fa = vec![0u32, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3];
    let fb = vec![0u32, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5];
    let n_obs = fa.len();

    let mut cats = ndarray::Array2::<u32>::zeros((n_obs, 2));
    for i in 0..n_obs {
        cats[[i, 0]] = fa[i];
        cats[[i, 1]] = fb[i];
    }

    let y: Vec<f64> = (0..n_obs).map(|i| (i as f64 * 0.41).cos()).collect();

    let params = LsmrOptions {
        tol: 1e-8,
        maxiter: 500,
        ..LsmrOptions::default()
    };
    let precond = PreconditionerConfig::default();
    let result = solve(cats.view(), &y, None, &params, &precond).expect("solve should not error");

    assert!(
        result.converged,
        "disconnected larger design did not converge (residual: {:.2e})",
        result.residual
    );
}

#[test]
fn test_disconnected_design_solve_converges() {
    use within::{solve, LsmrOptions, PreconditionerConfig};

    let n_obs = 4;
    let mut cats = ndarray::Array2::<u32>::zeros((n_obs, 2));
    cats[[0, 0]] = 0;
    cats[[1, 0]] = 0;
    cats[[2, 0]] = 1;
    cats[[3, 0]] = 1;
    cats[[0, 1]] = 0;
    cats[[1, 1]] = 1;
    cats[[2, 1]] = 2;
    cats[[3, 1]] = 3;

    let y = vec![1.0, 2.0, 3.0, 4.0];

    let params = LsmrOptions {
        tol: 1e-8,
        maxiter: 500,
        ..LsmrOptions::default()
    };
    let precond = PreconditionerConfig::default();
    let result = solve(cats.view(), &y, None, &params, &precond).expect("solve should not error");

    assert!(
        result.converged,
        "disconnected design solver did not converge (residual: {:.2e})",
        result.residual
    );
}

// ---------------------------------------------------------------------------
// 3. Single-factor design
// ---------------------------------------------------------------------------

#[test]
fn test_single_factor_design_construction() {
    let frame = ObservationFrame::new(vec![vec![0u32, 1, 2, 0, 1].into()], Vec::new())
        .expect("valid frame");
    let dm = Design::from_frame(frame).expect("valid single-factor design");

    assert_eq!(dm.n_factors(), 1, "expected 1 factor");
    assert_eq!(dm.n_dofs(), 3, "expected 3 DOFs (levels 0,1,2)");
    assert_eq!(dm.n_obs(), 5, "expected 5 rows");
}

/// A single-factor design has no factor pairs, so the additive Schwarz
/// preconditioner has no subdomains to work with. The solver should still
/// function (falling back to unpreconditioned LSMR) or be able to solve the
/// trivial normal equations directly.
#[test]
fn test_single_factor_design_solve_without_precond() {
    use within::{solve, LsmrOptions};

    let n_obs = 5usize;
    let mut cats = ndarray::Array2::<u32>::zeros((n_obs, 1));
    let levels = [0u32, 1, 2, 0, 1];
    for i in 0..n_obs {
        cats[[i, 0]] = levels[i];
    }
    let y = vec![10.0, 20.0, 30.0, 10.0, 20.0];

    let params = LsmrOptions {
        tol: 1e-8,
        maxiter: 500,
        ..LsmrOptions::default()
    };
    let result = solve(cats.view(), &y, None, &params, None).expect("solve should not error");

    assert!(
        result.converged,
        "single-factor unpreconditioned solve did not converge (residual: {:.2e})",
        result.residual
    );
}

// ---------------------------------------------------------------------------
// 4. Effect-term design API (issue #58)
// ---------------------------------------------------------------------------

/// Intercept-only `Effect` design vs. the categories path. Both run through the
/// same `from_frame` locality sort, so rows sum in the same order and the result
/// is bit-identical — hence the exact `assert_eq`, not a tolerance.
#[test]
fn test_intercept_only_effects_match_categories_bitwise() {
    use within::{Effect, LsmrOptions, PreconditionerConfig, Solver};

    // Non-monotonic dominant factor so the locality sort is genuinely exercised.
    let col0: Vec<u32> = vec![3, 0, 2, 1, 3, 0, 2, 1, 3, 0, 2, 1];
    let col1: Vec<u32> = vec![0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1];
    let n_obs = col0.len();
    let y: Vec<f64> = (0..n_obs)
        .map(|i| (i as f64 * 1.3 - 2.0).sin() + 0.5)
        .collect();
    let params = LsmrOptions::default();
    let precond = PreconditionerConfig::default();

    let categories = Design::from_frame(
        ObservationFrame::new(vec![col0.clone().into(), col1.clone().into()], Vec::new())
            .expect("frame"),
    )
    .expect("categories design");
    let cat = Solver::new(categories, None, &precond)
        .expect("categories solver")
        .solve(&y, &params)
        .expect("categories solve");

    let eff = Solver::new(
        vec![
            Effect::new(&col0, true, []).expect("effect 0"),
            Effect::new(&col1, true, []).expect("effect 1"),
        ],
        None,
        &precond,
    )
    .expect("effect solver")
    .solve(&y, &params)
    .expect("effect solve");

    // Bit-identity is only meaningful if both solves actually converged:
    // `Solver::solve` returns `Ok` even at `maxiter`, so without this the
    // assert_eqs could pass on two identical non-converged states.
    assert!(
        eff.converged && cat.converged,
        "both solves must converge for bit-identity to be meaningful"
    );
    assert_eq!(eff.x, cat.x, "coefficients must be bit-identical");
    assert_eq!(
        eff.demeaned, cat.demeaned,
        "residuals must be bit-identical"
    );
}

/// Weakly-connected slope design: firms form a chain linked only by two
/// mover observations per adjacent pair, and the worker factor carries a
/// slope. The intercept and slope subdomains overlap on the firm block;
/// multiplicity-weighted partition weights on that overlap blow this design
/// up to hundreds of LSMR iterations that grow with slope count and scale
/// (issue #94). With uniform weights on slope-carrying subdomains it
/// converges in a few dozen. Regression guard for the slope-chain blind spot.
#[test]
fn test_slope_chain_design_converges_fast() {
    use within::{Effect, LsmrOptions, PreconditionerConfig, Solver};

    let (n_firms, wpf, t) = (60usize, 3usize, 4usize);
    let n_workers = n_firms * wpf;
    let n_obs = n_workers * t;
    let mut worker = Vec::with_capacity(n_obs);
    let mut firm = Vec::with_capacity(n_obs);
    for w in 0..n_workers {
        let home = (w / wpf) as u32;
        for obs in 0..t {
            worker.push(w as u32);
            // first worker of each firm block spends its last obs at the
            // next firm in the chain
            let moves = w % wpf == 0 && obs == t - 1 && (home as usize) < n_firms - 1;
            firm.push(if moves { home + 1 } else { home });
        }
    }
    let z: Vec<f64> = (0..n_obs).map(|i| (i as f64 * 3.7 + 0.5).sin()).collect();
    let y: Vec<f64> = (0..n_obs).map(|i| (i as f64 * 0.17 + 1.0).sin()).collect();

    let effects = vec![
        Effect::new(&worker, true, [&z[..]]).expect("slope effect"),
        Effect::new(&firm, true, []).expect("plain effect"),
    ];
    let solver = Solver::new(effects, None, PreconditionerConfig::default()).expect("solver build");
    let params = LsmrOptions {
        tol: 1e-8,
        maxiter: 500,
        ..Default::default()
    };
    let result = solver.solve(&y, &params).expect("solve");

    assert!(
        result.converged,
        "slope-chain solve did not converge (residual: {:.2e})",
        result.residual
    );
    assert!(
        result.iterations < 100,
        "slope-chain solve took {} iterations — preconditioner regression on \
         weakly-connected slope designs",
        result.iterations
    );
}
