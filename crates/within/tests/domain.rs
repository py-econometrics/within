//! Integration tests for the domain layer: solver convergence through the
//! public `solve` API for designs that exercise partition-of-unity weights
//! and disconnected bipartite structure.

use within::observation::FactorMajorStore;
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

    let store = FactorMajorStore::new(vec![fa, fb, fc], n_obs).expect("valid 3-factor store");
    let dm = Design::from_store(store).expect("valid 3-factor design");

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
    let categories = vec![vec![0u32, 1, 2, 0, 1]];
    let store = FactorMajorStore::new(categories, 5).expect("valid store");
    let dm = Design::from_store(store).expect("valid single-factor design");

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
