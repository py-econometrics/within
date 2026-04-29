//! Integration tests for the domain layer: Design operations,
//! adjoint properties, gramian diagonal identity, and convergence through
//! the solve API for designs that exercise partition-of-unity weights and
//! disconnected bipartite structure.

use proptest::prelude::*;
use schwarz_precond::Operator;
use within::observation::FactorMajorStore;
use within::operator::gramian::Gramian;
use within::{Design, DesignOperator, WeightedDesignOperator};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_design(categories: Vec<Vec<u32>>, n_obs: usize) -> Design<FactorMajorStore> {
    let store = FactorMajorStore::new(categories, n_obs).expect("valid factor-major store");
    Design::from_store(store).expect("valid design")
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// ---------------------------------------------------------------------------
// 1. Parallel gather/scatter behavioral test (n_rows > PAR_THRESHOLD = 10,000)
// ---------------------------------------------------------------------------

/// Build a 15,000-row design with two factors (~50 levels each).
/// This exercises the parallel code paths in `gather_add` (par_chunks_mut)
/// and `scatter_add` (Fold strategy: n_rows > 10,000, n_levels < 100,000).
fn make_large_design() -> Design<FactorMajorStore> {
    let n_obs = 15_000;
    let n_levels_a = 50usize;
    let n_levels_b = 50usize;
    let fa: Vec<u32> = (0..n_obs).map(|i| (i % n_levels_a) as u32).collect();
    let fb: Vec<u32> = (0..n_obs).map(|i| (i % n_levels_b) as u32).collect();
    make_design(vec![fa, fb], n_obs)
}

#[test]
fn test_large_design_adjoint_property_matvec_d_rmatvec_dt() {
    // Verify <D·x, r> == <x, D^T·r> for random-looking deterministic vectors.
    let dm = make_large_design();
    let n_dofs = dm.n_dofs;
    let n_rows = dm.n_rows;

    let x: Vec<f64> = (0..n_dofs).map(|i| (i as f64 * 0.17 + 1.0).sin()).collect();
    let r: Vec<f64> = (0..n_rows).map(|i| (i as f64 * 0.23 + 2.0).cos()).collect();

    let mut dx = vec![0.0f64; n_rows];
    DesignOperator::new(&dm).apply(&x, &mut dx).expect("apply");

    let mut dtr = vec![0.0f64; n_dofs];
    DesignOperator::new(&dm)
        .apply_adjoint(&r, &mut dtr)
        .expect("apply");

    let lhs = dot(&dx, &r);
    let rhs = dot(&x, &dtr);

    assert!(
        (lhs - rhs).abs() < 1e-8,
        "Adjoint property violated: <D·x, r>={lhs} vs <x, D^T·r>={rhs}"
    );
}

#[test]
fn test_large_design_matvec_correctness() {
    // Sanity: D·e_j should equal the j-th column of D (a vector of 1.0s at
    // rows whose factor-level assignment corresponds to DOF j).
    let dm = make_large_design();
    let n_dofs = dm.n_dofs;
    let n_rows = dm.n_rows;

    // Pick DOF index 0 (factor 0, level 0): observations i where i % 50 == 0.
    let mut ej = vec![0.0f64; n_dofs];
    ej[0] = 1.0;
    let mut y = vec![0.0f64; n_rows];
    DesignOperator::new(&dm).apply(&ej, &mut y).expect("apply");

    for (i, &yi) in y.iter().enumerate() {
        let expected = if i % 50 == 0 { 1.0 } else { 0.0 };
        assert_eq!(
            yi, expected,
            "matvec_d(e_0)[{i}]: expected {expected}, got {yi}"
        );
    }
}

#[test]
fn test_large_design_rmatvec_dt_correctness() {
    // D^T·1 should equal the per-level observation count for each factor.
    let dm = make_large_design();
    let n_dofs = dm.n_dofs;
    let n_rows = dm.n_rows;

    let ones = vec![1.0f64; n_rows];
    let mut x = vec![0.0f64; n_dofs];
    DesignOperator::new(&dm)
        .apply_adjoint(&ones, &mut x)
        .expect("apply");

    // Each factor has 50 levels, 15,000 obs cycling → each level appears 300 times.
    let expected_count = (n_rows / 50) as f64;
    for (j, &xj) in x.iter().enumerate() {
        assert!(
            (xj - expected_count).abs() < 1e-10,
            "D^T·1 at DOF {j}: expected {expected_count}, got {xj}"
        );
    }
}

#[test]
fn test_large_design_gramian_diagonal_from_unit_vectors() {
    // Verify: gramian_diagonal()[j] == e_j^T · (D^T · D) · e_j
    // which equals ||D · e_j||^2 for unweighted designs.
    let dm = make_large_design();
    let n_dofs = dm.n_dofs;
    let n_rows = dm.n_rows;

    let diag = Gramian::build(&dm).diagonal();

    for j in 0..n_dofs {
        let mut ej = vec![0.0f64; n_dofs];
        ej[j] = 1.0;

        let mut dej = vec![0.0f64; n_rows];
        DesignOperator::new(&dm)
            .apply(&ej, &mut dej)
            .expect("apply");

        let norm_sq: f64 = dej.iter().map(|v| v * v).sum();
        assert!(
            (diag[j] - norm_sq).abs() < 1e-10,
            "gramian_diagonal()[{j}]={} != ||D·e_j||^2={norm_sq}",
            diag[j]
        );
    }
}

// ---------------------------------------------------------------------------
// 2. Gramian diagonal algebraic identity (property test)
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10))]

    /// For random weights and factor structures, gramian_diagonal()[j] must equal
    /// e_j^T · D^T · W · D · e_j for all DOFs j.
    #[test]
    fn prop_gramian_diagonal_matches_unit_vector_quadratic_form(
        n_obs in 20usize..=200,
        n_levels_a in 2usize..=15,
        n_levels_b in 2usize..=15,
        seed in 0u64..1000,
    ) {
        // Build deterministic-ish factor arrays from seed + indices.
        let fa: Vec<u32> = (0..n_obs)
            .map(|i| ((i * 3 + seed as usize * 7) % n_levels_a) as u32)
            .collect();
        let fb: Vec<u32> = (0..n_obs)
            .map(|i| ((i * 5 + seed as usize * 11) % n_levels_b) as u32)
            .collect();

        // Weights: non-uniform, vary by observation index.
        let weights: Vec<f64> = (0..n_obs)
            .map(|i| 0.5 + (i as f64 * 0.1 + seed as f64 * 0.3).sin().abs())
            .collect();

        let store = FactorMajorStore::new(vec![fa, fb], n_obs).unwrap();
        let dm = Design::from_store(store).unwrap();

        let n_dofs = dm.n_dofs;
        let n_rows = dm.n_rows;
        let diag = Gramian::build_weighted(&dm, &weights).diagonal();

        for j in 0..n_dofs {
            let mut ej = vec![0.0f64; n_dofs];
            ej[j] = 1.0;

            // D · e_j
            let mut dej = vec![0.0f64; n_rows];
            DesignOperator::new(&dm).apply(&ej, &mut dej).expect("apply");

            // e_j^T · D^T · W · D · e_j = sum_i w_i * (D·e_j)[i]^2
            let quadratic: f64 = (0..n_rows).map(|i| weights[i] * dej[i] * dej[i]).sum();

            prop_assert!(
                (diag[j] - quadratic).abs() < 1e-10,
                "gramian_diagonal()[{j}]={} != e_j^T·D^T·W·D·e_j={quadratic}",
                diag[j]
            );
        }
    }

    /// The adjoint property must hold for random designs:
    /// <D·x, W·r> == <x, D^T·W·r>  (i.e., <D·x, W·r> == <x, rmatvec_wdt(r)>)
    #[test]
    fn prop_weighted_adjoint_property(
        n_obs in 20usize..=200,
        n_levels_a in 2usize..=15,
        n_levels_b in 2usize..=15,
        seed in 0u64..1000,
    ) {
        let fa: Vec<u32> = (0..n_obs)
            .map(|i| ((i * 3 + seed as usize * 7) % n_levels_a) as u32)
            .collect();
        let fb: Vec<u32> = (0..n_obs)
            .map(|i| ((i * 5 + seed as usize * 11) % n_levels_b) as u32)
            .collect();

        let weights: Vec<f64> = (0..n_obs)
            .map(|i| 0.5 + (i as f64 * 0.13 + seed as f64 * 0.41).sin().abs())
            .collect();

        let store = FactorMajorStore::new(vec![fa, fb], n_obs).unwrap();
        let dm = Design::from_store(store).unwrap();

        let n_dofs = dm.n_dofs;
        let n_rows = dm.n_rows;

        let x: Vec<f64> = (0..n_dofs)
            .map(|i| (i as f64 * 0.37 + seed as f64 * 0.13).sin())
            .collect();
        let r: Vec<f64> = (0..n_rows)
            .map(|i| (i as f64 * 0.29 + seed as f64 * 0.07).cos())
            .collect();

        // <D·x, W·r>
        let mut dx = vec![0.0f64; n_rows];
        DesignOperator::new(&dm).apply(&x, &mut dx).expect("apply");
        let lhs: f64 = dx
            .iter()
            .zip(r.iter())
            .enumerate()
            .map(|(i, (dxi, ri))| weights[i] * dxi * ri)
            .sum();

        // <x, D^T·W·r>
        let mut wdtr = vec![0.0f64; n_dofs];
        let weighted_op = WeightedDesignOperator::new(&dm, &weights);
        weighted_op.apply_adjoint(&weighted_op.weighted_rhs(&r), &mut wdtr).expect("apply");
        let rhs: f64 = x.iter().zip(wdtr.iter()).map(|(xi, wi)| xi * wi).sum();

        prop_assert!(
            (lhs - rhs).abs() < 1e-8,
            "<D·x, W·r>={lhs} != <x, D^T·W·r>={rhs}"
        );
    }
}

// ---------------------------------------------------------------------------
// 3. Three-factor design: overlap creates NonUniform partition weights
// ---------------------------------------------------------------------------
// In a 3-factor design, pairs (0,1), (0,2), (1,2) all share DOFs from
// the respective factors. Factor 0's DOFs appear in both the (0,1) and
// (0,2) subdomains → NonUniform weights (1/2) must be assigned.
// We verify this through the public solve API by checking convergence,
// which validates that the partition of unity is correct.

#[test]
fn test_three_factor_design_solve_converges() {
    use within::{solve, LocalSolverConfig, Preconditioner, SolverParams};

    let n_obs = 60;
    let n_lev = 5usize;
    // Fully connected: every level of each factor appears with every level
    // of every other factor → significant subdomain overlap.
    let fa: Vec<u32> = (0..n_obs).map(|i| (i % n_lev) as u32).collect();
    let fb: Vec<u32> = (0..n_obs).map(|i| ((i / n_lev) % n_lev) as u32).collect();
    let fc: Vec<u32> = (0..n_obs).map(|i| ((i * 3) % n_lev) as u32).collect();

    let store = FactorMajorStore::new(vec![fa, fb, fc], n_obs).expect("valid 3-factor store");
    let dm = Design::from_store(store).expect("valid 3-factor design");

    assert_eq!(dm.n_factors(), 3);

    // Build y = D·1 so the true normal-equation solution is 1.
    let x_true = vec![1.0f64; dm.n_dofs];
    let mut y = vec![0.0f64; dm.n_rows];
    DesignOperator::new(&dm)
        .apply(&x_true, &mut y)
        .expect("apply");

    // Use ndarray array2 as required by the solve() API.
    let n_factors = 3;
    let mut cats = ndarray::Array2::<u32>::zeros((n_obs, n_factors));
    for i in 0..n_obs {
        cats[[i, 0]] = (i % n_lev) as u32;
        cats[[i, 1]] = ((i / n_lev) % n_lev) as u32;
        cats[[i, 2]] = ((i * 3) % n_lev) as u32;
    }

    let params = SolverParams {
        tol: 1e-8,
        maxiter: 500,
        ..SolverParams::default()
    };
    let precond = Preconditioner::Additive(
        LocalSolverConfig::solver_default(),
        within::ReductionStrategy::Auto,
    );
    let result =
        solve(cats.view(), &y, None, &params, Some(&precond)).expect("solve should not error");

    assert!(
        result.converged,
        "3-factor solver did not converge (residual: {:.2e})",
        result.final_residual
    );
}

/// Test that the 3-factor design converges with additive Schwarz, which
/// internally validates that the partition of unity is correct (overlapping
/// subdomains with NonUniform weights). Multiplicative Schwarz is also tested
/// to cover more of the preconditioner path.
#[test]
fn test_three_factor_design_multiplicative_schwarz_converges() {
    use within::{solve, LocalSolverConfig, Preconditioner, SolverParams};

    let n_obs = 60;
    let n_lev = 5usize;
    let mut cats = ndarray::Array2::<u32>::zeros((n_obs, 3));
    for i in 0..n_obs {
        cats[[i, 0]] = (i % n_lev) as u32;
        cats[[i, 1]] = ((i / n_lev) % n_lev) as u32;
        cats[[i, 2]] = ((i * 3) % n_lev) as u32;
    }

    // Build y = D·1 so the true solution is 1.
    let store = FactorMajorStore::new(
        vec![
            (0..n_obs).map(|i| (i % n_lev) as u32).collect(),
            (0..n_obs).map(|i| ((i / n_lev) % n_lev) as u32).collect(),
            (0..n_obs).map(|i| ((i * 3) % n_lev) as u32).collect(),
        ],
        n_obs,
    )
    .expect("valid 3-factor store");
    let dm = Design::from_store(store).expect("valid 3-factor design");
    let x_true = vec![1.0f64; dm.n_dofs];
    let mut y = vec![0.0f64; dm.n_rows];
    DesignOperator::new(&dm)
        .apply(&x_true, &mut y)
        .expect("apply");

    let params = SolverParams {
        tol: 1e-8,
        maxiter: 500,
        ..SolverParams::default()
    };
    let precond = Preconditioner::Multiplicative(LocalSolverConfig::solver_default());
    let result =
        solve(cats.view(), &y, None, &params, Some(&precond)).expect("solve should not error");

    assert!(
        result.converged,
        "3-factor multiplicative Schwarz did not converge (residual: {:.2e})",
        result.final_residual
    );
}

// ---------------------------------------------------------------------------
// 4. Disconnected bipartite graph → multiple subdomains per factor pair
// ---------------------------------------------------------------------------
// Factor 0: [0, 0, 1, 1]
// Factor 1: [0, 1, 2, 3]
// Levels {0} of factor 0 co-occurs only with {0,1} of factor 1, and
// level {1} of factor 0 co-occurs only with {2,3} of factor 1.
// This creates two disconnected bipartite components for pair (0,1).

/// Verify that a disconnected bipartite design converges under additive Schwarz.
/// The disconnected structure means `build_local_domains` splits pair (0,1) into
/// 2 subdomains — correctness is validated indirectly through convergence.
#[test]
fn test_disconnected_design_larger_converges() {
    use within::{solve, LocalSolverConfig, Preconditioner, SolverParams};

    // Extend the disconnected example to more observations so the solve is
    // non-trivial: component A has factor-0 levels {0,1}, factor-1 levels {0,1,2};
    // component B has factor-0 levels {2,3}, factor-1 levels {3,4,5}.
    let fa = vec![0u32, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3];
    let fb = vec![0u32, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5];
    let n_obs = fa.len();

    let mut cats = ndarray::Array2::<u32>::zeros((n_obs, 2));
    for i in 0..n_obs {
        cats[[i, 0]] = fa[i];
        cats[[i, 1]] = fb[i];
    }

    let store = FactorMajorStore::new(vec![fa.clone(), fb.clone()], n_obs)
        .expect("valid disconnected store");
    let dm = Design::from_store(store).expect("valid disconnected design");

    let x_true = vec![1.0f64; dm.n_dofs];
    let mut y = vec![0.0f64; dm.n_rows];
    DesignOperator::new(&dm)
        .apply(&x_true, &mut y)
        .expect("apply");

    let params = SolverParams {
        tol: 1e-8,
        maxiter: 500,
        ..SolverParams::default()
    };
    let precond = Preconditioner::Additive(
        LocalSolverConfig::solver_default(),
        within::ReductionStrategy::Auto,
    );
    let result =
        solve(cats.view(), &y, None, &params, Some(&precond)).expect("solve should not error");

    assert!(
        result.converged,
        "disconnected larger design did not converge (residual: {:.2e})",
        result.final_residual
    );
}

#[test]
fn test_disconnected_design_solve_converges() {
    use within::{solve, LocalSolverConfig, Preconditioner, SolverParams};

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

    let params = SolverParams {
        tol: 1e-8,
        maxiter: 500,
        ..SolverParams::default()
    };
    let precond = Preconditioner::Additive(
        LocalSolverConfig::solver_default(),
        within::ReductionStrategy::Auto,
    );
    let result =
        solve(cats.view(), &y, None, &params, Some(&precond)).expect("solve should not error");

    assert!(
        result.converged,
        "disconnected design solver did not converge (residual: {:.2e})",
        result.final_residual
    );
}

// ---------------------------------------------------------------------------
// 5. Single-factor design
// ---------------------------------------------------------------------------

#[test]
fn test_single_factor_design_construction() {
    let categories = vec![vec![0u32, 1, 2, 0, 1]];
    let store = FactorMajorStore::new(categories, 5).expect("valid store");
    let dm = Design::from_store(store).expect("valid single-factor design");

    assert_eq!(dm.n_factors(), 1, "expected 1 factor");
    assert_eq!(dm.n_dofs, 3, "expected 3 DOFs (levels 0,1,2)");
    assert_eq!(dm.n_rows, 5, "expected 5 rows");
}

#[test]
fn test_single_factor_design_adjoint_property() {
    let categories = vec![vec![0u32, 1, 2, 0, 1]];
    let store = FactorMajorStore::new(categories, 5).expect("valid store");
    let dm = Design::from_store(store).expect("valid single-factor design");

    let n_dofs = dm.n_dofs;
    let n_rows = dm.n_rows;

    let x: Vec<f64> = vec![1.0, 2.0, 3.0];
    let r: Vec<f64> = vec![0.5, 1.5, -0.5, 2.0, -1.0];

    let mut dx = vec![0.0f64; n_rows];
    DesignOperator::new(&dm).apply(&x, &mut dx).expect("apply");

    let mut dtr = vec![0.0f64; n_dofs];
    DesignOperator::new(&dm)
        .apply_adjoint(&r, &mut dtr)
        .expect("apply");

    let lhs = dot(&dx, &r);
    let rhs = dot(&x, &dtr);

    assert!(
        (lhs - rhs).abs() < 1e-12,
        "<D·x, r>={lhs} != <x, D^T·r>={rhs}"
    );
}

#[test]
fn test_single_factor_design_gramian_diagonal_is_level_counts() {
    // For unweighted single-factor: gramian_diagonal = observation count per level.
    // levels: [0, 1, 2, 0, 1] → counts [2, 2, 1]
    let categories = vec![vec![0u32, 1, 2, 0, 1]];
    let store = FactorMajorStore::new(categories, 5).expect("valid store");
    let dm = Design::from_store(store).expect("valid single-factor design");

    let diag = Gramian::build(&dm).diagonal();
    assert_eq!(diag, vec![2.0, 2.0, 1.0]);
}

/// A single-factor design has no factor pairs, so the additive Schwarz
/// preconditioner has no subdomains to work with. The solver should still
/// function (falling back to unpreconditioned CG) or be able to solve the
/// trivial normal equations directly.
#[test]
fn test_single_factor_design_solve_without_precond() {
    use within::{solve, SolverParams};

    // y = [10, 20, 30, 10, 20] with levels [0,1,2,0,1] → normal equations
    // are diagonal → converges in 1 iteration unpreconditioned.
    let n_obs = 5usize;
    let mut cats = ndarray::Array2::<u32>::zeros((n_obs, 1));
    let levels = [0u32, 1, 2, 0, 1];
    for i in 0..n_obs {
        cats[[i, 0]] = levels[i];
    }
    let y = vec![10.0, 20.0, 30.0, 10.0, 20.0];

    let params = SolverParams {
        tol: 1e-8,
        maxiter: 500,
        ..SolverParams::default()
    };
    // No preconditioner: the normal equations for a single-factor design are
    // diagonal and CG converges immediately.
    let result = solve(cats.view(), &y, None, &params, None).expect("solve should not error");

    assert!(
        result.converged,
        "single-factor unpreconditioned solve did not converge (residual: {:.2e})",
        result.final_residual
    );
}

#[test]
fn test_single_factor_matvec_d_values() {
    // D·[a, b, c] with levels [0,1,2,0,1] should give [a, b, c, a, b]
    let categories = vec![vec![0u32, 1, 2, 0, 1]];
    let store = FactorMajorStore::new(categories, 5).expect("valid store");
    let dm = Design::from_store(store).expect("valid single-factor design");

    let x = vec![10.0, 20.0, 30.0];
    let mut y = vec![0.0f64; 5];
    DesignOperator::new(&dm).apply(&x, &mut y).expect("apply");
    assert_eq!(y, vec![10.0, 20.0, 30.0, 10.0, 20.0]);
}

#[test]
fn test_single_factor_rmatvec_dt_values() {
    // D^T·[1,2,3,4,5] with levels [0,1,2,0,1] should give [1+4, 2+5, 3] = [5, 7, 3]
    let categories = vec![vec![0u32, 1, 2, 0, 1]];
    let store = FactorMajorStore::new(categories, 5).expect("valid store");
    let dm = Design::from_store(store).expect("valid single-factor design");

    let r = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let mut x = vec![0.0f64; 3];
    DesignOperator::new(&dm)
        .apply_adjoint(&r, &mut x)
        .expect("apply");
    assert_eq!(x, vec![5.0, 7.0, 3.0]);
}
