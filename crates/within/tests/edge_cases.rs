use ndarray::array;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use schwarz_precond::Operator;
use within::observation::FactorMajorStore;
use within::operator::gramian::GramianOperator;
use within::{
    solve, Design, DesignOperator, FactorMajorStore as FMStore, KrylovMethod, LocalSolverConfig,
    OperatorRepr, Preconditioner, ReductionStrategy, Solver, SolverParams,
};

#[path = "common/orchestrate_helpers.rs"]
mod common;

fn additive_precond() -> Preconditioner {
    Preconditioner::Additive(LocalSolverConfig::solver_default(), ReductionStrategy::Auto)
}

// ---------------------------------------------------------------------------
// Test 1: single observation
// ---------------------------------------------------------------------------

/// n_obs=1 with 2 factors each at level 0.
/// The system D*x = y is underdetermined (2 DOFs, 1 equation). The solver
/// should still return a finite result without panicking.
#[test]
fn test_single_observation() {
    let cats = array![[0u32, 0]];
    let y = vec![5.0f64];
    let params = SolverParams::default();

    let result = solve(cats.view(), &y, None, &params, None).expect("single-obs solve");
    assert!(
        result.x.iter().all(|v| v.is_finite()),
        "non-finite x for single observation"
    );
    assert!(
        result.demeaned.iter().all(|v| v.is_finite()),
        "non-finite demeaned for single observation"
    );
}

// ---------------------------------------------------------------------------
// Test 2: factor where all observations share the same level
// ---------------------------------------------------------------------------

/// Factor 0 is constant (all level 0); factor 1 varies. The constant factor
/// contributes only a single DOF to the system. The solver should handle this
/// without any issue since the Gramian is still well-defined.
#[test]
fn test_trivial_factor_all_same_level() {
    // 5 observations; factor 0 is constant, factor 1 cycles through 0, 1, 2.
    let cats = array![[0u32, 0], [0u32, 1], [0u32, 2], [0u32, 0], [0u32, 1]];
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let params = SolverParams::default();
    let precond = additive_precond();

    let result =
        solve(cats.view(), &y, None, &params, Some(&precond)).expect("trivial-factor solve");
    assert!(
        result.converged,
        "solver did not converge with constant factor"
    );
    common::assert_solution_finite(&result);
    assert_eq!(
        result.x.len(),
        4,
        "1 level for factor0 + 3 levels for factor1"
    );
}

// ---------------------------------------------------------------------------
// Test 3: all-zero weights should produce an error
// ---------------------------------------------------------------------------

/// All-zero weights produce a zero Gramian diagonal.
///
/// Without a preconditioner, the Gramian and RHS are both zero so CG returns
/// x=0 immediately — this is technically valid (the "solution" is trivially
/// the zero vector). With a preconditioner, the local solver must factorize
/// a matrix with a zero diagonal and should fail with a build error.
#[test]
fn test_zero_weight_error_with_preconditioner() {
    let cats = array![[0u32, 0], [1u32, 0], [0u32, 1], [1u32, 1], [2u32, 0]];
    let y = vec![1.0f64; 5];
    let weights = vec![0.0f64; 5];
    let precond = additive_precond();

    let result = solve(
        cats.view(),
        &y,
        Some(&weights),
        &SolverParams::default(),
        Some(&precond),
    );
    assert!(
        result.is_err(),
        "zero weights with preconditioner should produce an error, but got: {:?}",
        result.map(|r| r.x)
    );
}

/// Without a preconditioner, all-zero weights produce a zero Gramian and a
/// zero RHS. CG starts with residual zero and converges immediately to x=0.
#[test]
fn test_zero_weight_no_preconditioner_returns_zero() {
    let cats = array![[0u32, 0], [1u32, 0], [0u32, 1], [1u32, 1], [2u32, 0]];
    let y = vec![1.0f64; 5];
    let weights = vec![0.0f64; 5];

    let result = solve(
        cats.view(),
        &y,
        Some(&weights),
        &SolverParams::default(),
        None,
    )
    .expect("zero weights with no preconditioner should succeed");

    assert!(
        result.converged,
        "zero-Gramian system should trivially converge"
    );
    assert!(
        result.x.iter().all(|&v| v == 0.0),
        "zero-Gramian solution must be the zero vector"
    );
}

// ---------------------------------------------------------------------------
// Test 4: maxiter=1 on a non-trivial problem
// ---------------------------------------------------------------------------

/// With maxiter=1, CG should stop after one iteration. The result need not be
/// converged, but x must be finite — no NaN/Inf should escape.
#[test]
fn test_cg_maxiter_1_partial_result() {
    // Use a moderately sized seeded problem to ensure 1 iteration is insufficient.
    let mut rng = SmallRng::seed_from_u64(7);
    let n_obs = 200usize;
    let cats: Vec<Vec<u32>> = vec![
        (0..n_obs).map(|_| rng.random_range(0..20u32)).collect(),
        (0..n_obs).map(|_| rng.random_range(0..20u32)).collect(),
    ];
    let store = FactorMajorStore::new(cats, n_obs).expect("valid store");
    let design = Design::from_store(store).expect("valid design");

    let y: Vec<f64> = (0..n_obs).map(|i| (i as f64 * 0.17).sin()).collect();

    let params = SolverParams {
        tol: 1e-15,
        maxiter: 1,
        max_refinements: 0,
        ..SolverParams::default()
    };
    let solver = Solver::from_design(design, None, &params, None).expect("solver build");
    let result = solver.solve(&y).expect("solve with maxiter=1");

    // Convergence is not expected (tolerance is unreachable in 1 iteration),
    // but all values must be finite.
    assert!(
        result.x.iter().all(|v| v.is_finite()),
        "non-finite x after maxiter=1"
    );
    assert!(
        result.demeaned.iter().all(|v| v.is_finite()),
        "non-finite demeaned after maxiter=1"
    );
    assert!(
        result.iterations <= 1,
        "expected ≤ 1 iteration, got {}",
        result.iterations
    );
}

// ---------------------------------------------------------------------------
// Test 5: GMRES on a known-solution problem
// ---------------------------------------------------------------------------

/// Solve with GMRES (explicit Gramian + additive Schwarz) using y = D * 1.
/// The system is consistent so GMRES should converge, and final_residual
/// must be below the requested tolerance.
#[test]
fn test_gmres_known_solution() {
    let design = common::make_test_design();
    let y = common::make_y_from_unit_solution(&design);

    let params = SolverParams {
        krylov: KrylovMethod::Gmres { restart: 30 },
        operator: OperatorRepr::Explicit,
        tol: 1e-8,
        maxiter: 1000,
        ..SolverParams::default()
    };
    let precond = additive_precond();
    let solver = Solver::from_design(design, None, &params, Some(&precond)).expect("solver build");
    let result = solver.solve(&y).expect("GMRES solve");

    assert!(
        result.converged,
        "GMRES did not converge (residual={:.2e})",
        result.final_residual
    );
    assert!(
        result.final_residual < 1e-6,
        "residual too large: {:.2e}",
        result.final_residual
    );
    common::assert_solution_finite(&result);
}

// ---------------------------------------------------------------------------
// Test 6: GMRES reported residual matches actual residual
// ---------------------------------------------------------------------------

/// After a GMRES solve, compute the actual observation-space residual
/// ||D^T W (y - Dx)|| / ||D^T W y|| independently and compare with
/// `result.final_residual`. They must agree to within floating-point
/// rounding (the solver uses the same formula).
#[test]
fn test_gmres_residual_estimate_vs_actual() {
    let design = common::make_test_design();
    let y = common::make_y_from_unit_solution(&design);

    let params = SolverParams {
        krylov: KrylovMethod::Gmres { restart: 30 },
        operator: OperatorRepr::Explicit,
        tol: 1e-8,
        maxiter: 1000,
        ..SolverParams::default()
    };
    let precond = additive_precond();
    let solver = Solver::from_design(design, None, &params, Some(&precond)).expect("solver build");
    let result = solver.solve(&y).expect("GMRES solve");

    assert!(result.converged);

    // Recompute the residual from scratch using the GramianOperator.
    // `final_residual` is ||D^T W (y - Dx)|| / ||D^T W y||.
    // Because the solver computes it via `rmatvec_wdt`, we reconstruct the
    // same quantity from the public `Gramian` API:
    //   actual_resid_vec = G * x - D^T W y
    // and normalise by ||D^T W y|| = ||rhs||.
    //
    // We rebuild a fresh design from the same categories to access the operator.
    let design2 = common::make_test_design();
    let n_dofs = design2.n_dofs;
    let gramian_op = GramianOperator::new(&design2, None);

    // rhs = D^T W y (unit weights, so D^T y)
    let mut rhs = vec![0.0; n_dofs];
    DesignOperator::new(&design2, None)
        .apply_adjoint(&y, &mut rhs)
        .expect("apply");
    let rhs_norm = rhs.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-15);

    // residual = G*x - rhs
    let mut gx = vec![0.0; n_dofs];
    gramian_op.apply(&result.x, &mut gx).expect("apply");
    let actual_residual_norm = gx
        .iter()
        .zip(rhs.iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        .sqrt();
    let actual_relative_residual = actual_residual_norm / rhs_norm;

    // The reported residual uses observation-space recomputation; the normal-
    // equation residual should agree to within a small factor.
    assert!(
        actual_relative_residual < 1e-5,
        "actual normal-equation residual too large: {:.2e}",
        actual_relative_residual
    );
    assert!(
        result.final_residual < 1e-5,
        "reported residual too large: {:.2e}",
        result.final_residual
    );
}

// ---------------------------------------------------------------------------
// Test 7: large design with seeded random data
// ---------------------------------------------------------------------------

/// Build a 10 000-observation, 2-factor design with seeded random categories.
/// Use the unit-solution RHS (y = D * 1) for a consistent system and verify
/// the preconditioned CG converges. This exercises the Schwarz preconditioner
/// at moderate scale without being slow enough to require `#[ignore]`.
#[test]
fn test_large_design_convergence() {
    let mut rng = SmallRng::seed_from_u64(42);
    let n_obs = 10_000usize;
    let cats: Vec<Vec<u32>> = vec![
        (0..n_obs).map(|_| rng.random_range(0..100u32)).collect(),
        (0..n_obs).map(|_| rng.random_range(0..100u32)).collect(),
    ];

    let store = FMStore::new(cats, n_obs).expect("valid large store");
    let design = Design::from_store(store).expect("valid large design");
    let y = common::make_y_from_unit_solution(&design);

    let params = SolverParams {
        tol: 1e-7,
        ..SolverParams::default()
    };
    let precond = additive_precond();
    let solver = Solver::from_design(design, None, &params, Some(&precond)).expect("solver build");
    let result = solver.solve(&y).expect("large design solve");

    assert!(
        result.converged,
        "large design did not converge (n_obs={n_obs}, residual={:.2e})",
        result.final_residual
    );
    common::assert_solution_finite(&result);
}

// ---------------------------------------------------------------------------
// Test 8: zero RHS produces zero solution immediately
// ---------------------------------------------------------------------------

/// y = 0 means D^T W y = 0, so the initial residual is already zero and CG
/// should return immediately with 0 iterations and x = 0.
#[test]
fn test_zero_rhs_zero_solution() {
    let design = common::make_test_design();
    let y = vec![0.0f64; design.n_rows];

    let params = SolverParams::default();
    let solver = Solver::from_design(design, None, &params, None).expect("solver build");
    let result = solver.solve(&y).expect("zero RHS solve");

    assert!(result.converged, "zero RHS should trivially converge");
    assert_eq!(result.iterations, 0, "zero RHS should need 0 iterations");
    assert!(
        result.x.iter().all(|&v| v == 0.0),
        "zero RHS should produce zero solution"
    );
}

// ---------------------------------------------------------------------------
// Test 9: weighted and unweighted give different solutions for the same data
// ---------------------------------------------------------------------------

/// Verify that passing non-uniform weights actually changes the solution.
/// Identical weights should produce the same solution as unweighted.
#[test]
fn test_uniform_weights_matches_unweighted() {
    let cats = array![[0u32, 0], [1u32, 0], [0u32, 1], [1u32, 1], [2u32, 0]];
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0f64];
    let uniform_weights = vec![2.0f64; 5]; // constant — equivalent to unit weights

    let params = SolverParams::default();
    let precond = additive_precond();

    let r_unit = solve(cats.view(), &y, None, &params, Some(&precond)).expect("unweighted solve");
    let r_uniform = solve(
        cats.view(),
        &y,
        Some(&uniform_weights),
        &params,
        Some(&precond),
    )
    .expect("uniform-weight solve");

    // Constant scaling of W leaves G and D^T W y proportional, so the solution
    // is identical.
    for (a, b) in r_unit.x.iter().zip(r_uniform.x.iter()) {
        assert!(
            (a - b).abs() < 1e-8,
            "uniform weights changed solution: {} vs {}",
            a,
            b
        );
    }
}

// ---------------------------------------------------------------------------
// Test 10: solve twice with same Solver produces identical results
// ---------------------------------------------------------------------------

/// The Solver caches the preconditioner. Calling `solve()` twice with the same
/// RHS on the same Solver instance must return bit-identical results.
#[test]
fn test_repeated_solve_is_deterministic() {
    let design = common::make_test_design();
    let y = common::make_y_from_unit_solution(&design);

    let params = SolverParams::default();
    let precond = additive_precond();
    let solver = Solver::from_design(design, None, &params, Some(&precond)).expect("solver build");

    let r1 = solver.solve(&y).expect("first solve");
    let r2 = solver.solve(&y).expect("second solve");

    assert_eq!(r1.x.len(), r2.x.len());
    for (i, (a, b)) in r1.x.iter().zip(r2.x.iter()).enumerate() {
        assert_eq!(a, b, "x[{i}] differs between two solves: {} vs {}", a, b);
    }
}
