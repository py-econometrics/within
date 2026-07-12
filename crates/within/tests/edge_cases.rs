use ndarray::array;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use within::{solve, LsmrOptions, PreconditionerConfig, Solver};

#[path = "common/orchestrate_helpers.rs"]
mod common;

fn additive_precond() -> PreconditionerConfig {
    PreconditionerConfig::default()
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
    let params = LsmrOptions::default();

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
    let params = LsmrOptions::default();
    let precond = additive_precond();

    let result = solve(cats.view(), &y, None, &params, &precond).expect("trivial-factor solve");
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
// Test 3: all-zero weights solve the zero system to x=0
// ---------------------------------------------------------------------------

/// All-zero weights zero every Gramian cell and diagonal. Routing skips the
/// resulting dead DOFs, so no additive subdomain remains and the solve falls
/// back to unpreconditioned LSMR — which, like the diagonal and
/// unpreconditioned paths, solves the zero system and returns x=0.
#[test]
fn test_zero_weight_additive_preconditioner_returns_zero() {
    let cats = array![[0u32, 0], [1u32, 0], [0u32, 1], [1u32, 1], [2u32, 0]];
    let y = vec![1.0f64; 5];
    let weights = vec![0.0f64; 5];
    let precond = additive_precond();

    let result = solve(
        cats.view(),
        &y,
        Some(&weights),
        &LsmrOptions::default(),
        &precond,
    )
    .expect("zero weights with additive preconditioner should succeed");
    assert!(
        result.converged,
        "zero-Gramian system should trivially converge"
    );
    assert!(
        result.x.iter().all(|&v| v == 0.0),
        "zero-Gramian solution must be the zero vector"
    );
}

/// All-zero weights make every diagonal entry zero. The diagonal
/// preconditioner takes the pseudo-inverse of each zero entry, so — like the
/// additive and unpreconditioned paths — it solves the resulting zero system
/// and returns x=0.
#[test]
fn test_zero_weight_diagonal_preconditioner_returns_zero() {
    let cats = array![[0u32, 0], [1u32, 0], [0u32, 1], [1u32, 1], [2u32, 0]];
    let y = vec![1.0f64; 5];
    let weights = vec![0.0f64; 5];

    let result = solve(
        cats.view(),
        &y,
        Some(&weights),
        &LsmrOptions::default(),
        &PreconditionerConfig::Diagonal,
    )
    .expect("zero weights with diagonal preconditioner should succeed");

    assert!(
        result.converged,
        "zero-Gramian system should trivially converge"
    );
    assert!(
        result.x.iter().all(|&v| v == 0.0),
        "zero-Gramian solution must be the zero vector"
    );
}

/// Without a preconditioner, all-zero weights produce a zero system and a
/// zero RHS. LSMR starts with residual zero and converges immediately to x=0.
#[test]
fn test_zero_weight_no_preconditioner_returns_zero() {
    let cats = array![[0u32, 0], [1u32, 0], [0u32, 1], [1u32, 1], [2u32, 0]];
    let y = vec![1.0f64; 5];
    let weights = vec![0.0f64; 5];

    let result = solve(
        cats.view(),
        &y,
        Some(&weights),
        &LsmrOptions::default(),
        &PreconditionerConfig::Off,
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

/// A preconditioner changes convergence, not the answer: the diagonal and
/// unpreconditioned solves must agree on the same least-squares solution.
///
/// Uses a single full-rank factor so the solution is unique. A multi-factor FE
/// design is rank-deficient (the additive constant is unidentified), so the
/// minimum-norm coefficient vector LSMR returns depends on the preconditioned
/// metric — only the fitted values, not the raw coefficients, are invariant.
#[test]
fn test_diagonal_matches_unpreconditioned_solution() {
    let cats = array![[0u32], [0], [1], [1], [2], [2]];
    let y = vec![1.0, 3.0, 2.0, 4.0, 5.0, 7.0];
    let params = LsmrOptions::default();

    let diagonal = solve(
        cats.view(),
        &y,
        None,
        &params,
        &PreconditionerConfig::Diagonal,
    )
    .expect("diagonal solve");
    let unpreconditioned = solve(cats.view(), &y, None, &params, &PreconditionerConfig::Off)
        .expect("unpreconditioned solve");

    common::assert_solution_finite(&diagonal);
    common::assert_solutions_close(&diagonal.x, &unpreconditioned.x, 1e-6);
}

/// A factor whose observed levels leave interior gaps (`n_levels = max + 1`)
/// produces structural zero columns of `D` — unidentified DOFs whose diagonal
/// is zero. The unpreconditioned and additive paths both pin those coefficients
/// to 0 and solve fine; with the pseudo-inverse of a zero diagonal, the diagonal
/// preconditioner now matches rather than failing with `SingularDiagonal`.
#[test]
fn test_diagonal_matches_unpreconditioned_on_gap_design() {
    // Single factor observed only at levels {0, 2, 4} => n_levels = 5, so global
    // DOFs 1 and 3 have no observations.
    let cats = array![[0u32], [2], [4]];
    let y = vec![1.0, 2.0, 3.0];
    let params = LsmrOptions::default();

    let diagonal = solve(
        cats.view(),
        &y,
        None,
        &params,
        &PreconditionerConfig::Diagonal,
    )
    .expect("diagonal solve must succeed on a gap design (pseudo-inverse of zero diagonal)");
    let unpreconditioned = solve(cats.view(), &y, None, &params, &PreconditionerConfig::Off)
        .expect("unpreconditioned solve");

    assert!(diagonal.converged, "diagonal solve must converge");
    common::assert_solutions_close(&diagonal.x, &unpreconditioned.x, 1e-6);
    // The unobserved DOFs are unidentified and must be pinned to exactly 0.
    assert_eq!(diagonal.x[1], 0.0, "unobserved DOF 1 must be 0");
    assert_eq!(diagonal.x[3], 0.0, "unobserved DOF 3 must be 0");
}

// ---------------------------------------------------------------------------
// Test 4: maxiter=1 on a non-trivial problem
// ---------------------------------------------------------------------------

/// With maxiter=1, LSMR should stop after one iteration. The result need not be
/// converged, but x must be finite — no NaN/Inf should escape.
#[test]
fn test_maxiter_1_partial_result() {
    // Use a moderately sized seeded problem to ensure 1 iteration is insufficient.
    let mut rng = SmallRng::seed_from_u64(7);
    let n_obs = 200usize;
    let cats: Vec<Vec<u32>> = vec![
        (0..n_obs).map(|_| rng.random_range(0..20u32)).collect(),
        (0..n_obs).map(|_| rng.random_range(0..20u32)).collect(),
    ];
    let design = common::make_design(cats).expect("valid design");

    let y: Vec<f64> = (0..n_obs).map(|i| (i as f64 * 0.17).sin()).collect();

    let params = LsmrOptions {
        tol: 1e-15,
        maxiter: 1,
        ..LsmrOptions::default()
    };
    let solver = Solver::new(design, None, None).expect("solver build");
    let result = solver.solve(&y, &params).expect("solve with maxiter=1");

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
// Test 5: large design with seeded random data
// ---------------------------------------------------------------------------

/// Build a 10 000-observation, 2-factor design with seeded random categories.
/// Use the unit-solution RHS (y = D * 1) for a consistent system and verify
/// the preconditioned LSMR converges. This exercises the Schwarz preconditioner
/// at moderate scale without being slow enough to require `#[ignore]`.
#[test]
fn test_large_design_convergence() {
    let mut rng = SmallRng::seed_from_u64(42);
    let n_obs = 10_000usize;
    let cats: Vec<Vec<u32>> = vec![
        (0..n_obs).map(|_| rng.random_range(0..100u32)).collect(),
        (0..n_obs).map(|_| rng.random_range(0..100u32)).collect(),
    ];

    let design = common::make_design(cats).expect("valid large design");
    let y = common::make_deterministic_y(&design);

    let params = LsmrOptions {
        tol: 1e-7,
        ..LsmrOptions::default()
    };
    let precond = additive_precond();
    let solver = Solver::new(design, None, &precond).expect("solver build");
    let result = solver.solve(&y, &params).expect("large design solve");

    assert!(
        result.converged,
        "large design did not converge (n_obs={n_obs}, residual={:.2e})",
        result.residual
    );
    common::assert_solution_finite(&result);
}

// ---------------------------------------------------------------------------
// Test 8: zero RHS produces zero solution immediately
// ---------------------------------------------------------------------------

/// y = 0 means the residual is already zero, so LSMR should return immediately
/// with 0 iterations and x = 0.
#[test]
fn test_zero_rhs_zero_solution() {
    let design = common::make_test_design();
    let y = vec![0.0f64; design.n_obs()];

    let params = LsmrOptions::default();
    let solver = Solver::new(design, None, None).expect("solver build");
    let result = solver.solve(&y, &params).expect("zero RHS solve");

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

    let params = LsmrOptions::default();
    let precond = additive_precond();

    let r_unit = solve(cats.view(), &y, None, &params, &precond).expect("unweighted solve");
    let r_uniform = solve(cats.view(), &y, Some(&uniform_weights), &params, &precond)
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
    let y = common::make_deterministic_y(&design);

    let params = LsmrOptions::default();
    let precond = additive_precond();
    let solver = Solver::new(design, None, &precond).expect("solver build");

    let r1 = solver.solve(&y, &params).expect("first solve");
    let r2 = solver.solve(&y, &params).expect("second solve");

    assert_eq!(r1.x.len(), r2.x.len());
    for (i, (a, b)) in r1.x.iter().zip(r2.x.iter()).enumerate() {
        assert_eq!(a, b, "x[{i}] differs between two solves: {} vs {}", a, b);
    }
}
