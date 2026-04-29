use schwarz_precond::Operator;
use within::operator::gramian::Gramian;
use within::operator::preconditioner::{
    additive_reduction_strategy, additive_schwarz_diagnostics, build_preconditioner,
    resolved_additive_reduction_strategy, FePreconditioner,
};
use within::operator::schwarz::build_schwarz;
use within::{
    KrylovMethod, LocalSolverConfig, OperatorRepr, Preconditioner, ReductionStrategy, Solver,
    SolverParams,
};

#[path = "common/orchestrate_helpers.rs"]
mod common;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn build_multiplicative_solver() -> Solver<within::FactorMajorStore> {
    let design = common::make_test_design();
    let params = SolverParams {
        krylov: KrylovMethod::Gmres { restart: 30 },
        operator: OperatorRepr::Explicit,
        tol: 1e-8,
        maxiter: 1000,
        ..Default::default()
    };
    let precond = Preconditioner::Multiplicative(LocalSolverConfig::default());
    Solver::from_design(design, None, &params, Some(&precond)).expect("multiplicative solver build")
}

// ---------------------------------------------------------------------------
// Test 1: additive_reduction_strategy returns None for multiplicative
// ---------------------------------------------------------------------------

#[test]
fn test_additive_reduction_strategy_none_for_multiplicative() {
    let solver = build_multiplicative_solver();
    let preconditioner = solver.preconditioner().expect("should have preconditioner");

    let result = additive_reduction_strategy(preconditioner);
    assert!(
        result.is_none(),
        "additive_reduction_strategy should return None for multiplicative preconditioner"
    );
}

// ---------------------------------------------------------------------------
// Test 2: resolved_additive_reduction_strategy returns None for multiplicative
// ---------------------------------------------------------------------------

#[test]
fn test_resolved_additive_reduction_strategy_none_for_multiplicative() {
    let solver = build_multiplicative_solver();
    let preconditioner = solver.preconditioner().expect("should have preconditioner");

    let result = resolved_additive_reduction_strategy(preconditioner);
    assert!(
        result.is_none(),
        "resolved_additive_reduction_strategy should return None for multiplicative preconditioner"
    );
}

// ---------------------------------------------------------------------------
// Test 3: additive_schwarz_diagnostics returns None for multiplicative
// ---------------------------------------------------------------------------

#[test]
fn test_additive_schwarz_diagnostics_none_for_multiplicative() {
    let solver = build_multiplicative_solver();
    let preconditioner = solver.preconditioner().expect("should have preconditioner");

    let result = additive_schwarz_diagnostics(preconditioner);
    assert!(
        result.is_none(),
        "additive_schwarz_diagnostics should return None for multiplicative preconditioner"
    );
}

// ---------------------------------------------------------------------------
// Test 4: build_preconditioner multiplicative succeeds with explicit gramian
// ---------------------------------------------------------------------------

#[test]
fn test_build_preconditioner_multiplicative_success() {
    let design = common::make_test_design();
    let gramian = Gramian::build(&design);

    let result = build_preconditioner(
        &design,
        None,
        Some(&gramian),
        &Preconditioner::Multiplicative(LocalSolverConfig::default()),
    );

    let precond = result.expect("multiplicative build_preconditioner should succeed");

    assert!(
        matches!(precond, FePreconditioner::Multiplicative(_)),
        "expected Multiplicative variant"
    );
    assert_eq!(precond.nrows(), design.n_dofs);
    assert_eq!(precond.ncols(), design.n_dofs);
}

// ---------------------------------------------------------------------------
// Test 5: solver with None preconditioner returns None from preconditioner()
// ---------------------------------------------------------------------------

#[test]
fn test_build_preconditioner_none_returns_none() {
    let design = common::make_test_design();
    let params = SolverParams::default();

    let solver = Solver::from_design(design, None, &params, None).expect("solver build");
    assert!(
        solver.preconditioner().is_none(),
        "solver with no preconditioner should return None"
    );
}

// ---------------------------------------------------------------------------
// Test 6: FeSchwarz with_reduction_strategy changes strategy, apply is finite
// ---------------------------------------------------------------------------

#[test]
fn test_fe_schwarz_with_reduction_strategy() {
    let design = common::make_test_design();
    let schwarz = build_schwarz(&design, None, &LocalSolverConfig::default())
        .expect("build_schwarz should succeed");

    let new_schwarz = schwarz.with_reduction_strategy(ReductionStrategy::AtomicScatter);

    assert_eq!(
        new_schwarz.reduction_strategy(),
        ReductionStrategy::AtomicScatter,
        "reduction_strategy should reflect the new strategy"
    );

    let x = vec![1.0; design.n_dofs];
    let mut y = vec![0.0; design.n_dofs];
    new_schwarz.apply(&x, &mut y);
    assert!(
        y.iter().all(|v| v.is_finite()),
        "apply output should be finite after with_reduction_strategy"
    );
}

// ---------------------------------------------------------------------------
// Test 7: FeSchwarz subdomains count is positive
// ---------------------------------------------------------------------------

#[test]
fn test_fe_schwarz_subdomains_count() {
    let design = common::make_test_design();
    let schwarz = build_schwarz(&design, None, &LocalSolverConfig::default())
        .expect("build_schwarz should succeed");

    assert!(
        !schwarz.subdomains().is_empty(),
        "FeSchwarz should have at least one subdomain"
    );
}

// ---------------------------------------------------------------------------
// Test 8: FeSchwarz with Auto strategy resolves to a concrete strategy
// ---------------------------------------------------------------------------

#[test]
fn test_fe_schwarz_auto_resolves() {
    let design = common::make_test_design();
    let schwarz = build_schwarz(&design, None, &LocalSolverConfig::default())
        .expect("build_schwarz should succeed");

    // Default strategy is Auto.
    assert_eq!(
        schwarz.reduction_strategy(),
        ReductionStrategy::Auto,
        "default build should use Auto strategy"
    );

    // Resolved strategy must be something concrete.
    let resolved = schwarz.resolved_reduction_strategy();
    assert_ne!(
        resolved,
        ReductionStrategy::Auto,
        "resolved_reduction_strategy should not be Auto"
    );
    assert!(
        matches!(
            resolved,
            ReductionStrategy::AtomicScatter | ReductionStrategy::ParallelReduction
        ),
        "resolved strategy should be AtomicScatter or ParallelReduction, got {:?}",
        resolved
    );
}

// ---------------------------------------------------------------------------
// Test 9: FeSchwarz diagnostics returns sensible values
// ---------------------------------------------------------------------------

#[test]
fn test_fe_schwarz_diagnostics_valid() {
    let design = common::make_test_design();
    let schwarz = build_schwarz(&design, None, &LocalSolverConfig::default())
        .expect("build_schwarz should succeed");

    let diag = schwarz.diagnostics();

    assert!(
        diag.n_subdomains() > 0,
        "diagnostics should report at least one subdomain, got {}",
        diag.n_subdomains()
    );
    assert_eq!(
        diag.n_dofs(),
        design.n_dofs,
        "diagnostics n_dofs should match design n_dofs"
    );
}

// ---------------------------------------------------------------------------
// Test 10: FeSchwarz try_apply matches apply
// ---------------------------------------------------------------------------

#[test]
fn test_fe_schwarz_try_apply_matches_apply() {
    let design = common::make_test_design();
    let schwarz = build_schwarz(&design, None, &LocalSolverConfig::default())
        .expect("build_schwarz should succeed");

    let x = vec![1.0; design.n_dofs];

    let mut y_apply = vec![0.0; design.n_dofs];
    schwarz.apply(&x, &mut y_apply);

    let mut y_try = vec![0.0; design.n_dofs];
    schwarz
        .try_apply(&x, &mut y_try)
        .expect("try_apply should not fail");

    for (a, b) in y_apply.iter().zip(y_try.iter()) {
        assert_eq!(a, b, "apply and try_apply should produce identical results");
    }
}

// ---------------------------------------------------------------------------
// Bonus: additive_reduction_strategy / resolved / diagnostics return Some
//        for an additive preconditioner
// ---------------------------------------------------------------------------

#[test]
fn test_additive_preconditioner_functions_return_some() {
    let design = common::make_test_design();
    let gramian = Gramian::build(&design);

    let precond = build_preconditioner(
        &design,
        None,
        Some(&gramian),
        &Preconditioner::Additive(LocalSolverConfig::default(), ReductionStrategy::Auto),
    )
    .expect("additive build_preconditioner should succeed");

    assert!(
        additive_reduction_strategy(&precond).is_some(),
        "additive_reduction_strategy should return Some for additive preconditioner"
    );
    assert!(
        resolved_additive_reduction_strategy(&precond).is_some(),
        "resolved_additive_reduction_strategy should return Some for additive preconditioner"
    );
    assert!(
        additive_schwarz_diagnostics(&precond).is_some(),
        "additive_schwarz_diagnostics should return Some for additive preconditioner"
    );
}
