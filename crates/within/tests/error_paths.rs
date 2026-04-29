use std::error::Error;

use ndarray::Array2;
use schwarz_precond::{BuildError, SolveError};
use within::observation::FactorMajorStore;
use within::{
    solve, Design, LocalSolverConfig, Preconditioner, ReductionStrategy, SolverParams, WithinError,
};

#[test]
fn test_empty_observations_error() {
    // FactorMajorStore::new allows 0 rows; EmptyObservations is raised by Design::from_store
    let store = FactorMajorStore::new(vec![vec![], vec![]], 0).expect("store ok");
    let result = Design::from_store(store);
    assert!(result.is_err());
    match result.unwrap_err() {
        WithinError::EmptyObservations => {}
        other => panic!("Expected EmptyObservations, got: {:?}", other),
    }
}

#[test]
fn test_observation_count_mismatch_error() {
    // Factor columns have different lengths
    let result = FactorMajorStore::new(vec![vec![0, 1, 2], vec![0, 1]], 3);
    assert!(result.is_err());
    match result.unwrap_err() {
        WithinError::ObservationCountMismatch { .. } => {}
        other => panic!("Expected ObservationCountMismatch, got: {:?}", other),
    }
}

#[test]
fn test_weight_count_mismatch_error() {
    use within::Design;
    let store = FactorMajorStore::new(vec![vec![0, 1, 2], vec![0, 1, 0]], 3).expect("valid store");
    let design = Design::from_store(store).expect("valid design");
    let params = SolverParams::default();
    let err = within::Solver::from_design(design, Some(vec![1.0, 2.0]), &params, None)
        .err()
        .expect("expected mismatch error");
    match err {
        WithinError::WeightCountMismatch { .. } => {}
        other => panic!("Expected WeightCountMismatch, got: {:?}", other),
    }
}

#[test]
fn test_empty_categories_via_solve() {
    let cats = Array2::<u32>::zeros((0, 2));
    let y: Vec<f64> = vec![];
    let params = SolverParams::default();
    let precond =
        Preconditioner::Additive(LocalSolverConfig::solver_default(), ReductionStrategy::Auto);
    let result = solve(cats.view(), &y, None, &params, Some(&precond));
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// Display tests for all WithinError variants
// ---------------------------------------------------------------------------

#[test]
fn test_within_error_display_empty_observations() {
    let e = WithinError::EmptyObservations;
    assert_eq!(e.to_string(), "no observations provided");
}

#[test]
fn test_within_error_display_observation_count_mismatch() {
    let e = WithinError::ObservationCountMismatch {
        factor: 1,
        expected: 10,
        got: 5,
    };
    let s = e.to_string();
    assert!(s.contains("factor 1"));
    assert!(s.contains("5"));
    assert!(s.contains("10"));
}

#[test]
fn test_within_error_display_weight_count_mismatch() {
    let e = WithinError::WeightCountMismatch {
        expected: 10,
        got: 5,
    };
    let s = e.to_string();
    assert!(s.contains("5"));
    assert!(s.contains("10"));
}

#[test]
fn test_within_error_display_overflow() {
    let e = WithinError::Overflow("test overflow".to_string());
    assert!(e.to_string().contains("test overflow"));
}

#[test]
fn test_within_error_display_singular_diagonal() {
    let e = WithinError::SingularDiagonal {
        block: "test_block",
        index: 42,
    };
    let s = e.to_string();
    assert!(s.contains("test_block"));
    assert!(s.contains("42"));
}

#[test]
fn test_within_error_display_local_solver_build() {
    let e = WithinError::LocalSolverBuild("factorization failed".to_string());
    assert!(e.to_string().contains("factorization failed"));
}

#[test]
fn test_within_error_display_preconditioner_build() {
    let inner = BuildError::GlobalIndexOutOfBounds {
        subdomain: 0,
        local_index: 1,
        global_index: 5,
        n_dofs: 3,
    };
    let e = WithinError::PreconditionerBuild(inner);
    let s = e.to_string();
    assert!(s.contains("5"));
    assert!(s.contains("3"));
}

#[test]
fn test_within_error_display_iterative_solve() {
    let inner = SolveError::Synchronization { context: "test" };
    let e = WithinError::IterativeSolve(inner);
    assert!(e.to_string().contains("test"));
}

// ---------------------------------------------------------------------------
// Error::source() tests
// ---------------------------------------------------------------------------

#[test]
fn test_within_error_source_none_variants() {
    let variants: Vec<WithinError> = vec![
        WithinError::EmptyObservations,
        WithinError::ObservationCountMismatch {
            factor: 0,
            expected: 1,
            got: 2,
        },
        WithinError::WeightCountMismatch {
            expected: 1,
            got: 2,
        },
        WithinError::Overflow("x".to_string()),
        WithinError::SingularDiagonal {
            block: "b",
            index: 0,
        },
        WithinError::LocalSolverBuild("x".to_string()),
    ];
    for e in &variants {
        assert!(e.source().is_none(), "expected None source for {:?}", e);
    }
}

#[test]
fn test_within_error_source_preconditioner_build() {
    let inner = BuildError::GlobalIndexOutOfBounds {
        subdomain: 0,
        local_index: 1,
        global_index: 5,
        n_dofs: 3,
    };
    let e = WithinError::PreconditionerBuild(inner);
    assert!(e.source().is_some());
}

#[test]
fn test_within_error_source_iterative_solve() {
    let inner = SolveError::Synchronization { context: "test" };
    let e = WithinError::IterativeSolve(inner);
    assert!(e.source().is_some());
}

// ---------------------------------------------------------------------------
// From conversions
// ---------------------------------------------------------------------------

#[test]
fn test_within_error_from_preconditioner_build_error() {
    let inner = BuildError::GlobalIndexOutOfBounds {
        subdomain: 0,
        local_index: 0,
        global_index: 100,
        n_dofs: 50,
    };
    let e: WithinError = inner.into();
    match e {
        WithinError::PreconditionerBuild(_) => {}
        other => panic!("expected PreconditionerBuild, got: {:?}", other),
    }
}

#[test]
fn test_within_error_from_solve_error() {
    let inner = SolveError::Synchronization { context: "test" };
    let e: WithinError = inner.into();
    match e {
        WithinError::IterativeSolve(_) => {}
        other => panic!("expected IterativeSolve, got: {:?}", other),
    }
}
