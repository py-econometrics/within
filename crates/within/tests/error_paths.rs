use std::error::Error;

use ndarray::Array2;
use schwarz_precond::SolveError;
use within::observation::ObservationFrame;
use within::{solve, BuildError, Design, LsmrOptions, PreconditionerConfig, Solver, WithinError};

// Behavior: a malformed input produces the right typed error. The Display /
// source() / From plumbing is covered by a single wiring check per enum below,
// not by pinning every message string.

#[test]
fn test_empty_observations_error() {
    // A zero-row frame is valid; EmptyObservations is raised by Design::from_frame.
    let frame =
        ObservationFrame::new(vec![vec![].into(), vec![].into()], Vec::new()).expect("frame ok");
    let result = Design::from_frame(frame);
    assert!(result.is_err());
    match result.unwrap_err() {
        BuildError::EmptyObservations => {}
        other => panic!("Expected EmptyObservations, got: {:?}", other),
    }
}

#[test]
fn test_observation_count_mismatch_error() {
    // Factor columns have different lengths
    let result = ObservationFrame::new(
        vec![vec![0u32, 1, 2].into(), vec![0u32, 1].into()],
        Vec::new(),
    );
    assert!(result.is_err());
    match result.unwrap_err() {
        BuildError::ObservationCountMismatch { .. } => {}
        other => panic!("Expected ObservationCountMismatch, got: {:?}", other),
    }
}

#[test]
fn test_weight_count_mismatch_error() {
    // Weights of wrong length are caught at Solver construction time.
    let frame = ObservationFrame::new(
        vec![vec![0u32, 1, 2].into(), vec![0u32, 1, 0].into()],
        Vec::new(),
    )
    .expect("frame ok");
    let design = Design::from_frame(frame).expect("valid design");
    let result = Solver::new(design, Some(vec![1.0, 2.0]), None);
    let err = result.expect_err("expected WeightCountMismatch error, got Ok");
    match err {
        BuildError::WeightCountMismatch { .. } => {}
        other => panic!("Expected WeightCountMismatch, got: {:?}", other),
    }
}

#[test]
fn test_empty_categories_via_solve() {
    let cats = Array2::<u32>::zeros((0, 2));
    let y: Vec<f64> = vec![];
    let params = LsmrOptions::default();
    let precond = PreconditionerConfig::default();
    let result = solve(cats.view(), &y, None, &params, &precond);
    assert!(result.is_err());
    match result.unwrap_err() {
        WithinError::Build(BuildError::EmptyObservations) => {}
        other => panic!(
            "Expected Build(EmptyObservations) via solve(), got: {:?}",
            other
        ),
    }
}

#[test]
fn test_preconditioner_dimension_mismatch_error() {
    // Build a preconditioner against a larger design, then try to reuse it
    // with a smaller design — the dim check in Solver::new should fire.
    let big = Array2::from_shape_vec((4, 2), vec![0u32, 0, 1, 1, 2, 0, 3, 1]).expect("big array");
    let small = Array2::from_shape_vec((3, 2), vec![0u32, 0, 1, 1, 0, 0]).expect("small array");

    let big_solver = Solver::new(big.view(), None, None).expect("big solver");
    let prebuilt = big_solver
        .preconditioner()
        .expect("default solver has a preconditioner")
        .clone();

    let result = Solver::new(small.view(), None, prebuilt);
    let err = result.expect_err("expected PreconditionerDimensionMismatch, got Ok");
    match err {
        BuildError::PreconditionerDimensionMismatch {
            expected,
            actual_rows,
            actual_cols,
        } => {
            assert_ne!(expected, actual_rows);
            assert_eq!(actual_rows, actual_cols);
        }
        other => panic!("Expected PreconditionerDimensionMismatch, got: {:?}", other),
    }
}

#[test]
fn test_solver_accepts_slope_bearing_design_alongside_other_terms() {
    let levels = [0u32, 1, 0, 1];
    let slope = [1.0, 2.0, 3.0, 4.0];
    let effects = vec![
        within::Effect::new(&levels, true, []).expect("plain effect"),
        within::Effect::new(&levels, true, [&slope[..]]).expect("slope effect"),
    ];
    // Cross-factor routing (#61): slope terms alongside other terms build.
    let design = Design::new(effects).expect("slope design builds");
    Solver::new(design, None, None).expect("signed routing builds");
}

// ---------------------------------------------------------------------------
// Error-type plumbing: one wiring check per enum (From conversions and the
// transparent source() forward), not per-message pinning.
// ---------------------------------------------------------------------------

#[test]
fn test_within_error_from_build_error() {
    let inner = BuildError::EmptyObservations;
    let e: WithinError = inner.into();
    match e {
        WithinError::Build(BuildError::EmptyObservations) => {}
        other => panic!("expected Build(EmptyObservations), got: {:?}", other),
    }
}

#[test]
fn test_within_error_from_solve_error() {
    let inner = SolveError::Synchronization { context: "test" };
    let e: WithinError = inner.into();
    match e {
        WithinError::Solve(_) => {}
        other => panic!("expected Solve, got: {:?}", other),
    }
}

#[test]
fn test_within_error_source_chains_through_transparent_wrapper() {
    // Transparent: WithinError -> (BuildError::Preconditioner via #[source]) -> schwarz_precond::BuildError
    let inner = schwarz_precond::BuildError::ScratchSizeTooSmall {
        scratch_size: 1,
        required_min: 2,
    };
    let e = WithinError::Build(BuildError::Preconditioner(inner));
    assert!(e.source().is_some());
}
