use std::error::Error;

use ndarray::Array2;
use schwarz_precond::SolveError;
use within::observation::ObservationFrame;
use within::{
    solve, solve_batch, BuildError, Design, Effect, LsmrOptions, PreconditionerConfig, Solver,
    WithinError,
};

// The Display/source()/From plumbing has one wiring check per enum, not per message.

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
    // Reusing a larger design's preconditioner must trip the dim check in Solver::new.
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
fn test_non_finite_response_rejected() {
    let cats = Array2::from_shape_vec((3, 1), vec![0u32, 1, 2]).expect("cats");
    let params = LsmrOptions::default();
    let precond = PreconditionerConfig::default();

    let y = [1.0, f64::NAN, 3.0];
    match solve(cats.view(), &y, None, &params, &precond).unwrap_err() {
        WithinError::Solve(SolveError::InvalidInput { message, .. }) => {
            assert!(
                message.contains("index 1"),
                "message names the index: {message}"
            );
        }
        other => panic!("Expected Solve(InvalidInput) via solve(), got: {other:?}"),
    }

    // The persistent Solver API funnels through the same guard, so it cannot be bypassed.
    let solver = Solver::new(cats.view(), None, &precond).expect("solver");
    match solver.solve(&y, &params).unwrap_err() {
        SolveError::InvalidInput { message, .. } => {
            assert!(
                message.contains("index 1"),
                "message names the index: {message}"
            );
        }
        other => panic!("Expected InvalidInput via Solver::solve(), got: {other:?}"),
    }

    // solve_batch funnels every column through Solver::solve; the bad value is in the 2nd RHS.
    let good = [1.0, 2.0, 3.0];
    let bad = [1.0, 2.0, f64::INFINITY];
    match solve_batch(cats.view(), &[&good[..], &bad[..]], None, &params, &precond).unwrap_err() {
        WithinError::Solve(SolveError::InvalidInput { message, .. }) => {
            assert!(
                message.contains("index 2"),
                "message names the index: {message}"
            );
        }
        other => panic!("Expected Solve(InvalidInput) via solve_batch(), got: {other:?}"),
    }
}

#[test]
fn test_collinear_finite_slope_is_unidentified_not_rejected() {
    // A duplicated but FINITE slope is rank deficiency, not malformed input (#122).
    let levels = [0u32, 0, 1, 1];
    let z = [1.0, 3.0, 2.0, 5.0];
    let y = [1.0, 2.0, 3.0, 4.0];
    let effects = vec![Effect::new(&levels, true, [&z[..], &z[..]]).expect("effect")];
    let params = LsmrOptions::default();
    let precond = PreconditionerConfig::default();

    let r = solve(effects, &y, None, &params, &precond).expect("collinear-but-finite must solve");
    assert!(
        !r.unidentified.is_empty(),
        "a duplicated finite slope must report an unidentified direction"
    );
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

// One wiring check per enum: From conversions and the transparent source() forward.

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
    // Transparent: WithinError -> BuildError::Preconditioner -> schwarz_precond::BuildError
    let inner = schwarz_precond::BuildError::ScratchSizeTooSmall {
        scratch_size: 1,
        required_min: 2,
    };
    let e = WithinError::Build(BuildError::Preconditioner(inner));
    assert!(e.source().is_some());
}
