#![allow(dead_code)]

use within::observation::ObservationFrame;
use within::{Design, SolveResult};

pub fn make_test_design() -> Design<'static> {
    make_design(vec![vec![0, 1, 0, 1, 2], vec![0, 0, 1, 1, 0]]).expect("valid test design")
}

pub fn make_design(categories: Vec<Vec<u32>>) -> Result<Design<'static>, within::BuildError> {
    let frame =
        ObservationFrame::new(categories.into_iter().map(Into::into).collect(), Vec::new())?;
    Design::from_frame(frame)
}

/// Deterministic, non-trivial RHS sized to the design's observation count.
/// Used to drive convergence assertions where the exact x is irrelevant.
pub fn make_deterministic_y(design: &Design<'_>) -> Vec<f64> {
    (0..design.n_obs())
        .map(|i| (i as f64 * 0.17 + 1.0).sin())
        .collect()
}

pub fn assert_converged_with_small_residual(result: &SolveResult, tol: f64) {
    assert!(result.converged, "solver did not converge");
    assert!(
        result.residual < tol,
        "residual too large: {}",
        result.residual
    );
}

pub fn assert_solution_finite(result: &SolveResult) {
    assert!(
        result.x.iter().all(|v| v.is_finite()),
        "Non-finite solution"
    );
}

/// Assert two solution vectors agree element-wise within `tol`.
pub fn assert_solutions_close(a: &[f64], b: &[f64], tol: f64) {
    assert_eq!(a.len(), b.len(), "solution lengths differ");
    for (i, (&ai, &bi)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (ai - bi).abs() <= tol,
            "solutions differ at index {i}: {ai} vs {bi} (tol {tol})"
        );
    }
}
