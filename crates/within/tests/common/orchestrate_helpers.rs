#![allow(dead_code)]

use ndarray::Array2;
use within::config::{LocalSolverConfig, ReductionStrategy, Staleness};
use within::observation::ObservationFrame;
use within::{Design, PreconditionerConfig, SolveResult};

/// One-level additive Schwarz, pinned so a test names the rung the ladder would escalate to.
pub fn additive() -> PreconditionerConfig {
    PreconditionerConfig::Additive {
        local_solver: LocalSolverConfig::default(),
        reduction: ReductionStrategy::default(),
    }
}

/// The diagonal→Schwarz ladder, named rather than reached through `default()`.
pub fn adaptive() -> PreconditionerConfig {
    PreconditionerConfig::Adaptive {
        local_solver: LocalSolverConfig::default(),
        reduction: ReductionStrategy::default(),
        stall: Staleness::default(),
    }
}

/// The canonical 2-factor, 5-observation categorical structure used across the
/// orchestration tests. `categories[f][i]` is observation `i`'s level in factor `f`.
pub fn test_categories() -> Vec<Vec<u32>> {
    vec![vec![0, 1, 0, 1, 2], vec![0, 0, 1, 1, 0]]
}

/// [`test_categories`] as the observation-major `(n_obs, n_factors)` matrix the
/// high-level `solve` / `solve_batch` entry points consume.
pub fn test_categories_array() -> Array2<u32> {
    let cats = test_categories();
    let (n_factors, n_obs) = (cats.len(), cats[0].len());
    Array2::from_shape_fn((n_obs, n_factors), |(i, f)| cats[f][i])
}

pub fn make_test_design() -> Design<'static> {
    make_design(test_categories()).expect("valid test design")
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

/// Independent optimality oracle for a purely categorical fixed-effects design:
/// recomputes the relative normal-equation residual `‖DᵀW(y−Dx)‖ / ‖DᵀWy‖` from
/// `result.demeaned` (`y − Dx`, produced by a matvec separate from the LSMR
/// recurrence) and the raw categories. Unlike `result.residual`, which is now the
/// solver's own stopping estimate, this cannot be satisfied by the solver merely
/// reporting convergence. `categories[f][i]` is observation `i`'s level in factor `f`.
pub fn assert_normal_equations_satisfied(
    categories: &[Vec<u32>],
    weights: Option<&[f64]>,
    y: &[f64],
    result: &SolveResult,
    tol: f64,
) {
    // ‖DᵀW v‖² = Σ_f Σ_level (Σ_{i in level} w_i v_i)²  for a categorical D.
    let dtw_sq_norm = |v: &[f64]| -> f64 {
        let mut acc = 0.0;
        for levels in categories {
            let n_levels = levels.iter().copied().max().map_or(0, |m| m as usize + 1);
            let mut sums = vec![0.0f64; n_levels];
            for (i, &level) in levels.iter().enumerate() {
                let w = weights.map_or(1.0, |ws| ws[i]);
                sums[level as usize] += w * v[i];
            }
            acc += sums.iter().map(|s| s * s).sum::<f64>();
        }
        acc
    };

    let numerator = dtw_sq_norm(&result.demeaned).sqrt();
    let denominator = dtw_sq_norm(y).sqrt().max(1e-15);
    let relative = numerator / denominator;
    assert!(
        relative < tol,
        "independent normal-equation residual {relative} exceeds {tol} \
         (‖DᵀW(y−Dx)‖ = {numerator}, ‖DᵀWy‖ = {denominator})"
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
