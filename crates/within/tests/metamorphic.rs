use ndarray::Array2;
use proptest::prelude::*;
use within::{solve, solve_batch, Channel, CoefficientAddress, LsmrOptions};

#[path = "common/property_strategies.rs"]
mod strategies;
use strategies::{additive_precond, random_fe_problem_strategy};

fn at(term: usize, level: usize, column: usize) -> CoefficientAddress {
    CoefficientAddress {
        channel: Channel { term, column },
        level,
    }
}

// Well-conditioned plain-FE designs, so convergence is asserted, never a silent skip.
fn tight_params() -> LsmrOptions {
    LsmrOptions {
        tol: 1e-11,
        maxiter: 3000,
        local_size: Some(10),
    }
}

/// L2 agreement with a mixed absolute+relative tolerance: returns the actual
/// discrepancy and the tolerance it must stay under. A near-saturated design
/// drives the residual (hence `expected`) toward zero, where a purely relative
/// check amplifies machine-precision noise into a spurious failure; the `atol`
/// floor absorbs that regime while `rtol` still catches real divergence.
fn l2_close(actual: &[f64], expected: &[f64]) -> (f64, f64) {
    const ATOL: f64 = 1e-9;
    const RTOL: f64 = 1e-6;
    let num = actual
        .iter()
        .zip(expected)
        .map(|(a, e)| (a - e).powi(2))
        .sum::<f64>()
        .sqrt();
    let expected_norm = expected.iter().map(|e| e * e).sum::<f64>().sqrt();
    (num, ATOL + RTOL * expected_norm)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(12))]

    /// Response-scaling equivariance: the residual `r = y − Dx` is linear in the
    /// response, so `r(c·y) = c·r(y)` for any nonzero scalar `c`.
    #[test]
    fn prop_response_scaling_equivariance(
        (cats, y) in random_fe_problem_strategy(),
        c in prop_oneof![-8.0f64..=-0.25, 0.25f64..=8.0],
    ) {
        let params = tight_params();
        let precond = additive_precond();

        let base = solve(cats.view(), &y, None, &params, &precond).unwrap();
        prop_assert!(base.converged);

        let y_scaled: Vec<f64> = y.iter().map(|v| c * v).collect();
        let scaled = solve(cats.view(), &y_scaled, None, &params, &precond).unwrap();
        prop_assert!(scaled.converged);

        let expected: Vec<f64> = base.demeaned.iter().map(|v| c * v).collect();
        let (num, tol) = l2_close(&scaled.demeaned, &expected);
        prop_assert!(
            num <= tol,
            "response-scaling equivariance violated: |Δ| = {num:.3e} > tol {tol:.3e} (c={c})"
        );
    }

    /// Uniform weight-scaling invariance: `argmin ∑ k·wᵢ rᵢ² = argmin ∑ wᵢ rᵢ²`
    /// for any `k > 0`, so scaling every weight by a constant leaves the fit —
    /// and hence the residual — unchanged.
    #[test]
    fn prop_weight_scaling_invariance(
        (cats, y, w) in random_fe_problem_strategy().prop_flat_map(|(cats, y)| {
            let n = y.len();
            (Just(cats), Just(y), proptest::collection::vec(0.2f64..3.0, n))
        }),
        k in 0.25f64..=6.0,
    ) {
        let params = tight_params();
        let precond = additive_precond();

        let base = solve(cats.view(), &y, Some(w.as_slice()), &params, &precond).unwrap();
        prop_assert!(base.converged);

        let w_scaled: Vec<f64> = w.iter().map(|v| k * v).collect();
        let scaled = solve(cats.view(), &y, Some(w_scaled.as_slice()), &params, &precond).unwrap();
        prop_assert!(scaled.converged);

        let (num, tol) = l2_close(&scaled.demeaned, &base.demeaned);
        prop_assert!(
            num <= tol,
            "weight-scaling invariance violated: |Δ| = {num:.3e} > tol {tol:.3e} (k={k})"
        );
    }

    /// `solve_batch` must agree column-for-column with independent `solve` calls
    /// on the same design: batching only shares the preconditioner, it must not
    /// change the fit of any single RHS.
    #[test]
    fn prop_batch_matches_columnwise_solve(
        (cats, ys) in random_fe_problem_strategy().prop_flat_map(|(cats, y0)| {
            let n = y0.len();
            (
                Just(cats),
                proptest::collection::vec(proptest::collection::vec(-10.0f64..10.0, n), 2..=4),
            )
        }),
    ) {
        let params = tight_params();
        let precond = additive_precond();

        let refs: Vec<&[f64]> = ys.iter().map(Vec::as_slice).collect();
        let batch = solve_batch(cats.view(), &refs, None, &params, &precond).unwrap();
        prop_assert!(batch.converged.iter().all(|&c| c));

        for (j, y) in ys.iter().enumerate() {
            let single = solve(cats.view(), y, None, &params, &precond).unwrap();
            prop_assert!(single.converged);
            let (num, tol) = l2_close(batch.demeaned(j), &single.demeaned);
            prop_assert!(
                num <= tol,
                "batch vs column-wise residual mismatch (column {j}): |Δ| = {num:.3e} > tol {tol:.3e}"
            );
        }
    }

    /// Minimal-norm gauge: first-order optimality pins every *identified*
    /// coefficient, but leaves the null-space representative free. `within`
    /// documents the minimal-norm choice — unidentified directions held at
    /// exactly `0` — so assert that directly (the equivariance and residual
    /// checks are gauge-invariant and cannot see it).
    #[test]
    fn prop_unidentified_slots_are_zero((cats, y) in random_fe_problem_strategy()) {
        let params = tight_params();
        let precond = additive_precond();
        let result = solve(cats.view(), &y, None, &params, &precond).unwrap();
        prop_assert!(result.converged);

        for u in &result.unidentified {
            let slot = result.layout.index(*u).unwrap();
            prop_assert_eq!(
                result.x[slot],
                0.0,
                "unidentified slot (term {}, level {}, col {}) = {}, expected exactly 0",
                u.channel.term,
                u.level,
                u.channel.column,
                result.x[slot]
            );
        }
    }
}

/// Known-answer oracle: a saturated single-factor design has no gauge freedom,
/// so each level's fixed effect is exactly the mean of that level's responses.
/// This pins actual coefficient VALUES at their `layout` addresses — the one
/// thing the gauge-invariant optimality and residual checks cannot verify, and
/// the cheap guard against a weighting/labeling misconception shared between the
/// solver and a self-referential oracle.
#[test]
fn saturated_single_factor_recovers_level_means() {
    // Level means: {1,3}→2, {2,4,6}→4, {5}→5.
    let cats = Array2::from_shape_vec((6, 1), vec![0u32, 0, 1, 1, 1, 2]).unwrap();
    let y = vec![1.0, 3.0, 2.0, 4.0, 6.0, 5.0];
    let params = tight_params();
    let precond = additive_precond();
    let result = solve(cats.view(), &y, None, &params, &precond).unwrap();
    assert!(result.converged);

    for (level, &mean) in [2.0, 4.0, 5.0].iter().enumerate() {
        let slot = result.layout.index(at(0, level, 0)).unwrap();
        assert!(
            (result.x[slot] - mean).abs() < 1e-6,
            "level {level}: coefficient {} != level mean {mean}",
            result.x[slot]
        );
    }
}
