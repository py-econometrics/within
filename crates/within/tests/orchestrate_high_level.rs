use ndarray::array;
use within::{solve, solve_batch, LsmrOptions, PreconditionerConfig};

#[path = "common/orchestrate_helpers.rs"]
mod common;

#[test]
fn test_high_level_solve() {
    let categories = array![[0u32, 0], [1, 0], [0, 1], [1, 1], [2, 0]];
    let y = [1.0, 2.0, 3.0, 4.0, 5.0];

    let params = LsmrOptions::default();
    let precond = PreconditionerConfig::default();
    let result = solve(categories.view(), &y, None, &params, &precond).expect("solve");
    common::assert_converged_with_small_residual(&result, 1e-6);
    common::assert_solution_finite(&result);
}

#[test]
fn test_high_level_solve_weighted() {
    let categories = array![[0u32, 0], [1, 0], [0, 1], [1, 1], [2, 0]];
    let y = [1.0, 2.0, 3.0, 4.0, 5.0];
    let weights = vec![1.0, 2.0, 1.5, 0.5, 3.0];

    let params = LsmrOptions::default();
    let precond = PreconditionerConfig::default();
    let result =
        solve(categories.view(), &y, Some(&weights), &params, &precond).expect("solve weighted");
    common::assert_converged_with_small_residual(&result, 1e-6);
    common::assert_solution_finite(&result);
}

#[test]
fn test_solve_batch_matches_individual() {
    let categories = array![[0u32, 0], [1, 0], [0, 1], [1, 1], [2, 0]];
    let y1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y2 = vec![5.0, 4.0, 3.0, 2.0, 1.0];

    let params = LsmrOptions::default();
    let precond = PreconditionerConfig::default();

    let r1 = solve(categories.view(), &y1, None, &params, &precond).expect("solve y1");
    let r2 = solve(categories.view(), &y2, None, &params, &precond).expect("solve y2");

    let batch =
        solve_batch(categories.view(), &[&y1, &y2], None, &params, &precond).expect("solve batch");

    assert_eq!(batch.converged.len(), 2);
    for (a, b) in batch.x(0).iter().zip(r1.x.iter()) {
        assert!((a - b).abs() < 1e-12, "batch vs individual x mismatch");
    }
    for (a, b) in batch.x(1).iter().zip(r2.x.iter()) {
        assert!((a - b).abs() < 1e-12, "batch vs individual x mismatch");
    }
}

#[test]
fn test_solve_batch_single_rhs() {
    let categories = array![[0u32, 0], [1, 0], [0, 1], [1, 1], [2, 0]];
    let y = [1.0, 2.0, 3.0, 4.0, 5.0];

    let params = LsmrOptions::default();
    let precond = PreconditionerConfig::default();

    let batch = solve_batch(categories.view(), &[&y[..]], None, &params, &precond)
        .expect("solve batch single");

    assert_eq!(batch.converged.len(), 1);
    assert!(batch.converged[0]);
    assert!(batch.x(0).iter().all(|v| v.is_finite()));
}

#[test]
fn test_solve_batch_weighted() {
    let categories = array![[0u32, 0], [1, 0], [0, 1], [1, 1], [2, 0]];
    let y1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y2 = vec![5.0, 4.0, 3.0, 2.0, 1.0];
    let weights = vec![1.0, 2.0, 1.5, 0.5, 3.0];

    let params = LsmrOptions::default();
    let precond = PreconditionerConfig::default();

    let batch = solve_batch(
        categories.view(),
        &[&y1, &y2],
        Some(&weights),
        &params,
        &precond,
    )
    .expect("solve batch weighted");

    assert_eq!(batch.converged.len(), 2);
    assert!(batch.converged.iter().all(|&c| c));
}

#[test]
fn test_batch_result_accessors() {
    let categories = array![[0u32, 0], [1, 0], [0, 1], [1, 1], [2, 0]];
    let y1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y2 = vec![5.0, 4.0, 3.0, 2.0, 1.0];

    let params = LsmrOptions::default();
    let precond = PreconditionerConfig::default();

    let batch =
        solve_batch(categories.view(), &[&y1, &y2], None, &params, &precond).expect("solve batch");

    // n_rhs
    assert_eq!(batch.converged.len(), 2);

    // x_all length
    let n_dofs = batch.x(0).len();
    assert_eq!(batch.x.len(), 2 * n_dofs);

    // demeaned accessor
    let n_obs = 5;
    assert_eq!(batch.demeaned(0).len(), n_obs);
    assert_eq!(batch.demeaned(1).len(), n_obs);
    assert_eq!(batch.demeaned.len(), 2 * n_obs);

    // converged
    assert_eq!(batch.converged.len(), 2);
    assert!(batch.converged.iter().all(|&c| c));

    // iterations
    assert_eq!(batch.iterations.len(), 2);

    // residual
    assert_eq!(batch.residual.len(), 2);
    assert!(batch.residual.iter().all(|&r| r.is_finite() && r >= 0.0));

    // time_solve
    assert_eq!(batch.time_solve.len(), 2);
    assert!(batch.time_solve.iter().all(|&t| t >= 0.0));

    // time_total
    assert!(batch.time_total >= 0.0);

    // All values finite
    assert!(batch.x.iter().all(|v| v.is_finite()));
    assert!(batch.demeaned.iter().all(|v| v.is_finite()));
}
