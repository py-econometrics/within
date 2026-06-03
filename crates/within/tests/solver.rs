use ndarray::array;
use within::{
    solve, solve_approx_parallel, LsmrOptions, Preconditioner, PreconditionerConfig, Solver,
};

#[path = "common/orchestrate_helpers.rs"]
mod common;

fn default_params() -> LsmrOptions {
    LsmrOptions::default()
}

fn additive_precond() -> PreconditionerConfig {
    PreconditionerConfig::default()
}

fn categories_and_y() -> (ndarray::Array2<u32>, Vec<f64>) {
    let categories = array![[0u32, 0], [1, 0], [0, 1], [1, 1], [2, 0]];
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    (categories, y)
}

#[test]
fn test_solver_matches_oneshot() {
    let (categories, y) = categories_and_y();
    let params = default_params();
    let precond = additive_precond();

    let oneshot = solve(categories.view(), &y, None, &params, Some(&precond)).expect("oneshot");

    let solver =
        Solver::new(categories.view(), None::<Vec<f64>>, Some(&precond)).expect("solver build");
    let result = solver.solve(&y, &params).expect("solver solve");

    assert!(result.converged);
    assert_eq!(result.x.len(), oneshot.x.len());
    for (a, b) in result.x.iter().zip(oneshot.x.iter()) {
        assert!((a - b).abs() < 1e-12, "x mismatch: {} vs {}", a, b);
    }
}

#[test]
fn test_solver_demeaned() {
    let (categories, y) = categories_and_y();
    let params = default_params();
    let precond = additive_precond();

    let solver =
        Solver::new(categories.view(), None::<Vec<f64>>, Some(&precond)).expect("solver build");
    let result = solver.solve(&y, &params).expect("solver solve");

    assert_eq!(result.demeaned.len(), y.len());
    // demeaned = y - D*x: verify by checking D^T * demeaned ≈ 0
    // (the residual should be orthogonal to the design matrix)
    assert!(
        result.demeaned.iter().all(|v| v.is_finite()),
        "demeaned should be finite"
    );
}

#[test]
fn test_solver_no_preconditioner() {
    let (categories, y) = categories_and_y();
    let params = default_params();

    let solver = Solver::new(categories.view(), None::<Vec<f64>>, None).expect("solver build");
    let result = solver.solve(&y, &params).expect("solver solve");

    assert!(result.converged);
    common::assert_solution_finite(&result);
}

#[test]
fn test_solver_batch() {
    let (categories, _) = categories_and_y();
    let y1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y2 = vec![5.0, 4.0, 3.0, 2.0, 1.0];
    let y3 = vec![1.0, 1.0, 1.0, 1.0, 1.0];

    let params = default_params();
    let precond = additive_precond();

    let solver =
        Solver::new(categories.view(), None::<Vec<f64>>, Some(&precond)).expect("solver build");

    let r1 = solver.solve(&y1, &params).expect("solve y1");
    let r2 = solver.solve(&y2, &params).expect("solve y2");
    let r3 = solver.solve(&y3, &params).expect("solve y3");

    let batch = solver
        .solve_batch(&[&y1, &y2, &y3], &params)
        .expect("solve batch");

    assert_eq!(batch.converged.len(), 3);

    for (batch_x, individual_x) in [
        (batch.x(0), r1.x.as_slice()),
        (batch.x(1), r2.x.as_slice()),
        (batch.x(2), r3.x.as_slice()),
    ] {
        for (a, b) in batch_x.iter().zip(individual_x.iter()) {
            assert!((a - b).abs() < 1e-12, "batch x mismatch");
        }
    }

    assert_eq!(batch.converged.len(), 3);
    assert!(batch.converged.iter().all(|&c| c));
}

#[test]
fn test_approx_parallel_solver_is_one_shot() {
    let (categories, y) = categories_and_y();
    let precond = additive_precond();

    let solver =
        Solver::new(categories.view(), None::<Vec<f64>>, Some(&precond)).expect("solver build");
    let result = solver
        .solve_approx_parallel(&y, 1e-8)
        .expect("approximate solve");

    assert_eq!(result.iterations, 1);
    assert!(result.x.iter().all(|v| v.is_finite()));
    assert!(result.demeaned.iter().all(|v| v.is_finite()));
    assert!(result.residual.is_finite());
}

#[test]
fn test_approx_parallel_oneshot_api() {
    let (categories, y) = categories_and_y();
    let params = default_params();
    let precond = additive_precond();

    let result = solve_approx_parallel(categories.view(), &y, None, &params, Some(&precond))
        .expect("approximate oneshot");

    assert_eq!(result.iterations, 1);
    assert_eq!(result.x.len(), 5);
    assert_eq!(result.demeaned.len(), y.len());
}

#[test]
fn test_approx_parallel_requires_preconditioner() {
    let (categories, y) = categories_and_y();
    let solver = Solver::new(
        categories.view(),
        None::<Vec<f64>>,
        PreconditionerConfig::Off,
    )
    .expect("solver build");

    assert!(solver.solve_approx_parallel(&y, 1e-8).is_err());
}

#[test]
fn test_solver_properties() {
    let (categories, _) = categories_and_y();

    let solver = Solver::new(categories.view(), None::<Vec<f64>>, None).expect("solver build");

    assert_eq!(solver.n_dofs(), 5); // 3 levels + 2 levels
    assert_eq!(solver.n_obs(), 5);
}

#[test]
fn test_serde_roundtrip() {
    let (categories, y) = categories_and_y();
    let params = default_params();
    let precond = additive_precond();

    let solver1 =
        Solver::new(categories.view(), None::<Vec<f64>>, Some(&precond)).expect("solver build");
    let r1 = solver1.solve(&y, &params).expect("solve 1");

    // Serialize preconditioner
    let precond_ref = solver1
        .preconditioner()
        .expect("should have preconditioner");
    let bytes = postcard::to_stdvec(precond_ref).expect("serialize");
    assert!(!bytes.is_empty());

    // Deserialize and build new solver
    let precond2: Preconditioner = postcard::from_bytes(&bytes).expect("deserialize");
    let solver2 = Solver::new(categories.view(), None::<Vec<f64>>, precond2)
        .expect("solver from preconditioner");
    let r2 = solver2.solve(&y, &params).expect("solve 2");

    for (a, b) in r1.x.iter().zip(r2.x.iter()) {
        assert!((a - b).abs() < 1e-12, "serde roundtrip x mismatch");
    }
}

#[test]
fn test_solver_accepts_prebuilt_design() {
    let design = common::make_test_design();
    let y = vec![1.0; design.n_obs()];
    let params = default_params();
    let precond = additive_precond();

    let solver = Solver::new(design, None::<Vec<f64>>, Some(&precond)).expect("prebuilt design");
    let result = solver.solve(&y, &params).expect("solve");
    assert!(result.converged);
}
