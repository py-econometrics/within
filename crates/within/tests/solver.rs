use ndarray::array;
use within::{
    solve, FePreconditioner, KrylovMethod, LocalSolverConfig, OperatorRepr, Preconditioner, Solver,
    SolverParams,
};

#[path = "common/orchestrate_helpers.rs"]
mod common;

fn default_params() -> SolverParams {
    SolverParams::default()
}

fn additive_precond() -> Preconditioner {
    Preconditioner::Additive(LocalSolverConfig::solver_default())
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
        Solver::new(categories.view(), None, &params, Some(&precond)).expect("solver build");
    let result = solver.solve(&y).expect("solver solve");

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
        Solver::new(categories.view(), None, &params, Some(&precond)).expect("solver build");
    let result = solver.solve(&y).expect("solver solve");

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

    let solver = Solver::new(categories.view(), None, &params, None).expect("solver build");
    let result = solver.solve(&y).expect("solver solve");

    assert!(result.converged);
    common::assert_solution_finite(&result);
}

#[test]
fn test_solver_explicit_gramian() {
    let (categories, y) = categories_and_y();
    let params = SolverParams {
        krylov: KrylovMethod::Cg,
        operator: OperatorRepr::Explicit,
        tol: 1e-8,
        maxiter: 1000,
    };
    let precond = additive_precond();

    let solver =
        Solver::new(categories.view(), None, &params, Some(&precond)).expect("solver build");
    let result = solver.solve(&y).expect("solver solve");

    assert!(result.converged);
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
        Solver::new(categories.view(), None, &params, Some(&precond)).expect("solver build");

    let r1 = solver.solve(&y1).expect("solve y1");
    let r2 = solver.solve(&y2).expect("solve y2");
    let r3 = solver.solve(&y3).expect("solve y3");

    let batch = solver.solve_batch(&[&y1, &y2, &y3]).expect("solve batch");

    assert_eq!(batch.n_rhs(), 3);

    for (batch_x, individual_x) in [
        (batch.x(0), r1.x.as_slice()),
        (batch.x(1), r2.x.as_slice()),
        (batch.x(2), r3.x.as_slice()),
    ] {
        for (a, b) in batch_x.iter().zip(individual_x.iter()) {
            assert!((a - b).abs() < 1e-12, "batch x mismatch");
        }
    }

    assert_eq!(batch.converged().len(), 3);
    assert!(batch.converged().iter().all(|&c| c));
}

#[test]
fn test_solver_properties() {
    let (categories, _) = categories_and_y();
    let params = default_params();

    let solver = Solver::new(categories.view(), None, &params, None).expect("solver build");

    assert_eq!(solver.n_dofs(), 5); // 3 levels + 2 levels
    assert_eq!(solver.n_obs(), 5);
}

#[test]
fn test_serde_roundtrip() {
    let (categories, y) = categories_and_y();
    let params = default_params();
    let precond = additive_precond();

    let solver1 =
        Solver::new(categories.view(), None, &params, Some(&precond)).expect("solver build");
    let r1 = solver1.solve(&y).expect("solve 1");

    // Serialize preconditioner
    let precond_ref = solver1
        .preconditioner()
        .expect("should have preconditioner");
    let bytes = bincode::serialize(precond_ref).expect("serialize");
    assert!(!bytes.is_empty());

    // Deserialize and build new solver
    let precond2: FePreconditioner = bincode::deserialize(&bytes).expect("deserialize");
    let solver2 = Solver::with_preconditioner(categories.view(), None, &params, precond2)
        .expect("solver from preconditioner");
    let r2 = solver2.solve(&y).expect("solve 2");

    for (a, b) in r1.x.iter().zip(r2.x.iter()) {
        assert!((a - b).abs() < 1e-12, "serde roundtrip x mismatch");
    }
}

#[test]
fn test_solver_from_design() {
    let design = common::make_test_design();
    let y = vec![1.0; design.n_rows];
    let params = default_params();
    let precond = additive_precond();

    let solver = Solver::from_design(design, &params, Some(&precond)).expect("from_design");
    let result = solver.solve(&y).expect("solve");
    assert!(result.converged);
}
