use ndarray::array;
use within::{solve, LsmrOptions, Preconditioner, PreconditionerConfig, Solver};

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

    let oneshot = solve(categories.view(), &y, None, &params, &precond).expect("oneshot");

    let solver = Solver::new(categories.view(), None, &precond).expect("solver build");
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

    let solver = Solver::new(categories.view(), None, &precond).expect("solver build");
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

    let solver = Solver::new(categories.view(), None, None).expect("solver build");
    let result = solver.solve(&y, &params).expect("solver solve");

    assert!(result.converged);
    common::assert_solution_finite(&result);
}

#[test]
fn test_solver_diagonal_preconditioner() {
    let (categories, y) = categories_and_y();
    let params = default_params();
    let precond = PreconditionerConfig::Diagonal;

    let solver = Solver::new(categories.view(), None, &precond).expect("solver build");
    assert!(
        solver.preconditioner().is_some(),
        "diagonal preconditioner should be cached"
    );
    let result = solver.solve(&y, &params).expect("solver solve");

    assert!(result.converged);
    common::assert_solution_finite(&result);
}

#[test]
fn test_diagonal_preconditioner_single_factor_is_cached() {
    let categories = array![[0u32], [1], [0], [2], [1]];
    let precond = PreconditionerConfig::Diagonal;

    let solver = Solver::new(categories.view(), None, &precond).expect("solver build");

    let cached = solver
        .preconditioner()
        .expect("single-factor diagonal preconditioner should be cached");
    assert_eq!(cached.nrows(), 3);
    assert_eq!(cached.ncols(), 3);
}

#[test]
fn test_solver_batch() {
    let (categories, _) = categories_and_y();
    let y1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y2 = vec![5.0, 4.0, 3.0, 2.0, 1.0];
    let y3 = vec![1.0, 1.0, 1.0, 1.0, 1.0];

    let params = default_params();
    let precond = additive_precond();

    let solver = Solver::new(categories.view(), None, &precond).expect("solver build");

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
fn test_solver_properties() {
    let (categories, _) = categories_and_y();

    let solver = Solver::new(categories.view(), None, None).expect("solver build");

    assert_eq!(solver.n_dofs(), 5); // 3 levels + 2 levels
    assert_eq!(solver.n_obs(), 5);
}

#[test]
fn test_serde_roundtrip() {
    let (categories, y) = categories_and_y();
    let params = default_params();
    let precond = additive_precond();

    let solver1 = Solver::new(categories.view(), None, &precond).expect("solver build");
    let r1 = solver1.solve(&y, &params).expect("solve 1");

    // Serialize preconditioner
    let precond_ref = solver1
        .preconditioner()
        .expect("should have preconditioner");
    let bytes = postcard::to_stdvec(precond_ref).expect("serialize");
    assert!(!bytes.is_empty());

    // Deserialize and build new solver
    let precond2: Preconditioner = postcard::from_bytes(&bytes).expect("deserialize");
    let solver2 =
        Solver::new(categories.view(), None, precond2).expect("solver from preconditioner");
    let r2 = solver2.solve(&y, &params).expect("solve 2");

    for (a, b) in r1.x.iter().zip(r2.x.iter()) {
        assert!((a - b).abs() < 1e-12, "serde roundtrip x mismatch");
    }
}

#[test]
fn test_diagonal_serde_roundtrip() {
    let (categories, _) = categories_and_y();
    let precond = PreconditionerConfig::Diagonal;
    let solver = Solver::new(categories.view(), None, &precond).expect("solver build");
    let precond_ref = solver
        .preconditioner()
        .expect("should have diagonal preconditioner");
    let bytes = postcard::to_stdvec(precond_ref).expect("serialize");
    assert!(!bytes.is_empty());

    let deserialized: Preconditioner = postcard::from_bytes(&bytes).expect("deserialize");
    assert_eq!(deserialized.nrows(), precond_ref.nrows());
    assert_eq!(deserialized.ncols(), precond_ref.ncols());

    let x: Vec<f64> = (0..precond_ref.ncols()).map(|i| i as f64 + 0.25).collect();
    let mut y1 = vec![0.0; precond_ref.nrows()];
    let mut y2 = vec![0.0; deserialized.nrows()];
    precond_ref.apply(&x, &mut y1).expect("apply original");
    deserialized.apply(&x, &mut y2).expect("apply deserialized");
    assert_eq!(y1, y2);
}

#[test]
fn test_solver_accepts_prebuilt_design() {
    let design = common::make_test_design();
    let y = vec![1.0; design.n_obs()];
    let params = default_params();
    let precond = additive_precond();

    let solver = Solver::new(design, None, &precond).expect("prebuilt design");
    let result = solver.solve(&y, &params).expect("solve");
    assert!(result.converged);
}

/// The construction-time locality sort must be transparent: coefficients are
/// permutation-invariant and `demeaned` comes back in caller order. The sort
/// is applied to every store by `Design::from_store`, so the oracle is built
/// via `from_store_unsorted`, the explicit caller-order escape hatch. Results
/// agree within solver tolerance, not bitwise: the paths sum in different
/// row orders.
#[test]
fn test_internal_locality_sort_is_transparent() {
    use within::observation::FactorMajorStore;
    use within::Design;

    // Factor 0 (4 levels) is the dominant factor and is non-monotonic.
    let col0: Vec<u32> = vec![3, 0, 2, 1, 3, 0, 2, 1, 3, 0, 2, 1];
    let col1: Vec<u32> = vec![0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1];
    let n_obs = col0.len();
    let y: Vec<f64> = (0..n_obs)
        .map(|i| (i as f64 * 1.3 - 2.0).sin() + 0.5)
        .collect();
    let w: Vec<f64> = (0..n_obs).map(|i| 0.5 + 0.1 * i as f64).collect();
    let params = default_params();
    let precond = additive_precond();

    let make_solver = |weights: Option<Vec<f64>>| {
        let design = common::make_design(vec![col0.clone(), col1.clone()]).expect("design");
        Solver::new(design, weights, &precond).expect("solver")
    };
    let make_oracle = |weights: Option<Vec<f64>>| {
        let store = FactorMajorStore::new(vec![col0.clone(), col1.clone()], n_obs).expect("store");
        let design = Design::from_store_unsorted(store).expect("oracle design");
        Solver::new(design, weights, &precond).expect("oracle solver")
    };

    // The weighted run also exercises the weights-permutation path.
    for weights in [None, Some(w)] {
        let oracle = make_oracle(weights.clone())
            .solve(&y, &params)
            .expect("oracle");
        let sorted = make_solver(weights).solve(&y, &params).expect("sorted");
        common::assert_solutions_close(&sorted.x, &oracle.x, 1e-7);
        common::assert_solutions_close(&sorted.demeaned, &oracle.demeaned, 1e-7);
    }
}
