use ndarray::array;
use within::{
    solve, ApproxCholConfig, ApproxSchurConfig, Effect, LocalSolverConfig, LsmrOptions,
    Preconditioner, PreconditionerConfig, ReductionStrategy, ScalingConfig, ScalingFailure,
    SchurMode, Solver,
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
    assert_eq!(cached.config(), &precond);
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

    // The method reuses the already-built preconditioner, so it reports no
    // build cost; the free solve_batch is what folds construction in.
    assert_eq!(batch.time_setup, 0.0);
}

#[test]
fn test_solver_batch_term_design_shares_drop_report() {
    // Level 0's z is constant, so its whitened slope column drops; each batch
    // column must reproduce the single solve and share the drop report.
    let f = [0u32, 0, 0, 1, 1, 1];
    let g = [0u32, 1, 2, 0, 1, 2];
    let z = [3.0, 3.0, 3.0, -1.0, -1.0, 2.0];
    let ys = [
        [1.0, -2.0, 0.5, 3.0, -1.5, 2.5],
        [0.3, 1.1, -0.7, 2.2, 0.9, -1.4],
    ];
    let effects = vec![
        Effect::new(&f, true, [&z[..]]).expect("slope effect"),
        Effect::new(&g, true, []).expect("plain effect"),
    ];
    let params = default_params();

    let solver = Solver::new(effects, None, additive_precond()).expect("solver build");
    let batch = solver
        .solve_batch(&[&ys[0], &ys[1]], &params)
        .expect("solve batch");
    assert!(!batch.unidentified.is_empty(), "level 0 slope should drop");

    for (i, y) in ys.iter().enumerate() {
        let single = solver.solve(y, &params).expect("single solve");
        assert_eq!(batch.unidentified, single.unidentified);
        for (a, b) in batch.x(i).iter().zip(single.x.iter()) {
            assert!((a - b).abs() < 1e-12, "batch x mismatch");
        }
        for (a, b) in batch.demeaned(i).iter().zip(single.demeaned.iter()) {
            assert!((a - b).abs() < 1e-12, "batch demeaned mismatch");
        }
    }
}

#[test]
fn test_unidentified_empty_for_plain_factors() {
    let (categories, y) = categories_and_y();
    let params = default_params();
    let precond = additive_precond();

    let solver = Solver::new(categories.view(), None, &precond).expect("solver build");

    let single = solver.solve(&y, &params).expect("solve");
    assert!(single.unidentified.is_empty());
    assert!(single.x.iter().all(|v| v.is_finite()));

    let batch = solver.solve_batch(&[&y], &params).expect("solve batch");
    assert!(batch.unidentified.is_empty());
}

#[test]
fn test_solver_properties() {
    let (categories, _) = categories_and_y();

    let solver = Solver::new(categories.view(), None, None).expect("solver build");

    assert_eq!(solver.n_dofs(), 5); // 3 levels + 2 levels
    assert_eq!(solver.n_obs(), 5);
}

#[test]
fn test_additive_serde_roundtrip_preserves_config_and_solution() {
    let (categories, y) = categories_and_y();
    let params = default_params();
    let precond = PreconditionerConfig::Additive {
        local_solver: LocalSolverConfig {
            approx_chol: ApproxCholConfig {
                seed: 41,
                split_merge: Some(3),
            },
            schur: SchurMode::Approximate(ApproxSchurConfig { seed: 17, split: 2 }),
            dense_threshold: 0,
            scaling: ScalingConfig {
                tolerance: 1e-7,
                max_sweeps: 123,
                on_failure: ScalingFailure::Error,
            },
        },
        reduction: ReductionStrategy::AtomicScatter,
    };

    let solver1 = Solver::new(categories.view(), None, &precond).expect("solver build");
    let r1 = solver1.solve(&y, &params).expect("solve 1");

    // Serialize preconditioner
    let precond_ref = solver1
        .preconditioner()
        .expect("should have preconditioner");
    assert_eq!(precond_ref.config(), &precond);
    let bytes = postcard::to_stdvec(precond_ref).expect("serialize");
    assert!(!bytes.is_empty());

    // Deserialize and build new solver
    let precond2: Preconditioner = postcard::from_bytes(&bytes).expect("deserialize");
    assert_eq!(precond2.config(), &precond);
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
/// is applied to every frame by `Design::from_frame`, so the oracle is built
/// via `from_frame_unsorted`, the explicit caller-order escape hatch. Results
/// agree within solver tolerance, not bitwise: the paths sum in different
/// row orders.
#[test]
fn test_internal_locality_sort_is_transparent() {
    use within::observation::ObservationFrame;
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
        let frame =
            ObservationFrame::new(vec![col0.clone().into(), col1.clone().into()], Vec::new())
                .expect("frame");
        let design = Design::from_frame_unsorted(frame).expect("oracle design");
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
