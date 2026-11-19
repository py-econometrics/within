use ndarray::array;
use rstest::rstest;
use within::{
    solve, ApproxCholConfig, ApproxSchurConfig, Effect, LocalSolverConfig, LsmrOptions,
    Preconditioner, PreconditionerConfig, ScalingConfig, ScalingFailure, SchurMode, Solver,
};

#[path = "common/orchestrate_helpers.rs"]
mod common;

fn default_params() -> LsmrOptions {
    LsmrOptions::default()
}

fn categories_and_y() -> (ndarray::Array2<u32>, Vec<f64>) {
    let categories = array![[0u32, 0], [1, 0], [0, 1], [1, 1], [2, 0]];
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    (categories, y)
}

/// `Solver::new().solve()` and one-shot `solve()` agree under every preconditioner, and the
/// solver holds a map exactly when one was requested.
#[rstest]
fn test_solver_matches_oneshot(
    #[values(
        PreconditionerConfig::Off,
        PreconditionerConfig::Diagonal,
        common::additive(),
        common::adaptive()
    )]
    precond: PreconditionerConfig,
) {
    let (categories, y) = categories_and_y();
    let params = default_params();

    let oneshot = solve(categories.view(), &y, None, &params, &precond).expect("oneshot");

    let solver = Solver::new(categories.view(), None, &precond).expect("solver build");
    assert_eq!(
        solver.preconditioner().is_some(),
        precond != PreconditionerConfig::Off,
        "a map is cached exactly when a preconditioner was requested"
    );
    let result = solver.solve(&y, &params).expect("solver solve");

    assert!(result.converged);
    common::assert_solution_finite(&result);
    assert_eq!(result.x.len(), oneshot.x.len());
    for (a, b) in result.x.iter().zip(oneshot.x.iter()) {
        assert!((a - b).abs() < 1e-12, "x mismatch: {} vs {}", a, b);
    }
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
    assert_eq!(cached.config(), precond);
}

#[test]
fn test_solver_batch() {
    let (categories, _) = categories_and_y();
    let y1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y2 = vec![5.0, 4.0, 3.0, 2.0, 1.0];
    let y3 = vec![1.0, 1.0, 1.0, 1.0, 1.0];

    let params = default_params();
    let precond = PreconditionerConfig::default();

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
    // Level 0's z is constant, so its orthogonalized slope column drops; each batch
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

    let solver = Solver::new(effects, None, PreconditionerConfig::default()).expect("solver build");
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
    let precond = PreconditionerConfig::default();

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

/// Every non-default field set, so a roundtrip that silently reset one would be caught.
fn fully_specified_additive() -> PreconditionerConfig {
    PreconditionerConfig::Schwarz(within::Schwarz {
        local_solver: LocalSolverConfig {
            approx_chol: ApproxCholConfig {
                seed: 41,
                split_merge: Some(3),
            },
            schur: SchurMode::Approximate(ApproxSchurConfig { split: 2 }),
            dense_threshold: 0,
            scaling: ScalingConfig {
                tolerance: 1e-7,
                max_iterations: 123,
                on_failure: ScalingFailure::Error,
            },
        },
    })
}

/// A solved ladder hands out whichever rung it reached, so the axis is the two eager variants.
#[rstest]
#[case::diagonal(PreconditionerConfig::Diagonal)]
#[case::additive(fully_specified_additive())]
fn test_preconditioner_serde_roundtrip(#[case] precond: PreconditionerConfig) {
    let (categories, y) = categories_and_y();
    let params = default_params();

    let solver1 = Solver::new(categories.view(), None, &precond).expect("solver build");
    let r1 = solver1.solve(&y, &params).expect("solve 1");

    let precond_ref = solver1
        .preconditioner()
        .expect("should have preconditioner");
    assert_eq!(precond_ref.config(), precond);
    let bytes = postcard::to_stdvec(precond_ref).expect("serialize");
    assert!(!bytes.is_empty());

    let precond2: Preconditioner = postcard::from_bytes(&bytes).expect("deserialize");
    assert_eq!(precond2.config(), precond);
    assert_eq!(precond2.nrows(), precond_ref.nrows());
    assert_eq!(precond2.ncols(), precond_ref.ncols());
    assert_eq!(precond2.build_duration(), precond_ref.build_duration());

    let x: Vec<f64> = (0..precond_ref.ncols()).map(|i| i as f64 + 0.25).collect();
    let mut y1 = vec![0.0; precond_ref.nrows()];
    let mut y2 = vec![0.0; precond2.nrows()];
    precond_ref.apply(&x, &mut y1).expect("apply original");
    precond2.apply(&x, &mut y2).expect("apply deserialized");
    assert_eq!(y1, y2);

    let solver2 =
        Solver::new(categories.view(), None, precond2).expect("solver from preconditioner");
    assert_eq!(
        solver2
            .preconditioner()
            .expect("reused preconditioner")
            .build_duration(),
        precond_ref.build_duration(),
    );
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
    let precond = PreconditionerConfig::default();

    let solver = Solver::new(design, None, &precond).expect("prebuilt design");
    let result = solver.solve(&y, &params).expect("solve");
    assert!(result.converged);
}

/// Locality sort vs. the same rows given already sorted, a slope included: equal in caller order.
#[rstest]
#[case::unweighted(false)]
#[case::weighted(true)]
fn test_internal_locality_sort_is_transparent(#[case] weighted: bool) {
    // Factor 0 (4 levels) is the dominant factor and is non-monotonic.
    let col0: Vec<u32> = vec![3, 0, 2, 1, 3, 0, 2, 1, 3, 0, 2, 1];
    let col1: Vec<u32> = vec![0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1];
    let n_obs = col0.len();
    let y: Vec<f64> = (0..n_obs)
        .map(|i| (i as f64 * 1.3 - 2.0).sin() + 0.5)
        .collect();
    let weights: Option<Vec<f64>> =
        weighted.then(|| (0..n_obs).map(|i| 0.5 + 0.1 * i as f64).collect());
    let params = default_params();
    let precond = PreconditionerConfig::default();
    let slope: Vec<f64> = (0..n_obs).map(|i| (i as f64 * 0.7).cos() + 2.0).collect();
    let solve = |cols: Vec<Vec<u32>>, z: &[f64], y: &[f64], weights: Option<&[f64]>| {
        let design = within::Design::new([
            Effect::new(&cols[0], true, [z]).expect("effect 0"),
            Effect::new(&cols[1], true, []).expect("effect 1"),
        ])
        .expect("design");
        Solver::new(design, weights, &precond)
            .expect("solver")
            .solve(y, &params)
            .expect("solve")
    };

    let mut order: Vec<usize> = (0..n_obs).collect();
    order.sort_by_key(|&i| col0[i]);
    let gather = |v: &[u32]| order.iter().map(|&i| v[i]).collect::<Vec<_>>();
    let gather_f = |v: &[f64]| order.iter().map(|&i| v[i]).collect::<Vec<_>>();
    let presorted = solve(
        vec![gather(&col0), gather(&col1)],
        &gather_f(&slope),
        &gather_f(&y),
        weights.as_deref().map(gather_f).as_deref(),
    );
    let sorted = solve(
        vec![col0.clone(), col1.clone()],
        &slope,
        &y,
        weights.as_deref(),
    );
    let mut oracle_demeaned = vec![0.0; n_obs];
    for (k, &i) in order.iter().enumerate() {
        oracle_demeaned[i] = presorted.demeaned[k];
    }
    common::assert_solutions_close(&sorted.x, &presorted.x, 1e-7);
    common::assert_solutions_close(&sorted.demeaned, &oracle_demeaned, 1e-7);
}

/// `demeaned` is `y − D x` for the returned `x`, whichever exit produced it.
#[rstest]
fn test_demeaned_equals_residual_of_returned_x(
    #[values(
        PreconditionerConfig::Off,
        PreconditionerConfig::Diagonal,
        common::additive(),
        common::adaptive()
    )]
    precond: PreconditionerConfig,
    #[values(
        None,
        Some(vec![1.0; 5]),
        Some(vec![0.5, 2.0, 1.0, 3.0, 0.25]),
        Some(vec![0.0, 2.0, 1.0, 3.0, 0.25])
    )]
    weights: Option<Vec<f64>>,
) {
    let (categories, y) = categories_and_y();
    let result = solve(
        categories.view(),
        &y,
        weights.as_deref(),
        &default_params(),
        &precond,
    )
    .expect("solve");

    common::assert_demeaned_is_residual(categories.view(), &y, &result, 1e-12);
}
