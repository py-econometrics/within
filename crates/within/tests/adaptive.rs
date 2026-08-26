//! Behavior of the `Adaptive` (diagonal→Schwarz) preconditioner strategy (#260).

use within::config::{LocalSolverConfig, ReductionStrategy};
use within::{BuildWarning, Design, Effect, LsmrOptions, PreconditionerConfig, Solver, Staleness};

#[path = "common/orchestrate_helpers.rs"]
mod common;

/// Escalates after any single non-vanishing contraction, so a handoff is deterministic.
fn eager_stall() -> Staleness {
    Staleness::try_new(1, 0.0).expect("valid staleness")
}

fn adaptive(stall: Staleness) -> PreconditionerConfig {
    PreconditionerConfig::Adaptive {
        local_solver: LocalSolverConfig::default(),
        reduction: ReductionStrategy::Auto,
        stall,
    }
}

fn params() -> LsmrOptions {
    LsmrOptions {
        tol: 1e-8,
        maxiter: 1000,
        ..Default::default()
    }
}

#[test]
fn the_escalated_answer_matches_a_cold_additive_solve() {
    let cats = common::test_categories();
    let y = common::make_deterministic_y(&common::make_test_design());

    for weights in [None, Some(vec![1.0, 2.0, 1.5, 0.5, 3.0])] {
        let ladder = Solver::new(
            common::make_design(cats.clone()).expect("design"),
            weights.clone(),
            adaptive(eager_stall()),
        )
        .expect("solver");
        let escalated = ladder.solve(&y, &params()).expect("adaptive solve");
        let cold = Solver::new(
            common::make_design(cats.clone()).expect("design"),
            weights.clone(),
            PreconditionerConfig::default(),
        )
        .expect("solver");
        let reference = cold.solve(&y, &params()).expect("additive solve");

        // Visible in the solver's lifecycle...
        assert!(ladder.has_escalated(), "eager stall must hand off");
        assert!(!cold.has_escalated());

        // ...invisible in the answer: the fitted values `Dx` agree with a cold Schwarz solve.
        // Raw coefficients agree only up to `null(D)` — a warm-started resume shares the unique
        // fitted values but its min-norm correction picks a different null-space representative.
        common::assert_converged_with_small_residual(&escalated, 1e-6);
        common::assert_normal_equations_satisfied(&cats, weights.as_deref(), &y, &escalated, 1e-6);
        common::assert_solutions_close(&escalated.demeaned, &reference.demeaned, 1e-6);
    }
}

#[test]
fn without_a_target_it_behaves_like_diagonal() {
    // A single factor has no cross-factor pair, so there is nothing to escalate to.
    let design = || common::make_design(vec![vec![0, 1, 2, 0, 1, 2]]).expect("design");
    let y = common::make_deterministic_y(&design());

    let solver = Solver::new(design(), None, adaptive(eager_stall())).expect("solver");
    // Settled at construction, so even the first solve is plain diagonal rather than a probe.
    assert_eq!(
        solver.preconditioner().expect("a base map").variant_name(),
        "Diagonal"
    );
    let first = solver.solve(&y, &params()).expect("first solve");
    let second = solver.solve(&y, &params()).expect("second solve");
    let diagonal = Solver::new(design(), None, &PreconditionerConfig::Diagonal)
        .expect("solver")
        .solve(&y, &params())
        .expect("diagonal solve");

    assert!(
        !solver.has_escalated(),
        "no factor-pair target means no escalation"
    );
    // No probe means no restart: the answer and the iteration count match `Diagonal` outright.
    assert_eq!(first.iterations, diagonal.iterations);
    assert_eq!(second.iterations, diagonal.iterations);
    common::assert_solutions_close(&first.demeaned, &diagonal.demeaned, 1e-6);
    common::assert_solutions_close(&second.demeaned, &diagonal.demeaned, 1e-6);
}

#[test]
fn total_iteration_budget_is_honored_across_rungs() {
    let y = common::make_deterministic_y(&common::make_test_design());
    let budget = LsmrOptions {
        tol: 1e-12,
        maxiter: 5,
        ..Default::default()
    };
    let result = Solver::new(common::make_test_design(), None, adaptive(eager_stall()))
        .expect("solver")
        .solve(&y, &budget)
        .expect("solve");

    assert!(
        result.iterations <= budget.maxiter,
        "ladder used {} iterations, exceeding the budget of {}",
        result.iterations,
        budget.maxiter
    );
}

#[test]
fn batch_escalation_builds_schwarz_once() {
    let y0 = common::make_deterministic_y(&common::make_test_design());
    let y1: Vec<f64> = y0.iter().map(|v| v * 0.5 + 0.3).collect();

    let solver =
        Solver::new(common::make_test_design(), None, adaptive(eager_stall())).expect("solver");
    let result = solver
        .solve_batch(&[&y0[..], &y1[..]], &params())
        .expect("batch solve");

    assert!(result.converged.iter().all(|&c| c));
    assert!(solver.has_escalated());
    // The single deferred build has landed, so the active preconditioner is Schwarz.
    assert_eq!(
        solver.preconditioner().expect("built").variant_name(),
        "Additive"
    );
}

/// Design screening runs at construction, so its warnings must survive the deferred
/// Schwarz build rather than being replaced by it (#260 over #283).
#[test]
fn screening_warnings_survive_escalation() {
    let n = 4000;
    let a: Vec<u32> = (0..n).map(|i| (i % 40) as u32).collect();
    let b: Vec<u32> = (0..n).map(|i| ((i / 40) % 25) as u32).collect();
    let z: Vec<f64> = (0..n).map(|i| (i as f64 * 0.17 + 1.0).sin()).collect();
    let design = Design::new(vec![
        Effect::new(&a, true, [&z[..]]).unwrap(),
        Effect::new(&b, true, [&z[..]]).unwrap(),
    ])
    .unwrap();
    let collinear = |w: &[BuildWarning]| {
        w.iter()
            .filter(|w| matches!(w, BuildWarning::CollinearSlopeCovariate { .. }))
            .count()
    };

    let solver = Solver::new(design, None, adaptive(eager_stall())).expect("solver");
    let before = collinear(solver.warnings());
    assert!(
        before > 0,
        "shared covariate must warn: {:?}",
        solver.warnings()
    );

    let y: Vec<f64> = (0..n).map(|i| z[i] + (i % 7) as f64).collect();
    let _ = solver.solve(&y, &params()).expect("adaptive solve");
    assert!(solver.has_escalated(), "eager stall must hand off");
    assert_eq!(
        collinear(solver.warnings()),
        before,
        "escalation dropped screening warnings: {:?}",
        solver.warnings()
    );
}

/// `PreconditionerConfig` is persisted, so `Staleness` must survive the wire — and its
/// hand-written validating decoder must still pair with the derived encoder.
#[test]
fn adaptive_config_round_trips_and_rejects_an_invalid_stall() {
    let config = adaptive(Staleness::try_new(3, 0.25).expect("valid staleness"));
    let bytes = postcard::to_stdvec(&config).expect("serialize");
    let restored: PreconditionerConfig = postcard::from_bytes(&bytes).expect("deserialize");
    assert_eq!(restored, config);

    // `stall` is the variant's last field, so the tail is `window` (varint) then `threshold` (f64).
    let mut corrupt = bytes;
    let window = corrupt.len() - 9;
    assert_eq!(
        corrupt[window], 3,
        "expected the encoded window at the tail"
    );
    corrupt[window] = 0;
    assert!(
        postcard::from_bytes::<PreconditionerConfig>(&corrupt).is_err(),
        "a zero window must not decode into a Staleness that never escalates"
    );
}

/// Once the map is built the ladder is over: later solves must cost exactly what a cold
/// Schwarz solver costs, not a fresh diagonal stall streak plus a handoff.
#[test]
fn later_solves_cost_the_same_as_a_cold_schwarz_solver() {
    // Crossed workers/firms with periodic mobility: the diagonal needs enough iterations
    // for a re-probe to be visible, and Schwarz converges in strictly fewer.
    let n = 20_000;
    let workers: Vec<u32> = (0..n).map(|i| (i / 8) as u32).collect();
    let firms: Vec<u32> = (0..n)
        .map(|i| ((i / 8 + i % 8 * 977) % 500) as u32)
        .collect();
    let design = || common::make_design(vec![workers.clone(), firms.clone()]).expect("design");
    let y = common::make_deterministic_y(&design());
    let lsmr = LsmrOptions {
        tol: 1e-10,
        maxiter: 5000,
        ..Default::default()
    };

    // A wide window makes the wasted streak long, so a re-probe cannot hide in the noise.
    let stall = Staleness::try_new(8, 0.5).expect("valid staleness");
    let solver = Solver::new(design(), None, adaptive(stall)).expect("solver");
    let escalating = solver.solve(&y, &lsmr).expect("escalating solve");
    assert!(solver.has_escalated(), "the ladder must hand off");
    let later = solver.solve(&y, &lsmr).expect("later solve");

    let cold = Solver::new(design(), None, PreconditionerConfig::default())
        .expect("solver")
        .solve(&y, &lsmr)
        .expect("cold schwarz solve");

    assert_eq!(
        later.iterations, cold.iterations,
        "a built ladder re-probed the diagonal: {} iterations vs a cold {}",
        later.iterations, cold.iterations
    );
    assert!(
        later.iterations < escalating.iterations,
        "the escalating solve should be the expensive one ({} vs {})",
        escalating.iterations,
        later.iterations
    );
}

/// `time_setup` is this call's build, not the slot's lifetime record: a batch that reuses an
/// already-built rung did no setup, and must not re-report the original build's cost.
#[test]
fn batch_setup_time_is_not_recharged_on_later_batches() {
    let cats = common::test_categories();
    let ys: Vec<Vec<f64>> = (0..3)
        .map(|k| (0..cats[0].len()).map(|i| ((i + k) as f64).sin()).collect())
        .collect();
    let refs: Vec<&[f64]> = ys.iter().map(Vec::as_slice).collect();

    let solver = Solver::new(
        common::make_design(cats).expect("design"),
        None,
        adaptive(eager_stall()),
    )
    .expect("solver");

    let escalating = solver.solve_batch(&refs, &params()).expect("first batch");
    assert!(solver.has_escalated(), "eager stall must hand off");
    assert!(
        escalating.time_setup > 0.0,
        "the batch that built the rung must charge for it"
    );

    let later = solver.solve_batch(&refs, &params()).expect("second batch");
    assert_eq!(
        later.time_setup, 0.0,
        "a batch that built nothing reported {} s of setup",
        later.time_setup
    );
}

/// Escalating on the last permitted iteration leaves rung 2 no budget, so the map must not
/// be built at all — a factorization for zero iterations is pure waste.
#[test]
fn a_stall_on_the_final_iteration_does_not_build() {
    let y = common::make_deterministic_y(&common::make_test_design());
    // This design's eager stall fires on iteration 2, so that is exactly the budget to grant.
    let exhausted = LsmrOptions {
        tol: 1e-14,
        maxiter: 2,
        ..Default::default()
    };

    let solver =
        Solver::new(common::make_test_design(), None, adaptive(eager_stall())).expect("solver");
    let result = solver.solve(&y, &exhausted).expect("solve");

    assert_eq!(result.iterations, 2, "the budget should be spent on rung 1");
    assert!(
        !solver.has_escalated(),
        "no budget remained, so nothing should have been built"
    );
}
