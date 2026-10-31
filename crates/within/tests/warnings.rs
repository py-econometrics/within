//! One-shot `solve`/`solve_batch` entry points surface preconditioner
//! [`BuildWarning`]s, not just [`Solver::warnings`] (#165).

use within::{
    solve, solve_batch, BuildWarning, Effect, LocalSolverConfig, PreconditionerConfig,
    ScalingConfig, ScalingFailure, Solver,
};

// Two crossed slope-only factors: level a0's cross entries sum to 4 > its diagonal 2,
// so the block needs a dominance scaling. A zero-tolerance / zero-iteration policy reports
// that as an uncertified-scaling warning deterministically, without relying on
// the (deliberately rare) default-tolerance path.
const LA: [u32; 4] = [0, 0, 1, 1];
const LB: [u32; 4] = [0, 1, 0, 1];
const ZA: [f64; 4] = [1.0, 1.0, 1.0, 1.0];
const ZB: [f64; 4] = [3.0, -1.0, 2.0, -4.0];

fn warning_effects() -> Vec<Effect<'static>> {
    vec![
        Effect::new(&LA, false, [&ZA[..]]).unwrap(),
        Effect::new(&LB, false, [&ZB[..]]).unwrap(),
    ]
}

fn has_scaling_warning(warnings: &[BuildWarning]) -> bool {
    warnings
        .iter()
        .any(|w| matches!(w, BuildWarning::UnscalableComponent { .. }))
}

#[test]
fn free_solve_surfaces_build_warnings() {
    let y = [1.0, 2.0, 0.5, -1.0];
    let cfg = scaling(0.0, ScalingFailure::Warn);

    let result = solve(warning_effects(), &y, None, None, &cfg).expect("solve");
    assert!(
        has_scaling_warning(&result.warnings),
        "free solve dropped the preconditioner build warning: {:?}",
        result.warnings,
    );

    // The result field mirrors the solver accessor exactly.
    let solver = Solver::new(warning_effects(), None, &cfg).expect("solver");
    assert_eq!(result.warnings.as_slice(), solver.warnings());
}

#[test]
fn free_solve_batch_surfaces_build_warnings() {
    let y0 = [1.0, 2.0, 0.5, -1.0];
    let y1 = [0.2, -0.7, 1.3, 0.4];
    let ys: [&[f64]; 2] = [&y0, &y1];
    let cfg = scaling(0.0, ScalingFailure::Warn);

    let result = solve_batch(warning_effects(), &ys, None, None, &cfg).expect("solve_batch");
    assert!(
        has_scaling_warning(&result.warnings),
        "free solve_batch dropped the preconditioner build warning: {:?}",
        result.warnings,
    );

    // The result field mirrors the solver accessor exactly.
    let solver = Solver::new(warning_effects(), None, &cfg).expect("solver");
    assert_eq!(result.warnings.as_slice(), solver.warnings());
}

fn scaling(tolerance: f64, on_failure: ScalingFailure) -> PreconditionerConfig {
    PreconditionerConfig::Additive {
        local_solver: LocalSolverConfig {
            scaling: ScalingConfig {
                tolerance,
                max_iterations: 0,
                on_failure,
            },
            ..LocalSolverConfig::default()
        },
    }
}

// The fixture's unscaled violation is about 0.078.
#[rstest::rstest]
#[case::within_tolerance(0.1, true)]
#[case::past_tolerance(0.05, false)]
fn only_a_violation_past_tolerance_fails_the_build(#[case] tolerance: f64, #[case] builds: bool) {
    let built = Solver::new(
        warning_effects(),
        None,
        scaling(tolerance, ScalingFailure::Error),
    );
    assert_eq!(built.is_ok(), builds);
}
