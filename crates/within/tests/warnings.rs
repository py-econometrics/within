//! One-shot `solve`/`solve_batch` entry points surface preconditioner
//! [`BuildWarning`]s, not just [`Solver::warnings`] (#165).

use within::{
    solve, solve_batch, BuildWarning, Design, Effect, LocalSolverConfig, PreconditionerConfig,
    ReductionStrategy, ScalingConfig, ScalingFailure, Solver,
};

// Two crossed slope-only factors: their signed slope cross-block is not
// diagonally dominant. A zero-tolerance / zero-iteration scaling policy reports
// that as an uncertified-scaling warning deterministically, without relying on
// the (deliberately rare) default-tolerance path.
const LA: [u32; 4] = [0, 0, 1, 1];
const LB: [u32; 4] = [0, 1, 0, 1];
const ZA: [f64; 4] = [1.0, 1.0, 1.0, 1.0];
const ZB: [f64; 4] = [1.0, -1.0, 2.0, -2.0];

fn warning_effects() -> Vec<Effect<'static>> {
    vec![
        Effect::new(&LA, false, [&ZA[..]]).unwrap(),
        Effect::new(&LB, false, [&ZB[..]]).unwrap(),
    ]
}

fn strict_scaling() -> PreconditionerConfig {
    PreconditionerConfig::Additive {
        local_solver: LocalSolverConfig {
            scaling: ScalingConfig {
                tolerance: 0.0,
                max_iterations: 0,
                on_failure: ScalingFailure::Warn,
            },
            ..LocalSolverConfig::default()
        },
        reduction: ReductionStrategy::default(),
    }
}

fn has_scaling_warning(warnings: &[BuildWarning]) -> bool {
    warnings
        .iter()
        .any(|w| matches!(w, BuildWarning::UnscalableComponent { .. }))
}

#[test]
fn free_solve_surfaces_build_warnings() {
    let y = [1.0, 2.0, 0.5, -1.0];
    let cfg = strict_scaling();

    let result = solve(warning_effects(), &y, None, None, &cfg).expect("solve");
    assert!(
        has_scaling_warning(&result.warnings),
        "free solve dropped the preconditioner build warning: {:?}",
        result.warnings,
    );

    // The result field mirrors the solver accessor exactly.
    let design = Design::new(warning_effects()).expect("design");
    let solver = Solver::new(&design, None, &cfg).expect("solver");
    assert_eq!(result.warnings.as_slice(), solver.warnings());
}

#[test]
fn free_solve_batch_surfaces_build_warnings() {
    let y0 = [1.0, 2.0, 0.5, -1.0];
    let y1 = [0.2, -0.7, 1.3, 0.4];
    let ys: [&[f64]; 2] = [&y0, &y1];
    let cfg = strict_scaling();

    let result = solve_batch(warning_effects(), &ys, None, None, &cfg).expect("solve_batch");
    assert!(
        has_scaling_warning(&result.warnings),
        "free solve_batch dropped the preconditioner build warning: {:?}",
        result.warnings,
    );

    // The result field mirrors the solver accessor exactly.
    let design = Design::new(warning_effects()).expect("design");
    let solver = Solver::new(&design, None, &cfg).expect("solver");
    assert_eq!(result.warnings.as_slice(), solver.warnings());
}
