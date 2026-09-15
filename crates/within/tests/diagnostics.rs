//! The diagnostics channel is bit-neutral and reports one record per iteration.

use std::sync::Arc;

use within::{LsmrOptions, Solver};

#[path = "common/diagnostics_helpers.rs"]
mod common;
use common::{problem, Recorder};

const ITERATION: &str = "schwarz_precond::lsmr";

fn solve_once() -> within::SolveResult {
    let (categories, y) = problem();
    let solver = Solver::new(categories.view(), None, None).expect("build");
    solver.solve(&y, &LsmrOptions::default()).expect("solve")
}

// One test: a scoped subscriber caches callsites as never when another thread hits them first.
#[test]
fn trace_subscriber_sees_every_record_and_leaves_the_solution_bitwise_identical() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("single-thread pool");
    let quiet = pool.install(solve_once);
    let recorder = Arc::new(Recorder::default());
    let observed = pool.install(|| tracing::subscriber::with_default(recorder.clone(), solve_once));

    let bits = |x: &[f64]| x.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    assert!(quiet.iterations > 1, "problem must take several iterations");
    assert_eq!(quiet.iterations, observed.iterations);
    assert_eq!(quiet.converged, observed.converged);
    assert_eq!(quiet.residual.to_bits(), observed.residual.to_bits());
    assert_eq!(bits(&quiet.x), bits(&observed.x));
    assert_eq!(bits(&quiet.demeaned), bits(&observed.demeaned));

    assert_eq!(
        (
            recorder.count(ITERATION),
            recorder.count("solved"),
            recorder.count("preconditioner built"),
            recorder.count("design"),
        ),
        (observed.iterations, 1, 1, 1)
    );
}
