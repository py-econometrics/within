//! Which thread emits what: a global subscriber sees every thread, unlike a scoped one.

use std::collections::HashSet;
use std::sync::{Arc, LazyLock};
use std::thread;

use within::{LsmrOptions, Solver};

#[path = "common/diagnostics_helpers.rs"]
mod common;
use common::{problem, Recorder};

static RECORDER: LazyLock<Arc<Recorder>> = LazyLock::new(Default::default);

fn off_thread(caller: thread::ThreadId) -> HashSet<String> {
    RECORDER
        .events
        .lock()
        .unwrap()
        .iter()
        .filter(|(t, _)| *t != caller)
        .map(|(_, m)| m.clone())
        .collect()
}

#[test]
fn workers_emit_only_the_per_rhs_completion_record() {
    tracing::subscriber::set_global_default(RECORDER.clone())
        .expect("first and only global subscriber");
    let (categories, y) = problem();
    let y2: Vec<f64> = y.iter().map(|v| 2.0 * v).collect();
    let caller = thread::current().id();

    let solver = Solver::new(categories.view(), None, None).expect("build");
    let result = solver.solve(&y, &LsmrOptions::default()).expect("solve");
    // Domain builds run on the pool; everything else in a single solve stays here.
    assert!(off_thread(caller).is_subset(&HashSet::from(["domain".to_owned()])));
    assert_eq!(RECORDER.count("schwarz_precond::lsmr"), result.iterations);
    assert_eq!(RECORDER.count("solving"), 1);
    RECORDER.clear();

    // Called from outside the global pool, so every RHS solve runs on a worker, not here.
    solver
        .solve_batch(&[&y[..], &y2[..]], &LsmrOptions::default())
        .expect("batch");
    assert!(off_thread(caller).is_subset(&HashSet::from(["solved".to_owned()])));
    assert_eq!(RECORDER.count("solved"), 2);
    assert_eq!(RECORDER.count("schwarz_precond::lsmr"), 0);
}
