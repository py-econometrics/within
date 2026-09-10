//! Which thread emits what: a global subscriber sees every thread, unlike a scoped one.

use std::collections::HashSet;
use std::sync::{Arc, LazyLock};
use std::thread;

use within::{LsmrOptions, Solver};

#[path = "common/diagnostics_helpers.rs"]
mod common;
use common::{problem, Recorder};

static RECORDER: LazyLock<Arc<Recorder>> = LazyLock::new(Default::default);

#[test]
fn workers_emit_only_the_per_rhs_completion_record() {
    tracing::subscriber::set_global_default(RECORDER.clone())
        .expect("first and only global subscriber");
    let (categories, y) = problem();
    let y2: Vec<f64> = y.iter().map(|v| 2.0 * v).collect();
    let caller = thread::current().id();
    let all_on_caller = || {
        RECORDER
            .span_callbacks
            .lock()
            .unwrap()
            .iter()
            .all(|t| *t == caller)
    };

    let solver = Solver::new(categories.view(), None, None).expect("build");
    let result = solver.solve(&y, &LsmrOptions::default()).expect("solve");
    {
        let events = RECORDER.events.lock().unwrap();
        let foreign: Vec<_> = events.iter().filter(|(t, _)| *t != caller).collect();
        assert!(
            foreign.is_empty(),
            "single solve emitted off-thread: {foreign:?}"
        );
    }
    assert_eq!(RECORDER.count("schwarz_precond::lsmr"), result.iterations);
    assert_eq!(RECORDER.count("solving"), 1);
    assert!(all_on_caller());
    RECORDER.clear();

    // Called from outside the global pool, so every RHS solve runs on a worker, not here.
    let rhs = [&y[..], &y2[..]];
    solver
        .solve_batch(&rhs, &LsmrOptions::default())
        .expect("batch");
    {
        let events = RECORDER.events.lock().unwrap();
        let off_thread: HashSet<&str> = events
            .iter()
            .filter(|(t, _)| *t != caller)
            .map(|(_, m)| m.as_str())
            .collect();
        assert!(
            off_thread.is_empty() || off_thread == HashSet::from(["solved"]),
            "workers emitted {off_thread:?}"
        );
    }
    assert_eq!(RECORDER.count("solved"), 2);
    assert_eq!(RECORDER.count("schwarz_precond::lsmr"), 0);
    assert!(all_on_caller());
}
