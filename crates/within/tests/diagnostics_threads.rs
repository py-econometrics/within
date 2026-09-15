//! Which thread emits what: a global subscriber sees every thread, unlike a scoped one.

use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex};
use std::thread::{self, ThreadId};

use tracing::span::{Attributes, Id, Record};
use tracing::{Event, Metadata, Subscriber};
use within::{LsmrOptions, Solver};

/// Every event, tagged with the emitting thread and labelled by its message (its target if none).
#[derive(Default)]
struct Recorder {
    events: Mutex<Vec<(ThreadId, String)>>,
    spans: AtomicUsize,
}

impl Recorder {
    fn count(&self, label: &str) -> usize {
        self.events
            .lock()
            .unwrap()
            .iter()
            .filter(|(_, m)| m == label)
            .count()
    }

    fn off_thread(&self, caller: ThreadId) -> HashSet<String> {
        let events = self.events.lock().unwrap();
        events
            .iter()
            .filter(|(t, _)| *t != caller)
            .map(|(_, m)| m.clone())
            .collect()
    }
}

struct Message(Option<String>);

impl tracing::field::Visit for Message {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            self.0 = Some(format!("{value:?}"));
        }
    }
}

impl Subscriber for Recorder {
    fn enabled(&self, _: &Metadata<'_>) -> bool {
        true
    }
    fn new_span(&self, _: &Attributes<'_>) -> Id {
        Id::from_u64(1 + self.spans.fetch_add(1, Ordering::Relaxed) as u64)
    }
    fn record(&self, _: &Id, _: &Record<'_>) {}
    fn record_follows_from(&self, _: &Id, _: &Id) {}
    fn event(&self, event: &Event<'_>) {
        let mut message = Message(None);
        event.record(&mut message);
        let label = message
            .0
            .unwrap_or_else(|| event.metadata().target().to_owned());
        self.events
            .lock()
            .unwrap()
            .push((thread::current().id(), label));
    }
    fn enter(&self, _: &Id) {}
    fn exit(&self, _: &Id) {}
}

static RECORDER: LazyLock<Arc<Recorder>> = LazyLock::new(Default::default);

/// Three crossed factors with a deterministic pattern: takes several LSMR iterations.
fn problem() -> (ndarray::Array2<u32>, Vec<f64>) {
    let n = 600;
    let categories = ndarray::Array2::from_shape_fn((n, 3), |(i, f)| {
        [(i % 40) as u32, (i * 7 % 25) as u32, (i * 13 % 7) as u32][f]
    });
    let y = (0..n).map(|i| (i * 31 % 17) as f64 - 8.0).collect();
    (categories, y)
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
    // Domain factors are built on the pool; everything else in a single solve stays here.
    let pool_only = HashSet::from(["reduced factor".to_owned()]);
    assert!(RECORDER.off_thread(caller).is_subset(&pool_only));
    assert!(
        result.iterations > 1,
        "problem must take several iterations"
    );
    assert_eq!(RECORDER.count("schwarz_precond::lsmr"), result.iterations);
    assert_eq!(RECORDER.count("solved"), 1);
    RECORDER.events.lock().unwrap().clear();

    // Called from outside the global pool, so every RHS solve runs on a worker, not here.
    solver
        .solve_batch(&[&y[..], &y2[..]], &LsmrOptions::default())
        .expect("batch");
    let worker_only = HashSet::from(["solved".to_owned()]);
    assert!(RECORDER.off_thread(caller).is_subset(&worker_only));
    assert_eq!(RECORDER.count("solved"), 2);
    assert_eq!(RECORDER.count("schwarz_precond::lsmr"), 0);
}
