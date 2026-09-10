//! Which thread emits what: a global subscriber sees every thread, unlike a scoped one.

use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::thread::{self, ThreadId};

use tracing::span::{Attributes, Id, Record};
use tracing::{Event, Metadata, Subscriber};
use within::{LsmrOptions, Solver};

/// Every event and span callback, tagged with the thread that made it.
#[derive(Default)]
struct Recorder {
    events: Mutex<Vec<(ThreadId, String)>>,
    span_callbacks: Mutex<Vec<ThreadId>>,
    spans: AtomicUsize,
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
        self.span_callbacks
            .lock()
            .unwrap()
            .push(thread::current().id());
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
    fn enter(&self, _: &Id) {
        self.span_callbacks
            .lock()
            .unwrap()
            .push(thread::current().id());
    }
    fn exit(&self, _: &Id) {
        self.span_callbacks
            .lock()
            .unwrap()
            .push(thread::current().id());
    }
}

static RECORDER: std::sync::LazyLock<std::sync::Arc<Recorder>> =
    std::sync::LazyLock::new(Default::default);

fn problem() -> (ndarray::Array2<u32>, Vec<f64>) {
    let n = 600;
    let mut categories = ndarray::Array2::<u32>::zeros((n, 3));
    let mut y = Vec::with_capacity(n);
    let mut state = 12345u64;
    for i in 0..n {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        categories[[i, 0]] = (state >> 33) as u32 % 40;
        categories[[i, 1]] = (state >> 17) as u32 % 25;
        categories[[i, 2]] = (state >> 5) as u32 % 7;
        y.push(((state >> 40) as f64 / (1u64 << 24) as f64) - 0.5);
    }
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
    {
        let events = RECORDER.events.lock().unwrap();
        let foreign: Vec<_> = events.iter().filter(|(t, _)| *t != caller).collect();
        assert!(
            foreign.is_empty(),
            "single solve emitted off-thread: {foreign:?}"
        );
        let iterations = events
            .iter()
            .filter(|(_, m)| m == "schwarz_precond::lsmr")
            .count();
        assert_eq!(iterations, result.iterations);
        assert_eq!(events.iter().filter(|(_, m)| m == "solving").count(), 1);
        assert!(RECORDER
            .span_callbacks
            .lock()
            .unwrap()
            .iter()
            .all(|t| *t == caller));
    }
    RECORDER.events.lock().unwrap().clear();
    RECORDER.span_callbacks.lock().unwrap().clear();

    // Called from outside the global pool, so every RHS solve runs on a worker, not here.
    let rhs = [&y[..], &y2[..]];
    solver
        .solve_batch(&rhs, &LsmrOptions::default())
        .expect("batch");
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
    assert_eq!(events.iter().filter(|(_, m)| m == "solved").count(), 2);
    assert_eq!(
        events
            .iter()
            .filter(|(_, m)| m == "schwarz_precond::lsmr")
            .count(),
        0
    );
    assert!(RECORDER
        .span_callbacks
        .lock()
        .unwrap()
        .iter()
        .all(|t| *t == caller));
}
