//! The diagnostics channel: emitted from the driving thread, bit-neutral, one record per iteration.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use tracing::span::{Attributes, Id, Record};
use tracing::{Event, Metadata, Subscriber};
use within::{LsmrOptions, Solver};

/// Counts events by message, accepting every level so trace-level iteration records are seen.
#[derive(Default)]
struct Counter {
    iterations: AtomicUsize,
    solved: AtomicUsize,
    built: AtomicUsize,
    design: AtomicUsize,
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

impl Subscriber for Counter {
    fn enabled(&self, _: &Metadata<'_>) -> bool {
        true
    }
    fn new_span(&self, _: &Attributes<'_>) -> Id {
        Id::from_u64(1 + self.spans.fetch_add(1, Ordering::Relaxed) as u64)
    }
    fn record(&self, _: &Id, _: &Record<'_>) {}
    fn record_follows_from(&self, _: &Id, _: &Id) {}
    fn event(&self, event: &Event<'_>) {
        if event.metadata().target() == "schwarz_precond::lsmr" {
            self.iterations.fetch_add(1, Ordering::Relaxed);
            return;
        }
        let mut message = Message(None);
        event.record(&mut message);
        let counter = match message.0.as_deref() {
            Some("solved") => &self.solved,
            Some("preconditioner built") => &self.built,
            Some("design") => &self.design,
            _ => return,
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }
    fn enter(&self, _: &Id) {}
    fn exit(&self, _: &Id) {}
}

/// Three crossed factors with a deterministic congruential pattern: takes several LSMR iterations.
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
    let counter = Arc::new(Counter::default());
    let observed = pool.install(|| tracing::subscriber::with_default(counter.clone(), solve_once));

    let bits = |x: &[f64]| x.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    assert!(quiet.iterations > 1, "problem must take several iterations");
    assert_eq!(quiet.iterations, observed.iterations);
    assert_eq!(quiet.converged, observed.converged);
    assert_eq!(quiet.residual.to_bits(), observed.residual.to_bits());
    assert_eq!(bits(&quiet.x), bits(&observed.x));
    assert_eq!(bits(&quiet.demeaned), bits(&observed.demeaned));

    let load = |c: &AtomicUsize| c.load(Ordering::Relaxed);
    assert_eq!(
        (
            load(&counter.iterations),
            load(&counter.solved),
            load(&counter.built),
            load(&counter.design),
        ),
        (observed.iterations, 1, 1, 1)
    );

    // A batch reports once per RHS and never per iteration.
    let (categories, y) = problem();
    let solver = Solver::new(categories.view(), None, None).expect("build");
    let y2: Vec<f64> = y.iter().map(|v| 2.0 * v).collect();
    pool.install(|| {
        tracing::subscriber::with_default(counter.clone(), || {
            solver
                .solve_batch(&[&y, &y2], &LsmrOptions::default())
                .expect("batch")
        })
    });
    assert_eq!(
        (load(&counter.iterations), load(&counter.solved)),
        (observed.iterations, 3)
    );
}
