#![allow(dead_code)]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::thread::{self, ThreadId};

use tracing::span::{Attributes, Id, Record};
use tracing::{Event, Metadata, Subscriber};

/// Every event and span callback, tagged with the thread that made it.
/// An event is labelled by its message, or by its target when it has none.
#[derive(Default)]
pub struct Recorder {
    pub events: Mutex<Vec<(ThreadId, String)>>,
    pub span_callbacks: Mutex<Vec<ThreadId>>,
    spans: AtomicUsize,
}

impl Recorder {
    pub fn count(&self, label: &str) -> usize {
        self.events
            .lock()
            .unwrap()
            .iter()
            .filter(|(_, m)| m == label)
            .count()
    }

    pub fn clear(&self) {
        self.events.lock().unwrap().clear();
        self.span_callbacks.lock().unwrap().clear();
    }

    fn span_callback(&self) {
        self.span_callbacks
            .lock()
            .unwrap()
            .push(thread::current().id());
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
        self.span_callback();
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
        self.span_callback();
    }
    fn exit(&self, _: &Id) {
        self.span_callback();
    }
}

/// Three crossed factors with a deterministic congruential pattern: takes several LSMR iterations.
pub fn problem() -> (ndarray::Array2<u32>, Vec<f64>) {
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
