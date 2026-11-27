//! Times of the grouping probe, on the system allocator.
#[path = "shared/probe.rs"]
mod probe;
fn main() {
    probe::run(|| 0, || 0);
}
