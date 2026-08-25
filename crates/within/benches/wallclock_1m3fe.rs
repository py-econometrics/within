//! Tier-2 wall-clock harness (#147): times the large 1M-row 3-way-FE solve and
//! prints the best of N runs. `best-of-N` rejects one-sided contention noise —
//! runner contention can only slow a run, so the minimum is the cleanest
//! throughput estimate. Multi-threaded on purpose (default Rayon width): this
//! tier is the multi-threaded safety net, not the deterministic path.
//!
//! Mirrors the `LSMR-AC` variant of the `fixest` Criterion smoke bench so the
//! committed reference stays comparable. Prints `WALLCLOCK_BEST_NS=<ns>` for
//! `scripts/check-wallclock-reference.py` to consume.

use std::hint::black_box;
use std::time::Instant;

use within::config::{LocalSolverConfig, LsmrOptions, PreconditionerConfig, ReductionStrategy};
use within::Solver;

#[path = "shared/fixest_dgp.rs"]
mod fixest_dgp;
use fixest_dgp::generate_fixest_like_case;

const N_OBS: usize = 1_000_000;
const N_FE: usize = 3;
const SEED: u64 = 42;
const RUNS: usize = 5;
const MAXITER: usize = 200;
const TOL: f64 = 1e-6;

fn main() {
    let (design, y) = generate_fixest_like_case(N_OBS, N_FE, true, SEED);
    let params = LsmrOptions {
        tol: TOL,
        maxiter: MAXITER,
        ..Default::default()
    };
    let precond = PreconditionerConfig::Additive {
        local_solver: LocalSolverConfig::default(),
        reduction: ReductionStrategy::Auto,
    };

    let mut best = u128::MAX;
    for _ in 0..RUNS {
        let start = Instant::now();
        let solver = Solver::new(&design, None, &precond).expect("solver build");
        let result = solver.solve(&y, &params).expect("solve");
        best = best.min(start.elapsed().as_nanos());
        black_box(&result);
    }

    println!("WALLCLOCK_BEST_NS={best}");
}
