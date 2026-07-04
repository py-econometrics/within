//! Standalone profiling harness for the LSMR iteration hot loop.
//!
//! Builds the `Solver` (and thus the preconditioner) ONCE, then runs many
//! `solve()` calls in a tight loop so a sampling profiler (samply, perf,
//! Instruments) sees the steady-state iteration loop, not the one-time
//! preconditioner build.
//!
//! Usage:
//!   cargo build --profile profiling --example profile_hot_loop -p within
//!   samply record ./target/profiling/examples/profile_hot_loop [n_obs] [n_fe] [s|d] [n_solves]
//!
//! Defaults: n_obs=1_000_000, n_fe=3, dgp=d(ifficult), n_solves=30.

use std::hint::black_box;
use std::time::Instant;

use within::config::{LocalSolverConfig, LsmrOptions, PreconditionerConfig, ReductionStrategy};
use within::Solver;

#[path = "../benches/shared/fixest_dgp.rs"]
mod fixest_dgp;
use fixest_dgp::generate_fixest_like_case;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n_obs: usize = args
        .get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1_000_000);
    let n_fe: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(3);
    let difficult = args.get(3).map(|s| s.starts_with('d')).unwrap_or(true);
    let n_solves: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(30);

    eprintln!(
        "case: n_obs={n_obs} n_fe={n_fe} dgp={} n_solves={n_solves}",
        if difficult { "difficult" } else { "simple" }
    );

    let (design, y) = generate_fixest_like_case(n_obs, n_fe, difficult, 0xC0FFEE);
    eprintln!("n_dofs={}", design.n_dofs());

    let params = LsmrOptions {
        tol: 1e-6,
        maxiter: 200,
        ..Default::default()
    };
    let precond = PreconditionerConfig::Additive {
        local_solver: LocalSolverConfig::default(),
        reduction: ReductionStrategy::Auto,
    };

    let t_build = Instant::now();
    let solver = Solver::new(design, None, &precond).expect("solver build");
    eprintln!(
        "preconditioner build: {:.3}s",
        t_build.elapsed().as_secs_f64()
    );

    // Warmup + report convergence regime.
    let r = solver.solve(&y, &params).expect("solve");
    eprintln!(
        "warmup: iterations={} converged={} residual={:.2e} time_solve={:.4}s time_setup={:.4}s",
        r.iterations, r.converged, r.residual, r.time_solve, r.time_setup
    );

    // Profiled region: pure iteration hot loop (preconditioner reused).
    let t_loop = Instant::now();
    let mut acc = 0.0f64;
    for _ in 0..n_solves {
        let r = solver.solve(&y, &params).expect("solve");
        acc += black_box(r.x[0]);
    }
    let elapsed = t_loop.elapsed().as_secs_f64();
    eprintln!(
        "loop: {n_solves} solves in {elapsed:.3}s => {:.4}s/solve (acc={acc:.6})",
        elapsed / n_solves as f64
    );
}
