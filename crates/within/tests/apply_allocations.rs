//! A solve allocates nothing per operator or preconditioner apply (#511).

use within::{Design, Effect, LsmrOptions, PreconditionerConfig, Solver};

#[path = "common/orchestrate_helpers.rs"]
mod common;

/// Counts every allocation in the process, so this file holds one test.
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

/// Allocations of one solve capped at `maxiter`, which it must spend in full.
fn allocations(solver: &Solver<'_>, y: &[f64], maxiter: usize) -> u64 {
    let lsmr = LsmrOptions {
        tol: 1e-15,
        maxiter,
        local_size: None,
    };
    let before = dhat::HeapStats::get().total_blocks;
    let result = solver.solve(y, &lsmr).expect("solve");
    let spent = dhat::HeapStats::get().total_blocks - before;
    assert_eq!(
        result.iterations, maxiter,
        "the solve stopped before its cap"
    );
    spent
}

/// Twice the iterations, the same allocations: every apply runs on the run's stack.
#[test]
fn a_solve_allocates_nothing_per_apply() {
    let _profiler = dhat::Profiler::builder().testing().build();
    let n = 60_000;
    let worker: Vec<u32> = (0..n).map(|i| (i / 3) as u32).collect();
    let firm: Vec<u32> = (0..n).map(|i| ((i * 7_919) % 1_500) as u32).collect();
    let region: Vec<u32> = (0..n).map(|i| ((i * 104_729) % 9_000) as u32).collect();
    let tenure: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.37).sin()).collect();
    let y: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.11).cos()).collect();
    let design = || {
        Design::new([
            Effect::new(&worker, true, []).expect("worker"),
            Effect::new(&firm, true, [&tenure[..]]).expect("firm"),
            Effect::new(&region, true, []).expect("region"),
        ])
        .expect("design")
    };
    // Every pool lives through every count, so no worker exits while one is taken.
    let pools = [1, 4].map(|threads| {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build();
        (threads, pool.expect("pool"))
    });
    for (name, precond) in [
        ("off", PreconditionerConfig::Off),
        ("diagonal", PreconditionerConfig::Diagonal),
        ("schwarz", common::additive()),
    ] {
        for (threads, pool) in &pools {
            pool.install(|| {
                let solver = Solver::new(design(), None, &precond).expect("solver");
                // Workers grow their queues on their first jobs.
                allocations(&solver, &y, 5);
                let short = allocations(&solver, &y, 5);
                let long = allocations(&solver, &y, 10);
                assert_eq!(long, short, "{name} at {threads} threads");
            });
        }
    }
}
