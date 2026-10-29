//! A solve returns the same bits at any thread count, run after run (#472).

use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use rstest::rstest;
use within::{Design, Effect, LsmrOptions, PreconditionerConfig, Solver};

#[path = "common/orchestrate_helpers.rs"]
mod common;

/// Worker + firm with a slope + region: one sorted term and both unsorted adjoint splits.
struct Panel {
    worker: Vec<u32>,
    firm: Vec<u32>,
    tenure: Vec<f64>,
    region: Vec<u32>,
    ys: [Vec<f64>; 2],
    weights: Vec<f64>,
}

const WORKERS: u32 = 40_000;
const SPELL: usize = 3;
const FIRMS: u32 = 2_000;
// 4 rows per region and 60 per firm: below and above the 12 rows per level that split by row.
const REGIONS: u32 = 30_000;

impl Panel {
    /// 120k rows and 74k coefficients: both exceed the length below which a sum runs on one thread.
    fn new() -> Self {
        let mut rng = SmallRng::seed_from_u64(472);
        let n = WORKERS as usize * SPELL;
        let mut firm = Vec::with_capacity(n);
        for _ in 0..WORKERS {
            let (from, to) = (rng.random_range(0..FIRMS), rng.random_range(0..FIRMS));
            let moves_at = rng.random_range(1..=SPELL);
            firm.extend((0..SPELL).map(|t| if t < moves_at { from } else { to }));
        }
        let mut draw =
            |lo: f64, hi: f64| -> Vec<f64> { (0..n).map(|_| rng.random_range(lo..hi)).collect() };
        Self {
            worker: (0..n).map(|i| (i / SPELL) as u32).collect(),
            firm,
            tenure: draw(-0.5, 0.5),
            region: (0..n).map(|i| (i as u32 * 7_919) % REGIONS).collect(),
            ys: [draw(-1.0, 1.0), draw(-1.0, 1.0)],
            weights: draw(0.1, 2.0),
        }
    }

    fn design(&self) -> Design<'_> {
        Design::new([
            Effect::new(&self.worker, true, []).expect("worker"),
            Effect::new(&self.firm, true, [&self.tenure[..]]).expect("firm"),
            Effect::new(&self.region, true, []).expect("region"),
        ])
        .expect("design")
    }
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[rstest]
fn a_solve_returns_the_same_bits_at_any_thread_count(
    #[values(
        PreconditionerConfig::Off,
        PreconditionerConfig::Diagonal,
        common::additive(),
        common::adaptive()
    )]
    precond: PreconditionerConfig,
    #[values(false, true)] weighted: bool,
) {
    let panel = Panel::new();
    let lsmr = LsmrOptions {
        tol: 1e-6,
        maxiter: 200,
        local_size: None,
    };
    // Design, preconditioner and both right-hand sides are all built inside the pool.
    let solve = || {
        let weights = weighted.then_some(&panel.weights[..]);
        let solver = Solver::new(panel.design(), weights, &precond).expect("solver");
        let ys = panel.ys.each_ref().map(Vec::as_slice);
        let r = solver.solve_batch(&ys, &lsmr).expect("solve");
        (
            bits(&r.x),
            bits(&r.demeaned),
            bits(&r.residual),
            r.iterations,
            r.converged,
        )
    };
    let in_pool = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("pool")
            .install(solve)
    };
    let reference = in_pool(1);
    // 8 twice: before #472, a fixed width already changed bits between runs.
    for threads in [2, 3, 8, 8] {
        // Not `assert_eq!`: a printed solution vector would bury the thread count.
        assert!(in_pool(threads) == reference, "{threads} threads");
    }
}
