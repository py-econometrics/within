//! A solve returns the same bits at any thread count, run after run (#472).

use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use rstest::{fixture, rstest};
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

/// 120k rows and 74k coefficients: both exceed the length below which a sum runs on one thread.
#[fixture]
#[once]
fn panel() -> Panel {
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
    Panel {
        worker: (0..n).map(|i| (i / SPELL) as u32).collect(),
        firm,
        tenure: draw(-0.5, 0.5),
        region: (0..n).map(|i| (i as u32 * 7_919) % REGIONS).collect(),
        ys: [draw(-1.0, 1.0), draw(-1.0, 1.0)],
        weights: draw(0.1, 2.0),
    }
}

/// Everything a caller reads off a batch solve, floats as bits.
#[derive(PartialEq)]
struct Fingerprint {
    x: Vec<u64>,
    demeaned: Vec<u64>,
    residual: Vec<u64>,
    iterations: Vec<usize>,
    converged: Vec<bool>,
    escalated: bool,
}

impl Panel {
    fn design(&self) -> Design<'_> {
        Design::new([
            Effect::new(&self.worker, true, []).expect("worker"),
            Effect::new(&self.firm, true, [&self.tenure[..]]).expect("firm"),
            Effect::new(&self.region, true, []).expect("region"),
        ])
        .expect("design")
    }

    fn solver(&self, precond: &PreconditionerConfig, weighted: bool) -> Solver<'_> {
        let weights = weighted.then_some(&self.weights[..]);
        Solver::new(self.design(), weights, precond).expect("solver")
    }

    fn fingerprint(&self, solver: &Solver<'_>) -> Fingerprint {
        let lsmr = LsmrOptions {
            tol: 1e-6,
            maxiter: 200,
            local_size: None,
        };
        let r = solver
            .solve_batch(&self.ys.each_ref().map(Vec::as_slice), &lsmr)
            .expect("solve");
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect();
        Fingerprint {
            x: bits(&r.x),
            demeaned: bits(&r.demeaned),
            residual: bits(&r.residual),
            iterations: r.iterations,
            converged: r.converged,
            escalated: solver.has_escalated(),
        }
    }
}

fn with_threads<T: Send>(threads: usize, run: impl FnOnce() -> T + Send) -> T {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("pool")
        .install(run)
}

/// Off and the diagonal run different LSMR drivers; Adaptive builds a weighted Schwarz on a stall.
#[rstest]
#[case::off(PreconditionerConfig::Off, false)]
#[case::diagonal_weighted(PreconditionerConfig::Diagonal, true)]
#[case::additive(common::additive(), false)]
#[case::adaptive_weighted(common::adaptive(), true)]
fn a_solve_returns_the_same_bits_at_any_thread_count(
    panel: &Panel,
    #[case] precond: PreconditionerConfig,
    #[case] weighted: bool,
) {
    let adaptive = matches!(precond, PreconditionerConfig::Adaptive { .. });
    let solver = with_threads(1, || panel.solver(&precond, weighted));
    let reference = with_threads(1, || panel.fingerprint(&solver));
    assert_eq!(
        reference.escalated, adaptive,
        "only the weighted ladder reaches its Schwarz build"
    );
    // Not `assert_eq!`: a printed solution vector would bury the thread count.
    for threads in [2, 3, 8] {
        let fresh = with_threads(threads, || {
            panel.fingerprint(&panel.solver(&precond, weighted))
        });
        assert!(fresh == reference, "{threads} threads");
    }
    // Adaptive's second solve starts from the Schwarz its first one built (#370).
    if !adaptive {
        let again = with_threads(8, || panel.fingerprint(&solver));
        assert!(again == reference, "built at 1 thread, solved again at 8");
    }
}
