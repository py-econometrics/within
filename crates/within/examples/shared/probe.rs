//! Shared body of the grouping probes: the fixest-like difficult panel, rows optionally shuffled
//! by a fixed seed; times `Design::new` (grouping by level) and `Solver::new` apart.
//!
//! Usage: <probe> <n_obs> <sorted|shuffled> [--solve]

use std::time::Instant;

use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use within::config::{
    ApproxCholConfig, ApproxSchurConfig, LocalSolverConfig, LsmrOptions, PreconditionerConfig,
    SchurMode,
};
use within::{Design, Effect, Schwarz, Solver};

pub fn run(mark: impl Fn() -> usize, peak: impl Fn() -> usize) {
    let args: Vec<String> = std::env::args().collect();
    let n_obs: usize = args.get(1).and_then(|s| s.parse().ok()).expect("n_obs");
    let order = args.get(2).map(String::as_str).expect("sorted|shuffled");
    let solve = args.iter().any(|a| a == "--solve");

    let n_years = 10usize;
    let n_indiv = (n_obs as f64 / n_years as f64).round() as usize;
    let n_firm = (n_indiv as f64 / 23.0).round() as usize;
    let mut cols: [Vec<u32>; 3] = [
        (0..n_obs).map(|i| (i / n_years) as u32).collect(),
        (0..n_obs).map(|i| (i % n_years) as u32).collect(),
        (0..n_obs).map(|i| (i % n_firm) as u32).collect(),
    ];
    let mut rng = SmallRng::seed_from_u64(0xC0FFEE);
    let mut y: Vec<f64> = (0..n_obs).map(|_| rng.random_range(-1.0..1.0)).collect();
    match order {
        "sorted" => {}
        "shuffled" => {
            let mut rng = SmallRng::seed_from_u64(0x5EED);
            for i in (1..n_obs).rev() {
                let j = rng.random_range(0..=i);
                for c in &mut cols {
                    c.swap(i, j);
                }
                y.swap(i, j);
            }
        }
        other => panic!("unknown order {other}"),
    }
    let local_solver = LocalSolverConfig {
        approx_chol: ApproxCholConfig {
            seed: 0,
            split_merge: Some(8),
        },
        schur: SchurMode::Approximate(ApproxSchurConfig::default()),
        ..Default::default()
    };
    let precond = PreconditionerConfig::Schwarz(Schwarz { local_solver });

    let before = mark();
    let t = Instant::now();
    let design = Design::new(
        cols.iter()
            .map(|c| Effect::new(c, true, []).expect("effect")),
    )
    .expect("design");
    let design_s = t.elapsed().as_secs_f64();
    let design_peak = peak() - before;
    let before = mark();
    let t = Instant::now();
    let solver = Solver::new(design, None, &precond).expect("solver build");
    let setup_s = t.elapsed().as_secs_f64();
    let setup_peak = peak() - before;
    let per = |b: usize| b as f64 / n_obs as f64;
    print!(
        "n_obs={n_obs} order={order} design_s={design_s:.4} setup_s={setup_s:.4} \
         design_bpo={:.2} setup_bpo={:.2}",
        per(design_peak),
        per(setup_peak)
    );
    if solve {
        let params = LsmrOptions {
            tol: 1e-6,
            maxiter: 200,
            ..Default::default()
        };
        let t = Instant::now();
        let r = solver.solve(&y, &params).expect("solve");
        let solve_s = t.elapsed().as_secs_f64();
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for v in &r.x {
            for b in v.to_bits().to_le_bytes() {
                h ^= u64::from(b);
                h = h.wrapping_mul(0x100_0000_01b3);
            }
        }
        print!(
            " iters={} solve_s={solve_s:.3} checksum={h:016x}",
            r.iterations
        );
    }
    println!();
}
