//! Head-to-head comparison of CG and LSMR across multiple FE structures.
//!
//! Build: `cargo build --profile profiling --example lsmr_vs_cg -p within`
//! Run:   `./target/profiling/examples/lsmr_vs_cg [n_solves] [shape_filter]`
//!
//! For each (shape, size, method, tol) combination:
//!   1. Build the design from a deterministic DGP.
//!   2. Compute a reference demeaned vector per RHS with LSMR tol=1e-14.
//!   3. Time `n_solves` solves under each config.
//!   4. Report wall time + demean error vs reference (‖demean − demean_ref‖₂ / ‖demean_ref‖₂,
//!      worst case across RHSes).
//!
//! Shapes:
//!   - `highcard-1fe` — single factor with n_obs/10 levels (baseline, well-conditioned)
//!   - `panel-2fe`    — firm × year (classic panel, firm ≫ year)
//!   - `panel-3fe`    — indiv × year × firm (within-default benchmark DGP)
//!   - `akm`          — worker × firm (AKM-style, limited mobility → ill-conditioned)

use schwarz_precond::Operator;
use std::env;
use std::time::Instant;

use ndarray::Array2;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use within::config::LocalSolverConfig;
use within::domain::Design;
use within::observation::FactorMajorStore;
use within::operator::DesignOperator;
use within::{KrylovMethod, Preconditioner, ReductionStrategy, Solver, SolverParams};

#[derive(Clone, Copy)]
struct Shape {
    name: &'static str,
    /// Build `(categories, n_factors, description)`.
    ctor: fn(n_obs: usize, seed: u64) -> (Array2<u32>, usize, String),
}

fn build_3fe(n_obs: usize, _seed: u64) -> (Array2<u32>, usize, String) {
    let n_years = 10usize;
    let n_indiv_per_firm = 23usize;
    let n_indiv = (n_obs / n_years).max(1);
    let n_firm = (n_indiv / n_indiv_per_firm).max(1);
    let mut c = Array2::<u32>::zeros((n_obs, 3));
    for i in 0..n_obs {
        c[[i, 0]] = (i / n_years) as u32;
        c[[i, 1]] = (i % n_years) as u32;
        c[[i, 2]] = (i % n_firm) as u32;
    }
    (
        c,
        3,
        format!("indiv={n_indiv} × year={n_years} × firm={n_firm}"),
    )
}

fn build_2fe_panel(n_obs: usize, _seed: u64) -> (Array2<u32>, usize, String) {
    let n_years = 20usize;
    let n_firms = (n_obs / n_years).max(1);
    let mut c = Array2::<u32>::zeros((n_obs, 2));
    for i in 0..n_obs {
        c[[i, 0]] = (i / n_years) as u32;
        c[[i, 1]] = (i % n_years) as u32;
    }
    (c, 2, format!("firm={n_firms} × year={n_years}"))
}

fn build_1fe_highcard(n_obs: usize, _seed: u64) -> (Array2<u32>, usize, String) {
    let n_levels = (n_obs / 10).max(1);
    let mut c = Array2::<u32>::zeros((n_obs, 1));
    for i in 0..n_obs {
        c[[i, 0]] = (i % n_levels) as u32;
    }
    (c, 1, format!("levels={n_levels}"))
}

fn build_akm(n_obs: usize, seed: u64) -> (Array2<u32>, usize, String) {
    // Two-factor worker × firm. Each worker observed at multiple firms over
    // time, but with limited mobility (most workers stay at one firm). This
    // creates the characteristic AKM condition-number blowup.
    let mut rng = SmallRng::seed_from_u64(seed.wrapping_add(0xA4747));
    let n_workers = (n_obs / 4).max(1);
    let n_firms = (n_obs / 40).max(1);
    let mobility_prob = 0.10_f64;

    let mut c = Array2::<u32>::zeros((n_obs, 2));
    let mut firm_for_worker = vec![0u32; n_workers];
    for f in firm_for_worker.iter_mut() {
        *f = rng.random_range(0..n_firms) as u32;
    }
    for i in 0..n_obs {
        let w = i % n_workers;
        if rng.random::<f64>() < mobility_prob {
            firm_for_worker[w] = rng.random_range(0..n_firms) as u32;
        }
        c[[i, 0]] = w as u32;
        c[[i, 1]] = firm_for_worker[w];
    }
    (c, 2, format!("worker={n_workers} × firm={n_firms}"))
}

/// AKM with very low mobility and a heavy-tailed firm-size distribution.
/// Most workers never switch firms, and many firms have only a handful of
/// workers, producing a near-disconnected bipartite mobility graph — i.e.
/// a worker-firm Laplacian with many tiny eigenvalues.
fn build_akm_sparse(n_obs: usize, seed: u64) -> (Array2<u32>, usize, String) {
    let mut rng = SmallRng::seed_from_u64(seed.wrapping_add(0xA4748));
    let n_workers = (n_obs / 6).max(1);
    let n_firms = (n_obs / 200).max(1);
    let mobility_prob = 0.005_f64;

    // Heavy-tailed firm assignment: log-uniform pick → many small firms,
    // a few huge ones. Increases the spread of cell sizes.
    let pick_firm = |rng: &mut SmallRng| -> u32 {
        let u: f64 = rng.random();
        let exp = u.powi(3); // skew toward small index
        ((exp * n_firms as f64) as usize).min(n_firms - 1) as u32
    };

    let mut c = Array2::<u32>::zeros((n_obs, 2));
    let mut firm_for_worker = vec![0u32; n_workers];
    for f in firm_for_worker.iter_mut() {
        *f = pick_firm(&mut rng);
    }
    for i in 0..n_obs {
        let w = i % n_workers;
        if rng.random::<f64>() < mobility_prob {
            firm_for_worker[w] = pick_firm(&mut rng);
        }
        c[[i, 0]] = w as u32;
        c[[i, 1]] = firm_for_worker[w];
    }
    (
        c,
        2,
        format!("worker={n_workers} × firm={n_firms} (sparse mobility, heavy-tailed)"),
    )
}

/// Three high-cardinality factors with low pairwise overlap (worker × firm
/// × city). Each worker stays mostly in one firm and one city; rare
/// switches generate the only off-diagonal coupling.
fn build_3fe_highcard(n_obs: usize, seed: u64) -> (Array2<u32>, usize, String) {
    let mut rng = SmallRng::seed_from_u64(seed.wrapping_add(0x0003_FE4C));
    let n_workers = (n_obs / 3).max(1);
    let n_firms = (n_obs / 30).max(1);
    let n_cities = (n_obs / 100).max(1);
    let firm_switch = 0.01_f64;
    let city_switch = 0.005_f64;

    let mut c = Array2::<u32>::zeros((n_obs, 3));
    let mut firm_for_worker = vec![0u32; n_workers];
    let mut city_for_worker = vec![0u32; n_workers];
    for w in 0..n_workers {
        firm_for_worker[w] = rng.random_range(0..n_firms) as u32;
        city_for_worker[w] = rng.random_range(0..n_cities) as u32;
    }
    for i in 0..n_obs {
        let w = i % n_workers;
        if rng.random::<f64>() < firm_switch {
            firm_for_worker[w] = rng.random_range(0..n_firms) as u32;
        }
        if rng.random::<f64>() < city_switch {
            city_for_worker[w] = rng.random_range(0..n_cities) as u32;
        }
        c[[i, 0]] = w as u32;
        c[[i, 1]] = firm_for_worker[w];
        c[[i, 2]] = city_for_worker[w];
    }
    (
        c,
        3,
        format!("worker={n_workers} × firm={n_firms} × city={n_cities}"),
    )
}

/// Four nested factors (worker × firm × occupation × year) with limited
/// occupation/firm churn. The four-way structure plus the modest mobility
/// rates lengthens the absorption cycles and inflates the condition number.
fn build_4fe_chain(n_obs: usize, seed: u64) -> (Array2<u32>, usize, String) {
    let mut rng = SmallRng::seed_from_u64(seed.wrapping_add(0x4FE));
    let n_years = 10usize;
    let n_occs = 200usize;
    let n_firms = (n_obs / 40).max(1);
    let n_workers = (n_obs / 4).max(1);
    let firm_switch = 0.02_f64;
    let occ_switch = 0.005_f64;

    let mut c = Array2::<u32>::zeros((n_obs, 4));
    let mut firm_for_worker = vec![0u32; n_workers];
    let mut occ_for_worker = vec![0u32; n_workers];
    for w in 0..n_workers {
        firm_for_worker[w] = rng.random_range(0..n_firms) as u32;
        occ_for_worker[w] = rng.random_range(0..n_occs) as u32;
    }
    for i in 0..n_obs {
        let w = i % n_workers;
        if rng.random::<f64>() < firm_switch {
            firm_for_worker[w] = rng.random_range(0..n_firms) as u32;
        }
        if rng.random::<f64>() < occ_switch {
            occ_for_worker[w] = rng.random_range(0..n_occs) as u32;
        }
        c[[i, 0]] = w as u32;
        c[[i, 1]] = firm_for_worker[w];
        c[[i, 2]] = occ_for_worker[w];
        c[[i, 3]] = (i % n_years) as u32;
    }
    (
        c,
        4,
        format!("worker={n_workers} × firm={n_firms} × occ={n_occs} × year={n_years}"),
    )
}

const SHAPES: &[Shape] = &[
    Shape {
        name: "1fe-highcard",
        ctor: build_1fe_highcard,
    },
    Shape {
        name: "2fe-panel",
        ctor: build_2fe_panel,
    },
    Shape {
        name: "3fe-panel",
        ctor: build_3fe,
    },
    Shape {
        name: "akm",
        ctor: build_akm,
    },
    Shape {
        name: "akm-sparse",
        ctor: build_akm_sparse,
    },
    Shape {
        name: "3fe-highcard",
        ctor: build_3fe_highcard,
    },
    Shape {
        name: "4fe-chain",
        ctor: build_4fe_chain,
    },
];

fn vec_norm2(v: &[f64]) -> f64 {
    v.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

fn rel_l2_diff(a: &[f64], b: &[f64]) -> f64 {
    let diff_norm: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt();
    diff_norm / vec_norm2(b).max(f64::MIN_POSITIVE)
}

fn run_one(
    shape: &Shape,
    n_obs: usize,
    n_solves: usize,
    rng_seed: u64,
    size_label: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = SmallRng::seed_from_u64(rng_seed);
    let (categories, n_factors, desc) = (shape.ctor)(n_obs, rng_seed);

    // Build design for y generation.
    let factor_levels: Vec<Vec<u32>> = (0..n_factors)
        .map(|q| categories.column(q).to_vec())
        .collect();
    let store = FactorMajorStore::new(factor_levels, n_obs)?;
    let design = Design::from_store(store)?;
    let x_true: Vec<f64> = (0..design.n_dofs)
        .map(|_| rng.random_range(-1.0..1.0))
        .collect();
    let rhses: Vec<Vec<f64>> = (0..(n_solves + 1))
        .map(|_| {
            let mut y = vec![0.0; n_obs];
            DesignOperator::new(&design, None)
                .apply(&x_true, &mut y)
                .expect("apply");
            for yi in &mut y {
                *yi += 0.1 * rng.random_range(-1.0..1.0);
            }
            y
        })
        .collect();

    let precond =
        Preconditioner::Additive(LocalSolverConfig::solver_default(), ReductionStrategy::Auto);

    println!();
    println!(
        "=== shape={} | size={} | n_obs={} | dofs={} | {} ===",
        shape.name, size_label, n_obs, design.n_dofs, desc
    );

    // Reference demean
    let ref_params = SolverParams {
        krylov: KrylovMethod::Lsmr { local_size: None },
        tol: 1e-14,
        maxiter: 2000,
        ..Default::default()
    };
    let t_ref = Instant::now();
    let ref_solver = Solver::new(categories.view(), None, &ref_params, Some(&precond))?;
    let mut ref_iters = 0usize;
    let mut ref_converged = true;
    let ref_demeans: Vec<Vec<f64>> = rhses[1..=n_solves]
        .iter()
        .map(|y| {
            let r = ref_solver.solve(y).expect("ref solve");
            ref_iters += r.iterations();
            ref_converged &= r.converged();
            r.into_parts().1
        })
        .collect();
    let ref_wall = t_ref.elapsed().as_secs_f64();
    let ref_iter_avg = ref_iters as f64 / n_solves as f64;
    println!(
        "reference (LSMR tol=1e-14): avg iters={:.1}, wall={:.2}s{}",
        ref_iter_avg,
        ref_wall,
        if ref_converged {
            ""
        } else {
            " (some non-converged!)"
        }
    );

    let configs: &[(&str, KrylovMethod, f64)] = &[
        ("CG   tol=1e-8 ", KrylovMethod::Cg, 1e-8),
        ("CG   tol=1e-12", KrylovMethod::Cg, 1e-12),
        ("CG   tol=1e-13", KrylovMethod::Cg, 1e-13),
        (
            "LSMR tol=1e-8 ",
            KrylovMethod::Lsmr { local_size: None },
            1e-8,
        ),
        (
            "LSMR tol=1e-12",
            KrylovMethod::Lsmr { local_size: None },
            1e-12,
        ),
    ];

    println!(
        "{:<16} {:>10} {:>10} {:>6} {:>11} {:>11}",
        "config", "per_solve", "per_iter", "iters", "demean_rel", "demean_max"
    );
    println!("{}", "-".repeat(72));

    for (name, krylov, tol) in configs {
        let params = SolverParams {
            krylov: *krylov,
            tol: *tol,
            maxiter: 2000,
            ..Default::default()
        };
        let solver = Solver::new(categories.view(), None, &params, Some(&precond))?;
        // Warmup
        let _ = solver.solve(&rhses[0])?;

        let mut demeans: Vec<Vec<f64>> = Vec::with_capacity(n_solves);
        let mut iter_sum = 0usize;
        let mut all_converged = true;
        let t = Instant::now();
        for y in &rhses[1..=n_solves] {
            let r = solver.solve(y)?;
            iter_sum += r.iterations();
            all_converged &= r.converged();
            demeans.push(r.into_parts().1);
        }
        let wall = t.elapsed().as_secs_f64();

        let (mut worst_rel, mut worst_max) = (0.0_f64, 0.0_f64);
        for (d, dref) in demeans.iter().zip(&ref_demeans) {
            worst_rel = worst_rel.max(rel_l2_diff(d, dref));
            worst_max = worst_max.max(max_abs_diff(d, dref));
        }

        let per_solve_ms = 1000.0 * wall / n_solves as f64;
        let per_iter_ms = 1000.0 * wall / iter_sum.max(1) as f64;
        let avg_iters = iter_sum as f64 / n_solves as f64;
        let flag = if all_converged { "" } else { " !" };

        println!(
            "{:<16} {:>8.1}ms {:>8.1}ms {:>6.1} {:>11.2e} {:>11.2e}{}",
            name, per_solve_ms, per_iter_ms, avg_iters, worst_rel, worst_max, flag
        );
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let n_solves: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(10);
    let shape_filter: Option<String> = args.next();

    // (label, n_obs)
    let sizes: &[(&str, usize)] = &[("500k", 500_000), ("1M", 1_000_000), ("5M", 5_000_000)];

    println!(
        "CG vs LSMR sweep: n_solves={} per config{}",
        n_solves,
        if let Some(f) = &shape_filter {
            format!(", shape_filter={f}")
        } else {
            String::new()
        }
    );

    for shape in SHAPES {
        if let Some(filter) = &shape_filter {
            if !shape.name.contains(filter.as_str()) {
                continue;
            }
        }
        for (size_label, n_obs) in sizes {
            run_one(shape, *n_obs, n_solves, 0xBADC0FFEE, size_label)?;
        }
    }
    Ok(())
}
