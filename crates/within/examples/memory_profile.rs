//! Memory profiling for setup and solve phases.
//!
//! Run with: `cargo run --example memory_profile -p within --release -- [n_obs]`
//!
//! Uses the same fixest-style DGP as the benchmarks. Reports RSS at each phase.

use std::env;
use std::time::Instant;

use approx_chol::Config;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use schwarz_precond::solve::cg::cg_solve_preconditioned;
use within::config::SchwarzConfig;
use within::domain::WeightedDesign;
use within::observation::{FactorMajorStore, ObservationWeights};
use within::{build_schwarz, GramianOperator};

fn rss_mb() -> f64 {
    let pid = std::process::id();
    if let Ok(output) = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &pid.to_string()])
        .output()
    {
        if let Ok(s) = String::from_utf8(output.stdout) {
            if let Ok(kb) = s.trim().parse::<f64>() {
                return kb / 1024.0;
            }
        }
    }
    0.0
}

fn generate_fixest_3fe(n_obs: usize, seed: u64) -> (WeightedDesign<FactorMajorStore>, Vec<f64>) {
    let mut rng = SmallRng::seed_from_u64(seed);
    let n_years = 10usize;
    let n_indiv_per_firm = 23usize;

    let n_indiv = ((n_obs as f64 / n_years as f64).round() as usize).max(1);
    let n_firm = ((n_indiv as f64 / n_indiv_per_firm as f64).round() as usize).max(1);

    let mut indiv_id = Vec::with_capacity(n_obs);
    let mut year = Vec::with_capacity(n_obs);
    let mut firm_id = Vec::with_capacity(n_obs);

    for i in 0..n_obs {
        indiv_id.push((i / n_years) as u32);
        year.push((i % n_years) as u32);
        firm_id.push((i % n_firm) as u32);
    }

    let factor_levels = vec![indiv_id, year, firm_id];
    let n_levels = vec![n_indiv, n_years, n_firm];

    let store = FactorMajorStore::new(factor_levels, ObservationWeights::Unit, n_obs)
        .expect("valid factor-major store");
    let design = WeightedDesign::from_store(store, &n_levels).expect("valid design");

    let mut x_true = vec![0.0; design.n_dofs];
    for x in &mut x_true {
        *x = rng.random_range(-1.0..1.0);
    }

    let mut y = vec![0.0; n_obs];
    design.matvec_d(&x_true, &mut y);
    for yi in &mut y {
        *yi += 0.1 * rng.random_range(-1.0..1.0);
    }

    (design, y)
}

fn main() {
    let n_obs: usize = env::args()
        .nth(1)
        .and_then(|s| s.replace(['_', ','], "").parse().ok())
        .unwrap_or(5_000_000);

    println!("=== Memory Profile: n_obs={n_obs} (difficult 3FE) ===\n");

    let rss_baseline = rss_mb();
    println!("Baseline RSS: {rss_baseline:.1} MB");

    // Phase 1: Generate data
    let t = Instant::now();
    let (design, y) = generate_fixest_3fe(n_obs, 42);
    let rss_data = rss_mb();
    let dt_data = t.elapsed().as_secs_f64();
    println!("\n[1] Data Generation ({dt_data:.3}s)");
    println!(
        "    n_obs={n_obs}, n_dofs={}, n_factors={}",
        design.n_dofs,
        design.n_factors()
    );
    println!(
        "    RSS: {rss_data:.1} MB  (+{:.1} MB from baseline)",
        rss_data - rss_baseline
    );
    let est_store = (design.n_factors() * n_obs * 4) as f64 / (1024.0 * 1024.0);
    let est_y = (n_obs * 8) as f64 / (1024.0 * 1024.0);
    println!(
        "    Estimated: store={est_store:.1} MB, y={est_y:.1} MB, x_true={:.1} MB",
        (design.n_dofs * 8) as f64 / (1024.0 * 1024.0)
    );

    // Phase 2: Build Schwarz preconditioner
    let t = Instant::now();
    let schwarz = build_schwarz(
        &design,
        &SchwarzConfig {
            smoother: Config {
                seed: 42,
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .expect("build schwarz preconditioner");
    let rss_schwarz = rss_mb();
    let dt_schwarz = t.elapsed().as_secs_f64();
    println!("\n[2] Build Schwarz Preconditioner ({dt_schwarz:.3}s)");
    println!(
        "    RSS: {rss_schwarz:.1} MB  (+{:.1} MB from data)",
        rss_schwarz - rss_data
    );

    // Phase 3: Build GramianOperator (just a reference + scratch)
    let gramian_op = GramianOperator::new(&design);
    let rss_gramian = rss_mb();
    println!("\n[3] GramianOperator");
    println!(
        "    RSS: {rss_gramian:.1} MB  (+{:.1} MB from schwarz)",
        rss_gramian - rss_schwarz
    );

    // Phase 4: Build rhs = D^T y (via DesignOperator adjoint)
    let t = Instant::now();
    let mut rhs = vec![0.0; design.n_dofs];
    {
        use schwarz_precond::Operator;
        let design_op = within::DesignOperator::new(&design);
        design_op.apply_adjoint(&y, &mut rhs);
    }
    let rss_rhs = rss_mb();
    let dt_rhs = t.elapsed().as_secs_f64();
    println!("\n[4] Build RHS ({dt_rhs:.6}s)");
    println!(
        "    RSS: {rss_rhs:.1} MB  (+{:.1} MB)",
        rss_rhs - rss_gramian
    );

    // Phase 5: CG solve
    let t = Instant::now();
    let cg_result =
        cg_solve_preconditioned(&gramian_op, &schwarz, &rhs, 1e-8, 100).expect("cg solve");
    let rss_solve = rss_mb();
    let dt_solve = t.elapsed().as_secs_f64();
    println!("\n[5] CG Solve ({dt_solve:.3}s)");
    println!(
        "    converged={}, iters={}, residual={:.2e}",
        cg_result.converged, cg_result.iterations, cg_result.residual_norm
    );
    println!(
        "    RSS: {rss_solve:.1} MB  (+{:.1} MB from setup)",
        rss_solve - rss_rhs
    );

    // Phase 6: Drop preconditioner, measure retained
    drop(schwarz);
    let rss_no_schwarz = rss_mb();
    println!("\n[6] After dropping Schwarz");
    println!(
        "    RSS: {rss_no_schwarz:.1} MB  (-{:.1} MB)",
        rss_solve - rss_no_schwarz
    );

    // Summary table
    println!("\n=== Summary (bytes/obs) ===");
    let bpo = |mb: f64| mb * 1024.0 * 1024.0 / n_obs as f64;
    println!(
        "  Data (store+y):     {:6.1} B/obs  ({:.1} MB)",
        bpo(rss_data - rss_baseline),
        rss_data - rss_baseline
    );
    println!(
        "  Schwarz build:      {:6.1} B/obs  ({:.1} MB)",
        bpo(rss_schwarz - rss_data),
        rss_schwarz - rss_data
    );
    println!(
        "  CG vectors:         {:6.1} B/obs  ({:.1} MB)",
        bpo(rss_solve - rss_rhs),
        rss_solve - rss_rhs
    );
    println!(
        "  Total peak:         {:6.1} B/obs  ({:.1} MB)",
        bpo(rss_solve - rss_baseline),
        rss_solve - rss_baseline
    );
    println!(
        "  Schwarz retained:   {:6.1} B/obs  ({:.1} MB)",
        bpo(rss_schwarz - rss_data),
        rss_schwarz - rss_data
    );
    println!(
        "  Schwarz freed:      {:6.1} B/obs  ({:.1} MB)",
        bpo(rss_solve - rss_no_schwarz),
        rss_solve - rss_no_schwarz
    );
}
