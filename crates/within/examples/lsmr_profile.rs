//! LSMR profiling workload. Build once, solve many times.
//!
//! Use: `cargo build --profile profiling --example lsmr_profile -p within`
//! then `samply record ./target/profiling/examples/lsmr_profile`.
//!
//! Runs a 3FE panel DGP (indiv × year × firm) and calls the LSMR solver with
//! an Additive Schwarz preconditioner on the same design for N RHSes. Setup is
//! done once up front so the profile is dominated by LSMR iteration work.

use std::env;
use std::time::Instant;

use ndarray::Array2;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use within::observation::{FactorMajorStore, ObservationWeights};
use within::{
    KrylovMethod, LocalSolverConfig, Preconditioner, ReductionStrategy, Solver, SolverParams,
    WeightedDesign,
};

fn main() {
    let mut args = env::args().skip(1);
    let n_obs: usize = args
        .next()
        .and_then(|s| s.replace(['_', ','], "").parse().ok())
        .unwrap_or(1_000_000);
    let n_solves: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(20);

    println!("LSMR profile: n_obs={n_obs}, n_solves={n_solves}");

    // -- DGP: 3 factors mirroring the fixest-style panel used elsewhere --
    let mut rng = SmallRng::seed_from_u64(0xBADC0FFEE);
    let n_years = 10usize;
    let n_indiv_per_firm = 23usize;
    let n_indiv = (n_obs / n_years).max(1);
    let n_firm = (n_indiv / n_indiv_per_firm).max(1);

    let mut categories = Array2::<u32>::zeros((n_obs, 3));
    for i in 0..n_obs {
        categories[[i, 0]] = (i / n_years) as u32; // indiv
        categories[[i, 1]] = (i % n_years) as u32; // year
        categories[[i, 2]] = (i % n_firm) as u32; // firm
    }

    // Build the design for generating realistic y = D x + noise.
    let factor_levels: Vec<Vec<u32>> = (0..3).map(|q| categories.column(q).to_vec()).collect();
    let store = FactorMajorStore::new(factor_levels, ObservationWeights::Unit, n_obs)
        .expect("valid factor-major store");
    let design = WeightedDesign::from_store(store).expect("valid design");
    let x_true: Vec<f64> = (0..design.n_dofs)
        .map(|_| rng.random_range(-1.0..1.0))
        .collect();

    // Generate n_solves RHSes — fresh noise per solve so the solver actually iterates.
    let rhses: Vec<Vec<f64>> = (0..n_solves)
        .map(|_| {
            let mut y = vec![0.0; n_obs];
            design.matvec_d(&x_true, &mut y);
            for yi in &mut y {
                *yi += 0.1 * rng.random_range(-1.0..1.0);
            }
            y
        })
        .collect();

    // -- Build solver once (preconditioner reuse) --
    let params = SolverParams {
        krylov: KrylovMethod::Lsmr { local_size: None },
        tol: 1e-10,
        maxiter: 500,
        ..Default::default()
    };
    let precond =
        Preconditioner::Additive(LocalSolverConfig::solver_default(), ReductionStrategy::Auto);

    let t_setup = Instant::now();
    let solver =
        Solver::new(categories.view(), None, &params, Some(&precond)).expect("solver construction");
    let setup_s = t_setup.elapsed().as_secs_f64();
    println!("setup (preconditioner build): {setup_s:.3} s");

    // -- Drive the LSMR hot path --
    let t_solve = Instant::now();
    let mut total_iters: usize = 0;
    for (i, y) in rhses.iter().enumerate() {
        let r = solver.solve(y).expect("lsmr solve");
        total_iters += r.iterations;
        if i == 0 {
            println!(
                "first solve: iters={}, converged={}, residual={:.2e}",
                r.iterations, r.converged, r.final_residual
            );
        }
    }
    let solve_s = t_solve.elapsed().as_secs_f64();
    let per_solve_ms = 1000.0 * solve_s / n_solves as f64;
    let per_iter_ms = 1000.0 * solve_s / total_iters.max(1) as f64;
    println!(
        "solve: total={solve_s:.3} s, per-solve={per_solve_ms:.2} ms, \
         total_iters={total_iters}, per-iter={per_iter_ms:.2} ms"
    );
}
