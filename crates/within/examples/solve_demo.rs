//! Simplest usage of the `within` crate.
//!
//! Generates synthetic fixed-effects data with two factors (100 levels each,
//! 10 000 observations), then solves with the default solver.
//!
//! Run with: `cargo run --example solve_demo -p within`

use within::{solve, SolverParams};

fn main() {
    // Two factors, each with 100 levels, 10 000 observations.
    let n_obs = 10_000;
    let n_levels = [100, 100];

    // Category vectors via modular arithmetic (deterministic, no randomness).
    let factor_0: Vec<u32> = (0..n_obs).map(|i| (i % 100) as u32).collect();
    let factor_1: Vec<u32> = (0..n_obs).map(|i| (i / 100) as u32).collect();
    let categories = vec![factor_0, factor_1];

    // True coefficient vector: x_true[j] = (j mod 7) - 3.
    let total_dofs: usize = n_levels.iter().sum();
    let x_true: Vec<f64> = (0..total_dofs).map(|j| (j % 7) as f64 - 3.0).collect();

    // Build y = D * x_true + noise (deterministic "noise" from a simple function).
    // We construct the design manually to compute D * x_true.
    use within::observation::{FactorMajorStore, ObservationWeights};
    use within::WeightedDesign;

    let store = FactorMajorStore::new(categories.clone(), ObservationWeights::Unit, n_obs)
        .expect("valid factor-major store");
    let design = WeightedDesign::from_store(store, &n_levels).expect("valid design");
    let mut y = vec![0.0; n_obs];
    design.matvec_d(&x_true, &mut y);
    // Add small deterministic perturbation so the system is not trivially exact.
    for (i, yi) in y.iter_mut().enumerate() {
        *yi += 0.01 * ((i * 7 + 3) % 13) as f64 - 0.06;
    }

    // Solve with default parameters (CG + additive Schwarz, implicit operator).
    let params = SolverParams::default();
    let result = solve(&categories, &n_levels, &y, None, &params).expect("solve");

    println!("=== Basic solve (default params) ===");
    println!("  converged:  {}", result.converged);
    println!("  iterations: {}", result.iterations);
    println!("  residual:   {:.3e}", result.final_residual);
    println!("  time_total: {:.3} s", result.time_total);
    println!("  time_setup: {:.3} s", result.time_setup);
    println!("  time_solve: {:.3} s", result.time_solve);
}
