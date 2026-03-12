//! Simplest usage of the `within` crate.
//!
//! Generates synthetic fixed-effects data with two factors (100 levels each,
//! 10 000 observations), then solves with the default solver.
//!
//! Run with: `cargo run --example solve_demo -p within`

use ndarray::Array2;
use within::{solve, LocalSolverConfig, Preconditioner, ReductionStrategy, SolverParams};

fn main() {
    // Two factors, each with 100 levels, 10 000 observations.
    let n_obs = 10_000;
    // Category array via modular arithmetic (deterministic, no randomness).
    let mut categories = Array2::<u32>::zeros((n_obs, 2));
    for i in 0..n_obs {
        categories[[i, 0]] = (i % 100) as u32;
        categories[[i, 1]] = (i / 100) as u32;
    }

    // Build design to compute D * x_true.
    use within::observation::{FactorMajorStore, ObservationWeights};
    use within::WeightedDesign;

    let factor_levels = vec![categories.column(0).to_vec(), categories.column(1).to_vec()];
    let store = FactorMajorStore::new(factor_levels, ObservationWeights::Unit, n_obs)
        .expect("valid factor-major store");
    let design = WeightedDesign::from_store(store).expect("valid design");

    // True coefficient vector: x_true[j] = (j mod 7) - 3.
    let total_dofs = design.n_dofs;
    let x_true: Vec<f64> = (0..total_dofs).map(|j| (j % 7) as f64 - 3.0).collect();
    let mut y = vec![0.0; n_obs];
    design.matvec_d(&x_true, &mut y);
    // Add small deterministic perturbation so the system is not trivially exact.
    for (i, yi) in y.iter_mut().enumerate() {
        *yi += 0.01 * ((i * 7 + 3) % 13) as f64 - 0.06;
    }

    // Solve with default parameters (CG + additive Schwarz, implicit operator).
    let params = SolverParams::default();
    let precond =
        Preconditioner::Additive(LocalSolverConfig::solver_default(), ReductionStrategy::Auto);
    let result = solve(categories.view(), &y, None, &params, Some(&precond)).expect("solve");

    println!("=== Basic solve (default params) ===");
    println!("  converged:  {}", result.converged);
    println!("  iterations: {}", result.iterations);
    println!("  residual:   {:.3e}", result.final_residual);
    println!("  time_total: {:.3} s", result.time_total);
    println!("  time_solve: {:.3} s", result.time_solve);
}
