//! CG with one-level additive Schwarz preconditioning, compared to unpreconditioned LSMR.
//!
//! Generates the same synthetic data as `basic_solve` (two factors, 100 levels
//! each, 10 000 observations), then solves with both methods and prints the
//! results side by side.
//!
//! Run with: `cargo run --example preconditioned_solve`

use within::{
    solve, OperatorRepr, Preconditioner, SchwarzConfig, SolveResult, SolverMethod, SolverParams,
};

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

    // Build y = D * x_true + noise (deterministic perturbation).
    use within::observation::{FactorMajorStore, ObservationWeights};
    use within::WeightedDesign;

    let store = FactorMajorStore::new(categories.clone(), ObservationWeights::Unit, n_obs)
        .expect("valid factor-major store");
    let design = WeightedDesign::from_store(store, &n_levels).expect("valid design");
    let mut y = vec![0.0; n_obs];
    design.matvec_d(&x_true, &mut y);
    for (i, yi) in y.iter_mut().enumerate() {
        *yi += 0.01 * ((i * 7 + 3) % 13) as f64 - 0.06;
    }

    // -----------------------------------------------------------------------
    // 1. Unpreconditioned LSMR (default)
    // -----------------------------------------------------------------------
    let lsmr_params = SolverParams::default();
    let lsmr_result = solve(&categories, &n_levels, &y, &lsmr_params, None).expect("lsmr solve");

    // -----------------------------------------------------------------------
    // 2. CG with one-level additive Schwarz
    // -----------------------------------------------------------------------
    let cg_params = SolverParams {
        method: SolverMethod::Cg {
            preconditioner: Preconditioner::Additive(SchwarzConfig::default()),
            operator: OperatorRepr::Implicit,
        },
        tol: 1e-8,
        maxiter: 1000,
    };
    let cg_result = solve(&categories, &n_levels, &y, &cg_params, None).expect("cg solve");

    // -----------------------------------------------------------------------
    // Print comparison
    // -----------------------------------------------------------------------
    print_comparison(
        "LSMR (default)",
        &lsmr_result,
        "CG + 1-level Schwarz",
        &cg_result,
    );
}

fn print_comparison(name_a: &str, a: &SolveResult, name_b: &str, b: &SolveResult) {
    println!("{:<30} {:>12} {:>12}", "", name_a, name_b,);
    println!("{}", "-".repeat(54));
    println!(
        "{:<30} {:>12} {:>12}",
        "converged", a.converged, b.converged,
    );
    println!(
        "{:<30} {:>12} {:>12}",
        "iterations", a.iterations, b.iterations,
    );
    println!(
        "{:<30} {:>12.3e} {:>12.3e}",
        "residual", a.final_residual, b.final_residual,
    );
    println!(
        "{:<30} {:>12.3} {:>12.3}",
        "time_total (s)", a.time_total, b.time_total,
    );
    println!(
        "{:<30} {:>12.3} {:>12.3}",
        "time_setup (s)", a.time_setup, b.time_setup,
    );
    println!(
        "{:<30} {:>12.3} {:>12.3}",
        "time_solve (s)", a.time_solve, b.time_solve,
    );
}
