//! Compare the default CG + additive Schwarz configuration to a GMRES +
//! multiplicative Schwarz variant.
//!
//! Generates the same synthetic data as `basic_solve` (two factors, 100 levels
//! each, 10 000 observations), then solves with both methods and prints the
//! results side by side.
//!
//! Run with: `cargo run --example preconditioned_solve`

use ndarray::Array2;
use within::{
    solve, KrylovMethod, LocalSolverConfig, OperatorRepr, Preconditioner, ReductionStrategy,
    SolveResult, SolverParams,
};

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
    for (i, yi) in y.iter_mut().enumerate() {
        *yi += 0.01 * ((i * 7 + 3) % 13) as f64 - 0.06;
    }

    let cg_params = SolverParams::default();
    let cg_precond =
        Preconditioner::Additive(LocalSolverConfig::solver_default(), ReductionStrategy::Auto);
    let cg_result =
        solve(categories.view(), &y, None, &cg_params, Some(&cg_precond)).expect("cg solve");

    let gmres_params = SolverParams {
        krylov: KrylovMethod::Gmres { restart: 30 },
        operator: OperatorRepr::Implicit,
        tol: 1e-8,
        maxiter: 1000,
        ..Default::default()
    };
    let gmres_precond = Preconditioner::Multiplicative(LocalSolverConfig::solver_default());
    let gmres_result = solve(
        categories.view(),
        &y,
        None,
        &gmres_params,
        Some(&gmres_precond),
    )
    .expect("gmres solve");

    // -----------------------------------------------------------------------
    // Print comparison
    // -----------------------------------------------------------------------
    print_comparison(
        "CG + additive Schwarz",
        &cg_result,
        "GMRES + mult. Schwarz",
        &gmres_result,
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
