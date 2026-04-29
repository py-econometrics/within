//! Tests for iterative refinement in the solver.
//!
//! Iterative refinement recomputes the normal-equation residual from observation
//! space and solves for a correction, closing the gap between normal-equation
//! residual accuracy and observation-space demeaning quality.

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use within::{
    Design, FactorMajorStore, KrylovMethod, LocalSolverConfig, Preconditioner, ReductionStrategy,
    Solver, SolverParams, Store,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a chain-like 2-FE design where factor 1 has `n_chain` levels connected
/// sequentially through factor 2. This creates a poorly connected graph with
/// high condition number — exactly the setting where iterative refinement helps.
///
/// Structure: observation i connects factor1_level=i to factor2_level=i,
/// and also factor1_level=i to factor2_level=i+1, forming a chain.
fn make_chain_design(n_chain: usize, seed: u64) -> (Design<FactorMajorStore>, Vec<f64>) {
    let mut rng = SmallRng::seed_from_u64(seed);

    let n_levels_1 = n_chain;
    let n_levels_2 = n_chain + 1;

    // Each level of factor 1 has exactly 2 observations connecting to
    // adjacent levels of factor 2, plus some extra observations for density.
    let mut f1 = Vec::new();
    let mut f2 = Vec::new();

    // Chain edges: level i of f1 connects to levels i and i+1 of f2
    for i in 0..n_chain {
        // Two observations per chain link (minimum for connectivity)
        f1.push(i as u32);
        f2.push(i as u32);
        f1.push(i as u32);
        f2.push((i + 1) as u32);
    }

    // Add a few extra observations per level to avoid singular systems
    for _ in 0..n_chain {
        let l1 = rng.random_range(0..n_levels_1 as u32);
        let l2 = rng.random_range(0..n_levels_2 as u32);
        f1.push(l1);
        f2.push(l2);
    }

    let n_obs = f1.len();
    let y: Vec<f64> = (0..n_obs).map(|_| rng.random::<f64>()).collect();

    let store = FactorMajorStore::new(vec![f1, f2], n_obs).expect("valid store");
    let design = Design::from_store(store).expect("valid design");

    (design, y)
}

/// Compute max absolute weighted group mean of demeaned values.
/// This is the observation-space quality metric: if demeaning is perfect,
/// the mean of demeaned values within every level of every factor is zero.
fn max_abs_group_mean(design: &Design<FactorMajorStore>, demeaned: &[f64]) -> f64 {
    let mut worst = 0.0f64;
    for q in 0..design.factors.len() {
        let n_levels = design.factors[q].n_levels;
        let mut sums = vec![0.0f64; n_levels];
        let mut counts = vec![0u32; n_levels];
        for (i, &d) in demeaned.iter().enumerate().take(design.n_rows) {
            let level = design.store.level(i, q) as usize;
            sums[level] += d;
            counts[level] += 1;
        }
        for (s, &c) in sums.iter().zip(counts.iter()) {
            if c > 0 {
                worst = worst.max((s / c as f64).abs());
            }
        }
    }
    worst
}

fn params_with_refinement(krylov: KrylovMethod, max_refinements: usize) -> SolverParams {
    SolverParams {
        krylov,
        tol: 1e-8,
        maxiter: 2000,
        max_refinements,
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------
// CG tests
// ---------------------------------------------------------------------------

#[test]
fn test_refinement_cg_improves_demeaning_quality() {
    // Chain design with n=200 creates κ(G) ~ O(n²), causing a significant gap
    // between normal-equation residual and demeaning quality.
    let (design, y) = make_chain_design(200, 42);
    let precond =
        Preconditioner::Additive(LocalSolverConfig::solver_default(), ReductionStrategy::Auto);

    // Without refinement
    let params_no_refine = params_with_refinement(KrylovMethod::Cg, 0);
    let solver_no =
        Solver::from_design(design, None, &params_no_refine, Some(&precond)).expect("build solver");
    // Need a fresh design for the second solver (design was moved)
    let (design2, _) = make_chain_design(200, 42);
    let result_no = solver_no.solve(&y).expect("solve without refinement");

    // With refinement
    let params_refine = params_with_refinement(KrylovMethod::Cg, 2);
    let solver_yes =
        Solver::from_design(design2, None, &params_refine, Some(&precond)).expect("build solver");
    let result_yes = solver_yes.solve(&y).expect("solve with refinement");

    assert!(result_no.converged, "initial solve should converge");
    assert!(result_yes.converged, "refined solve should converge");

    let (design3, _) = make_chain_design(200, 42);
    let demean_no = max_abs_group_mean(&design3, &result_no.demeaned);
    let demean_yes = max_abs_group_mean(&design3, &result_yes.demeaned);

    // Refinement should improve demeaning quality
    assert!(
        demean_yes <= demean_no,
        "refinement should not worsen demeaning: {demean_yes:.2e} > {demean_no:.2e}"
    );

    // The refined residual should be at least as good
    assert!(
        result_yes.final_residual <= result_no.final_residual + 1e-15,
        "refinement should not worsen residual: {} > {}",
        result_yes.final_residual,
        result_no.final_residual
    );
}

#[test]
fn test_refinement_cg_zero_means_old_behavior() {
    // max_refinements=0 should produce identical results to the old code path.
    let (design, y) = make_chain_design(50, 99);
    let precond =
        Preconditioner::Additive(LocalSolverConfig::solver_default(), ReductionStrategy::Auto);
    let params = params_with_refinement(KrylovMethod::Cg, 0);
    let solver = Solver::from_design(design, None, &params, Some(&precond)).expect("build solver");
    let result = solver.solve(&y).expect("solve");

    assert!(result.converged);
    assert!(result.final_residual < 1e-6);
}

#[test]
fn test_refinement_cg_well_conditioned_no_extra_iters() {
    // On a well-conditioned problem, the refinement check should find the
    // residual already small enough and not trigger any correction solves.
    let store = FactorMajorStore::new(vec![vec![0, 1, 0, 1, 2], vec![0, 0, 1, 1, 0]], 5)
        .expect("valid store");
    let design = Design::from_store(store).expect("design");
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let precond =
        Preconditioner::Additive(LocalSolverConfig::solver_default(), ReductionStrategy::Auto);

    // With refinement
    let params_refine = params_with_refinement(KrylovMethod::Cg, 2);
    let solver =
        Solver::from_design(design, None, &params_refine, Some(&precond)).expect("build solver");
    let result = solver.solve(&y).expect("solve");

    // Should converge with very few iterations (small problem)
    assert!(result.converged);
    assert!(
        result.iterations <= 10,
        "too many iterations on small problem: {}",
        result.iterations
    );
}

// ---------------------------------------------------------------------------
// GMRES tests
// ---------------------------------------------------------------------------

#[test]
fn test_refinement_gmres_improves_demeaning_quality() {
    let (design, y) = make_chain_design(200, 77);
    let precond = Preconditioner::Multiplicative(LocalSolverConfig::solver_default());

    // Without refinement
    let params_no_refine = params_with_refinement(KrylovMethod::Gmres { restart: 30 }, 0);
    let solver_no =
        Solver::from_design(design, None, &params_no_refine, Some(&precond)).expect("build solver");
    let (design2, _) = make_chain_design(200, 77);
    let result_no = solver_no.solve(&y).expect("solve without refinement");

    // With refinement
    let params_refine = params_with_refinement(KrylovMethod::Gmres { restart: 30 }, 2);
    let solver_yes =
        Solver::from_design(design2, None, &params_refine, Some(&precond)).expect("build solver");
    let result_yes = solver_yes.solve(&y).expect("solve with refinement");

    assert!(result_no.converged, "initial GMRES solve should converge");
    assert!(result_yes.converged, "refined GMRES solve should converge");

    let (design3, _) = make_chain_design(200, 77);
    let demean_no = max_abs_group_mean(&design3, &result_no.demeaned);
    let demean_yes = max_abs_group_mean(&design3, &result_yes.demeaned);

    assert!(
        demean_yes <= demean_no,
        "refinement should not worsen GMRES demeaning: {demean_yes:.2e} > {demean_no:.2e}"
    );
}

#[test]
fn test_refinement_gmres_well_conditioned() {
    let store = FactorMajorStore::new(vec![vec![0, 1, 0, 1, 2], vec![0, 0, 1, 1, 0]], 5)
        .expect("valid store");
    let design = Design::from_store(store).expect("design");
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let precond = Preconditioner::Multiplicative(LocalSolverConfig::solver_default());
    let params = params_with_refinement(KrylovMethod::Gmres { restart: 30 }, 2);
    let solver = Solver::from_design(design, None, &params, Some(&precond)).expect("build solver");
    let result = solver.solve(&y).expect("solve");

    assert!(result.converged);
    assert!(result.final_residual < 1e-6);
}

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

#[test]
fn test_refinement_zero_rhs() {
    let store = FactorMajorStore::new(vec![vec![0, 1, 0, 1, 2], vec![0, 0, 1, 1, 0]], 5)
        .expect("valid store");
    let design = Design::from_store(store).expect("design");
    let y = vec![0.0; 5];

    let params = params_with_refinement(KrylovMethod::Cg, 2);
    let solver = Solver::from_design(design, None, &params, None).expect("build solver");
    let result = solver.solve(&y).expect("solve");

    assert!(result.converged);
    assert_eq!(result.iterations, 0);
    assert!(result.x.iter().all(|&v| v == 0.0));
}

#[test]
fn test_refinement_unpreconditioned() {
    // Refinement should work without a preconditioner too.
    let (design, y) = make_chain_design(50, 123);
    let params = params_with_refinement(KrylovMethod::Cg, 2);
    let solver = Solver::from_design(design, None, &params, None).expect("build solver");
    let result = solver.solve(&y).expect("solve");

    assert!(result.converged);
    assert!(result.final_residual < 1e-6);
}

#[test]
fn test_refinement_batch_applies() {
    // Refinement should apply to each RHS in a batch solve.
    let (design, _) = make_chain_design(50, 456);
    let precond =
        Preconditioner::Additive(LocalSolverConfig::solver_default(), ReductionStrategy::Auto);
    let params = params_with_refinement(KrylovMethod::Cg, 2);
    let solver = Solver::from_design(design, None, &params, Some(&precond)).expect("build solver");

    let mut rng = SmallRng::seed_from_u64(789);
    let n_obs = solver.n_obs();
    let y1: Vec<f64> = (0..n_obs).map(|_| rng.random::<f64>()).collect();
    let y2: Vec<f64> = (0..n_obs).map(|_| rng.random::<f64>()).collect();

    let single1 = solver.solve(&y1).expect("solve y1");
    let single2 = solver.solve(&y2).expect("solve y2");
    let batch = solver.solve_batch(&[&y1, &y2]).expect("batch solve");

    assert_eq!(batch.n_rhs(), 2);
    assert!(batch.converged().iter().all(|&c| c));

    // Batch results should match single solves
    for (a, b) in batch.x(0).iter().zip(single1.x.iter()) {
        assert!((a - b).abs() < 1e-12, "batch/single x mismatch: {a} vs {b}");
    }
    for (a, b) in batch.x(1).iter().zip(single2.x.iter()) {
        assert!((a - b).abs() < 1e-12, "batch/single x mismatch: {a} vs {b}");
    }
}

#[test]
fn test_refinement_correction_tolerance_scaling() {
    // Batch solve with a structured
    // RHS (y = signal + noise, converges easily) alongside unstructured RHS columns
    // (pure noise, harder to demean). Without scaled correction tolerance, the
    // refinement solve overshoots by applying tol=1e-8 *relative to the small
    // correction RHS*, wasting iterations. With scaling, the correction only needs
    // to bring the total residual below the original tolerance.
    let (design, y_structured) = make_chain_design(200, 42);
    let n_obs = y_structured.len();

    // Generate unstructured columns (pure random noise — harder for CG)
    let mut rng = SmallRng::seed_from_u64(123);
    let x0: Vec<f64> = (0..n_obs).map(|_| rng.random::<f64>() - 0.5).collect();
    let x1: Vec<f64> = (0..n_obs).map(|_| rng.random::<f64>() - 0.5).collect();

    let precond =
        Preconditioner::Additive(LocalSolverConfig::solver_default(), ReductionStrategy::Auto);
    let params = params_with_refinement(KrylovMethod::Cg, 2);
    let solver = Solver::from_design(design, None, &params, Some(&precond)).expect("build solver");

    let batch = solver
        .solve_batch(&[&y_structured, &x0, &x1])
        .expect("batch solve");

    // All columns must converge
    for (i, &c) in batch.converged().iter().enumerate() {
        assert!(c, "column {i} did not converge");
    }

    // All residuals must be within tolerance
    for (i, &r) in batch.final_residual().iter().enumerate() {
        assert!(r < 1e-6, "column {i} residual too large: {r:.2e}");
    }

    // Refinement should not waste iterations: the correction solve for harder
    // columns should add only a few iterations, not double the count.
    let base_iters = batch.iterations()[0];
    for (i, &iters) in batch.iterations().iter().enumerate().skip(1) {
        assert!(
            iters <= base_iters * 2,
            "column {i} took {iters} iters (base={base_iters}), correction tolerance may not be scaled"
        );
    }
}
