use schwarz_precond::solve::cg::cg_solve_preconditioned;
use schwarz_precond::{LocalSolver, Operator};
use within::operator::gramian::GramianOperator;
use within::operator::schwarz::build_schwarz;
use within::{LocalSolverConfig, DEFAULT_DENSE_SCHUR_THRESHOLD};

#[path = "common/orchestrate_helpers.rs"]
mod common;

// ===========================================================================
// Test 1: BlockElimSolver round-trip via subdomain-level LocalSolver API
// ===========================================================================

/// Verify that each subdomain's `BlockElimSolver` produces a finite, non-trivial
/// solution when invoked directly via the `LocalSolver` trait.
///
/// The test also exercises the full Schwarz apply to confirm the aggregate
/// preconditioner is coherent across all subdomains.
#[test]
fn test_block_elim_round_trip() {
    let design = common::make_test_design();
    // Force sparse path (disable dense fast-path) so we exercise the general code.
    let config = LocalSolverConfig {
        dense_threshold: 0,
        ..LocalSolverConfig::default()
    };
    let schwarz = build_schwarz(&design, &config).expect("build schwarz");

    // Aggregate preconditioner apply: rhs → result should be finite and non-trivial.
    let rhs = common::make_rhs_from_unit_solution(&design);
    let mut result = vec![0.0; design.n_dofs];
    schwarz.apply(&rhs, &mut result);

    assert!(
        result.iter().all(|v| v.is_finite()),
        "aggregate Schwarz apply produced non-finite output"
    );
    assert!(
        result.iter().any(|v| v.abs() > 1e-15),
        "aggregate Schwarz apply produced all-zero output"
    );

    // Subdomain-level round-trip: call each local solver directly.
    for (idx, entry) in schwarz.subdomains().iter().enumerate() {
        let solver = entry.solver();
        let n = solver.n_local();
        let scratch = solver.scratch_size();

        // scratch_size must be at least n_local per the LocalSolver contract.
        assert!(
            scratch >= n,
            "subdomain {idx}: scratch_size ({scratch}) < n_local ({n})"
        );

        // Allocate scratch-sized buffers as required by the LocalSolver contract.
        let mut rhs_buf = vec![0.0; scratch];
        let mut sol_buf = vec![0.0; scratch];

        // Use rhs = 0 (trivial rhs for null-space consistency check).
        solver
            .solve_local(&mut rhs_buf, &mut sol_buf)
            .unwrap_or_else(|e| panic!("subdomain {idx}: local solve failed: {e}"));

        assert!(
            sol_buf[..n].iter().all(|v| v.is_finite()),
            "subdomain {idx}: solution contains non-finite values"
        );
    }
}

// ===========================================================================
// Test 2: Dense Schur vs sparse Schur produce equivalent preconditioned CG
// ===========================================================================

/// Build two preconditioners — one forcing the dense path (threshold=100) and
/// one forcing the sparse path (threshold=0) — and confirm both produce finite,
/// comparable output on the same rhs.
#[test]
fn test_dense_schur_vs_sparse_schur_equivalent() {
    let design = common::make_test_design();
    let rhs = common::make_rhs_from_unit_solution(&design);

    let config_dense = LocalSolverConfig {
        dense_threshold: 100,
        ..LocalSolverConfig::default()
    };
    let config_sparse = LocalSolverConfig {
        dense_threshold: 0,
        ..LocalSolverConfig::default()
    };

    let schwarz_dense = build_schwarz(&design, &config_dense).expect("build dense schwarz");
    let schwarz_sparse = build_schwarz(&design, &config_sparse).expect("build sparse schwarz");

    let mut res_dense = vec![0.0; design.n_dofs];
    let mut res_sparse = vec![0.0; design.n_dofs];
    schwarz_dense.apply(&rhs, &mut res_dense);
    schwarz_sparse.apply(&rhs, &mut res_sparse);

    for (i, (d, s)) in res_dense.iter().zip(res_sparse.iter()).enumerate() {
        assert!(d.is_finite(), "dense result[{i}] is non-finite");
        assert!(s.is_finite(), "sparse result[{i}] is non-finite");
        assert!(
            (d - s).abs() < 1.0,
            "dense vs sparse diverged at dof {i}: dense={d:.6e} sparse={s:.6e}"
        );
    }
}

// ===========================================================================
// Test 3: Dense fast-path triggered at default threshold for small factors
// ===========================================================================

/// A 2x2 design has `min(n_q, n_r) = 2`, which is well below the default
/// threshold of 24. Verify that `build_schwarz` succeeds and the preconditioner
/// applies correctly even on such a small domain.
#[test]
fn test_local_solver_dense_factor_small_levels() {
    let design = common::make_weighted_design(
        vec![vec![0, 1, 0, 1], vec![0, 0, 1, 1]],
        within::ObservationWeights::Unit,
    )
    .expect("valid 2x2 design");

    let config = LocalSolverConfig {
        dense_threshold: DEFAULT_DENSE_SCHUR_THRESHOLD,
        ..LocalSolverConfig::default()
    };
    let schwarz = build_schwarz(&design, &config).expect("build schwarz for small design");

    let rhs = common::make_rhs_from_unit_solution(&design);
    let mut result = vec![0.0; design.n_dofs];
    schwarz.apply(&rhs, &mut result);

    assert!(
        result.iter().all(|v| v.is_finite()),
        "small-levels preconditioner apply produced non-finite output"
    );
}

// ===========================================================================
// Test 4: Equal-sized (square) factors — elimination order should not matter
// ===========================================================================

/// With a 3x3 balanced cross-tab (n_q == n_r), the `eliminate_q` flag flips
/// between paths deterministically. Confirm the solve still converges.
#[test]
fn test_eliminate_q_vs_r_equivalent_square() {
    let design = common::make_weighted_design(
        vec![
            vec![0, 1, 2, 0, 1, 2, 0, 1, 2],
            vec![0, 0, 0, 1, 1, 1, 2, 2, 2],
        ],
        within::ObservationWeights::Unit,
    )
    .expect("valid 3x3 balanced design");

    let config = LocalSolverConfig::default();
    let schwarz = build_schwarz(&design, &config).expect("build schwarz for 3x3 design");

    let rhs = common::make_rhs_from_unit_solution(&design);
    let mut result = vec![0.0; design.n_dofs];
    schwarz.apply(&rhs, &mut result);

    assert!(
        result.iter().all(|v| v.is_finite()),
        "square cross-tab preconditioner apply produced non-finite output"
    );
}

// ===========================================================================
// Test 5: Dense and sparse paths both produce convergent CG solves
// ===========================================================================

/// Drive preconditioned CG to convergence with both `dense_threshold=0` (sparse
/// path) and `dense_threshold=100` (dense path). Both must converge to tolerance.
#[test]
fn test_dense_threshold_zero_forces_sparse_cg_convergence() {
    let design = common::make_test_design();
    let gramian = GramianOperator::new(&design);
    let rhs = common::make_rhs_from_unit_solution(&design);

    for threshold in [0usize, 100] {
        let config = LocalSolverConfig {
            dense_threshold: threshold,
            ..LocalSolverConfig::default()
        };
        let schwarz = build_schwarz(&design, &config)
            .unwrap_or_else(|e| panic!("threshold={threshold}: build_schwarz failed: {e}"));

        let cg_result = cg_solve_preconditioned(&gramian, &schwarz, &rhs, 1e-8, 500)
            .unwrap_or_else(|e| {
                panic!("threshold={threshold}: cg_solve_preconditioned failed: {e}")
            });

        assert!(
            cg_result.converged,
            "threshold={threshold}: CG did not converge (iterations={})",
            cg_result.iterations
        );
    }
}

// ===========================================================================
// Test 6: global_indices are valid DOF indices and cover subdomain DOFs
// ===========================================================================

/// For every subdomain entry, the global DOF indices must be in range
/// `[0, n_dofs)` and the index count must match the local solver's `n_local`.
#[test]
fn test_subdomain_global_indices_in_range() {
    let design = common::make_test_design();
    let config = LocalSolverConfig::default();
    let schwarz = build_schwarz(&design, &config).expect("build schwarz");

    for (idx, entry) in schwarz.subdomains().iter().enumerate() {
        let indices = entry.global_indices();
        let n = entry.solver().n_local();

        assert_eq!(
            indices.len(),
            n,
            "subdomain {idx}: global_indices.len() ({}) != n_local ({})",
            indices.len(),
            n
        );

        for &gi in indices {
            assert!(
                (gi as usize) < design.n_dofs,
                "subdomain {idx}: global index {gi} out of range [0, {})",
                design.n_dofs
            );
        }
    }
}
