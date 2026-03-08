mod common;

use rayon::prelude::*;
use schwarz_precond::domain::PartitionWeights;
use schwarz_precond::solve::cg::{cg_solve, cg_solve_preconditioned};
use schwarz_precond::{
    MultiplicativeSchwarzPreconditioner, Operator, OperatorResidualUpdater, SchwarzPreconditioner,
    SubdomainCore, SubdomainEntry,
};

use common::{make_schwarz_entries, DiagLocalSolver, DiagOperator, TridiagOperator};

#[test]
fn test_additive_schwarz_reduces_iterations() {
    let n = 20;
    let a = TridiagOperator::new(n, 3.0);
    let rhs = vec![1.0; n];

    let unprecond = cg_solve(&a, &rhs, 1e-8, 200).expect("unpreconditioned cg");
    assert!(unprecond.converged, "Unpreconditioned CG did not converge");

    let schwarz = SchwarzPreconditioner::new(make_schwarz_entries(n), n)
        .expect("valid additive schwarz preconditioner");
    let precond =
        cg_solve_preconditioned(&a, &schwarz, &rhs, 1e-8, 200).expect("preconditioned cg");
    assert!(precond.converged, "Preconditioned CG did not converge");

    assert!(
        precond.iterations <= unprecond.iterations,
        "Preconditioned ({}) should be <= unpreconditioned ({})",
        precond.iterations,
        unprecond.iterations
    );
}

#[test]
fn test_clone_produces_independent_preconditioner() {
    let n = 20;
    let original = SchwarzPreconditioner::new(make_schwarz_entries(n), n)
        .expect("valid additive schwarz preconditioner");
    let cloned = original.clone();

    // Apply on different inputs and verify both produce correct results.
    let r1: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
    let r2: Vec<f64> = (0..n).map(|i| ((n - i) as f64) * 0.5).collect();

    let mut z_orig = vec![0.0; n];
    let mut z_clone = vec![0.0; n];

    original.apply(&r1, &mut z_orig);
    cloned.apply(&r2, &mut z_clone);

    // Verify independently: apply the original with r2 to check the clone's result.
    let mut z_check = vec![0.0; n];
    original.apply(&r2, &mut z_check);

    for i in 0..n {
        assert!(
            (z_clone[i] - z_check[i]).abs() < 1e-14,
            "clone result differs at index {}: {} vs {}",
            i,
            z_clone[i],
            z_check[i],
        );
    }

    // Verify the original was not corrupted by the clone's apply.
    let mut z_orig2 = vec![0.0; n];
    original.apply(&r1, &mut z_orig2);
    for i in 0..n {
        assert!(
            (z_orig[i] - z_orig2[i]).abs() < 1e-14,
            "original result changed after clone apply at index {i}",
        );
    }
}

#[test]
fn test_additive_schwarz_operator_dimensions() {
    let n = 10;
    let schwarz = SchwarzPreconditioner::new(make_schwarz_entries(n), n)
        .expect("valid additive schwarz preconditioner");

    assert_eq!(schwarz.nrows(), n);
    assert_eq!(schwarz.ncols(), n);

    // apply and apply_adjoint should produce the same result (symmetric)
    let r = vec![1.0; n];
    let mut z1 = vec![0.0; n];
    let mut z2 = vec![0.0; n];
    schwarz.apply(&r, &mut z1);
    schwarz.apply_adjoint(&r, &mut z2);

    for i in 0..n {
        assert!(
            (z1[i] - z2[i]).abs() < 1e-14,
            "apply != apply_adjoint at index {}: {} vs {}",
            i,
            z1[i],
            z2[i]
        );
    }
}

#[test]
fn test_additive_schwarz_parallel_apply_stress_no_panics() {
    let n = 64;
    let schwarz = SchwarzPreconditioner::new(make_schwarz_entries(n), n)
        .expect("valid additive schwarz preconditioner");

    let rhs_columns: Vec<Vec<f64>> = (0..128)
        .map(|k| {
            (0..n)
                .map(|i| ((i + k) % 19) as f64 - 9.0)
                .collect::<Vec<_>>()
        })
        .collect();

    let outputs: Vec<Vec<f64>> = rhs_columns
        .par_iter()
        .map(|rhs| {
            let mut z = vec![0.0; n];
            for _ in 0..16 {
                schwarz.apply(rhs, &mut z);
            }
            z
        })
        .collect();

    assert_eq!(outputs.len(), rhs_columns.len());
    assert!(
        outputs.iter().flatten().all(|v| v.is_finite()),
        "all outputs should remain finite under concurrent apply stress",
    );
}

// ============================================================================
// Multiplicative Schwarz tests
// ============================================================================

#[test]
fn test_multiplicative_single_subdomain_exact() {
    let a = DiagOperator {
        values: vec![2.0, 3.0, 4.0],
    };
    let solver = DiagLocalSolver::new(&[2.0, 3.0, 4.0]);
    let core = SubdomainCore::uniform(vec![0, 1, 2]);
    let entry = SubdomainEntry::new(core, solver);

    let updater = OperatorResidualUpdater::new(&a, 3);
    let prec = MultiplicativeSchwarzPreconditioner::new(vec![entry], updater, 3, false)
        .expect("valid multiplicative preconditioner");

    let r = vec![6.0, 9.0, 12.0];
    let mut z = vec![0.0; 3];
    prec.apply(&r, &mut z);

    assert!((z[0] - 3.0).abs() < 1e-12);
    assert!((z[1] - 3.0).abs() < 1e-12);
    assert!((z[2] - 3.0).abs() < 1e-12);
}

#[test]
fn test_multiplicative_two_nonoverlapping_subdomains() {
    let a = DiagOperator {
        values: vec![2.0, 3.0, 1.0, 5.0],
    };

    let solver0 = DiagLocalSolver::new(&[2.0, 3.0]);
    let core0 = SubdomainCore::uniform(vec![0, 1]);
    let entry0 = SubdomainEntry::new(core0, solver0);

    let solver1 = DiagLocalSolver::new(&[1.0, 5.0]);
    let core1 = SubdomainCore::uniform(vec![2, 3]);
    let entry1 = SubdomainEntry::new(core1, solver1);

    let updater = OperatorResidualUpdater::new(&a, 4);
    let prec = MultiplicativeSchwarzPreconditioner::new(vec![entry0, entry1], updater, 4, false)
        .expect("valid multiplicative preconditioner");

    let r = vec![4.0, 9.0, 3.0, 10.0];
    let mut z = vec![0.0; 4];
    prec.apply(&r, &mut z);

    assert!((z[0] - 2.0).abs() < 1e-12);
    assert!((z[1] - 3.0).abs() < 1e-12);
    assert!((z[2] - 3.0).abs() < 1e-12);
    assert!((z[3] - 2.0).abs() < 1e-12);
}

#[test]
fn test_multiplicative_overlapping_residual_update() {
    let a = DiagOperator {
        values: vec![2.0, 3.0, 4.0],
    };

    let solver0 = DiagLocalSolver::new(&[2.0, 3.0]);
    let core0 = SubdomainCore {
        global_indices: vec![0, 1],
        partition_weights: PartitionWeights::NonUniform(vec![1.0, 0.5]),
    };
    let entry0 = SubdomainEntry::new(core0, solver0);

    let solver1 = DiagLocalSolver::new(&[3.0, 4.0]);
    let core1 = SubdomainCore {
        global_indices: vec![1, 2],
        partition_weights: PartitionWeights::NonUniform(vec![0.5, 1.0]),
    };
    let entry1 = SubdomainEntry::new(core1, solver1);

    let updater = OperatorResidualUpdater::new(&a, 3);
    let prec = MultiplicativeSchwarzPreconditioner::new(vec![entry0, entry1], updater, 3, false)
        .expect("valid multiplicative preconditioner");

    let r = vec![2.0, 6.0, 8.0];
    let mut z = vec![0.0; 3];
    prec.apply(&r, &mut z);

    for &v in &z {
        assert!(v.is_finite(), "z contains non-finite value: {}", v);
    }

    let mut residual = vec![0.0; 3];
    a.apply(&z, &mut residual);
    let resid_norm: f64 = (0..3)
        .map(|i| (r[i] - residual[i]).powi(2))
        .sum::<f64>()
        .sqrt();
    let r_norm: f64 = r.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(
        resid_norm < r_norm,
        "Preconditioner did not reduce residual: {} >= {}",
        resid_norm,
        r_norm
    );
}

#[test]
fn test_symmetric_multiplicative_reduces_residual_more() {
    let a = DiagOperator {
        values: vec![2.0, 3.0, 4.0, 5.0],
    };

    let make_entries = || {
        let s0 = DiagLocalSolver::new(&[2.0, 3.0]);
        let c0 = SubdomainCore {
            global_indices: vec![0, 1],
            partition_weights: PartitionWeights::NonUniform(vec![1.0, 0.5]),
        };
        let s1 = DiagLocalSolver::new(&[3.0, 4.0]);
        let c1 = SubdomainCore {
            global_indices: vec![1, 2],
            partition_weights: PartitionWeights::NonUniform(vec![0.5, 0.5]),
        };
        let s2 = DiagLocalSolver::new(&[4.0, 5.0]);
        let c2 = SubdomainCore {
            global_indices: vec![2, 3],
            partition_weights: PartitionWeights::NonUniform(vec![0.5, 1.0]),
        };
        vec![
            SubdomainEntry::new(c0, s0),
            SubdomainEntry::new(c1, s1),
            SubdomainEntry::new(c2, s2),
        ]
    };

    let r = vec![4.0, 9.0, 12.0, 15.0];

    let updater_fwd = OperatorResidualUpdater::new(&a, 4);
    let prec_fwd = MultiplicativeSchwarzPreconditioner::new(make_entries(), updater_fwd, 4, false)
        .expect("valid forward multiplicative preconditioner");
    let mut z_fwd = vec![0.0; 4];
    prec_fwd.apply(&r, &mut z_fwd);

    let updater_sym = OperatorResidualUpdater::new(&a, 4);
    let prec_sym = MultiplicativeSchwarzPreconditioner::new(make_entries(), updater_sym, 4, true)
        .expect("valid symmetric multiplicative preconditioner");
    let mut z_sym = vec![0.0; 4];
    prec_sym.apply(&r, &mut z_sym);

    let resid_norm = |z: &[f64]| -> f64 {
        let mut az = vec![0.0; 4];
        a.apply(z, &mut az);
        (0..4).map(|i| (r[i] - az[i]).powi(2)).sum::<f64>().sqrt()
    };

    let rn_fwd = resid_norm(&z_fwd);
    let rn_sym = resid_norm(&z_sym);
    assert!(
        rn_sym <= rn_fwd + 1e-12,
        "Symmetric should not be worse: sym={} > fwd={}",
        rn_sym,
        rn_fwd
    );
}

#[test]
fn test_multiplicative_with_tridiag() {
    let n = 6;
    let a = TridiagOperator::new(n, 3.0);
    let make_solver = |diag_val: f64| DiagLocalSolver::new(&[diag_val, diag_val]);

    let e0 = SubdomainEntry::new(SubdomainCore::uniform(vec![0, 1]), make_solver(3.0));
    let e1 = SubdomainEntry::new(SubdomainCore::uniform(vec![2, 3]), make_solver(3.0));
    let e2 = SubdomainEntry::new(SubdomainCore::uniform(vec![4, 5]), make_solver(3.0));

    let updater = OperatorResidualUpdater::new(&a, n);
    let prec = MultiplicativeSchwarzPreconditioner::new(vec![e0, e1, e2], updater, n, true)
        .expect("valid multiplicative preconditioner");

    let r = vec![1.0; n];
    let mut z = vec![0.0; n];
    prec.apply(&r, &mut z);

    for (i, &v) in z.iter().enumerate() {
        assert!(v.is_finite(), "z[{}] = {} is not finite", i, v);
    }
    let z_norm: f64 = z.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(z_norm > 1e-14, "z is zero");

    let mut az = vec![0.0; n];
    a.apply(&z, &mut az);
    let resid_norm: f64 = (0..n).map(|i| (r[i] - az[i]).powi(2)).sum::<f64>().sqrt();
    let r_norm: f64 = r.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(
        resid_norm < r_norm,
        "No residual reduction: {} >= {}",
        resid_norm,
        r_norm
    );
}

// ============================================================================
// IdentityOperator tests
// ============================================================================

use schwarz_precond::{
    ApplyError, IdentityOperator, LocalSolveError, PreconditionerBuildError, SolveError,
};
use std::error::Error;

#[test]
fn test_identity_operator_dimensions() {
    let id = IdentityOperator::new(7);
    assert_eq!(id.nrows(), 7);
    assert_eq!(id.ncols(), 7);
}

#[test]
fn test_identity_operator_apply() {
    let id = IdentityOperator::new(4);
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let mut y = vec![0.0; 4];
    id.apply(&x, &mut y);
    assert_eq!(y, x);
}

#[test]
fn test_identity_operator_apply_adjoint() {
    let id = IdentityOperator::new(4);
    let x = vec![5.0, 6.0, 7.0, 8.0];
    let mut y = vec![0.0; 4];
    id.apply_adjoint(&x, &mut y);
    assert_eq!(y, x);
}

// ============================================================================
// Default try_apply / try_apply_adjoint tests
// ============================================================================

#[test]
fn test_try_apply_default_succeeds() {
    let a = TridiagOperator::new(5, 3.0);
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let mut y_apply = vec![0.0; 5];
    let mut y_try = vec![0.0; 5];
    a.apply(&x, &mut y_apply);
    let result = a.try_apply(&x, &mut y_try);
    assert!(result.is_ok());
    assert_eq!(y_apply, y_try);
}

#[test]
fn test_try_apply_adjoint_default_succeeds() {
    let a = TridiagOperator::new(5, 3.0);
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let mut y_apply = vec![0.0; 5];
    let mut y_try = vec![0.0; 5];
    a.apply_adjoint(&x, &mut y_apply);
    let result = a.try_apply_adjoint(&x, &mut y_try);
    assert!(result.is_ok());
    assert_eq!(y_apply, y_try);
}

// ============================================================================
// Additive Schwarz edge cases
// ============================================================================

#[test]
fn test_additive_schwarz_subdomains_accessor() {
    let n = 10;
    let entries = make_schwarz_entries(n);
    let expected_len = entries.len();
    let schwarz =
        SchwarzPreconditioner::new(entries, n).expect("valid additive schwarz preconditioner");
    assert_eq!(schwarz.subdomains().len(), expected_len);
}

#[test]
fn test_additive_schwarz_apply_subdomain_empty_indices() {
    // SubdomainCore::uniform with empty indices produces Uniform(0) weights,
    // and a solver with n_local=0. Validation requires n_local == index_count
    // and scratch_size >= index_count, so n_local=0 with 0 indices should pass.
    let solver = common::UniformDiagLocalSolver::new(0, 1.0);
    let core = SubdomainCore::uniform(vec![]);
    let entry = SubdomainEntry::new(core, solver);
    let result = SchwarzPreconditioner::new(vec![entry], 5);
    match result {
        Ok(schwarz) => {
            // Verify apply works with empty subdomain
            let r = vec![1.0; 5];
            let mut z = vec![0.0; 5];
            schwarz.apply(&r, &mut z);
            // Empty subdomain contributes nothing
            for &v in &z {
                assert!((v - 0.0).abs() < 1e-14);
            }
        }
        Err(_) => {
            // Validation prevents empty subdomain — that's acceptable too
        }
    }
}

// ============================================================================
// Multiplicative Schwarz accessor
// ============================================================================

#[test]
fn test_multiplicative_schwarz_subdomains_accessor() {
    let a = DiagOperator {
        values: vec![2.0, 3.0, 4.0],
    };
    let solver = DiagLocalSolver::new(&[2.0, 3.0, 4.0]);
    let core = SubdomainCore::uniform(vec![0, 1, 2]);
    let entry = SubdomainEntry::new(core, solver);

    let updater = OperatorResidualUpdater::new(&a, 3);
    let prec = MultiplicativeSchwarzPreconditioner::new(vec![entry], updater, 3, false)
        .expect("valid multiplicative preconditioner");

    assert_eq!(prec.subdomains().len(), 1);
}

// ============================================================================
// Error Display and source() tests
// ============================================================================

#[test]
fn test_preconditioner_build_error_display_local_dof_mismatch() {
    let err = PreconditionerBuildError::LocalDofCountMismatch {
        subdomain: 2,
        index_count: 5,
        solver_n_local: 3,
    };
    let msg = err.to_string();
    assert!(
        msg.contains("subdomain 2"),
        "missing subdomain index: {msg}"
    );
    assert!(msg.contains("5"), "missing index_count: {msg}");
    assert!(msg.contains("3"), "missing solver_n_local: {msg}");
}

#[test]
fn test_preconditioner_build_error_display_scratch_size_too_small() {
    let err = PreconditionerBuildError::ScratchSizeTooSmall {
        subdomain: 1,
        scratch_size: 2,
        required_min: 4,
    };
    let msg = err.to_string();
    assert!(
        msg.contains("subdomain 1"),
        "missing subdomain index: {msg}"
    );
    assert!(msg.contains("2"), "missing scratch_size: {msg}");
    assert!(msg.contains("4"), "missing required_min: {msg}");
}

#[test]
fn test_preconditioner_build_error_display_partition_weight_mismatch() {
    let err = PreconditionerBuildError::PartitionWeightLengthMismatch {
        subdomain: 0,
        index_count: 3,
        weight_count: 5,
    };
    let msg = err.to_string();
    assert!(
        msg.contains("subdomain 0"),
        "missing subdomain index: {msg}"
    );
    assert!(msg.contains("3"), "missing index_count: {msg}");
    assert!(msg.contains("5"), "missing weight_count: {msg}");
}

#[test]
fn test_preconditioner_build_error_display_global_index_out_of_bounds() {
    let err = PreconditionerBuildError::GlobalIndexOutOfBounds {
        subdomain: 3,
        local_index: 1,
        global_index: 99,
        n_dofs: 50,
    };
    let msg = err.to_string();
    assert!(
        msg.contains("subdomain 3"),
        "missing subdomain index: {msg}"
    );
    assert!(msg.contains("99"), "missing global_index: {msg}");
    assert!(msg.contains("50"), "missing n_dofs: {msg}");
}

#[test]
fn test_local_solve_error_display() {
    let err = LocalSolveError::ApproxCholSolveFailed {
        context: "backsolve",
        message: "singular matrix".to_string(),
    };
    let msg = err.to_string();
    assert!(msg.contains("backsolve"), "missing context: {msg}");
    assert!(msg.contains("singular matrix"), "missing message: {msg}");
}

#[test]
fn test_apply_error_display_local_solve_failed() {
    let local_err = LocalSolveError::ApproxCholSolveFailed {
        context: "test",
        message: "fail".to_string(),
    };
    let err = ApplyError::LocalSolveFailed {
        subdomain: 7,
        source: local_err,
    };
    let msg = err.to_string();
    assert!(msg.contains("subdomain 7"), "missing subdomain: {msg}");
    assert!(
        msg.contains("local solve failed"),
        "missing description: {msg}"
    );
}

#[test]
fn test_apply_error_display_synchronization() {
    let err = ApplyError::Synchronization {
        context: "mutex.lock",
    };
    let msg = err.to_string();
    assert!(
        msg.contains("synchronization"),
        "missing sync keyword: {msg}"
    );
    assert!(msg.contains("mutex.lock"), "missing context: {msg}");
}

#[test]
fn test_apply_error_source() {
    let local_err = LocalSolveError::ApproxCholSolveFailed {
        context: "test",
        message: "err".to_string(),
    };
    let err = ApplyError::LocalSolveFailed {
        subdomain: 0,
        source: local_err,
    };
    assert!(
        err.source().is_some(),
        "LocalSolveFailed should have a source"
    );

    let err2 = ApplyError::Synchronization { context: "test" };
    assert!(
        err2.source().is_none(),
        "Synchronization should have no source"
    );
}

#[test]
fn test_solve_error_display() {
    let apply_err = ApplyError::Synchronization { context: "test" };
    let err = SolveError::Apply(apply_err);
    let msg = err.to_string();
    assert!(
        msg.contains("operator apply failed"),
        "missing prefix: {msg}"
    );
}

#[test]
fn test_solve_error_source() {
    let apply_err = ApplyError::Synchronization { context: "test" };
    let err = SolveError::Apply(apply_err);
    assert!(
        err.source().is_some(),
        "SolveError::Apply should have a source"
    );
}

#[test]
fn test_solve_error_from_apply_error() {
    let apply_err = ApplyError::Synchronization { context: "conv" };
    let solve_err: SolveError = apply_err.into();
    match solve_err {
        SolveError::Apply(ApplyError::Synchronization { context }) => {
            assert_eq!(context, "conv");
        }
        _ => panic!("expected SolveError::Apply(Synchronization)"),
    }
}

// ============================================================================
// Validation error tests
// ============================================================================

#[test]
fn test_validate_local_dof_count_mismatch() {
    // Build entry where solver n_local != index count
    let solver = common::UniformDiagLocalSolver::new(3, 1.0); // n_local=3
    let core = SubdomainCore::uniform(vec![0, 1]); // 2 indices
    let entry = SubdomainEntry::new(core, solver);
    let result = SchwarzPreconditioner::new(vec![entry], 10);
    match result {
        Err(PreconditionerBuildError::LocalDofCountMismatch { .. }) => {}
        Ok(_) => panic!("expected LocalDofCountMismatch, got Ok"),
        Err(other) => panic!("expected LocalDofCountMismatch, got: {:?}", other),
    }
}

#[test]
fn test_validate_global_index_out_of_bounds() {
    let solver = common::UniformDiagLocalSolver::new(2, 1.0);
    let core = SubdomainCore::uniform(vec![0, 10]); // index 10 >= n_dofs=5
    let entry = SubdomainEntry::new(core, solver);
    let result = SchwarzPreconditioner::new(vec![entry], 5);
    match result {
        Err(PreconditionerBuildError::GlobalIndexOutOfBounds { .. }) => {}
        Ok(_) => panic!("expected GlobalIndexOutOfBounds, got Ok"),
        Err(other) => panic!("expected GlobalIndexOutOfBounds, got: {:?}", other),
    }
}

#[test]
fn test_validate_partition_weight_length_mismatch() {
    use schwarz_precond::domain::PartitionWeights;
    let solver = common::UniformDiagLocalSolver::new(2, 1.0);
    let core = SubdomainCore {
        global_indices: vec![0, 1],
        partition_weights: PartitionWeights::NonUniform(vec![1.0, 0.5, 0.3]), // 3 weights, 2 indices
    };
    let entry = SubdomainEntry::new(core, solver);
    let result = SchwarzPreconditioner::new(vec![entry], 10);
    match result {
        Err(PreconditionerBuildError::PartitionWeightLengthMismatch { .. }) => {}
        Ok(_) => panic!("expected PartitionWeightLengthMismatch, got Ok"),
        Err(other) => panic!("expected PartitionWeightLengthMismatch, got: {:?}", other),
    }
}
