mod common;

use std::env;
use std::process::Command;
use std::thread;
use std::time::{Duration, Instant};

use rayon::prelude::*;
use rstest::{fixture, rstest};
use schwarz_precond::{
    lsmr, mlsmr, LocalSolveError, LocalSolver, MlsmrOptions, Operator, PartitionWeights,
    SchwarzPreconditioner, SubdomainCore, SubdomainEntry,
};

use common::{make_schwarz_entries, FailingLocalSolver, TridiagOperator, UniformDiagLocalSolver};

fn make_entry<S: LocalSolver>(core: SubdomainCore, solver: S) -> SubdomainEntry<S> {
    SubdomainEntry::try_new(core, solver).expect("valid subdomain entry")
}

fn weighted_core(global_indices: Vec<u32>, weights: Vec<f64>) -> SubdomainCore {
    SubdomainCore::with_partition_weights(global_indices, PartitionWeights::NonUniform(weights))
        .expect("matching partition weights")
}

const NESTED_RAYON_CHILD_ENV: &str = "WITHIN_TEST_NESTED_RAYON_CHILD";

struct NestedRayonIdentitySolver {
    n: usize,
}

impl NestedRayonIdentitySolver {
    const CHUNK_SIZE: usize = 256;

    fn new(n: usize) -> Self {
        Self { n }
    }
}

impl LocalSolver for NestedRayonIdentitySolver {
    fn n_local(&self) -> usize {
        self.n
    }

    fn workspace_len(&self) -> usize {
        0
    }

    fn solve_local(
        &self,
        x: &mut [f64],
        _workspace: &mut [f64],
        allow_inner_parallelism: bool,
    ) -> Result<(), LocalSolveError> {
        // Identity, forking per chunk so the apply runs nested Rayon work.
        if allow_inner_parallelism {
            x.par_chunks_mut(Self::CHUNK_SIZE).for_each(|chunk| {
                std::hint::black_box(chunk);
            });
        }
        Ok(())
    }

    fn inner_parallelism_work_estimate(&self) -> usize {
        self.n.saturating_mul(32)
    }
}

fn make_nested_parallel_entries(n: usize) -> Vec<SubdomainEntry<NestedRayonIdentitySolver>> {
    let full_domain: Vec<u32> = (0..n as u32).collect();
    (0..2)
        .map(|_| {
            make_entry(
                SubdomainCore::uniform(full_domain.clone()),
                NestedRayonIdentitySolver::new(n),
            )
        })
        .collect()
}

fn run_nested_rayon_regression_case() {
    let n = 16_384;
    let rhs: Vec<f64> = (0..n).map(|i| ((i % 13) as f64) - 6.0).collect();

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .expect("test rayon pool");
    pool.install(|| {
        let schwarz = SchwarzPreconditioner::new(make_nested_parallel_entries(n));

        for _ in 0..8 {
            let mut z = vec![0.0; n];
            schwarz.apply(&rhs, &mut z).expect("apply succeeds");
            for (i, (&zi, &ri)) in z.iter().zip(&rhs).enumerate() {
                assert!(
                    (zi - 2.0 * ri).abs() <= 1e-12,
                    "unexpected additive result at index {i}: got {zi}, expected {}",
                    2.0 * ri,
                );
            }
        }
    });
}

#[test]
fn test_nested_rayon_deadlock_child() {
    if env::var_os(NESTED_RAYON_CHILD_ENV).is_none() {
        return;
    }
    run_nested_rayon_regression_case();
}

#[test]
fn test_nested_rayon_does_not_deadlock() {
    let current_exe = env::current_exe().expect("test binary path");
    let mut child = Command::new(current_exe)
        .env(NESTED_RAYON_CHILD_ENV, "1")
        .arg("test_nested_rayon_deadlock_child")
        .arg("--exact")
        .arg("--nocapture")
        .spawn()
        .expect("spawn nested rayon regression child");

    let timeout = Duration::from_secs(15);
    let deadline = Instant::now() + timeout;
    loop {
        if let Some(status) = child.try_wait().expect("poll nested rayon child") {
            assert!(
                status.success(),
                "nested rayon regression child exited with status {status}"
            );
            break;
        }

        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            panic!("nested Rayon regression child exceeded {:?}", timeout);
        }

        thread::sleep(Duration::from_millis(25));
    }
}
/// Additive Schwarz over `make_schwarz_entries(n)`, 20 DOFs unless `#[with(n)]` says otherwise.
#[fixture]
fn schwarz(#[default(20)] n: usize) -> SchwarzPreconditioner<UniformDiagLocalSolver> {
    SchwarzPreconditioner::new(make_schwarz_entries(n))
}

#[rstest]
fn test_additive_schwarz_reduces_iterations(
    schwarz: SchwarzPreconditioner<UniformDiagLocalSolver>,
) {
    let n = schwarz.nrows();
    let a = TridiagOperator::new(n, 3.0);
    let rhs = vec![1.0; n];

    let unprecond = lsmr(&a, &rhs, 1e-8, 200, None).expect("unpreconditioned lsmr");
    assert!(
        unprecond.converged,
        "Unpreconditioned LSMR did not converge"
    );

    let options = MlsmrOptions::default();
    let precond = mlsmr(&a, &rhs, &schwarz, 1e-8, 200, options).expect("preconditioned lsmr");
    assert!(precond.converged, "Preconditioned LSMR did not converge");

    assert!(
        precond.iterations <= unprecond.iterations,
        "Preconditioned ({}) should be <= unpreconditioned ({})",
        precond.iterations,
        unprecond.iterations
    );
}

#[rstest]
fn test_clone_produces_independent_preconditioner(
    #[from(schwarz)] original: SchwarzPreconditioner<UniformDiagLocalSolver>,
) {
    let n = original.nrows();
    let cloned = original.clone();

    // Apply on different inputs and verify both produce correct results.
    let r1: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
    let r2: Vec<f64> = (0..n).map(|i| ((n - i) as f64) * 0.5).collect();

    let mut z_orig = vec![0.0; n];
    let mut z_clone = vec![0.0; n];

    original
        .apply(&r1, &mut z_orig)
        .expect("original apply succeeds");
    cloned
        .apply(&r2, &mut z_clone)
        .expect("cloned apply succeeds");

    // Verify independently: apply the original with r2 to check the clone's result.
    let mut z_check = vec![0.0; n];
    original
        .apply(&r2, &mut z_check)
        .expect("check apply succeeds");
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
    original.apply(&r1, &mut z_orig2).expect("apply succeeds");
    for i in 0..n {
        assert!(
            (z_orig[i] - z_orig2[i]).abs() < 1e-14,
            "original result changed after clone apply at index {i}",
        );
    }
}

#[rstest]
fn test_additive_schwarz_operator_dimensions(
    #[with(10)] schwarz: SchwarzPreconditioner<UniformDiagLocalSolver>,
) {
    let n = 10;
    assert_eq!(schwarz.nrows(), n);
    assert_eq!(schwarz.ncols(), n);

    // apply and apply_adjoint should produce the same result (symmetric)
    let r = vec![1.0; n];
    let mut z1 = vec![0.0; n];
    let mut z2 = vec![0.0; n];
    schwarz.apply(&r, &mut z1).expect("apply succeeds");
    schwarz
        .apply_adjoint(&r, &mut z2)
        .expect("apply_adjoint succeeds");
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

#[rstest]
fn test_additive_schwarz_parallel_apply_stress_no_panics(
    #[with(64)] schwarz: SchwarzPreconditioner<UniformDiagLocalSolver>,
) {
    let n = schwarz.nrows();

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
                schwarz.apply(rhs, &mut z).expect("apply succeeds");
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

const DEEP_OVERLAP_DOFS: usize = 20_000;

/// 64-DOF windows every 8 DOFs: interior DOFs sum 8 terms, so their order shows in the bits.
#[fixture]
fn deep_overlap_entries() -> Vec<SubdomainEntry<UniformDiagLocalSolver>> {
    (0..=DEEP_OVERLAP_DOFS - 64)
        .step_by(8)
        .map(|start| {
            let indices: Vec<u32> = (start as u32..start as u32 + 64).collect();
            // Weights over six decades, distinct per subdomain, so the summands differ in scale.
            let weights = indices
                .iter()
                .map(|&g| {
                    let h = (start as u64 * 31 + g as u64) * 2_654_435_761;
                    10f64.powf(((h % 1_000) as f64 / 1_000.0 - 0.5) * 6.0)
                })
                .collect();
            make_entry(
                weighted_core(indices, weights),
                UniformDiagLocalSolver::new(64, 3.0),
            )
        })
        .collect()
}

/// Entries spanning 30 decades, so a reordered sum rounds differently.
#[fixture]
fn wide_range_rhs() -> Vec<f64> {
    (0..DEEP_OVERLAP_DOFS)
        .map(|i| ((i * 7919) % 997) as f64 * 10f64.powi((i % 31) as i32 - 15))
        .collect()
}

/// `Σᵢ Rᵢᵀ D̃ᵢ Aᵢ⁻¹ D̃ᵢ Rᵢ r`, accumulated by hand in subdomain order.
#[fixture]
fn subdomain_order_sum(
    deep_overlap_entries: Vec<SubdomainEntry<UniformDiagLocalSolver>>,
    wide_range_rhs: Vec<f64>,
) -> Vec<f64> {
    let mut sum = vec![0.0; DEEP_OVERLAP_DOFS];
    for entry in &deep_overlap_entries {
        let PartitionWeights::NonUniform(w) = entry.partition_weights() else {
            unreachable!("deep-overlap entries are weighted")
        };
        for (&g, &wi) in entry.global_indices().iter().zip(w) {
            let g = g as usize;
            sum[g] += wide_range_rhs[g] * wi * (1.0 / 3.0) * wi;
        }
    }
    sum
}

#[rstest]
fn test_additive_apply_is_the_subdomain_order_sum_at_any_thread_count(
    deep_overlap_entries: Vec<SubdomainEntry<UniformDiagLocalSolver>>,
    wide_range_rhs: Vec<f64>,
    subdomain_order_sum: Vec<f64>,
    #[values(1, 2, 3, 8)] threads: usize,
) {
    let schwarz = SchwarzPreconditioner::new(deep_overlap_entries);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("test rayon pool");
    let mut z = vec![f64::NAN; DEEP_OVERLAP_DOFS];
    pool.install(|| schwarz.apply(&wide_range_rhs, &mut z))
        .expect("apply succeeds");
    assert!(z
        .iter()
        .map(|v| v.to_bits())
        .eq(subdomain_order_sum.iter().map(|v| v.to_bits())));
}

use schwarz_precond::{BuildError, SolveError};
use std::error::Error;

// ============================================================================
// Additive Schwarz edge cases
// ============================================================================
// Additive Schwarz edge cases
// ============================================================================

#[test]
fn test_additive_schwarz_subdomains_accessor() {
    let n = 10;
    let entries = make_schwarz_entries(n);
    let expected_len = entries.len();
    let schwarz = SchwarzPreconditioner::new(entries);
    assert_eq!(schwarz.subdomains().len(), expected_len);
}

#[test]
fn test_additive_schwarz_apply_subdomain_empty_indices() {
    // SubdomainCore::uniform with empty indices and a solver with n_local=0 is
    // degenerate but not erroneous. The resulting preconditioner has n_dofs=0;
    // apply against a zero-length r/z slice should be a no-op.
    let solver = common::UniformDiagLocalSolver::new(0, 1.0);
    let core = SubdomainCore::uniform(vec![]);
    let entry = make_entry(core, solver);
    let schwarz = SchwarzPreconditioner::new(vec![entry]);
    assert_eq!(schwarz.nrows(), 0);

    let r: Vec<f64> = vec![];
    let mut z: Vec<f64> = vec![];
    schwarz
        .apply(&r, &mut z)
        .expect("apply with empty subdomain succeeds");
}

#[test]
fn test_solve_local_gets_x_of_its_own_length_beside_larger_subdomains() {
    let entries = vec![
        make_entry(
            SubdomainCore::uniform(vec![0, 1, 2]),
            UniformDiagLocalSolver::new(3, 1.0),
        ),
        make_entry(
            SubdomainCore::uniform(vec![3]),
            UniformDiagLocalSolver::new(1, 1.0),
        ),
    ];
    let schwarz = SchwarzPreconditioner::new(entries);
    let mut z = vec![0.0; 4];
    schwarz
        .apply(&[1.0, 2.0, 3.0, 4.0], &mut z)
        .expect("apply succeeds");
    assert_eq!(z, [1.0, 2.0, 3.0, 4.0]);
}

// ============================================================================
// Error Display and source() tests
// ============================================================================

#[test]
fn test_subdomain_entry_build_error_display_local_dof_mismatch() {
    let err = BuildError::LocalDofCountMismatch {
        index_count: 5,
        solver_n_local: 3,
    };
    let msg = err.to_string();
    assert!(msg.contains("5"), "missing index_count: {msg}");
    assert!(msg.contains("3"), "missing solver_n_local: {msg}");
}

#[test]
fn test_subdomain_core_build_error_display_partition_weight_mismatch() {
    let err = BuildError::PartitionWeightLengthMismatch {
        index_count: 3,
        weight_count: 5,
    };
    let msg = err.to_string();
    assert!(msg.contains("3"), "missing index_count: {msg}");
    assert!(msg.contains("5"), "missing weight_count: {msg}");
}

#[test]
fn test_local_solve_error_display() {
    let err = LocalSolveError::BackendFailed {
        context: "backsolve",
        message: "singular matrix".to_string(),
    };
    let msg = err.to_string();
    assert!(msg.contains("backsolve"), "missing context: {msg}");
    assert!(msg.contains("singular matrix"), "missing message: {msg}");
}

#[test]
fn test_solve_error_display_local_solve_failed() {
    let local_err = LocalSolveError::BackendFailed {
        context: "test",
        message: "fail".to_string(),
    };
    let err = SolveError::LocalSolveFailed {
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
fn test_solve_error_source() {
    let local_err = LocalSolveError::BackendFailed {
        context: "test",
        message: "err".to_string(),
    };
    let err = SolveError::LocalSolveFailed {
        subdomain: 0,
        source: local_err,
    };
    assert!(
        err.source().is_some(),
        "LocalSolveFailed should have a source"
    );
}

#[test]
fn test_solve_error_invalid_input_display_and_source() {
    let err = SolveError::InvalidInput {
        context: "test",
        message: "bad dimension".to_string(),
    };
    let msg = err.to_string();
    assert!(msg.contains("test"));
    assert!(msg.contains("bad dimension"));
    assert!(err.source().is_none());
}
// ============================================================================
// Validation error tests
// ============================================================================

#[test]
fn test_validate_local_dof_count_mismatch() {
    // Build entry where solver n_local != index count
    let solver = common::UniformDiagLocalSolver::new(3, 1.0); // n_local=3
    let core = SubdomainCore::uniform(vec![0, 1]); // 2 indices
    let result = SubdomainEntry::try_new(core, solver);
    match result {
        Err(BuildError::LocalDofCountMismatch { .. }) => {}
        Ok(_) => panic!("expected LocalDofCountMismatch, got Ok"),
        Err(other) => panic!("expected LocalDofCountMismatch, got: {:?}", other),
    }
}

#[test]
fn test_validate_partition_weight_length_mismatch() {
    let result = SubdomainCore::with_partition_weights(
        vec![0, 1],
        PartitionWeights::NonUniform(vec![1.0, 0.5, 0.3]),
    );
    match result {
        Err(BuildError::PartitionWeightLengthMismatch { .. }) => {}
        Ok(_) => panic!("expected PartitionWeightLengthMismatch, got Ok"),
        Err(other) => panic!("expected PartitionWeightLengthMismatch, got: {:?}", other),
    }
}

// ============================================================================
// Parallel readout path (n > PAR_READOUT_THRESHOLD = 100_000)
// ============================================================================

#[test]
fn test_additive_schwarz_parallel_readout_large_n() {
    // n > 100_000 triggers the par_chunks_mut readout path.
    let n = 150_002usize;
    // Build non-overlapping 2-DOF subdomains with diag_val = 2.0.
    // UniformDiagLocalSolver computes sol = rhs / diag_val.
    // With uniform weights (w=1 per DOF), each subdomain contributes
    // w * (w * rhs[i] / diag_val) = rhs[i] / diag_val = rhs[i] * 0.5.
    let mut entries: Vec<SubdomainEntry<UniformDiagLocalSolver>> = Vec::new();
    let mut i = 0usize;
    while i + 1 < n {
        entries.push(make_entry(
            SubdomainCore::uniform(vec![i as u32, (i + 1) as u32]),
            UniformDiagLocalSolver::new(2, 2.0),
        ));
        i += 2;
    }
    if i < n {
        entries.push(make_entry(
            SubdomainCore::uniform(vec![i as u32]),
            UniformDiagLocalSolver::new(1, 2.0),
        ));
    }

    let schwarz = SchwarzPreconditioner::new(entries);
    assert_eq!(schwarz.nrows(), n);

    let rhs = vec![4.0; n];
    let mut z = vec![0.0; n];
    let result = schwarz.apply(&rhs, &mut z);
    assert!(result.is_ok(), "apply should succeed: {:?}", result);

    // Each DOF: output = 1.0 * (1.0 * 4.0 / 2.0) = 2.0
    for (i, &v) in z.iter().enumerate() {
        assert!((v - 2.0_f64).abs() < 1e-12, "z[{i}] = {v}, expected 2.0",);
    }
}

// ============================================================================
// apply propagates local-solver failure
// ============================================================================

#[test]
fn test_additive_schwarz_apply_returns_err_on_solver_failure() {
    // FailingLocalSolver always returns Err — apply must propagate the error
    // (previously this silently filled z with NaN).
    let solver = FailingLocalSolver { n_local: 2 };
    let core = SubdomainCore::uniform(vec![0u32, 1]);
    let entry = make_entry(core, solver);

    let schwarz = SchwarzPreconditioner::new(vec![entry]);

    let n = schwarz.nrows();
    let rhs = vec![1.0; n];
    let mut z = vec![0.0; n];
    let result = schwarz.apply(&rhs, &mut z);

    match result {
        Err(SolveError::LocalSolveFailed { subdomain, .. }) => {
            assert_eq!(subdomain, 0, "failure should be reported for subdomain 0");
        }
        other => panic!("expected LocalSolveFailed, got: {:?}", other),
    }
}

// ============================================================================
// with_n_dofs: explicit dimension may exceed the covered index span
// ============================================================================

#[test]
fn with_n_dofs_pads_uncovered_tail_and_applies_zero() {
    // One subdomain covers DOFs 0..2 (diagonal solver, diag 2.0). The operator
    // is two columns wider — a structural-null tail no subdomain touches.
    // `with_n_dofs` sizes the preconditioner to the full width; the uncovered
    // DOFs stay in the null space and apply to exactly 0 (whereas `new` would
    // infer the covered span of 2 and come up short of the operator).
    let entries = vec![make_entry(
        SubdomainCore::uniform(vec![0u32, 1]),
        UniformDiagLocalSolver::new(2, 2.0),
    )];
    let n_dofs = 4;
    let padded = SchwarzPreconditioner::with_n_dofs(entries, n_dofs);
    assert_eq!((padded.nrows(), padded.ncols()), (n_dofs, n_dofs));

    let rhs = vec![4.0; n_dofs];
    let mut z = vec![f64::NAN; n_dofs];
    padded.apply(&rhs, &mut z).expect("apply succeeds");
    // Covered DOFs: 1.0 * (4.0 / 2.0) = 2.0; uncovered tail stays exactly 0.
    assert_eq!(z, vec![2.0, 2.0, 0.0, 0.0]);
}
