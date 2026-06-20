mod common;

use std::env;
use std::process::Command;
use std::thread;
use std::time::{Duration, Instant};

use rayon::prelude::*;
use schwarz_precond::{
    lsmr, mlsmr, LocalSolveError, LocalSolver, Operator, PartitionWeights, ReductionStrategy,
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

fn make_overlapping_entries(n: usize) -> Vec<SubdomainEntry<UniformDiagLocalSolver>> {
    let mut entries = Vec::new();
    if n < 3 {
        return entries;
    }
    for start in (0..n.saturating_sub(2)).step_by(2) {
        let weights = if (start / 2) % 2 == 0 {
            vec![1.0, 0.5, 1.0]
        } else {
            vec![0.75, 1.0, 0.5]
        };
        entries.push(make_entry(
            weighted_core(
                vec![start as u32, (start + 1) as u32, (start + 2) as u32],
                weights,
            ),
            UniformDiagLocalSolver::new(3, 3.0),
        ));
    }
    // When n is even, the loop above leaves index n-1 uncovered; add a
    // trailing subdomain so the entries collectively cover [0, n).
    if entries
        .last()
        .and_then(|e| e.global_indices().iter().max())
        .copied()
        .is_none_or(|max_idx| (max_idx as usize) < n - 1)
    {
        let last_start = n - 3;
        entries.push(make_entry(
            weighted_core(
                vec![
                    last_start as u32,
                    (last_start + 1) as u32,
                    (last_start + 2) as u32,
                ],
                vec![1.0, 0.5, 1.0],
            ),
            UniformDiagLocalSolver::new(3, 3.0),
        ));
    }
    entries
}

fn assert_vec_close(lhs: &[f64], rhs: &[f64], tol: f64) {
    assert_eq!(lhs.len(), rhs.len(), "vector lengths differ");
    for (idx, (&a, &b)) in lhs.iter().zip(rhs.iter()).enumerate() {
        assert!(
            (a - b).abs() <= tol,
            "vectors differ at index {idx}: lhs={a}, rhs={b}, tol={tol}",
        );
    }
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

    fn scratch_size(&self) -> usize {
        self.n
    }

    fn solve_local(
        &self,
        rhs: &mut [f64],
        sol: &mut [f64],
        allow_inner_parallelism: bool,
    ) -> Result<(), LocalSolveError> {
        if allow_inner_parallelism {
            sol[..self.n]
                .par_chunks_mut(Self::CHUNK_SIZE)
                .enumerate()
                .for_each(|(chunk_idx, chunk)| {
                    let start = chunk_idx * Self::CHUNK_SIZE;
                    let end = start + chunk.len();
                    chunk.copy_from_slice(&rhs[start..end]);
                });
        } else {
            sol[..self.n].copy_from_slice(&rhs[..self.n]);
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

fn run_nested_parallel_reduction_regression_case() {
    let n = 16_384;
    let rhs: Vec<f64> = (0..n).map(|i| ((i % 13) as f64) - 6.0).collect();

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .expect("test rayon pool");
    pool.install(|| {
        let schwarz = SchwarzPreconditioner::new(
            make_nested_parallel_entries(n),
            ReductionStrategy::ParallelReduction,
        );

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
fn test_parallel_reduction_nested_rayon_deadlock_child() {
    if env::var_os(NESTED_RAYON_CHILD_ENV).is_none() {
        return;
    }
    run_nested_parallel_reduction_regression_case();
}

#[test]
fn test_parallel_reduction_nested_rayon_does_not_deadlock() {
    let current_exe = env::current_exe().expect("test binary path");
    let mut child = Command::new(current_exe)
        .env(NESTED_RAYON_CHILD_ENV, "1")
        .arg("test_parallel_reduction_nested_rayon_deadlock_child")
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
            panic!(
                "parallel reduction nested Rayon regression child exceeded {:?}",
                timeout
            );
        }

        thread::sleep(Duration::from_millis(25));
    }
}
#[test]
fn test_additive_schwarz_reduces_iterations() {
    let n = 20;
    let a = TridiagOperator::new(n, 3.0);
    let rhs = vec![1.0; n];

    let unprecond = lsmr(&a, &rhs, 1e-8, 200, None).expect("unpreconditioned lsmr");
    assert!(
        unprecond.converged,
        "Unpreconditioned LSMR did not converge"
    );

    let schwarz = SchwarzPreconditioner::new(make_schwarz_entries(n), ReductionStrategy::default());
    let precond = mlsmr(&a, &rhs, &schwarz, 1e-8, 200, None).expect("preconditioned lsmr");
    assert!(precond.converged, "Preconditioned LSMR did not converge");

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
    let original =
        SchwarzPreconditioner::new(make_schwarz_entries(n), ReductionStrategy::default());
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

#[test]
fn test_additive_schwarz_operator_dimensions() {
    let n = 10;
    let schwarz = SchwarzPreconditioner::new(make_schwarz_entries(n), ReductionStrategy::default());

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

#[test]
fn test_additive_schwarz_parallel_apply_stress_no_panics() {
    let n = 64;
    let schwarz = SchwarzPreconditioner::new(make_schwarz_entries(n), ReductionStrategy::default());

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

#[test]
fn test_additive_backends_match_on_overlapping_subdomains() {
    let n = 66;
    let rhs: Vec<f64> = (0..n).map(|i| ((i % 11) as f64) - 5.0).collect();

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .expect("test rayon pool");
    pool.install(|| {
        let atomic = SchwarzPreconditioner::new(
            make_overlapping_entries(n),
            ReductionStrategy::AtomicScatter,
        );
        let reduction = SchwarzPreconditioner::new(
            make_overlapping_entries(n),
            ReductionStrategy::ParallelReduction,
        );

        let mut z_atomic = vec![0.0; n];
        let mut z_reduction = vec![0.0; n];
        atomic
            .apply(&rhs, &mut z_atomic)
            .expect("atomic apply succeeds");
        reduction
            .apply(&rhs, &mut z_reduction)
            .expect("reduction apply succeeds");
        assert_vec_close(&z_atomic, &z_reduction, 1e-12);
    });
}

#[test]
fn test_additive_auto_matches_resolved_backend() {
    let n = 66;
    let rhs: Vec<f64> = (0..n).map(|i| ((3 * i) % 17) as f64 - 8.0).collect();

    for &n_threads in &[1usize, 4usize] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(n_threads)
            .build()
            .expect("test rayon pool");
        pool.install(|| {
            let auto =
                SchwarzPreconditioner::new(make_overlapping_entries(n), ReductionStrategy::Auto);
            let resolved = auto.reduction_strategy();
            assert_ne!(
                resolved,
                ReductionStrategy::Auto,
                "auto must resolve to a concrete backend"
            );

            let explicit = SchwarzPreconditioner::new(make_overlapping_entries(n), resolved);

            let mut z_auto = vec![0.0; n];
            let mut z_explicit = vec![0.0; n];
            auto.apply(&rhs, &mut z_auto).expect("auto apply succeeds");
            explicit
                .apply(&rhs, &mut z_explicit)
                .expect("explicit apply succeeds");
            assert_vec_close(&z_auto, &z_explicit, 1e-12);
        });
    }
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
    let schwarz = SchwarzPreconditioner::new(entries, ReductionStrategy::default());
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
    let schwarz = SchwarzPreconditioner::new(vec![entry], ReductionStrategy::default());
    assert_eq!(schwarz.nrows(), 0);

    let r: Vec<f64> = vec![];
    let mut z: Vec<f64> = vec![];
    schwarz
        .apply(&r, &mut z)
        .expect("apply with empty subdomain succeeds");
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
fn test_subdomain_entry_build_error_display_scratch_size_too_small() {
    let err = BuildError::ScratchSizeTooSmall {
        scratch_size: 2,
        required_min: 4,
    };
    let msg = err.to_string();
    assert!(msg.contains("2"), "missing scratch_size: {msg}");
    assert!(msg.contains("4"), "missing required_min: {msg}");
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
fn test_solve_error_display_synchronization() {
    let err = SolveError::Synchronization {
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

    let err2 = SolveError::Synchronization { context: "test" };
    assert!(
        err2.source().is_none(),
        "Synchronization should have no source"
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

    let schwarz = SchwarzPreconditioner::new(entries, ReductionStrategy::default());
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
    let solver = FailingLocalSolver {
        n_local: 2,
        scratch_size: 2,
    };
    let core = SubdomainCore::uniform(vec![0u32, 1]);
    let entry = make_entry(core, solver);

    let schwarz = SchwarzPreconditioner::new(vec![entry], ReductionStrategy::default());

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
