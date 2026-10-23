use std::env;
use std::process::Command;
use std::thread;
use std::time::{Duration, Instant};

use crate::config::{
    ApproxCholConfig, ApproxSchurConfig, LocalSolverConfig, SchurMode,
    DEFAULT_DENSE_SCHUR_THRESHOLD,
};
use crate::domain::{build_local_domains, Design, LocalDomain, PreparedDesign};
use crate::operator::schwarz::{build_additive, SchwarzConfig};
use crate::Effect;
use schwarz_precond::{LocalSolver, Operator, ReductionStrategy};

const BLOCK_ELIM_NESTED_RAYON_CHILD_ENV: &str = "WITHIN_TEST_BLOCK_ELIM_NESTED_RAYON_CHILD";

fn make_test_data() -> (PreparedDesign<'static>, Vec<LocalDomain>) {
    let design =
        PreparedDesign::from_levels_for_test(vec![vec![0, 1, 0, 1, 2], vec![0, 0, 1, 1, 0]]);
    let (domain_pairs, _) =
        build_local_domains(&design, &LocalSolverConfig::default()).expect("plain domains build");
    (design, domain_pairs)
}

/// Three weighted cells per eliminated level, so clique sampling is not deterministic-exact.
fn synthetic_sparse_design(n_keep: usize, elim_ratio: usize) -> PreparedDesign<'static> {
    let n_rows = n_keep * elim_ratio;
    let (mut rows, mut cols, mut weights) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..n_rows {
        let cells = [
            (i % n_keep, 1.0),
            ((i + 1) % n_keep, 0.8),
            ((i.wrapping_mul(17).wrapping_add(3)) % n_keep, 0.6),
        ];
        for (col, weight) in cells {
            rows.push(i as u32);
            cols.push(col as u32);
            weights.push(weight);
        }
    }
    let design = Design::new(vec![
        Effect::new(&rows, true, []).expect("eliminated effect"),
        Effect::new(&cols, true, []).expect("kept effect"),
    ])
    .expect("valid design")
    .into_owned();
    PreparedDesign::new(design, Some(&weights)).expect("valid weights")
}

fn nested_domains(
    prepared: &PreparedDesign<'_>,
    config: &LocalSolverConfig,
    n_subdomains: usize,
) -> Vec<LocalDomain> {
    (0..n_subdomains)
        .flat_map(|_| build_local_domains(prepared, config).expect("domains").0)
        .collect()
}

fn run_block_elim_parallel_reduction_regression_case() {
    let n_keep = 1_024;
    let elim_ratio = 32;
    let prepared = synthetic_sparse_design(n_keep, elim_ratio);
    let n_dofs = prepared.design.n_dofs;
    let config = LocalSolverConfig {
        approx_chol: ApproxCholConfig {
            split_merge: Some(8),
            seed: 42,
        },
        schur: SchurMode::Approximate(ApproxSchurConfig::default()),
        dense_threshold: DEFAULT_DENSE_SCHUR_THRESHOLD,
        ..Default::default()
    };
    let rhs: Vec<f64> = (0..n_dofs).map(|i| ((i % 29) as f64) - 14.0).collect();
    let domain_pairs = nested_domains(&prepared, &config, 2);
    let domain_pairs_atomic = nested_domains(&prepared, &config, 2);

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .expect("test rayon pool");
    pool.install(|| {
        let reduction = build_additive(
            domain_pairs,
            &SchwarzConfig {
                local_solver: config.clone(),
                reduction: ReductionStrategy::ParallelReduction,
            },
            n_dofs,
        )
        .expect("build block-elim additive preconditioner");
        let atomic = build_additive(
            domain_pairs_atomic,
            &SchwarzConfig {
                local_solver: config.clone(),
                reduction: ReductionStrategy::AtomicScatter,
            },
            n_dofs,
        )
        .expect("build block-elim atomic preconditioner");

        for _ in 0..4 {
            let mut z_reduction = vec![0.0; n_dofs];
            let mut z_atomic = vec![0.0; n_dofs];
            reduction
                .apply(&rhs, &mut z_reduction)
                .expect("reduction apply succeeds");
            atomic
                .apply(&rhs, &mut z_atomic)
                .expect("atomic apply succeeds");
            for (i, (&zr, &za)) in z_reduction.iter().zip(&z_atomic).enumerate() {
                assert!(
                    zr.is_finite() && za.is_finite(),
                    "non-finite result at index {i}: reduction={zr}, atomic={za}"
                );
                assert!(
                    (zr - za).abs() <= 1e-9,
                    "backend mismatch at index {i}: reduction={zr}, atomic={za}"
                );
            }
        }
    });
}

#[test]
fn test_block_elim_parallel_reduction_nested_rayon_child() {
    if env::var_os(BLOCK_ELIM_NESTED_RAYON_CHILD_ENV).is_none() {
        return;
    }
    run_block_elim_parallel_reduction_regression_case();
}

#[test]
fn test_block_elim_parallel_reduction_nested_rayon_does_not_deadlock() {
    let current_exe = env::current_exe().expect("test binary path");
    let mut child = Command::new(current_exe)
        .env(BLOCK_ELIM_NESTED_RAYON_CHILD_ENV, "1")
        .arg("test_block_elim_parallel_reduction_nested_rayon_child")
        .arg("--nocapture")
        .spawn()
        .expect("spawn block-elim nested Rayon regression child");

    let timeout = Duration::from_secs(30);
    let deadline = Instant::now() + timeout;
    loop {
        if let Some(status) = child.try_wait().expect("poll nested rayon child") {
            assert!(
                status.success(),
                "block-elim nested rayon regression child exited with status {status}"
            );
            break;
        }

        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            panic!(
                "block-elim parallel reduction nested Rayon regression child exceeded {:?}",
                timeout
            );
        }

        thread::sleep(Duration::from_millis(25));
    }
}

#[test]
fn test_build_additive() {
    let (design, domain_pairs) = make_test_data();
    let config = SchwarzConfig {
        local_solver: LocalSolverConfig::default(),
        reduction: ReductionStrategy::default(),
    };
    let schwarz = build_additive(domain_pairs, &config, design.design.n_dofs)
        .expect("build schwarz with explicit domains");
    let r = vec![1.0; design.design.n_dofs];
    let mut z = vec![0.0; design.design.n_dofs];
    schwarz.apply(&r, &mut z).expect("schwarz apply succeeds");
}

/// Three-entry eliminated stars, so clique sampling is not
/// deterministic-exact and the two routes are separable.
fn small_subdomain_solve(schur: SchurMode, dense_threshold: usize) -> Vec<f64> {
    let config = LocalSolverConfig {
        approx_chol: ApproxCholConfig::default(),
        schur,
        dense_threshold,
        ..Default::default()
    };
    let (domains, _) =
        build_local_domains(&synthetic_sparse_design(8, 4), &config).expect("domains");
    let [LocalDomain { solver, .. }] = &domains[..] else {
        panic!("one block expected, got {}", domains.len())
    };
    let mut rhs = vec![0.0; solver.scratch_size()];
    for (i, slot) in rhs.iter_mut().take(solver.n_local()).enumerate() {
        *slot = if i % 2 == 0 { 1.0 } else { -1.0 };
    }
    let mut solution = vec![0.0; solver.scratch_size()];
    solver
        .solve_local(&mut rhs, &mut solution, false)
        .expect("local solve");
    solution.truncate(solver.n_local());
    solution
}

#[test]
fn test_schur_mode_is_outranked_by_the_exact_route_for_small_reduced_system() {
    // Agreeing is only possible if both took the exact route.
    let exact = small_subdomain_solve(SchurMode::Exact, DEFAULT_DENSE_SCHUR_THRESHOLD);
    let sampled_mode = small_subdomain_solve(
        SchurMode::Approximate(ApproxSchurConfig::default()),
        DEFAULT_DENSE_SCHUR_THRESHOLD,
    );
    assert_eq!(exact, sampled_mode);
}

#[test]
fn test_dense_threshold_zero_disables_exact_route() {
    let exact = small_subdomain_solve(SchurMode::Exact, DEFAULT_DENSE_SCHUR_THRESHOLD);
    let approximate =
        small_subdomain_solve(SchurMode::Approximate(ApproxSchurConfig::default()), 0);
    assert_ne!(exact, approximate);
}
