use std::env;
use std::process::Command;
use std::thread;
use std::time::{Duration, Instant};

use crate::config::{
    ApproxCholConfig, ApproxSchurConfig, LocalSolverConfig, ScalingConfig, SchurMode,
    DEFAULT_DENSE_SCHUR_THRESHOLD,
};
use schwarz_precond::SubdomainCore;

use crate::csr_block::CsrBlock;
use crate::domain::{build_local_domains, Design, LocalDomain};
use crate::domain::{CrossTab, LocalComponent};
use crate::operator::schwarz::build_additive_with_strategy;
use schwarz_precond::{LocalSolver, Operator, ReductionStrategy};

const BLOCK_ELIM_NESTED_RAYON_CHILD_ENV: &str = "WITHIN_TEST_BLOCK_ELIM_NESTED_RAYON_CHILD";

fn make_test_data() -> (Design<'static>, Vec<LocalDomain>) {
    let design = Design::from_levels_for_test(vec![vec![0, 1, 0, 1, 2], vec![0, 0, 1, 1, 0]]);
    let (domain_pairs, _) =
        build_local_domains(&design, None, &ScalingConfig::default()).expect("plain domains build");
    (design, domain_pairs)
}

fn synthetic_sparse_cross_tab(n_keep: usize, elim_ratio: usize) -> (CrossTab, Vec<f64>) {
    let n_rows = n_keep * elim_ratio;
    let n_cols = n_keep;
    let mut indptr = Vec::with_capacity(n_rows + 1);
    let mut indices = Vec::with_capacity(n_rows * 3);
    let mut data = Vec::with_capacity(n_rows * 3);
    let mut row_diag = vec![0.0; n_rows];
    let mut col_diag = vec![0.0; n_cols];

    indptr.push(0);
    for (i, row_diag_i) in row_diag.iter_mut().enumerate().take(n_rows) {
        let mut row = [
            (i % n_cols, 1.0),
            ((i + 1) % n_cols, 0.8),
            ((i.wrapping_mul(17).wrapping_add(3)) % n_cols, 0.6),
        ];
        row.sort_unstable_by_key(|&(col, _)| col);

        let mut row_sum = 0.0;
        let mut cursor = 0usize;
        while cursor < row.len() {
            let col = row[cursor].0;
            let mut value = row[cursor].1;
            cursor += 1;
            while cursor < row.len() && row[cursor].0 == col {
                value += row[cursor].1;
                cursor += 1;
            }

            indices.push(col as u32);
            data.push(value);
            row_sum += value;
            col_diag[col] += value;
        }

        *row_diag_i = row_sum;
        indptr.push(indices.len() as u32);
    }

    let c = CsrBlock {
        indptr,
        indices,
        data,
        nrows: n_rows,
        ncols: n_cols,
    };
    let ct = c.transpose();
    (
        CrossTab { c, ct },
        row_diag.into_iter().chain(col_diag).collect(),
    )
}

fn make_nested_block_elim_domain_pairs(
    n_keep: usize,
    elim_ratio: usize,
    n_subdomains: usize,
) -> (usize, Vec<LocalDomain>) {
    let (cross_tab, block_diagonals) = synthetic_sparse_cross_tab(n_keep, elim_ratio);
    let n_local = cross_tab.n_local();
    let global_indices: Vec<u32> = (0..n_local as u32).collect();

    let domain_pairs = (0..n_subdomains)
        .map(|_| LocalDomain {
            core: SubdomainCore::uniform(global_indices.clone()),
            component: LocalComponent::plain_for_test(cross_tab.clone(), block_diagonals.clone()),
        })
        .collect();
    (n_local, domain_pairs)
}

fn run_block_elim_parallel_reduction_regression_case() {
    let n_keep = 1_024;
    let elim_ratio = 32;
    let (n_dofs, domain_pairs) = make_nested_block_elim_domain_pairs(n_keep, elim_ratio, 2);
    let config = LocalSolverConfig {
        approx_chol: ApproxCholConfig {
            split_merge: Some(8),
            seed: 42,
        },
        schur: SchurMode::Approximate(ApproxSchurConfig {
            seed: 7,
            ..Default::default()
        }),
        dense_threshold: DEFAULT_DENSE_SCHUR_THRESHOLD,
        scaling: Default::default(),
    };
    let rhs: Vec<f64> = (0..n_dofs).map(|i| ((i % 29) as f64) - 14.0).collect();

    let (_, domain_pairs_atomic) = make_nested_block_elim_domain_pairs(n_keep, elim_ratio, 2);

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .expect("test rayon pool");
    pool.install(|| {
        let reduction = build_additive_with_strategy(
            domain_pairs,
            &config,
            ReductionStrategy::ParallelReduction,
            n_dofs,
        )
        .expect("build block-elim additive preconditioner");
        let atomic = build_additive_with_strategy(
            domain_pairs_atomic,
            &config,
            ReductionStrategy::AtomicScatter,
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
fn test_build_additive_with_strategy() {
    let (design, domain_pairs) = make_test_data();
    let config = LocalSolverConfig::default();
    let strategy = schwarz_precond::ReductionStrategy::default();
    let schwarz = build_additive_with_strategy(domain_pairs, &config, strategy, design.n_dofs)
        .expect("build schwarz with explicit domains");
    let r = vec![1.0; design.n_dofs];
    let mut z = vec![0.0; design.n_dofs];
    schwarz.apply(&r, &mut z).expect("schwarz apply succeeds");
}

/// Three-entry eliminated stars, so clique sampling is not
/// deterministic-exact and the two routes are separable.
fn small_subdomain_solve(schur: SchurMode, dense_threshold: usize) -> Vec<f64> {
    let (cross_tab, block_diagonals) = synthetic_sparse_cross_tab(8, 4);
    let component = LocalComponent::plain_for_test(cross_tab, block_diagonals);
    let config = LocalSolverConfig {
        approx_chol: ApproxCholConfig::default(),
        schur,
        dense_threshold,
        scaling: Default::default(),
    };
    let solver =
        crate::block_elim::BlockElimSolver::build(component, &config).expect("block-elim build");
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
        SchurMode::Approximate(ApproxSchurConfig {
            seed: 7,
            ..Default::default()
        }),
        DEFAULT_DENSE_SCHUR_THRESHOLD,
    );
    assert_eq!(exact, sampled_mode);
}

#[test]
fn test_dense_threshold_zero_disables_exact_route() {
    let exact = small_subdomain_solve(SchurMode::Exact, DEFAULT_DENSE_SCHUR_THRESHOLD);
    let approximate = small_subdomain_solve(
        SchurMode::Approximate(ApproxSchurConfig {
            seed: 7,
            ..Default::default()
        }),
        0,
    );
    assert_ne!(exact, approximate);
}
