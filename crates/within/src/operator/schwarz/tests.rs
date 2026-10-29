use std::env;
use std::process::Command;
use std::thread;
use std::time::{Duration, Instant};

use crate::config::{
    ApproxCholConfig, ApproxSchurConfig, LocalSolverConfig, SchurMode,
    DEFAULT_DENSE_SCHUR_THRESHOLD,
};
use schwarz_precond::SubdomainCore;

use crate::csr_block::CsrBlock;
use crate::domain::{build_local_domains, LocalDomain, PreparedDesign};
use crate::domain::{CrossTab, LocalComponent};
use crate::operator::schwarz::build_additive;
use schwarz_precond::{LocalSolver, Operator};

const BLOCK_ELIM_NESTED_RAYON_CHILD_ENV: &str = "WITHIN_TEST_BLOCK_ELIM_NESTED_RAYON_CHILD";

fn make_test_data() -> (PreparedDesign<'static>, Vec<LocalDomain>) {
    let design =
        PreparedDesign::from_levels_for_test(vec![vec![0, 1, 0, 1, 2], vec![0, 0, 1, 1, 0]]);
    let (domain_pairs, _) =
        build_local_domains(&design, &LocalSolverConfig::default()).expect("plain domains build");
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
    (
        CrossTab::eager(c),
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

fn run_block_elim_nested_rayon_case() {
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
        ..Default::default()
    };
    let rhs: Vec<f64> = (0..n_dofs).map(|i| ((i % 29) as f64) - 14.0).collect();

    let schwarz =
        build_additive(domain_pairs, &config, n_dofs).expect("build block-elim preconditioner");

    // Two subdomains: inner parallelism is off in a 1-thread pool and on in a 4-thread pool.
    let apply_in = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("test rayon pool")
            .install(|| {
                let mut z = vec![0.0; n_dofs];
                schwarz.apply(&rhs, &mut z).expect("apply succeeds");
                z.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            })
    };
    let reference = apply_in(1);
    assert!(reference.iter().all(|&b| f64::from_bits(b).is_finite()));
    for _ in 0..4 {
        assert!(
            apply_in(4) == reference,
            "inner parallelism changed the bits"
        );
    }
}

#[test]
fn test_block_elim_nested_rayon_child() {
    if env::var_os(BLOCK_ELIM_NESTED_RAYON_CHILD_ENV).is_none() {
        return;
    }
    run_block_elim_nested_rayon_case();
}

#[test]
fn test_block_elim_nested_rayon_does_not_deadlock() {
    let current_exe = env::current_exe().expect("test binary path");
    let mut child = Command::new(current_exe)
        .env(BLOCK_ELIM_NESTED_RAYON_CHILD_ENV, "1")
        .arg("test_block_elim_nested_rayon_child")
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
                "block-elim nested Rayon regression child exceeded {:?}",
                timeout
            );
        }

        thread::sleep(Duration::from_millis(25));
    }
}

#[test]
fn test_build_additive() {
    let (design, domain_pairs) = make_test_data();
    let schwarz = build_additive(
        domain_pairs,
        &LocalSolverConfig::default(),
        design.design.n_dofs,
    )
    .expect("build schwarz with explicit domains");
    let r = vec![1.0; design.design.n_dofs];
    let mut z = vec![0.0; design.design.n_dofs];
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
        ..Default::default()
    };
    let solver =
        crate::block_elim::BlockElimSolver::build(component, &config).expect("block-elim build");
    let r: Vec<f64> = (0..solver.n_local())
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    solver.solve_for_test(&r, 0.0)
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
