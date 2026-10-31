use std::env;
use std::process::Command;
use std::thread;
use std::time::{Duration, Instant};

use crate::config::{
    ApproxCholConfig, ApproxSchurConfig, LocalSolverConfig, SchurMode,
    DEFAULT_DENSE_SCHUR_THRESHOLD,
};
use crate::domain::{build_local_domains, Design, LocalDomain, PreparedDesign};
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

/// One plain pair: `n_keep · elim_ratio` levels, each in three weighted cells of `n_keep`.
fn synthetic_star_design(n_keep: usize, elim_ratio: usize) -> PreparedDesign<'static> {
    let (mut rows, mut cols, mut weights) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..n_keep * elim_ratio {
        for (col, w) in [
            (i % n_keep, 1.0),
            ((i + 1) % n_keep, 0.8),
            ((i.wrapping_mul(17).wrapping_add(3)) % n_keep, 0.6),
        ] {
            rows.push(i as u32);
            cols.push(col as u32);
            weights.push(w);
        }
    }
    let design = Design::from_levels_for_test(vec![rows, cols]);
    PreparedDesign::new(design, Some(&weights)).expect("valid weights")
}

fn run_block_elim_nested_rayon_case() {
    let n_keep = 1_024;
    let elim_ratio = 32;
    let config = LocalSolverConfig {
        approx_chol: ApproxCholConfig {
            split_merge: Some(8),
            seed: 42,
        },
        dense_threshold: DEFAULT_DENSE_SCHUR_THRESHOLD,
        ..Default::default()
    };
    let design = synthetic_star_design(n_keep, elim_ratio);
    let n_dofs = design.design.n_dofs;
    let (mut domain_pairs, _) = build_local_domains(&design, &config).expect("domains");
    let [domain] = &domain_pairs[..] else {
        panic!("one component expected, got {}", domain_pairs.len())
    };
    domain_pairs.push(domain.clone());
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
    let config = LocalSolverConfig {
        approx_chol: ApproxCholConfig::default(),
        schur,
        dense_threshold,
        ..Default::default()
    };
    let (domains, _) = build_local_domains(&synthetic_star_design(8, 4), &config).expect("domains");
    let [domain] = &domains[..] else {
        panic!("one component expected, got {}", domains.len())
    };
    let mut x: Vec<f64> = (0..domain.solver.n_local())
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let mut workspace = vec![0.0; domain.solver.workspace_len()];
    domain
        .solver
        .solve_local(&mut x, &mut workspace, false)
        .expect("local solve");
    x
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

#[test]
fn signed_wire_fixture_contains_uniform_weights() {
    use crate::operator::schwarz::{Preconditioner, Variant};
    use schwarz_precond::PartitionWeights;

    let bytes = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/preconditioner_v21.postcard"
    ));
    let preconditioner: Preconditioner = postcard::from_bytes(bytes).expect("signed fixture");
    let Variant::Additive(additive) = &preconditioner.inner else {
        panic!("signed fixture must be additive Schwarz");
    };
    assert!(additive
        .inner
        .subdomains()
        .iter()
        .any(|subdomain| { matches!(subdomain.partition_weights(), PartitionWeights::Uniform) }));
}
