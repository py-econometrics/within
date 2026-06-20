mod design_tests {
    use crate::domain::Design;
    use crate::observation::FactorMajorStore;
    use crate::operator::DesignOperator;
    use schwarz_precond::Operator;

    fn make_test_design() -> Design<FactorMajorStore> {
        let store = FactorMajorStore::new(vec![vec![0, 1, 2, 0, 1], vec![0, 1, 2, 3, 0]], 5)
            .expect("valid factor-major store");
        Design::from_store(store).expect("valid test design")
    }

    #[test]
    fn test_design_operator_dimensions() {
        let schema = make_test_design();
        let op = DesignOperator::new(&schema, None);
        assert_eq!(op.nrows(), 5);
        assert_eq!(op.ncols(), 7);
    }

    #[test]
    fn test_design_operator_adjoint() {
        let schema = make_test_design();
        let op = DesignOperator::new(&schema, None);

        let x = vec![1.0, -0.5, 2.0, 0.3, -1.0, 0.7, 1.5];
        let r = vec![0.1, 0.2, -0.3, 0.4, -0.5];

        let mut dx = vec![0.0; 5];
        op.apply(&x, &mut dx).expect("apply succeeds");
        let lhs: f64 = dx.iter().zip(r.iter()).map(|(a, b)| a * b).sum();

        let mut dtr = vec![0.0; 7];
        op.apply_adjoint(&r, &mut dtr)
            .expect("apply_adjoint succeeds");
        let rhs: f64 = x.iter().zip(dtr.iter()).map(|(a, b)| a * b).sum();
        assert!((lhs - rhs).abs() < 1e-12);
    }

    #[test]
    fn test_apply_unweighted_values() {
        let schema = make_test_design();
        let op = DesignOperator::new(&schema, None);
        let x = vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 40.0];
        let mut y = vec![0.0; 5];
        op.apply(&x, &mut y).expect("apply succeeds");
        assert_eq!(y, vec![11.0, 22.0, 33.0, 41.0, 12.0]);
    }

    #[test]
    fn test_apply_adjoint_unweighted_values() {
        let schema = make_test_design();
        let op = DesignOperator::new(&schema, None);
        let r = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut x = vec![0.0; 7];
        op.apply_adjoint(&r, &mut x)
            .expect("apply_adjoint succeeds");
        assert_eq!(x, vec![5.0, 7.0, 3.0, 6.0, 2.0, 3.0, 4.0]);
    }

    fn dot(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    fn make_single_factor_design() -> Design<FactorMajorStore> {
        let store = FactorMajorStore::new(vec![vec![0u32, 1, 2, 0, 1]], 5).expect("valid store");
        Design::from_store(store).expect("valid single-factor design")
    }

    fn make_large_design() -> Design<FactorMajorStore> {
        let n_obs = 15_000;
        let n_levels_a = 50usize;
        let n_levels_b = 50usize;
        let fa: Vec<u32> = (0..n_obs).map(|i| (i % n_levels_a) as u32).collect();
        let fb: Vec<u32> = (0..n_obs).map(|i| (i % n_levels_b) as u32).collect();
        let store = FactorMajorStore::new(vec![fa, fb], n_obs).expect("valid factor-major store");
        Design::from_store(store).expect("valid design")
    }

    /// Verify <D·x, r> == <x, D^T·r> on a large design exercising the parallel
    /// gather/scatter paths (n_rows > PAR_THRESHOLD = 10_000).
    #[test]
    fn test_large_design_adjoint_property() {
        let dm = make_large_design();
        let n_dofs = dm.n_dofs;
        let n_rows = dm.n_obs;
        let op = DesignOperator::new(&dm, None);

        let x: Vec<f64> = (0..n_dofs).map(|i| (i as f64 * 0.17 + 1.0).sin()).collect();
        let r: Vec<f64> = (0..n_rows).map(|i| (i as f64 * 0.23 + 2.0).cos()).collect();

        let mut dx = vec![0.0f64; n_rows];
        op.apply(&x, &mut dx).expect("apply succeeds");

        let mut dtr = vec![0.0f64; n_dofs];
        op.apply_adjoint(&r, &mut dtr)
            .expect("apply_adjoint succeeds");

        let lhs = dot(&dx, &r);
        let rhs = dot(&x, &dtr);

        assert!(
            (lhs - rhs).abs() < 1e-8,
            "Adjoint property violated: <D·x, r>={lhs} vs <x, D^T·r>={rhs}"
        );
    }

    #[test]
    fn test_large_design_matvec_correctness() {
        let dm = make_large_design();
        let op = DesignOperator::new(&dm, None);

        let mut ej = vec![0.0f64; dm.n_dofs];
        ej[0] = 1.0;
        let mut y = vec![0.0f64; dm.n_obs];
        op.apply(&ej, &mut y).expect("apply succeeds");

        for (i, &yi) in y.iter().enumerate() {
            let expected = if i % 50 == 0 { 1.0 } else { 0.0 };
            assert_eq!(
                yi, expected,
                "D·e_0 at row {i}: expected {expected}, got {yi}"
            );
        }
    }

    #[test]
    fn test_large_design_apply_adjoint_correctness() {
        let dm = make_large_design();
        let op = DesignOperator::new(&dm, None);

        let ones = vec![1.0f64; dm.n_obs];
        let mut x = vec![0.0f64; dm.n_dofs];
        op.apply_adjoint(&ones, &mut x)
            .expect("apply_adjoint succeeds");

        let expected_count = (dm.n_obs / 50) as f64;
        for (j, &xj) in x.iter().enumerate() {
            assert!(
                (xj - expected_count).abs() < 1e-10,
                "D^T·1 at DOF {j}: expected {expected_count}, got {xj}"
            );
        }
    }

    #[test]
    fn test_single_factor_design_adjoint_property() {
        let dm = make_single_factor_design();
        let op = DesignOperator::new(&dm, None);

        let x: Vec<f64> = vec![1.0, 2.0, 3.0];
        let r: Vec<f64> = vec![0.5, 1.5, -0.5, 2.0, -1.0];

        let mut dx = vec![0.0f64; dm.n_obs];
        op.apply(&x, &mut dx).expect("apply succeeds");

        let mut dtr = vec![0.0f64; dm.n_dofs];
        op.apply_adjoint(&r, &mut dtr)
            .expect("apply_adjoint succeeds");

        let lhs = dot(&dx, &r);
        let rhs = dot(&x, &dtr);

        assert!(
            (lhs - rhs).abs() < 1e-12,
            "<D·x, r>={lhs} != <x, D^T·r>={rhs}"
        );
    }

    #[test]
    fn test_single_factor_apply_values() {
        let dm = make_single_factor_design();
        let op = DesignOperator::new(&dm, None);
        let x = vec![10.0, 20.0, 30.0];
        let mut y = vec![0.0f64; 5];
        op.apply(&x, &mut y).expect("apply succeeds");
        assert_eq!(y, vec![10.0, 20.0, 30.0, 10.0, 20.0]);
    }

    #[test]
    fn test_single_factor_apply_adjoint_values() {
        let dm = make_single_factor_design();
        let op = DesignOperator::new(&dm, None);
        let r = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut x = vec![0.0f64; 3];
        op.apply_adjoint(&r, &mut x)
            .expect("apply_adjoint succeeds");
        assert_eq!(x, vec![5.0, 7.0, 3.0]);
    }

    /// Single-factor design with `level(i) = i % n_levels`; when
    /// `n_obs >= n_levels` every level is populated so the inferred level count
    /// is exactly `n_levels` (which selects the scatter strategy).
    fn make_strategy_design(n_obs: usize, n_levels: usize) -> Design<FactorMajorStore> {
        let f: Vec<u32> = (0..n_obs).map(|i| (i % n_levels) as u32).collect();
        let store = FactorMajorStore::new(vec![f], n_obs).expect("valid store");
        Design::from_store(store).expect("valid design")
    }

    fn assert_all_close(actual: &[f64], expected: &[f64], ctx: &str) {
        assert_eq!(actual.len(), expected.len(), "{ctx}: length mismatch");
        for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            // Generous relative tolerance: parallel reductions may reorder FP
            // adds (~1e-13), but stale-scratch contamination is an O(value)
            // error, so this catches the bug class without flaking on FP noise.
            let tol = 1e-9 * e.abs().max(1.0);
            assert!(
                (a - e).abs() <= tol,
                "{ctx}: index {i}: {a} vs {e} (tol {tol})"
            );
        }
    }

    /// Regression guard for the scatter-scratch reuse bug class (cf. the removed
    /// `SCATTER_FOLD_POOL` leak). `apply_adjoint` reuses the operator's atomic
    /// scatter scratch across calls, so a second call on the *same* operator
    /// must match a freshly built operator — i.e. stale values from the first
    /// call must not bleed into the second. The three `(n_obs, n_levels)` pairs
    /// route the three scatter strategies: Sequential (`n_obs <= PAR_THRESHOLD`),
    /// Fold (parallel, `n_levels < SCATTER_LOCAL_THRESHOLD`), and Atomic
    /// (`n_levels >= SCATTER_LOCAL_THRESHOLD`) — the Atomic path, which owns the
    /// reused scratch, was otherwise never unit-tested.
    #[test]
    fn test_scatter_scratch_reuse_matches_fresh_operator() {
        for (n_obs, n_levels) in [(200usize, 16usize), (15_000, 64), (150_000, 100_000)] {
            let dm = make_strategy_design(n_obs, n_levels);
            let r: Vec<f64> = (0..dm.n_obs)
                .map(|i| (i as f64 * 0.37 + 1.0).sin())
                .collect();

            // Baseline: a fresh operator that has never applied before.
            let fresh = DesignOperator::new(&dm, None);
            let mut baseline = vec![0.0f64; dm.n_dofs];
            fresh
                .apply_adjoint(&r, &mut baseline)
                .expect("apply_adjoint succeeds");

            // Dirty the scratch with a first apply, then apply again on the same
            // operator; the second result must equal the fresh baseline.
            let op = DesignOperator::new(&dm, None);
            let mut warmup = vec![0.0f64; dm.n_dofs];
            op.apply_adjoint(&r, &mut warmup)
                .expect("apply_adjoint succeeds");
            let mut reused = vec![0.0f64; dm.n_dofs];
            op.apply_adjoint(&r, &mut reused)
                .expect("apply_adjoint succeeds");

            assert_all_close(
                &reused,
                &baseline,
                &format!("reused vs fresh (n_obs={n_obs}, n_levels={n_levels})"),
            );
        }
    }
}

// ===========================================================================
// weighted adjoint property test
// ===========================================================================

mod weighted_adjoint_proptests {
    use crate::domain::Design;
    use crate::observation::FactorMajorStore;
    use crate::operator::DesignOperator;
    use proptest::prelude::*;
    use schwarz_precond::Operator;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]

        /// The adjoint property must hold for random weighted designs:
        /// <D·x, W·r> == <x, D^T·W·r>, with D^T·W·r computed via
        /// DesignOperator::apply_adjoint(W^{1/2} r) = D^T W^{1/2} (W^{1/2} r) = D^T W r.
        #[test]
        fn prop_weighted_adjoint_property(
            n_obs in 20usize..=200,
            n_levels_a in 2usize..=15,
            n_levels_b in 2usize..=15,
            seed in 0u64..1000,
        ) {
            let fa: Vec<u32> = (0..n_obs)
                .map(|i| ((i * 3 + seed as usize * 7) % n_levels_a) as u32)
                .collect();
            let fb: Vec<u32> = (0..n_obs)
                .map(|i| ((i * 5 + seed as usize * 11) % n_levels_b) as u32)
                .collect();

            let weights: Vec<f64> = (0..n_obs)
                .map(|i| 0.5 + (i as f64 * 0.13 + seed as f64 * 0.41).sin().abs())
                .collect();

            let store = FactorMajorStore::new(vec![fa, fb], n_obs).unwrap();
            let dm = Design::from_store(store).unwrap();

            let n_dofs = dm.n_dofs;
            let n_rows = dm.n_obs;

            let x: Vec<f64> = (0..n_dofs)
                .map(|i| (i as f64 * 0.37 + seed as f64 * 0.13).sin())
                .collect();
            let r: Vec<f64> = (0..n_rows)
                .map(|i| (i as f64 * 0.29 + seed as f64 * 0.07).cos())
                .collect();

            let op_unweighted = DesignOperator::new(&dm, None);
            let mut dx = vec![0.0f64; n_rows];
            op_unweighted.apply(&x, &mut dx).unwrap();
            let lhs: f64 = dx
                .iter()
                .zip(r.iter())
                .enumerate()
                .map(|(i, (dxi, ri))| weights[i] * dxi * ri)
                .sum();

            let op_weighted = DesignOperator::new(&dm, Some(&weights));
            let wr = op_weighted.weighted_rhs(&r);
            let mut wdtr = vec![0.0f64; n_dofs];
            op_weighted.apply_adjoint(&wr, &mut wdtr).unwrap();
            let rhs: f64 = x.iter().zip(wdtr.iter()).map(|(xi, wi)| xi * wi).sum();

            prop_assert!(
                (lhs - rhs).abs() < 1e-8,
                "<D·x, W·r>={lhs} != <x, D^T·W·r>={rhs}"
            );
        }
    }
}

// ===========================================================================
// schwarz tests
// ===========================================================================

mod schwarz_tests {
    use std::env;
    use std::process::Command;
    use std::thread;
    use std::time::{Duration, Instant};

    use crate::config::{
        ApproxCholConfig, ApproxSchurConfig, LocalSolverConfig, DEFAULT_DENSE_SCHUR_THRESHOLD,
    };
    use schwarz_precond::SubdomainCore;

    use crate::block_elim::factor::ReducedFactor;
    use crate::csr_block::CsrBlock;
    use crate::domain::factor_pairs::Subdomain;
    use crate::domain::{build_local_domains, Design, LocalDomain};
    use crate::domain::{BlockDiagonals, CrossTab};
    use crate::observation::FactorMajorStore;
    use crate::operator::schwarz::{build_additive_with_strategy, build_entry};
    use schwarz_precond::{Operator, ReductionStrategy};

    const BLOCK_ELIM_NESTED_RAYON_CHILD_ENV: &str = "WITHIN_TEST_BLOCK_ELIM_NESTED_RAYON_CHILD";

    fn make_test_data() -> (Design<FactorMajorStore>, Vec<LocalDomain>) {
        let store = FactorMajorStore::new(vec![vec![0, 1, 0, 1, 2], vec![0, 0, 1, 1, 0]], 5)
            .expect("valid factor-major store");
        let design = Design::from_store(store).expect("valid fixed-effects design");
        let domain_pairs = build_local_domains(&design, None);
        (design, domain_pairs)
    }

    fn synthetic_sparse_cross_tab(n_keep: usize, elim_ratio: usize) -> (CrossTab, BlockDiagonals) {
        let n_q = n_keep * elim_ratio;
        let n_r = n_keep;
        let mut indptr = Vec::with_capacity(n_q + 1);
        let mut indices = Vec::with_capacity(n_q * 3);
        let mut data = Vec::with_capacity(n_q * 3);
        let mut diag_q = vec![0.0; n_q];
        let mut diag_r = vec![0.0; n_r];

        indptr.push(0);
        for (i, diag_q_i) in diag_q.iter_mut().enumerate().take(n_q) {
            let mut row = [
                (i % n_r, 1.0),
                ((i + 1) % n_r, 0.8),
                ((i.wrapping_mul(17).wrapping_add(3)) % n_r, 0.6),
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
                diag_r[col] += value;
            }

            *diag_q_i = row_sum;
            indptr.push(indices.len() as u32);
        }

        let c = CsrBlock {
            indptr,
            indices,
            data,
            nrows: n_q,
            ncols: n_r,
        };
        let ct = c.transpose();
        (
            CrossTab { c, ct },
            BlockDiagonals {
                q: diag_q,
                r: diag_r,
            },
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
                subdomain: Subdomain {
                    core: SubdomainCore::uniform(global_indices.clone()),
                },
                cross_tab: cross_tab.clone(),
                block_diagonals: block_diagonals.clone(),
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
            approx_schur: Some(ApproxSchurConfig {
                seed: 7,
                ..Default::default()
            }),
            dense_threshold: DEFAULT_DENSE_SCHUR_THRESHOLD,
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
            )
            .expect("build block-elim additive preconditioner");
            let atomic = build_additive_with_strategy(
                domain_pairs_atomic,
                &config,
                ReductionStrategy::AtomicScatter,
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
        let schwarz = build_additive_with_strategy(domain_pairs, &config, strategy)
            .expect("build schwarz with explicit domains");
        let r = vec![1.0; design.n_dofs];
        let mut z = vec![0.0; design.n_dofs];
        schwarz.apply(&r, &mut z).expect("schwarz apply succeeds");
    }

    #[test]
    fn test_exact_schur_uses_dense_fast_path_for_tiny_reduced_system() {
        let (_, mut domain_pairs) = make_test_data();
        let domain = domain_pairs.swap_remove(0);

        let config = LocalSolverConfig {
            approx_chol: ApproxCholConfig::default(),
            approx_schur: None,
            dense_threshold: DEFAULT_DENSE_SCHUR_THRESHOLD,
        };
        let entry = build_entry(domain, &config).expect("exact Schur entry build failed");
        assert!(matches!(
            entry.solver().reduced_factor,
            ReducedFactor::Dense(_)
        ));
    }

    #[test]
    fn test_approximate_schur_uses_dense_fast_path_for_tiny_reduced_system() {
        let (_, mut domain_pairs) = make_test_data();
        let domain = domain_pairs.swap_remove(0);

        let config = LocalSolverConfig {
            approx_chol: ApproxCholConfig::default(),
            approx_schur: Some(ApproxSchurConfig {
                seed: 7,
                ..Default::default()
            }),
            dense_threshold: DEFAULT_DENSE_SCHUR_THRESHOLD,
        };
        let entry = build_entry(domain, &config).expect("approximate Schur entry build failed");
        assert!(matches!(
            entry.solver().reduced_factor,
            ReducedFactor::Dense(_)
        ));
    }

    #[test]
    fn test_dense_threshold_zero_disables_dense_fast_path() {
        let (_, mut domain_pairs) = make_test_data();
        let domain = domain_pairs.swap_remove(0);

        let config = LocalSolverConfig {
            approx_chol: ApproxCholConfig::default(),
            approx_schur: None,
            dense_threshold: 0,
        };
        let entry = build_entry(domain, &config).expect("exact Schur entry build failed");
        assert!(!matches!(
            entry.solver().reduced_factor,
            ReducedFactor::Dense(_)
        ));
    }
}
