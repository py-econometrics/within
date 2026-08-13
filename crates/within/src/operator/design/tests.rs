mod design_tests {
    use crate::domain::Design;
    use crate::operator::DesignOperator;
    use schwarz_precond::Operator;

    fn make_test_design() -> Design<'static> {
        // Sorted on the dominant factor, so construction applies no locality permutation.
        Design::from_levels_for_test(vec![vec![0, 1, 1, 2, 0], vec![0, 0, 1, 2, 3]])
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
        assert_eq!(y, vec![11.0, 12.0, 22.0, 33.0, 41.0]);
    }

    #[test]
    fn test_apply_adjoint_unweighted_values() {
        let schema = make_test_design();
        let op = DesignOperator::new(&schema, None);
        let r = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut x = vec![0.0; 7];
        op.apply_adjoint(&r, &mut x)
            .expect("apply_adjoint succeeds");
        assert_eq!(x, vec![6.0, 5.0, 4.0, 3.0, 3.0, 4.0, 5.0]);
    }

    fn dot(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    fn make_single_factor_design() -> Design<'static> {
        // Sorted: a single-factor store is always dominated by its only factor.
        Design::from_levels_for_test(vec![vec![0u32, 0, 1, 1, 2]])
    }

    fn make_large_design() -> Design<'static> {
        // Sorted on both factors, so there is no construction-time permutation.
        let n_obs = 15_000;
        let block = 300u32;
        let fa: Vec<u32> = (0..n_obs).map(|i| i as u32 / block).collect();
        let fb = fa.clone();
        Design::from_levels_for_test(vec![fa, fb])
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

    /// Fold on an *unsorted* column. Construction only sorts by the dominant
    /// factor, so a small non-dominant factor legitimately reaches
    /// `ScatterStrategy::Fold` (parallel, `n_levels < SCATTER_LOCAL_THRESHOLD`)
    /// with interleaved levels — Fold must stay order-agnostic. The dominant
    /// factor here is pre-sorted (no permutation), leaving `fb = i % 50`
    /// interleaved.
    #[test]
    fn test_fold_unsorted_secondary_factor() {
        let n_obs = 15_000;
        let fa: Vec<u32> = (0..n_obs as u32).collect();
        let fb: Vec<u32> = (0..n_obs).map(|i| (i % 50) as u32).collect();
        let dm = Design::from_levels_for_test(vec![fa, fb]);
        assert!(dm.obs_perm.is_none(), "dominant factor is sorted; no perm");

        let op = DesignOperator::new(&dm, None);
        let x: Vec<f64> = (0..dm.n_dofs)
            .map(|i| (i as f64 * 0.17 + 1.0).sin())
            .collect();
        let r: Vec<f64> = (0..dm.n_obs)
            .map(|i| (i as f64 * 0.23 + 2.0).cos())
            .collect();

        let mut dx = vec![0.0f64; dm.n_obs];
        op.apply(&x, &mut dx).expect("apply succeeds");
        let mut dtr = vec![0.0f64; dm.n_dofs];
        op.apply_adjoint(&r, &mut dtr)
            .expect("apply_adjoint succeeds");

        let lhs = dot(&dx, &r);
        let rhs = dot(&x, &dtr);
        assert!(
            (lhs - rhs).abs() < 1e-8,
            "Adjoint property violated: <D·x, r>={lhs} vs <x, D^T·r>={rhs}"
        );

        assert_scratch_reuse_matches_fresh(&dm, "fold: non-dominant unsorted 50 levels");
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
            let expected = if i < 300 { 1.0 } else { 0.0 };
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
        assert_eq!(y, vec![10.0, 10.0, 20.0, 20.0, 30.0]);
    }

    #[test]
    fn test_single_factor_apply_adjoint_values() {
        let dm = make_single_factor_design();
        let op = DesignOperator::new(&dm, None);
        let r = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut x = vec![0.0f64; 3];
        op.apply_adjoint(&r, &mut x)
            .expect("apply_adjoint succeeds");
        assert_eq!(x, vec![3.0, 7.0, 5.0]);
    }

    /// Single-factor design with `level(i) = i % n_levels`; when
    /// `n_obs >= n_levels` every level is populated so the inferred level count
    /// is exactly `n_levels` (which selects the scatter strategy).
    fn make_strategy_design(n_obs: usize, n_levels: usize) -> Design<'static> {
        let f: Vec<u32> = (0..n_obs).map(|i| (i % n_levels) as u32).collect();
        Design::from_levels_for_test(vec![f])
    }

    fn assert_all_close(actual: &[f64], expected: &[f64], ctx: &str) {
        assert_eq!(actual.len(), expected.len(), "{ctx}: length mismatch");
        for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            // Stale-scratch contamination is O(value), so this cannot flake on FP noise.
            let tol = 1e-9 * e.abs().max(1.0);
            assert!(
                (a - e).abs() <= tol,
                "{ctx}: index {i}: {a} vs {e} (tol {tol})"
            );
        }
    }

    #[test]
    fn test_scatter_scratch_reuse_matches_fresh_operator() {
        // The three pairs route Sequential, Fold and SortedCoalesced respectively.
        for (n_obs, n_levels) in [(200usize, 16usize), (15_000, 64), (150_000, 100_000)] {
            let dm = make_strategy_design(n_obs, n_levels);
            assert_scratch_reuse_matches_fresh(&dm, &format!("n_obs={n_obs}, n_levels={n_levels}"));
        }

        // A large unsorted non-dominant factor cannot coalesce.
        let n_obs = 150_000usize;
        let fa: Vec<u32> = (0..n_obs as u32).collect();
        let fb: Vec<u32> = (0..n_obs).map(|i| ((i * 7919) % 100_000) as u32).collect();
        let dm = Design::from_levels_for_test(vec![fa, fb]);
        assert!(dm.obs_perm.is_none(), "dominant factor is sorted; no perm");
        assert_scratch_reuse_matches_fresh(&dm, "atomic: non-dominant unsorted 100K levels");
    }

    /// `apply_adjoint` reuses the operator's atomic scatter scratch across
    /// calls (cf. the removed `SCATTER_FOLD_POOL` leak): a second call on the
    /// *same* operator must match a freshly built operator — stale values from
    /// the first call must not bleed into the second.
    fn assert_scratch_reuse_matches_fresh(dm: &Design<'_>, ctx: &str) {
        let r: Vec<f64> = (0..dm.n_obs)
            .map(|i| (i as f64 * 0.37 + 1.0).sin())
            .collect();

        // Baseline: a fresh operator that has never applied before.
        let fresh = DesignOperator::new(dm, None);
        let mut baseline = vec![0.0f64; dm.n_dofs];
        fresh
            .apply_adjoint(&r, &mut baseline)
            .expect("apply_adjoint succeeds");

        // The second apply on a dirtied operator must equal the fresh baseline.
        let op = DesignOperator::new(dm, None);
        let mut warmup = vec![0.0f64; dm.n_dofs];
        op.apply_adjoint(&r, &mut warmup)
            .expect("apply_adjoint succeeds");
        let mut reused = vec![0.0f64; dm.n_dofs];
        op.apply_adjoint(&r, &mut reused)
            .expect("apply_adjoint succeeds");

        assert_all_close(&reused, &baseline, &format!("reused vs fresh ({ctx})"));
    }
}

mod slope_design_tests {
    use crate::domain::{Design, Effect, Loading};
    use crate::operator::DesignOperator;
    use schwarz_precond::Operator;

    /// Deterministic pseudo-random f64 in [-1, 1).
    fn noise(seed: usize) -> f64 {
        let mut z = seed as u64 ^ 0x9E37_79B9_7F4A_7C15;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        (z >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    }

    /// Dense design matrix from the design's internal (post-sort) columns —
    /// the reference the operator must agree with regardless of the locality
    /// permutation.
    fn dense_matrix(design: &Design<'_>) -> Vec<Vec<f64>> {
        let mut d = vec![vec![0.0; design.n_dofs]; design.n_obs];
        for (q, t) in design.terms.iter().enumerate() {
            let levels = design.frame.level_column(q);
            for (c, loading) in t.columns.iter().enumerate() {
                let base = t.offset + c * t.n_levels;
                for (i, &lev) in levels.iter().enumerate() {
                    d[i][base + lev as usize] = match loading {
                        Loading::Constant => 1.0,
                        Loading::Covariate(k) => design.frame.loading_column(*k as usize)[i],
                    };
                }
            }
        }
        d
    }

    fn assert_close(a: &[f64], b: &[f64]) {
        for (x, y) in a.iter().zip(b) {
            assert!(
                (x - y).abs() <= 1e-10 * x.abs().max(y.abs()).max(1.0),
                "{x} vs {y}"
            );
        }
    }

    /// Covers every kernel arm on the sequential path: plain, fused V=1/V=2,
    /// slope-only, and the generic V=3 fallback — against the dense reference.
    #[test]
    fn slope_matvec_and_adjoint_match_dense_reference() {
        let n = 6;
        let f0 = [0u32, 1, 2, 0, 1, 2];
        let f1 = [0u32, 0, 1, 1, 2, 2];
        let f2 = [0u32, 1, 0, 1, 0, 1];
        let f3 = [0u32, 0, 0, 1, 1, 1];
        let f4 = [1u32, 0, 2, 1, 0, 2];
        let zs: Vec<Vec<f64>> = (0..6)
            .map(|k| (0..n).map(|i| noise(k * 100 + i)).collect())
            .collect();
        let effects = vec![
            Effect::new(&f0, true, [&zs[0][..], &zs[1][..], &zs[2][..]]).unwrap(),
            Effect::new(&f1, true, [&zs[3][..]]).unwrap(),
            Effect::new(&f2, false, [&zs[4][..]]).unwrap(),
            Effect::new(&f3, true, []).unwrap(),
            Effect::new(&f4, true, [&zs[5][..], &zs[4][..]]).unwrap(),
        ];
        let design = Design::new(effects).unwrap();
        let dense = dense_matrix(&design);
        let op = DesignOperator::new(&design, None);

        let x: Vec<f64> = (0..design.n_dofs).map(|j| noise(7_000 + j)).collect();
        let mut got = vec![0.0; design.n_obs];
        op.apply(&x, &mut got).unwrap();
        let expect: Vec<f64> = dense
            .iter()
            .map(|row| row.iter().zip(&x).map(|(d, xj)| d * xj).sum())
            .collect();
        assert_close(&got, &expect);

        let r: Vec<f64> = (0..design.n_obs).map(|i| noise(9_000 + i)).collect();
        let mut got_t = vec![0.0; design.n_dofs];
        op.apply_adjoint(&r, &mut got_t).unwrap();
        let mut expect_t = vec![0.0; design.n_dofs];
        for (row, &ri) in dense.iter().zip(&r) {
            for (e, d) in expect_t.iter_mut().zip(row) {
                *e += d * ri;
            }
        }
        assert_close(&got_t, &expect_t);
    }

    /// Adjoint identity ⟨Dx, r⟩ = ⟨x, Dᵀr⟩ on a design large enough to take
    /// the parallel strategies — sorted-coalesced (C=2), atomic (C=2), and
    /// fold (C=3) — with weights in play. Gather and scatter share the layout
    /// logic but not the kernels, so a per-strategy addressing bug breaks the
    /// identity.
    #[test]
    fn slope_adjoint_property_parallel_strategies() {
        let n = 150_000;
        let l_big = 60_000usize;
        let sorted: Vec<u32> = (0..n).map(|i| (i * l_big / n) as u32).collect();
        let unsorted: Vec<u32> = (0..n).map(|i| ((i * 7919) % l_big) as u32).collect();
        let small: Vec<u32> = (0..n).map(|i| (i % 10) as u32).collect();
        let z: Vec<Vec<f64>> = (0..4)
            .map(|k| (0..n).map(|i| noise(k * n + i)).collect())
            .collect();
        let effects = vec![
            Effect::new(&sorted, true, [&z[0][..]]).unwrap(),
            Effect::new(&unsorted, true, [&z[1][..]]).unwrap(),
            Effect::new(&small, true, [&z[2][..], &z[3][..]]).unwrap(),
        ];
        let design = Design::new(effects).unwrap();
        let sqrt_weights: Vec<f64> = (0..n).map(|i| (0.5 + noise(i).abs()).sqrt()).collect();
        let op = DesignOperator::new(&design, Some(&sqrt_weights));

        let x: Vec<f64> = (0..design.n_dofs).map(|j| noise(13 * j + 1)).collect();
        let r: Vec<f64> = (0..n).map(|i| noise(29 * i + 5)).collect();

        let mut dx = vec![0.0; n];
        op.apply(&x, &mut dx).unwrap();
        let mut dtr = vec![0.0; design.n_dofs];
        op.apply_adjoint(&r, &mut dtr).unwrap();

        let lhs: f64 = dx.iter().zip(&r).map(|(a, b)| a * b).sum();
        let rhs: f64 = x.iter().zip(&dtr).map(|(a, b)| a * b).sum();
        assert!(
            (lhs - rhs).abs() <= 1e-9 * lhs.abs().max(rhs.abs()).max(1.0),
            "{lhs} vs {rhs}"
        );
    }
}

mod weighted_adjoint_proptests {
    use crate::domain::Design;
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

            let dm = Design::from_levels_for_test(vec![fa, fb]);

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

            let sqrt_weights: Vec<f64> = weights.iter().map(|w| w.sqrt()).collect();
            let op_weighted = DesignOperator::new(&dm, Some(&sqrt_weights));
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
