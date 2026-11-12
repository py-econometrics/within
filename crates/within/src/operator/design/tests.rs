/// Deterministic pseudo-random f64 in [-1, 1).
fn noise(seed: usize) -> f64 {
    let mut z = seed as u64 ^ 0x9E37_79B9_7F4A_7C15;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    (z >> 11) as f64 / (1u64 << 52) as f64 - 1.0
}

mod design_tests {
    use crate::domain::PreparedDesign;
    use crate::linalg::dot;
    use schwarz_precond::Operator;

    fn make_test_design() -> PreparedDesign<'static> {
        // Sorted on the dominant factor, so construction applies no locality permutation.
        PreparedDesign::from_levels_for_test(vec![vec![0, 1, 1, 2, 0], vec![0, 0, 1, 2, 3]])
    }

    #[test]
    fn test_design_operator_dimensions() {
        let schema = make_test_design();
        assert_eq!(schema.nrows(), 5);
        assert_eq!(schema.ncols(), 7);
    }

    #[test]
    fn test_design_operator_adjoint() {
        let schema = make_test_design();

        let x = vec![1.0, -0.5, 2.0, 0.3, -1.0, 0.7, 1.5];
        let r = vec![0.1, 0.2, -0.3, 0.4, -0.5];

        let mut dx = vec![0.0; 5];
        schema.apply(&x, &mut dx).expect("apply succeeds");
        let lhs: f64 = dx.iter().zip(r.iter()).map(|(a, b)| a * b).sum();

        let mut dtr = vec![0.0; 7];
        schema
            .apply_adjoint(&r, &mut dtr)
            .expect("apply_adjoint succeeds");
        let rhs: f64 = x.iter().zip(dtr.iter()).map(|(a, b)| a * b).sum();
        assert!((lhs - rhs).abs() < 1e-12);
    }

    #[test]
    fn test_apply_unweighted_values() {
        let schema = make_test_design();
        let x = vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 40.0];
        let mut y = vec![0.0; 5];
        schema.apply(&x, &mut y).expect("apply succeeds");
        assert_eq!(y, vec![11.0, 12.0, 22.0, 33.0, 41.0]);
    }

    #[test]
    fn test_apply_adjoint_unweighted_values() {
        let schema = make_test_design();
        let r = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut x = vec![0.0; 7];
        schema
            .apply_adjoint(&r, &mut x)
            .expect("apply_adjoint succeeds");
        assert_eq!(x, vec![6.0, 5.0, 4.0, 3.0, 3.0, 4.0, 5.0]);
    }

    fn make_single_factor_design() -> PreparedDesign<'static> {
        // Sorted: a single-factor store is always dominated by its only factor.
        PreparedDesign::from_levels_for_test(vec![vec![0u32, 0, 1, 1, 2]])
    }

    fn make_large_design() -> PreparedDesign<'static> {
        // Sorted on both factors, so there is no construction-time permutation.
        let n_obs = 15_000;
        let block = 300u32;
        let fa: Vec<u32> = (0..n_obs).map(|i| i as u32 / block).collect();
        let fb = fa.clone();
        PreparedDesign::from_levels_for_test(vec![fa, fb])
    }

    /// Verify <D·x, r> == <x, D^T·r> on a large design exercising the parallel
    /// gather/scatter paths (n_rows > PAR_THRESHOLD = 10_000).
    #[test]
    fn test_large_design_adjoint_property() {
        let dm = make_large_design();
        let n_dofs = dm.design.n_dofs;
        let n_rows = dm.design.n_obs;

        let x: Vec<f64> = (0..n_dofs).map(|i| (i as f64 * 0.17 + 1.0).sin()).collect();
        let r: Vec<f64> = (0..n_rows).map(|i| (i as f64 * 0.23 + 2.0).cos()).collect();

        let mut dx = vec![0.0f64; n_rows];
        dm.apply(&x, &mut dx).expect("apply succeeds");

        let mut dtr = vec![0.0f64; n_dofs];
        dm.apply_adjoint(&r, &mut dtr)
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

        let mut ej = vec![0.0f64; dm.design.n_dofs];
        ej[0] = 1.0;
        let mut y = vec![0.0f64; dm.design.n_obs];
        dm.apply(&ej, &mut y).expect("apply succeeds");

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

        let ones = vec![1.0f64; dm.design.n_obs];
        let mut x = vec![0.0f64; dm.design.n_dofs];
        dm.apply_adjoint(&ones, &mut x)
            .expect("apply_adjoint succeeds");

        let expected_count = (dm.design.n_obs / 50) as f64;
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

        let x: Vec<f64> = vec![1.0, 2.0, 3.0];
        let r: Vec<f64> = vec![0.5, 1.5, -0.5, 2.0, -1.0];

        let mut dx = vec![0.0f64; dm.design.n_obs];
        dm.apply(&x, &mut dx).expect("apply succeeds");

        let mut dtr = vec![0.0f64; dm.design.n_dofs];
        dm.apply_adjoint(&r, &mut dtr)
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
        let x = vec![10.0, 20.0, 30.0];
        let mut y = vec![0.0f64; 5];
        dm.apply(&x, &mut y).expect("apply succeeds");
        assert_eq!(y, vec![10.0, 10.0, 20.0, 20.0, 30.0]);
    }

    #[test]
    fn test_single_factor_apply_adjoint_values() {
        let dm = make_single_factor_design();
        let r = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut x = vec![0.0f64; 3];
        dm.apply_adjoint(&r, &mut x)
            .expect("apply_adjoint succeeds");
        assert_eq!(x, vec![3.0, 7.0, 5.0]);
    }
}

mod slope_design_tests {
    use super::noise;
    use crate::domain::{Design, Effect, PreparedDesign, ScatterSplit};
    use schwarz_precond::Operator;

    /// Dense design matrix from the design's internal (post-sort) columns —
    /// the reference the operator must agree with regardless of the locality
    /// permutation.
    fn dense_matrix(prepared: &PreparedDesign<'_>) -> Vec<Vec<f64>> {
        let design = &prepared.design;
        let mut d = vec![vec![0.0; design.n_dofs]; design.n_obs];
        for t in prepared.terms() {
            for c in 0..t.term.n_columns() {
                let base = t.term.column_dofs(c).start;
                let z = t.loading(c);
                for (i, &lev) in t.term.levels().iter().enumerate() {
                    d[i][base + lev as usize] = z.map_or(1.0, |z| z[i]);
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

    /// Every kernel arm on the sequential path, against the dense reference.
    #[test]
    fn slope_matvec_and_adjoint_match_dense_reference() {
        let n = 12;
        let f0: Vec<u32> = (0..n).map(|i| (i % 3) as u32).collect();
        let f1: Vec<u32> = (0..n).map(|i| ((i / 2) % 3) as u32).collect();
        let f2: Vec<u32> = (0..n).map(|i| (i % 2) as u32).collect();
        let f3: Vec<u32> = (0..n).map(|i| (i / 6) as u32).collect();
        let f4: Vec<u32> = (0..n).map(|i| [1u32, 0, 2, 1, 0, 2][i % 6]).collect();
        let f5: Vec<u32> = (0..n).map(|i| ((i / 3) % 2) as u32).collect();
        let zs: Vec<Vec<f64>> = (0..15)
            .map(|k| (0..n).map(|i| noise(k * 100 + i)).collect())
            .collect();
        let effects = vec![
            Effect::new(&f0, true, [&zs[0][..], &zs[1][..], &zs[2][..]]).unwrap(),
            Effect::new(&f1, true, [&zs[3][..]]).unwrap(),
            Effect::new(&f2, false, [&zs[4][..]]).unwrap(),
            Effect::new(&f3, true, []).unwrap(),
            Effect::new(&f4, true, [&zs[5][..], &zs[4][..]]).unwrap(),
            Effect::new(&f4, false, [&zs[6][..], &zs[7][..]]).unwrap(),
            Effect::new(&f1, false, [&zs[8][..], &zs[9][..], &zs[10][..]]).unwrap(),
            Effect::new(&f5, true, zs[11..15].iter().map(|z| &z[..])).unwrap(),
        ];
        let design = PreparedDesign::unweighted_for_test(Design::new(effects).unwrap());
        let dense = dense_matrix(&design);
        assert!(
            (0..design.design.n_dofs).all(|j| dense.iter().any(|row| row[j] != 0.0)),
            "whitening zeroed a column; the arm's addressing would go untested"
        );

        let x: Vec<f64> = (0..design.design.n_dofs)
            .map(|j| noise(7_000 + j))
            .collect();
        let mut got = vec![0.0; design.design.n_obs];
        design.apply(&x, &mut got).unwrap();
        let expect: Vec<f64> = dense
            .iter()
            .map(|row| row.iter().zip(&x).map(|(d, xj)| d * xj).sum())
            .collect();
        assert_close(&got, &expect);

        let r: Vec<f64> = (0..design.design.n_obs).map(|i| noise(9_000 + i)).collect();
        let mut got_t = vec![0.0; design.design.n_dofs];
        design.apply_adjoint(&r, &mut got_t).unwrap();
        let mut expect_t = vec![0.0; design.design.n_dofs];
        for (row, &ri) in dense.iter().zip(&r) {
            for (e, d) in expect_t.iter_mut().zip(row) {
                *e += d * ri;
            }
        }
        assert_close(&got_t, &expect_t);
    }

    /// By-level (sorted or indexed) and by-row terms per fused arity; levels over many leaves.
    fn scatter_split_design() -> PreparedDesign<'static> {
        let n: usize = 300_000;
        // Largest term and already sorted, so the locality sort leaves every fixture as built.
        let sorted: Vec<u32> = (0..n).map(|i| (i * 160_000 / n) as u32).collect();
        let sorted_small: Vec<u32> = (0..n).map(|i| (i * 30_000 / n) as u32).collect();
        let sorted_skewed: Vec<u32> = (0..n).map(|i| i.saturating_sub(150_000) as u32).collect();
        let unsorted_skewed: Vec<u32> = (0..n)
            .map(|i| (i % 2) as u32 * (1 + i / 2) as u32)
            .collect();
        let unsorted: Vec<u32> = (0..n).map(|i| ((i * 7919) % 100_000) as u32).collect();
        let unsorted_fused: Vec<u32> = (0..n).map(|i| ((i * 7919) % 60_000) as u32).collect();
        // 15 rows per level at C = 4: by row in two halves, not in leaves of 4·C·L rows.
        let unsorted_short: Vec<u32> = (0..n).map(|i| ((i * 7919) % 20_000) as u32).collect();
        let small: Vec<u32> = (0..n).map(|i| (i % 10) as u32).collect();
        let z: Vec<Vec<f64>> = (0..15)
            .map(|k| (0..n).map(|i| noise(k * n + i)).collect())
            .collect();
        let effects = vec![
            Effect::new(&sorted, true, [&z[0][..]]).unwrap(),
            Effect::new(&unsorted, true, [&z[1][..]]).unwrap(),
            Effect::new(&unsorted_fused, true, [&z[4][..]]).unwrap(),
            Effect::new(&small, true, [&z[2][..], &z[3][..]]).unwrap(),
            Effect::new(&sorted_small, true, [&z[5][..], &z[6][..], &z[7][..]]).unwrap(),
            Effect::new(&unsorted, false, [&z[8][..], &z[9][..], &z[10][..]]).unwrap(),
            Effect::new(&small, true, [&z[11][..], &z[12][..], &z[13][..]]).unwrap(),
            Effect::new(&sorted_skewed, true, []).unwrap(),
            Effect::new(&unsorted_skewed, true, []).unwrap(),
            Effect::new(&small, false, [&z[14][..]]).unwrap(),
            Effect::new(&unsorted_short, true, [&z[1][..], &z[2][..], &z[3][..]]).unwrap(),
        ];
        let weights: Vec<f64> = (0..n).map(|i| 0.5 + noise(i).abs()).collect();
        let design = Design::new(effects).unwrap().into_owned();
        PreparedDesign::new(design, Some(&weights)).unwrap()
    }

    /// The adjoint's bits must not depend on the pool it runs in.
    #[test]
    fn adjoint_is_pool_independent() {
        let design = scatter_split_design();
        let r: Vec<f64> = (0..design.design.n_obs)
            .map(|i| noise(29 * i + 5))
            .collect();
        let adjoint_in = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("test rayon pool")
                .install(|| {
                    let mut dtr = vec![0.0; design.design.n_dofs];
                    design.apply_adjoint(&r, &mut dtr).unwrap();
                    dtr.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                })
        };
        let reference = adjoint_in(1);
        for threads in [2, 3, 8] {
            // Not `assert_eq!`: a printed coefficient vector would bury the thread count.
            assert!(adjoint_in(threads) == reference, "{threads} threads");
        }
    }

    /// ⟨Dx, r⟩ = ⟨x, Dᵀr⟩, weighted, with each adjoint sum on each fused arity.
    #[test]
    fn slope_adjoint_property_each_scatter_split() {
        let design = scatter_split_design();
        let n = design.design.n_obs;

        // By level unless unsorted with at least 12 rows per level.
        let by_level = [
            true, true, true, false, true, true, false, true, true, false,
        ];
        for (t, by_level) in design.terms().zip(by_level) {
            assert_eq!(matches!(t.split, ScatterSplit::ByLevel(_)), by_level);
        }
        assert!(design.design.obs_perm.is_none());

        let x: Vec<f64> = (0..design.design.n_dofs)
            .map(|j| noise(13 * j + 1))
            .collect();
        let r: Vec<f64> = (0..n).map(|i| noise(29 * i + 5)).collect();

        let mut dx = vec![0.0; n];
        design.apply(&x, &mut dx).unwrap();
        let mut dtr = vec![0.0; design.design.n_dofs];
        design.apply_adjoint(&r, &mut dtr).unwrap();

        let lhs: f64 = dx.iter().zip(&r).map(|(a, b)| a * b).sum();
        let rhs: f64 = x.iter().zip(&dtr).map(|(a, b)| a * b).sum();
        assert!(
            (lhs - rhs).abs() <= 1e-9 * lhs.abs().max(rhs.abs()).max(1.0),
            "{lhs} vs {rhs}"
        );
    }
}

mod weighted_adjoint_proptests {
    use crate::domain::PreparedDesign;
    use proptest::prelude::*;
    use schwarz_precond::Operator;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]

        /// The weighted normal-equation identity must hold for random designs:
        /// <W^{1/2}D·x, W^{1/2}r> == <x, D^T W·r>.
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

            let design = crate::domain::Design::from_levels_for_test(vec![fa, fb]);
            let dm = PreparedDesign::new(design, Some(&weights)).unwrap();

            let n_dofs = dm.design.n_dofs;
            let n_rows = dm.design.n_obs;

            let x: Vec<f64> = (0..n_dofs)
                .map(|i| (i as f64 * 0.37 + seed as f64 * 0.13).sin())
                .collect();
            let r: Vec<f64> = (0..n_rows)
                .map(|i| (i as f64 * 0.29 + seed as f64 * 0.07).cos())
                .collect();

            let mut dx = vec![0.0f64; n_rows];
            dm.apply(&x, &mut dx).unwrap();
            let lhs: f64 = dx.iter().zip(&r).map(|(dxi, ri)| dxi * ri).sum();

            let mut dtr = vec![0.0f64; n_dofs];
            dm.apply_adjoint(&r, &mut dtr).unwrap();
            let rhs: f64 = x.iter().zip(dtr.iter()).map(|(xi, wi)| xi * wi).sum();

            prop_assert!(
                (lhs - rhs).abs() < 1e-8,
                "<W^1/2 D·x, r>={lhs} != <x, D^T W^1/2·r>={rhs}"
            );
        }
    }
}

mod elimination_tests {
    use rstest::rstest;
    use schwarz_precond::Operator;

    use super::noise;
    use crate::domain::{Design, Effect, PreparedDesign};
    use crate::linalg::dot;

    fn norm(v: &[f64]) -> f64 {
        dot(v, v).sqrt()
    }

    #[derive(Clone, Copy, Debug)]
    enum Dominant {
        InterceptOnly,
        /// Three near-collinear slopes with mean 1e6 beside the intercept.
        CollinearLargeMean,
        /// Near copies at the rank tolerance, so some levels drop one.
        RankEdge,
        SlopeOnly,
        /// A slope constant within level 0, so that level's slope is unidentified.
        DroppedSlope,
    }

    #[rstest]
    fn recovery_fits_the_eliminated_term_exactly(
        #[values(
            Dominant::InterceptOnly,
            Dominant::CollinearLargeMean,
            Dominant::RankEdge,
            Dominant::SlopeOnly,
            Dominant::DroppedSlope
        )]
        dominant: Dominant,
        #[values(false, true)] weighted: bool,
        #[values(false, true)] sorted: bool,
    ) {
        let n = 600;
        let n_levels = 40;
        let e: Vec<u32> = (0..n).map(|i| ((i * 7) % n_levels) as u32).collect();
        let f: Vec<u32> = (0..n).map(|i| (i % 5) as u32).collect();
        let g: Vec<u32> = (0..n).map(|i| ((i / 3) % 7) as u32).collect();
        let base: Vec<f64> = (0..n).map(|i| 1e6 + noise(i)).collect();
        let near = |k: usize, d: f64| -> Vec<f64> {
            (0..n).map(|i| base[i] + d * noise(k * n + i)).collect()
        };
        let zs: Vec<Vec<f64>> = match dominant {
            Dominant::InterceptOnly => vec![],
            Dominant::CollinearLargeMean => vec![base.clone(), near(1, 1e-4), near(2, 1e-4)],
            Dominant::RankEdge => vec![base.clone(), near(1, 1e-5), near(2, 3e-6)],
            Dominant::SlopeOnly => (1..3)
                .map(|k| (0..n).map(|i| noise(k * n + i)).collect())
                .collect(),
            Dominant::DroppedSlope => vec![(0..n)
                .map(|i| if e[i] == 0 { 2.0 } else { noise(n + i) })
                .collect()],
        };
        let intercept = !matches!(dominant, Dominant::SlopeOnly);
        // A slope-free design when the dominant term has none either.
        let gz: Vec<Vec<f64>> = match dominant {
            Dominant::InterceptOnly => vec![],
            _ => vec![(0..n).map(|i| noise(5 * n + i)).collect()],
        };
        let effects = vec![
            Effect::new(&f, true, []).unwrap(),
            Effect::new(&e, intercept, zs.iter().map(|z| &z[..])).unwrap(),
            Effect::new(&g, true, gz.iter().map(|z| &z[..])).unwrap(),
        ];
        let design = if sorted {
            Design::new(effects)
        } else {
            Design::new_unsorted(effects)
        }
        .unwrap();
        assert_eq!(design.dominant, 1);
        // Every row of level 3 and every fourth row elsewhere carry zero weight.
        let weights: Vec<f64> = (0..n)
            .map(|i| match (e[i], i % 4) {
                (3, _) | (_, 0) => 0.0,
                _ => 1.0 + noise(9 * n + i).abs(),
            })
            .collect();
        let prepared = PreparedDesign::new(design, weighted.then_some(&weights[..])).unwrap();
        let dofs = prepared.design.terms[1].dofs();
        let unidentified: Vec<usize> = prepared
            .diagonal(1)
            .iter()
            .enumerate()
            .filter(|&(_, &g)| g == 0.0)
            .map(|(j, _)| dofs.start + j)
            .collect();
        assert_eq!(
            unidentified.is_empty(),
            !weighted && !matches!(dominant, Dominant::DroppedSlope | Dominant::RankEdge)
        );

        let p = prepared.design.n_dofs;
        let kept: Vec<f64> = (0..p).map(|j| noise(11 * n + j)).collect();
        let residual = |x: &[f64], b: &[f64]| -> Vec<f64> {
            let mut fitted = vec![0.0; n];
            prepared.apply(x, &mut fitted).unwrap();
            b.iter().zip(&fitted).map(|(b, f)| b - f).collect()
        };
        let eliminated_normal = |r: &[f64]| {
            let mut normal = vec![0.0; p];
            prepared.apply_adjoint(r, &mut normal).unwrap();
            normal[dofs.clone()].to_vec()
        };

        // Any `b`: the recovered block solves the eliminated term's normal equations.
        let b: Vec<f64> = (0..n).map(|i| noise(13 * n + i)).collect();
        let mut without = kept.clone();
        without[dofs.clone()].fill(0.0);
        let mut x = kept.clone();
        prepared.recover_eliminated(&b, &mut x);
        let normal = eliminated_normal(&residual(&x, &b));
        let scale = eliminated_normal(&residual(&without, &b));
        assert!(
            norm(&normal) <= 1e-13 * norm(&scale),
            "‖A_eᵀ r‖ = {:.2e} of {:.2e}",
            norm(&normal),
            norm(&scale)
        );
        assert!(unidentified.iter().all(|&j| x[j] == 0.0));

        // `b` in the span: the kept coefficients of a solution recover a full solution.
        let mut b = vec![0.0; n];
        prepared.apply(&kept, &mut b).unwrap();
        let mut x = kept.clone();
        x[dofs.clone()].fill(f64::NAN);
        prepared.recover_eliminated(&b, &mut x);
        let r = residual(&x, &b);
        assert!(
            norm(&r) <= 1e-13 * norm(&b),
            "‖b − A x̂‖ = {:.2e} of {:.2e}",
            norm(&r),
            norm(&b)
        );
    }
}
