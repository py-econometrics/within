/// Deterministic pseudo-random f64 in [-1, 1).
fn noise(seed: usize) -> f64 {
    let mut z = seed as u64 ^ 0x9E37_79B9_7F4A_7C15;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    (z >> 11) as f64 / (1u64 << 52) as f64 - 1.0
}

/// Dense `A_r = (I − P_e) W^{1/2} D̃_k`, one column per unknown: each kept column with its
/// projection onto each eliminated level's columns removed, from the data alone, with each kept
/// column's norm `‖a_j‖²` before. `e`'s columns are `W`-orthogonal within a level, so projecting one
/// at a time is `P_e`.
fn dense_reduced(design: &crate::system::ReducedDesign<'_>) -> (Vec<Vec<f64>>, Vec<f64>) {
    let n = design.n_obs();
    let s = |i: usize| design.split.weights.sqrt().map_or(1.0, |s| s[i]);
    let mut columns = vec![vec![0.0; n]; design.unknowns().len()];
    for (channel, at) in design.unknowns().channels(design.layout()) {
        let t = design.split.term(channel.term);
        let z = t.loading(channel.column);
        for (i, &level) in t.levels().iter().enumerate() {
            columns[at.start + level as usize][i] = s(i) * z.map_or(1.0, |z| z[i]);
        }
    }
    let own = columns
        .iter()
        .map(|c| c.iter().map(|a| a * a).sum())
        .collect();
    let (e, segments) = (
        design.split.term(design.split.eliminated()),
        design.incidence().segments(),
    );
    for column in &mut columns {
        for level in 0..segments.n_levels() {
            let rows = segments.rows(level);
            for c in 0..e.layout().n_columns() {
                let z = e.loading(c);
                let a: Vec<f64> = rows
                    .clone()
                    .map(|i| s(i) * z.map_or(1.0, |z| z[i]))
                    .collect();
                let g: f64 = a.iter().map(|a| a * a).sum();
                if g == 0.0 {
                    continue;
                }
                let p = a
                    .iter()
                    .zip(&column[rows.clone()])
                    .map(|(a, v)| a * v)
                    .sum::<f64>()
                    / g;
                for (v, a) in column[rows.clone()].iter_mut().zip(&a) {
                    *v -= p * a;
                }
            }
        }
    }
    (columns, own)
}

/// `A_r x`, `A_rᵀ r` and `diag(S)` against [`dense_reduced`]; the adjoint is 0 on absorbed columns.
fn assert_matches_dense(design: &crate::system::ReducedDesign<'_>) {
    use crate::system::SystemMatrix;
    use schwarz_precond::Operator;
    let (columns, own) = dense_reduced(design);
    let norms: Vec<f64> = columns
        .iter()
        .map(|c| c.iter().map(|a| a * a).sum())
        .collect();
    // An absorbed column's dense energy is within rounding of its own norm, not the largest one.
    for ((&norm, &own), &energy) in norms.iter().zip(&own).zip(design.diag_s.values()) {
        let tolerance = 1e-10 * if energy == 0.0 { own } else { norm };
        assert!(
            (energy - norm).abs() <= tolerance,
            "diag(S) {energy} vs {norm}"
        );
    }
    let (n, n_kept) = (design.n_obs(), design.unknowns().len());
    let close = |got: &[f64], expect: &[f64]| {
        let scale = expect.iter().fold(1.0f64, |m, v| m.max(v.abs()));
        for (g, e) in got.iter().zip(expect) {
            assert!((g - e).abs() <= 1e-10 * scale, "{g} vs {e}");
        }
    };
    let x: Vec<f64> = (0..n_kept).map(|j| noise(7_000 + j)).collect();
    let mut got = vec![0.0; n];
    SystemMatrix::new(design).apply(&x, &mut got).unwrap();
    let mut expect = vec![0.0; n];
    for (column, &xj) in columns.iter().zip(&x) {
        for (e, a) in expect.iter_mut().zip(column) {
            *e += a * xj;
        }
    }
    close(&got, &expect);

    let r: Vec<f64> = (0..n).map(|i| noise(9_000 + i)).collect();
    let mut got = vec![0.0; n_kept];
    SystemMatrix::new(design)
        .apply_adjoint(&r, &mut got)
        .unwrap();
    let expect: Vec<f64> = columns
        .iter()
        .zip(design.diag_s.values())
        .map(|(column, &energy)| {
            let dot: f64 = column.iter().zip(&r).map(|(a, r)| a * r).sum();
            if energy == 0.0 {
                0.0
            } else {
                dot
            }
        })
        .collect();
    close(&got, &expect);
}

mod design_tests {
    use super::assert_matches_dense;
    use crate::design::Design;
    use crate::system::SystemMatrix;
    use schwarz_precond::Operator;

    #[test]
    fn two_factors_match_the_dense_reduced_matrix() {
        let design = Design::from_levels_for_test(vec![vec![0, 1, 1, 2, 0], vec![0, 0, 1, 2, 3]])
            .reduce(None)
            .unwrap();
        assert_eq!(design.unknowns().len(), 3);
        assert_matches_dense(&design);
    }

    /// One term leaves nothing to solve for: `A_r` has no columns.
    #[test]
    fn a_single_factor_leaves_no_unknowns() {
        let design = Design::from_levels_for_test(vec![vec![0u32, 0, 1, 1, 2]])
            .reduce(None)
            .unwrap();
        assert_eq!(design.unknowns().len(), 0);
        let mut y = vec![1.0; 5];
        SystemMatrix::new(&design).apply(&[], &mut y).unwrap();
        assert_eq!(y, [0.0; 5]);
        SystemMatrix::new(&design)
            .apply_adjoint(&[1.0; 5], &mut [])
            .unwrap();
    }

    /// A kept factor equal to the eliminated one is absorbed whole: `A_r = 0`, `diag(S) = 0`.
    #[test]
    fn a_kept_copy_of_the_eliminated_factor_is_absorbed() {
        let fa: Vec<u32> = (0..15_000).map(|i| i / 300).collect();
        let design = Design::from_levels_for_test(vec![fa.clone(), fa])
            .reduce(None)
            .unwrap();
        assert!(design.diag_s.values().iter().all(|&e| e == 0.0));
        assert_matches_dense(&design);
    }
}

mod slope_design_tests {
    use super::{assert_matches_dense, noise};
    use crate::design::{Design, Effect};
    use crate::system::ReducedDesign;
    use crate::system::SystemMatrix;
    use schwarz_precond::Operator;

    /// Every kept shape `with_split` fuses, and the dynamic ones, each with a fused and a dynamic
    /// eliminated term, weighted and not.
    #[test]
    fn every_fused_shape_matches_the_dense_reduced_matrix() {
        let n = 60;
        let e: Vec<u32> = (0..n).map(|i| (i / 4) as u32).collect();
        let f: Vec<u32> = (0..n).map(|i| ((i * 7) % 5) as u32).collect();
        let g: Vec<u32> = (0..n).map(|i| ((i * 11) % 3) as u32).collect();
        let h: Vec<u32> = (0..n).map(|i| ((i * 13) % 4) as u32).collect();
        let z: Vec<Vec<f64>> = (0..8)
            .map(|k| (0..n).map(|i| noise(k * 1_000 + i)).collect())
            .collect();
        let weights: Vec<f64> = (0..n).map(|i| 0.5 + noise(77 + i).abs()).collect();
        let slopes = |k: usize| z[..k].iter().map(|z| &z[..]);
        let kept_sets: Vec<Vec<Effect<'_>>> = vec![
            vec![Effect::new(&f, true, []).unwrap()],
            vec![Effect::new(&f, true, [&z[4][..]]).unwrap()],
            vec![
                Effect::new(&f, true, []).unwrap(),
                Effect::new(&g, true, []).unwrap(),
            ],
            vec![
                Effect::new(&f, true, [&z[4][..]]).unwrap(),
                Effect::new(&g, true, []).unwrap(),
            ],
            vec![
                Effect::new(&f, true, []).unwrap(),
                Effect::new(&g, true, [&z[5][..]]).unwrap(),
            ],
            vec![
                Effect::new(&f, true, [&z[4][..]]).unwrap(),
                Effect::new(&g, true, [&z[5][..]]).unwrap(),
            ],
            vec![Effect::new(&f, false, [&z[4][..]]).unwrap()],
            vec![
                Effect::new(&f, true, []).unwrap(),
                Effect::new(&g, true, []).unwrap(),
                Effect::new(&h, true, [&z[6][..]]).unwrap(),
            ],
        ];
        for kept in kept_sets {
            // One slope fuses `e`; four leave it dynamic.
            for n_slopes in [1, 4] {
                for weights in [None, Some(&weights[..])] {
                    let mut effects = vec![Effect::new(&e, true, slopes(n_slopes)).unwrap()];
                    effects.extend(kept.iter().cloned());
                    let design = Design::new(effects).unwrap().reduce(weights).unwrap();
                    assert_eq!(design.split.eliminated(), 0, "the fixture eliminates `e`");
                    assert_matches_dense(&design);
                }
            }
        }
    }

    /// Many terms and slopes of every width, against the dense reference.
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
        let design = Design::new(effects).unwrap().reduce(None).unwrap();
        assert_matches_dense(&design);
    }

    /// Kept slopes that `e`'s slopes span are absorbed on long levels, under heavy-tailed weights:
    /// their cancellation stays below the `rows·ε` cutoff.
    #[test]
    fn kept_slopes_in_the_eliminated_span_are_absorbed_on_long_levels() {
        let n: usize = 300_000;
        let levels: Vec<u32> = (0..n).map(|i| (i / 100_000) as u32).collect();
        let z: Vec<Vec<f64>> = [1e3, 0.0]
            .iter()
            .enumerate()
            .map(|(k, mean)| (0..n).map(|i| mean + noise(k * n + i)).collect())
            .collect();
        let spanned: Vec<Vec<f64>> = [[3.0, 1.37, 0.4], [-2.0, 0.6, 1.74]]
            .iter()
            .map(|[a, b, c]| (0..n).map(|i| a + b * z[0][i] + c * z[1][i]).collect())
            .collect();
        let weights: Vec<f64> = (0..n).map(|i| (6.0 * noise(2 * n + i)).exp()).collect();
        let effects = vec![
            Effect::new(&levels, true, spanned.iter().map(|z| &z[..])).unwrap(),
            Effect::new(&levels, true, z.iter().map(|z| &z[..])).unwrap(),
        ];
        let design = Design::new(effects)
            .unwrap()
            .reduce(Some(&weights))
            .unwrap();
        assert_eq!(design.split.eliminated(), 1, "the fixture eliminates `e`");
        assert!(design.diag_s.values().iter().all(|&e| e == 0.0));
        assert_matches_dense(&design);
    }

    /// Many terms, slopes of every width and collinear copies, over many tasks and leaves.
    fn level_reduction_design() -> ReducedDesign<'static> {
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
        Design::new(effects)
            .unwrap()
            .into_owned()
            .reduce(Some(&weights))
            .unwrap()
    }

    /// The adjoint's bits must not depend on the pool it runs in.
    #[test]
    fn adjoint_is_pool_independent() {
        let design = level_reduction_design();
        let r: Vec<f64> = (0..design.n_obs()).map(|i| noise(29 * i + 5)).collect();
        inflight::parallel::testing::assert_pool_independent(|| {
            let mut dtr = vec![0.0; design.unknowns().len()];
            SystemMatrix::new(&design)
                .apply_adjoint(&r, &mut dtr)
                .unwrap();
            dtr.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        });
    }

    /// ⟨A_r x, r⟩ = ⟨x, A_rᵀ r⟩, weighted, over many leaves.
    #[test]
    fn slope_adjoint_property_each_level_reduction() {
        let design = level_reduction_design();
        let n = design.n_obs();

        let x: Vec<f64> = (0..design.unknowns().len())
            .map(|j| noise(13 * j + 1))
            .collect();
        let r: Vec<f64> = (0..n).map(|i| noise(29 * i + 5)).collect();

        let mut dx = vec![0.0; n];
        SystemMatrix::new(&design).apply(&x, &mut dx).unwrap();
        let mut dtr = vec![0.0; design.unknowns().len()];
        SystemMatrix::new(&design)
            .apply_adjoint(&r, &mut dtr)
            .unwrap();

        let lhs: f64 = dx.iter().zip(&r).map(|(a, b)| a * b).sum();
        let rhs: f64 = x.iter().zip(&dtr).map(|(a, b)| a * b).sum();
        assert!(
            (lhs - rhs).abs() <= 1e-9 * lhs.abs().max(rhs.abs()).max(1.0),
            "{lhs} vs {rhs}"
        );
    }
}

mod weighted_adjoint_proptests {
    use crate::system::SystemMatrix;
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

            let design = crate::design::Design::from_levels_for_test(vec![fa, fb]);
            let design = design.reduce(Some(&weights)).unwrap();

            let n_dofs = design.unknowns().len();
            let n_rows = design.n_obs();

            let x: Vec<f64> = (0..n_dofs)
                .map(|i| (i as f64 * 0.37 + seed as f64 * 0.13).sin())
                .collect();
            let r: Vec<f64> = (0..n_rows)
                .map(|i| (i as f64 * 0.29 + seed as f64 * 0.07).cos())
                .collect();

            let mut dx = vec![0.0f64; n_rows];
            SystemMatrix::new(&design).apply(&x, &mut dx).unwrap();
            let lhs: f64 = dx.iter().zip(&r).map(|(dxi, ri)| dxi * ri).sum();

            let mut dtr = vec![0.0f64; n_dofs];
            SystemMatrix::new(&design).apply_adjoint(&r, &mut dtr).unwrap();
            let rhs: f64 = x.iter().zip(dtr.iter()).map(|(xi, wi)| xi * wi).sum();

            prop_assert!(
                (lhs - rhs).abs() < 1e-8,
                "<W^1/2 D·x, r>={lhs} != <x, D^T W^1/2·r>={rhs}"
            );
        }
    }
}
