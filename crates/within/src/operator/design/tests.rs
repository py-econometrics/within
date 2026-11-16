/// Deterministic pseudo-random f64 in [-1, 1).
fn noise(seed: usize) -> f64 {
    let mut z = seed as u64 ^ 0x9E37_79B9_7F4A_7C15;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    (z >> 11) as f64 / (1u64 << 52) as f64 - 1.0
}

/// Row `i`'s entries `(coefficient, value)` of `W^½ D` in solve coordinates, read off the terms
/// directly: the reference the operators must agree with.
fn row_entries(prepared: &crate::domain::PreparedDesign<'_>, i: usize) -> Vec<(usize, f64)> {
    let w = prepared.sqrt_weights().map_or(1.0, |s| s[i]);
    let mut entries = Vec::new();
    for t in prepared.terms() {
        for c in 0..t.term.n_columns() {
            let at = t.term.column_dofs(c).start + t.term.levels()[i] as usize;
            entries.push((at, w * t.loading(c).map_or(1.0, |z| z[i])));
        }
    }
    entries
}

/// `W^½ D x`.
fn product(prepared: &crate::domain::PreparedDesign<'_>, x: &[f64]) -> Vec<f64> {
    (0..prepared.design.n_obs)
        .map(|i| {
            row_entries(prepared, i)
                .iter()
                .map(|&(j, v)| v * x[j])
                .sum()
        })
        .collect()
}

/// `(W^½ D)ᵀ r`.
fn adjoint(prepared: &crate::domain::PreparedDesign<'_>, r: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; prepared.design.n_dofs];
    for (i, &ri) in r.iter().enumerate() {
        for (j, v) in row_entries(prepared, i) {
            out[j] += v * ri;
        }
    }
    out
}

/// Every gather arm, on a design the locality sort permutes: `demeaned` is `y − D x`.
#[test]
fn demeaned_is_the_response_less_the_design_product() {
    use crate::domain::{Design, Effect, PreparedDesign};
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
    let prepared = PreparedDesign::unweighted_for_test(Design::new(effects).unwrap());
    assert!(prepared.design.obs_perm.is_some());
    let x: Vec<f64> = (0..prepared.design.n_dofs)
        .map(|j| noise(7_000 + j))
        .collect();
    let y: Vec<f64> = (0..n).map(|i| noise(9_000 + i)).collect();
    let got = prepared.demeaned(&x, &y, None);
    let fitted = product(&prepared, &x);
    let y = prepared.design.permute_obs_in(&y);
    for ((g, f), yi) in got.iter().zip(&fitted).zip(y.iter()) {
        assert!((g - (yi - f)).abs() <= 1e-12, "{g} vs {}", yi - f);
    }
}

mod weighted_adjoint_proptests {
    use crate::operator::design::ReducedDesign;
    use proptest::prelude::*;
    use schwarz_precond::Operator;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]

        /// `⟨A_r x, r⟩ = ⟨x, A_rᵀ r⟩` on random weighted designs.
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
            let reduced = ReducedDesign::new(design, Some(&weights)).unwrap();

            let x: Vec<f64> = (0..reduced.ncols())
                .map(|i| (i as f64 * 0.37 + seed as f64 * 0.13).sin())
                .collect();
            let r: Vec<f64> = (0..reduced.nrows())
                .map(|i| (i as f64 * 0.29 + seed as f64 * 0.07).cos())
                .collect();

            let mut ax = vec![0.0f64; reduced.nrows()];
            reduced.apply(&x, &mut ax).unwrap();
            let lhs: f64 = ax.iter().zip(&r).map(|(a, ri)| a * ri).sum();

            let mut atr = vec![0.0f64; reduced.ncols()];
            reduced.apply_adjoint(&r, &mut atr).unwrap();
            let rhs: f64 = x.iter().zip(&atr).map(|(xi, ai)| xi * ai).sum();

            prop_assert!(
                (lhs - rhs).abs() < 1e-8,
                "<A_r x, r>={lhs} != <x, A_r^T r>={rhs}"
            );
        }
    }
}

mod elimination_tests {
    use rstest::rstest;
    use schwarz_precond::Operator;

    use super::{adjoint, noise, product};
    use crate::domain::{Design, Effect};
    use crate::linalg::dot;
    use crate::operator::design::ReducedDesign;

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

    const N: usize = 600;

    /// Three terms whose middle one, the dominant, is shaped by `dominant`.
    fn fixture(dominant: Dominant, weighted: bool) -> ReducedDesign<'static> {
        let n = N;
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
        let design = Design::new(effects).unwrap();
        assert_eq!(design.dominant, 1);
        // Every row of level 3 and every fourth row elsewhere carry zero weight.
        let weights: Vec<f64> = (0..n)
            .map(|i| match (e[i], i % 4) {
                (3, _) | (_, 0) => 0.0,
                _ => 1.0 + noise(9 * n + i).abs(),
            })
            .collect();
        ReducedDesign::new(design.into_owned(), weighted.then_some(&weights[..])).unwrap()
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
    ) {
        let n = N;
        let reduced = fixture(dominant, weighted);
        let prepared = reduced.prepared();
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
        let x_k: Vec<f64> = [&kept[..dofs.start], &kept[dofs.end..]].concat();
        // `f` is a function of `e` (`i mod 5` of `7i mod 40`) unless `e` has no intercept.
        let diagonal = reduced.gram_diagonal();
        assert_eq!(
            diagonal[..5].iter().all(|&d| d == 0.0),
            !matches!(dominant, Dominant::SlopeOnly)
        );
        let passed: Vec<f64> = (x_k.iter().zip(&diagonal))
            .map(|(&x, &d)| if d == 0.0 { 0.0 } else { x })
            .collect();
        let recover = |b: &[f64]| {
            let x = reduced.recover(|i| b[i], &x_k);
            assert_eq!([&x[..dofs.start], &x[dofs.end..]].concat(), passed);
            x
        };
        let residual = |x: &[f64], b: &[f64]| -> Vec<f64> {
            let fitted = product(prepared, x);
            b.iter().zip(&fitted).map(|(b, f)| b - f).collect()
        };
        let eliminated_normal = |r: &[f64]| adjoint(prepared, r)[dofs.clone()].to_vec();

        // Any `b`: the recovered block solves the eliminated term's normal equations.
        let b: Vec<f64> = (0..n).map(|i| noise(13 * n + i)).collect();
        let mut without = kept.clone();
        without[dofs.clone()].fill(0.0);
        let x = recover(&b);
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
        let b = product(prepared, &kept);
        let x = recover(&b);
        let r = residual(&x, &b);
        assert!(
            norm(&r) <= 1e-13 * norm(&b),
            "‖b − A x̂‖ = {:.2e} of {:.2e}",
            norm(&r),
            norm(&b)
        );
    }

    #[rstest]
    fn the_reduced_operator_projects_the_kept_columns_off_the_eliminated_span(
        #[values(
            Dominant::InterceptOnly,
            Dominant::CollinearLargeMean,
            Dominant::RankEdge,
            Dominant::SlopeOnly,
            Dominant::DroppedSlope
        )]
        dominant: Dominant,
        #[values(false, true)] weighted: bool,
    ) {
        let reduced = fixture(dominant, weighted);
        let prepared = reduced.prepared();
        let dofs = prepared.design.terms[1].dofs();
        let (p, k) = (prepared.design.n_dofs, reduced.ncols());
        assert_eq!(k, p - dofs.len());
        let embed = |x_k: &[f64]| {
            [
                &x_k[..dofs.start],
                &vec![0.0; dofs.len()],
                &x_k[dofs.start..],
            ]
            .concat()
        };
        let eliminated_normal = |r: &[f64]| adjoint(prepared, r)[dofs.clone()].to_vec();

        let x_k: Vec<f64> = (0..k).map(|j| noise(17 * N + j)).collect();
        let kept_image = product(prepared, &embed(&x_k));
        let mut image = vec![0.0; N];
        reduced.apply(&x_k, &mut image).unwrap();

        // `A_r x ⊥ A_e`.
        let normal = eliminated_normal(&image);
        let scale = eliminated_normal(&kept_image);
        assert!(
            norm(&normal) <= 1e-13 * norm(&scale),
            "‖A_eᵀ A_r x‖ = {:.2e} of {:.2e}",
            norm(&normal),
            norm(&scale)
        );
        // `(I − P_e) b ⊥ A_e`.
        let projected = reduced.project_rhs(|i| kept_image[i]);
        let normal = eliminated_normal(&projected);
        assert!(norm(&normal) <= 1e-13 * norm(&scale));
        // `A_k x − A_r x ∈ span(A_e)`: fitting it by `A_e` alone leaves nothing.
        let removed: Vec<f64> = kept_image.iter().zip(&image).map(|(a, r)| a - r).collect();
        let fit = reduced.recover(|i| removed[i], &vec![0.0; k]);
        let fitted = product(prepared, &fit);
        let left: Vec<f64> = removed.iter().zip(&fitted).map(|(d, f)| d - f).collect();
        assert!(norm(&left) <= 1e-13 * norm(&kept_image));

        // `⟨A_r x, v⟩ = ⟨x, A_rᵀ v⟩`.
        let v: Vec<f64> = (0..N).map(|i| noise(19 * N + i)).collect();
        let mut adjoint = vec![0.0; k];
        reduced.apply_adjoint(&v, &mut adjoint).unwrap();
        let gap = (dot(&image, &v) - dot(&x_k, &adjoint)).abs();
        assert!(
            gap <= 1e-13 * norm(&image) * norm(&v),
            "adjoint gap {gap:.2e}"
        );

        // `diag(A_rᵀA_r)` column by column, relative to the kept column's own norm.
        let diagonal = reduced.gram_diagonal();
        let kept_gram = [prepared.diagonal(0), prepared.diagonal(2)].concat();
        for j in 0..k {
            let mut unit = vec![0.0; k];
            unit[j] = 1.0;
            reduced.apply(&unit, &mut image).unwrap();
            let squared = dot(&image, &image);
            assert!(
                (diagonal[j] - squared).abs() <= 1e-12 * kept_gram[j],
                "column {j}: {:.6e} vs ‖A_r e_j‖² = {squared:.6e} of {:.2e}",
                diagonal[j],
                kept_gram[j]
            );
        }
    }

    /// Each cross-moment squares below the float range, but its share of the norm does not.
    #[test]
    fn the_diagonal_keeps_the_share_of_tiny_cross_moments() {
        let (e, f) = (vec![0, 0, 1, 1], vec![0, 1, 0, 1]);
        let effects = vec![
            Effect::new(&f, true, []).unwrap(),
            Effect::new(&e, true, []).unwrap(),
        ];
        let design = Design::new(effects).unwrap();
        let weights = Some(&[1e-200; 4][..]);
        let diagonal = ReducedDesign::new(design, weights).unwrap().gram_diagonal();
        assert_eq!(diagonal.len(), 2);
        for d in diagonal {
            assert!((d / 1e-200 - 1.0).abs() <= 1e-14, "{d:e}");
        }
    }

    /// A kept term nested in the eliminated one projects to nothing: every level is resummed to 0.
    #[test]
    fn a_kept_term_nested_in_the_eliminated_one_has_a_zero_diagonal() {
        let n = 200_000;
        let e: Vec<u32> = (0..n).map(|i| ((i * 7919) % 20_000) as u32).collect();
        let nested: Vec<u32> = e.iter().map(|&l| l / 4).collect();
        let g: Vec<u32> = (0..n).map(|i| (i % 7) as u32).collect();
        let effects = vec![
            Effect::new(&e, true, []).unwrap(),
            Effect::new(&nested, true, []).unwrap(),
            Effect::new(&g, true, []).unwrap(),
        ];
        let reduced = ReducedDesign::new(Design::new(effects).unwrap(), None).unwrap();
        let diagonal = reduced.gram_diagonal();
        let (k, n_nested) = (reduced.ncols(), 5_000);
        assert_eq!(diagonal.len(), k);
        assert!(diagonal[..n_nested].iter().all(|&d| d == 0.0));
        let mut image = vec![0.0; n];
        for j in n_nested..k {
            let mut unit = vec![0.0; k];
            unit[j] = 1.0;
            reduced.apply(&unit, &mut image).unwrap();
            let squared = dot(&image, &image);
            assert!(
                (diagonal[j] - squared).abs() <= 1e-12 * squared,
                "column {j}"
            );
        }
    }

    /// The products' and the diagonal's bits must not depend on the pool, the leaves capped.
    #[test]
    fn the_reduced_products_are_pool_independent() {
        let n = 600_000;
        let e: Vec<u32> = (0..n).map(|i| ((i * 7919) % 100_000) as u32).collect();
        let f: Vec<u32> = (0..n).map(|i| ((i / 3) % 30_000) as u32).collect();
        let g: Vec<u32> = (0..n).map(|i| (i % 11) as u32).collect();
        let z: Vec<Vec<f64>> = (0..2)
            .map(|k| (0..n).map(|i| noise(k * n + i)).collect())
            .collect();
        let effects = vec![
            Effect::new(&f, true, [&z[0][..]]).unwrap(),
            Effect::new(&e, true, [&z[1][..]]).unwrap(),
            Effect::new(&g, true, []).unwrap(),
        ];
        let weights: Vec<f64> = (0..n).map(|i| 0.5 + noise(2 * n + i).abs()).collect();
        // Built in each pool: construction computes the diagonal.
        let build = || ReducedDesign::new(Design::new(effects.clone()).unwrap(), Some(&weights));
        let k = Design::new(effects.clone()).unwrap().n_unknowns();
        // A leaf of 4·k rows exceeds an eighth of the design, so the cap sets the leaves.
        assert!(4 * k > n / 8);
        let x: Vec<f64> = (0..k).map(|j| noise(3 * n + j)).collect();
        let v: Vec<f64> = (0..n).map(|i| noise(5 * n + i)).collect();
        let products_in = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("test rayon pool")
                .install(|| {
                    let reduced = build().unwrap();
                    let (mut image, mut adjoint) = (vec![0.0; n], vec![0.0; k]);
                    reduced.apply(&x, &mut image).unwrap();
                    reduced.apply_adjoint(&v, &mut adjoint).unwrap();
                    let diagonal = reduced.gram_diagonal();
                    let bits = (image.iter().chain(&adjoint).chain(&diagonal)).map(|v| v.to_bits());
                    bits.collect::<Vec<_>>()
                })
        };
        let reference = products_in(1);
        for threads in [2, 3, 8] {
            // Not `assert_eq!`: a printed vector would bury the thread count.
            assert!(products_in(threads) == reference, "{threads} threads");
        }
    }

    /// Enough rows for several parallel tasks and accumulator leaves, against a hand projection.
    #[rstest]
    fn the_reduced_operator_matches_level_means_across_tasks(
        #[values(false, true)] weighted: bool,
    ) {
        let n = 150_000;
        let e: Vec<u32> = (0..n).map(|i| ((i * 7919) % 20_000) as u32).collect();
        let f: Vec<u32> = (0..n).map(|i| ((i / 3) % 700) as u32).collect();
        let g: Vec<u32> = (0..n).map(|i| (i % 11) as u32).collect();
        let effects = vec![
            Effect::new(&f, true, []).unwrap(),
            Effect::new(&e, true, []).unwrap(),
            Effect::new(&g, true, []).unwrap(),
        ];
        let design = Design::new(effects).unwrap();
        let weights: Vec<f64> = (0..n)
            .map(|i| {
                if i % 5 == 0 {
                    0.0
                } else {
                    0.5 + noise(i).abs()
                }
            })
            .collect();
        let reduced = ReducedDesign::new(design, weighted.then_some(&weights[..])).unwrap();
        let prepared = reduced.prepared();
        let dofs = prepared.design.terms[1].dofs();
        let k = reduced.ncols();
        let x_k: Vec<f64> = (0..k).map(|j| noise(3 * n + j)).collect();
        let full_x = [
            &x_k[..dofs.start],
            &vec![0.0; dofs.len()],
            &x_k[dofs.start..],
        ]
        .concat();
        let kept_image = product(prepared, &full_x);

        // `(I − P_e) u`: subtract each eliminated level's weighted mean, in internal row order.
        let sw = prepared.sqrt_weights();
        let w = |i: usize| sw.map_or(1.0, |s| s[i] * s[i]);
        let root = |i: usize| sw.map_or(1.0, |s| s[i]);
        let levels = prepared.design.terms[1].levels();
        let mut sums = vec![(0.0, 0.0); 20_000];
        for (i, &u) in kept_image.iter().enumerate() {
            let s = &mut sums[levels[i] as usize];
            s.0 += root(i) * u;
            s.1 += w(i);
        }
        let expected: Vec<f64> = (0..n)
            .map(|i| {
                let (su, sw) = sums[levels[i] as usize];
                kept_image[i] - if sw > 0.0 { root(i) * su / sw } else { 0.0 }
            })
            .collect();
        let mut image = vec![0.0; n];
        reduced.apply(&x_k, &mut image).unwrap();
        let gap: Vec<f64> = image.iter().zip(&expected).map(|(a, b)| a - b).collect();
        assert!(
            norm(&gap) <= 1e-12 * norm(&expected),
            "apply gap {:.2e}",
            norm(&gap)
        );

        let v: Vec<f64> = (0..n).map(|i| noise(5 * n + i)).collect();
        let mut adjoint = vec![0.0; k];
        reduced.apply_adjoint(&v, &mut adjoint).unwrap();
        let gap = (dot(&image, &v) - dot(&x_k, &adjoint)).abs();
        assert!(
            gap <= 1e-12 * norm(&image) * norm(&v),
            "adjoint gap {gap:.2e}"
        );
    }
}
