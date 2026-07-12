//! Seam tests for the per-level pivoted Gram–Schmidt (pivot bookkeeping,
//! rank-tolerance contract — neither reachable through the public API) and
//! for the per-term independence of the multi-term whitening.

use crate::domain::Effect;

use super::*;

#[test]
fn gram_schmidt_orthonormalizes_under_a_non_monotonic_pivot_order() {
    // Diagonals [2, 5, 3] force the pivot sequence 1 → 2 → 0: any
    // bookkeeping that assumes natural or physically swapped order breaks
    // the W·G·Wᵀ = I identity below (the prototype's ≥3-slope cliff).
    let g = [2.0, 1.0, 0.5, 1.0, 5.0, 2.0, 0.5, 2.0, 3.0];
    let (w, kept) = pivoted_gram_schmidt(&g, 3, RANK_TOL);
    assert_eq!(kept, [true; 3]);
    assert_eq!(w.len(), 9);

    for r in 0..3 {
        for s in 0..3 {
            let wgw: f64 = (0..3)
                .flat_map(|j| (0..3).map(move |k| (j, k)))
                .map(|(j, k)| w[r * 3 + j] * g[j * 3 + k] * w[s * 3 + k])
                .sum();
            let expected = if r == s { 1.0 } else { 0.0 };
            assert!(
                (wgw - expected).abs() < 1e-12,
                "(W·G·Wᵀ)[{r}][{s}] = {wgw}, expected {expected}"
            );
        }
    }
}

#[test]
fn zero_tolerance_keeps_a_near_degenerate_direction_the_default_drops() {
    let eps = 1e-12;
    let g = [1.0, 1.0 - eps, 1.0 - eps, 1.0];
    let (_, kept) = pivoted_gram_schmidt(&g, 2, RANK_TOL);
    assert_eq!(kept.iter().filter(|&&k| k).count(), 1);
    let (_, kept) = pivoted_gram_schmidt(&g, 2, 0.0);
    assert_eq!(kept, [true, true]);
}

const F0: [u32; 6] = [0, 0, 0, 1, 1, 1];
const Z0: [f64; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
const Z1: [f64; 6] = [1.0, 4.0, 9.0, 16.0, 25.0, 36.0];
const G: [u32; 6] = [0, 1, 2, 0, 1, 2];
const F2: [u32; 6] = [0, 1, 0, 1, 0, 1];
const Z2: [f64; 6] = [2.0, 7.0, 1.0, 8.0, 3.0, 9.0];

/// Two slope-bearing terms around a plain one; term 0 is dominant and sorted
/// so the locality sort stays a no-op.
fn three_term_effects() -> Vec<Effect<'static>> {
    vec![
        Effect::new(&F0, true, [&Z0[..], &Z1[..]]).unwrap(),
        Effect::new(&G, true, []).unwrap(),
        Effect::new(&F2, true, [&Z2[..]]).unwrap(),
    ]
}

#[test]
fn build_whitens_each_slope_bearing_term() {
    let mut design = Design::new(three_term_effects()).unwrap();
    let rp = SlopeReparam::build(&mut design, None).unwrap();
    assert!(rp.unidentified.is_empty());

    for term in [0, 2] {
        let meta = &design.terms[term];
        let levels = design.frame.level_column(term);
        let us: Vec<&[f64]> = meta
            .slopes
            .iter()
            .map(|&c| design.frame.loading_column(c))
            .collect();
        for level in 0..meta.n_levels {
            let obs: Vec<usize> = (0..levels.len())
                .filter(|&i| levels[i] as usize == level)
                .collect();
            for (j, uj) in us.iter().enumerate() {
                let sum: f64 = obs.iter().map(|&i| uj[i]).sum();
                assert!(sum.abs() < 1e-12, "term {term} level {level} Σu{j} = {sum}");
                for (k, uk) in us.iter().enumerate() {
                    let gram: f64 = obs.iter().map(|&i| uj[i] * uk[i]).sum();
                    let expected = if j == k { 1.0 } else { 0.0 };
                    assert!(
                        (gram - expected).abs() < 1e-12,
                        "term {term} level {level} ⟨u{j}, u{k}⟩ = {gram}"
                    );
                }
            }
        }
    }
}

#[test]
fn unidentified_directions_ascend_across_terms() {
    // Term 0 drops a slope at level 1, term 1 at the *earlier* level 0:
    // index-order term iteration keeps the list ascending in
    // (term, level, column) without any sort.
    let f0 = [0u32, 0, 1, 1, 2, 2];
    let z0 = [1.0, 2.0, 5.0, 5.0, 3.0, 7.0];
    let f1 = [0u32, 1, 0, 1, 0, 1];
    let z1 = [4.0, 1.0, 4.0, 2.0, 4.0, 3.0];
    let effects = vec![
        Effect::new(&f0, true, [&z0[..]]).unwrap(),
        Effect::new(&f1, true, [&z1[..]]).unwrap(),
    ];
    let mut design = Design::new(effects).unwrap();
    let rp = SlopeReparam::build(&mut design, None).unwrap();
    assert_eq!(
        rp.unidentified,
        vec![
            UnidentifiedDirection {
                term: 0,
                level: 1,
                column: 1,
            },
            UnidentifiedDirection {
                term: 1,
                level: 0,
                column: 1,
            },
        ]
    );
}

#[test]
fn back_transform_leaves_other_terms_untouched() {
    let mut design = Design::new(three_term_effects()).unwrap();
    let rp = SlopeReparam::build(&mut design, None).unwrap();

    let mut x: Vec<f64> = (0..design.n_dofs).map(|i| 1.0 + i as f64).collect();
    let before = x.clone();
    rp.back_transform(&mut x);

    // Plain term 1 sits between the two slope-bearing blocks.
    let (t1, t2) = (design.terms[1].offset, design.terms[2].offset);
    assert_eq!(x[t1..t2], before[t1..t2]);
    assert_ne!(x[..t1], before[..t1]);
    assert_ne!(x[t2..], before[t2..]);
}
