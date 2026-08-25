//! Seam tests for the per-term independence of the multi-term whitening,
//! not reachable through the public API.

use crate::channel::Channel;
use crate::domain::level_moments::TermMoments;
use crate::domain::Effect;
use crate::Design;

use super::*;

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
    let design = Design::new(three_term_effects()).unwrap();
    let raw_loadings: Vec<(usize, Vec<f64>)> = design
        .terms
        .iter()
        .flat_map(|term| term.columns.iter())
        .filter_map(|loading| loading.covariate())
        .map(|&column| {
            let column = column as usize;
            (column, design.loading_column(column).to_vec())
        })
        .collect();
    let moments = TermMoments::build(&design, None).unwrap();
    let mut solver_design = SolverDesign::new(design);
    let rp = SlopeReparam::build(&mut solver_design, &moments).unwrap();
    assert!(rp.unidentified.is_empty());
    for (column, expected) in raw_loadings {
        assert_eq!(
            solver_design.design().loading_column(column),
            expected.as_slice(),
            "canonical loading column {column} was mutated"
        );
    }

    let design = solver_design.design();
    for term in [0, 2] {
        let meta = &design.terms[term];
        let levels = design.level_column(term);
        let us: Vec<&[f64]> = meta
            .columns
            .iter()
            .filter_map(|c| c.covariate())
            .map(|&k| solver_design.loading_column(k as usize))
            .collect();
        for level in 0..meta.n_levels() {
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
    // Index-order term iteration keeps the list ascending without any sort.
    let f0 = [0u32, 0, 1, 1, 2, 2];
    let z0 = [1.0, 2.0, 5.0, 5.0, 3.0, 7.0];
    let f1 = [0u32, 1, 0, 1, 0, 1];
    let z1 = [4.0, 1.0, 4.0, 2.0, 4.0, 3.0];
    let effects = vec![
        Effect::new(&f0, true, [&z0[..]]).unwrap(),
        Effect::new(&f1, true, [&z1[..]]).unwrap(),
    ];
    let design = Design::new(effects).unwrap();
    let moments = TermMoments::build(&design, None).unwrap();
    let mut solver_design = SolverDesign::new(design);
    let rp = SlopeReparam::build(&mut solver_design, &moments).unwrap();
    assert_eq!(
        rp.unidentified,
        vec![
            CoefficientPosition {
                channel: Channel { term: 0, column: 1 },
                level: 1,
            },
            CoefficientPosition {
                channel: Channel { term: 1, column: 1 },
                level: 0,
            },
        ]
    );
}

#[test]
fn back_transform_leaves_other_terms_untouched() {
    let design = Design::new(three_term_effects()).unwrap();
    let moments = TermMoments::build(&design, None).unwrap();
    let mut solver_design = SolverDesign::new(design);
    let rp = SlopeReparam::build(&mut solver_design, &moments).unwrap();

    let design = solver_design.design();
    let mut x: Vec<f64> = (0..design.n_dofs).map(|i| 1.0 + i as f64).collect();
    let before = x.clone();
    rp.back_transform(&mut x);

    // Plain term 1 sits between the two slope-bearing blocks.
    let (t1, t2) = (design.terms[1].offset, design.terms[2].offset);
    assert_eq!(x[t1..t2], before[t1..t2]);
    assert_ne!(x[..t1], before[..t1]);
    assert_ne!(x[t2..], before[t2..]);
}
