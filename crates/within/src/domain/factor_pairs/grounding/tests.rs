use rstest::rstest;

use super::*;
use crate::channel::Channel;
use crate::domain::{CrossTab, Design, Effect};
use Grounding::{Floating, Grounded};

const FIRST_CHANNELS: ChannelPair = ChannelPair {
    rows: Channel { term: 0, column: 0 },
    cols: Channel { term: 1, column: 0 },
};

/// Each component's `(row levels, col levels, grounding)`, sorted.
fn classify_pair(
    rows: Effect<'_>,
    cols: Effect<'_>,
    weights: Option<&[f64]>,
) -> Vec<(usize, usize, Grounding)> {
    let design = Design::new([rows, cols]).expect("valid design");
    let prepared = PreparedDesign::new(design, weights).expect("valid weights");
    let (cross_tab, _) = CrossTab::build_for_pair(&prepared, FIRST_CHANNELS);
    let observations = PairObservations::new(&prepared, FIRST_CHANNELS);
    let forest = ScaledForest::grow(&observations, &cross_tab.c);
    let mut shapes: Vec<_> = (forest.into_components().into_iter())
        .map(|(component, grounding)| (component.rows.len(), component.cols.len(), grounding))
        .collect();
    shapes.sort_by_key(|&(rows, cols, grounding)| (rows, cols, grounding as u8));
    shapes
}

/// Three row levels over two columns with `z = a_row · b_col`: the slope aliases the intercept.
#[rstest]
#[case::unweighted(None)]
#[case::weighted(Some([0.3, 2.0, 1.0, 7.5, 0.01, 4.0]))]
#[case::weak_link(Some([1.0, 1e-12, 1.0, 1.0, 1.0, 1.0]))]
fn separable_loadings_float(#[case] weights: Option<[f64; 6]>) {
    let (rows, cols) = ([0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1]);
    let z = [1.0, 3.0, -2.0, -6.0, 0.5, 1.5];
    let shapes = classify_pair(
        Effect::new(&rows, false, [&z[..]]).unwrap(),
        Effect::new(&cols, true, []).unwrap(),
        weights.as_ref().map(|w| &w[..]),
    );
    assert_eq!(shapes, [(3, 2, Floating)]);
}

/// `ρ ≈ c·δ²` past an alias: below the bound the direction is numerically gone, above it stays.
#[rstest]
#[case::numerically_aliased(1e-9, Floating)]
#[case::identified(1e-4, Grounded)]
fn a_perturbed_alias_floats_only_below_the_bound(#[case] delta: f64, #[case] expected: Grounding) {
    let (rows, cols) = ([0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1]);
    let z = [1.0, 3.0, -2.0, -6.0 * (1.0 + delta), 0.5, 1.5];
    let shapes = classify_pair(
        Effect::new(&rows, false, [&z[..]]).unwrap(),
        Effect::new(&cols, true, []).unwrap(),
        None,
    );
    assert_eq!(shapes, [(3, 2, expected)]);
}

#[test]
fn a_frustrated_alias_floats() {
    let (rows, cols) = ([0, 0, 1, 1], [0, 1, 0, 1]);
    let (x, y) = ([1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, -1.0]);
    let shapes = classify_pair(
        Effect::new(&rows, false, [&x[..]]).unwrap(),
        Effect::new(&cols, false, [&y[..]]).unwrap(),
        None,
    );
    assert_eq!(shapes, [(2, 2, Floating)]);
}

/// Equal magnitudes, opposite signs in one cell: cancelled it splits apart, partial it grounds.
#[rstest]
#[case::cancelled(&[1.0, -1.0], vec![(0, 1, Grounded), (1, 0, Grounded)])]
#[case::partial(&[1.0, -1.0, 1.0], vec![(1, 1, Grounded)])]
fn mixed_signs_in_one_cell_ground(
    #[case] z: &[f64],
    #[case] expected: Vec<(usize, usize, Grounding)>,
) {
    let levels = vec![0; z.len()];
    let shapes = classify_pair(
        Effect::new(&levels, false, [z]).unwrap(),
        Effect::new(&levels, true, []).unwrap(),
        None,
    );
    assert_eq!(shapes, expected);
}

/// A zero loading leaves its intercept unmatched, unless it carries no weight.
#[rstest]
#[case::weighted([1.0, 1.0], Grounded)]
#[case::weightless([1.0, 0.0], Floating)]
fn a_zero_loading_grounds_its_cell(#[case] weights: [f64; 2], #[case] expected: Grounding) {
    let shapes = classify_pair(
        Effect::new(&[0, 0], false, [&[1.0, 0.0][..]]).unwrap(),
        Effect::new(&[0, 0], true, []).unwrap(),
        Some(&weights),
    );
    assert_eq!(shapes, [(1, 1, expected)]);
}

#[test]
fn a_zero_loading_grounds_only_the_component_it_leaves() {
    let shapes = classify_pair(
        Effect::new(&[0, 0], false, [&[1.0, 0.0][..]]).unwrap(),
        Effect::new(&[0, 1], true, []).unwrap(),
        None,
    );
    assert_eq!(shapes, [(0, 1, Grounded), (1, 1, Floating)]);
}

#[test]
fn a_large_tree_floats() {
    let n = 20_000;
    let rows = vec![0; n];
    let cols: Vec<u32> = (0..n as u32).collect();
    let z: Vec<f64> = (0..n).map(|j| 0.7 + 0.03 * (j % 17) as f64).collect();
    let weights: Vec<f64> = (0..n).map(|j| 1.0 + 0.01 * (j % 23) as f64).collect();
    let shapes = classify_pair(
        Effect::new(&rows, false, [&z[..]]).unwrap(),
        Effect::new(&cols, true, []).unwrap(),
        Some(&weights),
    );
    assert_eq!(shapes, [(1, n, Floating)]);
}

/// Loadings spanning 1e800 overflow the tree vector from any root, so the alias goes unproven.
#[test]
fn an_overflowing_tree_vector_grounds() {
    let rows = [0, 0, 1, 1, 2, 2, 3, 3];
    let cols = [0, 1, 1, 2, 2, 3, 3, 4];
    let z = [1e-200, 1.0, 1e-200, 1.0, 1e-200, 1.0, 1e-200, 1.0];
    let shapes = classify_pair(
        Effect::new(&rows, false, [&z[..]]).unwrap(),
        Effect::new(&cols, true, []).unwrap(),
        None,
    );
    assert_eq!(shapes, [(4, 5, Grounded)]);
}

/// Zero loadings, cancelled cells and weightless rows join nothing in either search.
#[test]
fn the_forest_finds_the_cross_tab_components_in_their_order() {
    let mut state = 0.61f64;
    let mut draw = || {
        state = (state * 997.0 + 0.311).fract();
        state
    };
    let (mut rows, mut cols, mut z, mut weights) = (vec![], vec![], vec![], vec![]);
    for _ in 0..40 {
        let (row, col) = ((draw() * 30.0) as u32, (draw() * 30.0) as u32);
        let loading = [0.0, 1.5, -0.5][(draw() * 3.0) as usize];
        // A repeat of opposite loading cancels its cell.
        let copies = if draw() < 0.2 { 2 } else { 1 };
        for k in 0..copies {
            rows.push(row);
            cols.push(col);
            z.push(if k == 0 { loading } else { -loading });
            weights.push(if draw() < 0.1 { 0.0 } else { 1.0 });
        }
    }
    let design = Design::new([
        Effect::new(&rows, false, [&z[..]]).unwrap(),
        Effect::new(&cols, true, []).unwrap(),
    ])
    .expect("valid design");
    let prepared = PreparedDesign::new(design, Some(&weights)).expect("valid weights");
    let (cross_tab, _) = CrossTab::build_for_pair(&prepared, FIRST_CHANNELS);
    let forest = ScaledForest::grow(
        &PairObservations::new(&prepared, FIRST_CHANNELS),
        &cross_tab.c,
    );

    let found: Vec<_> = (forest.into_components().into_iter())
        .map(|(component, _)| (component.rows, component.cols))
        .collect();
    let searched: Vec<_> = (cross_tab.bipartite_connected_components().into_iter())
        .map(|component| (component.rows, component.cols))
        .collect();
    assert!(
        found.len() > 2,
        "test is vacuous: build a pair with several components"
    );
    assert_eq!(found, searched);
}

/// Every join rescales the absorbed sums, so they stay the quotient at the forest's own `u`.
#[test]
fn the_forest_carries_the_rayleigh_quotient_of_its_own_scales() {
    let (n_rows, n_cols) = (4, 3);
    let mut state = 0.37f64;
    let mut draw = || {
        state = (state * 997.0 + 0.123).fract();
        0.1 + state
    };
    let (mut row_levels, mut col_levels) = (vec![], vec![]);
    let (mut x, mut y, mut sqrt_weights) = (vec![], vec![], vec![]);
    let mut table = vec![0.0; n_rows * n_cols];
    // Two blocks grow apart before the cross cells join them, so one absorbs a grown set.
    let block = |i: usize, j: usize| (i < 2) == (j < 2);
    let cells = (0..n_rows * n_cols).map(|k| (k / n_cols, k % n_cols));
    let (within, across): (Vec<_>, Vec<_>) = cells.partition(|&(i, j)| block(i, j));
    for (i, j) in within.into_iter().chain(across) {
        // A second observation of opposite sign on the diagonal cells cancels within them.
        for repeat in 0..2 {
            let sign = if repeat == 1 && i == j { -1.0 } else { 1.0 };
            let (xi, yj, s) = (sign * draw(), draw(), draw());
            row_levels.push(i as u32);
            col_levels.push(j as u32);
            x.push(xi);
            y.push(yj);
            sqrt_weights.push(s);
            table[i * n_cols + j] += s * s * xi * yj;
        }
    }
    let c = CsrBlock::from_dense_table(&table, n_rows, n_cols);
    let observations = PairObservations {
        row_levels: &row_levels,
        col_levels: &col_levels,
        row_load: Some(&x),
        col_load: Some(&y),
        sqrt_weights: Some(&sqrt_weights),
    };
    let mut forest = ScaledForest::grow(&observations, &c);
    let u: Vec<f64> = (0..n_rows + n_cols).map(|k| forest.find(k).1).collect();

    let mut diagonal = vec![0.0; n_rows + n_cols];
    for o in observations.iter() {
        diagonal[o.row] += o.w * o.x * o.x;
        diagonal[n_rows + o.col] += o.w * o.y * o.y;
    }
    let du: f64 = diagonal.iter().zip(&u).map(|(d, u)| d * u * u).sum();
    let cu: f64 = (0..n_rows)
        .flat_map(|i| c.row(i).map(move |(j, v)| (i, j, v)))
        .map(|(i, j, v)| u[i] * u[n_rows + j] * v.abs())
        .sum();
    let expected = (du - 2.0 * cu) / du;
    let (root, _) = forest.find(0);
    let rho = forest.numerator[root] / forest.denominator[root];
    assert!(expected > 1e-3, "fixture is near-singular: {expected:e}");
    assert!(
        (rho - expected).abs() <= 1e-12 * expected,
        "{rho:e} vs {expected:e}"
    );
}
