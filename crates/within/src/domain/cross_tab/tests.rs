use proptest::prelude::*;

use super::accumulate::{
    accumulate_dense_cross_block, accumulate_sparse_cross_block, PairColumns, Unit,
};
use super::CrossTab;
use crate::channel::{Channel, ChannelPair};
use crate::csr_block::CsrBlock;
use crate::domain::{Design, Effect, PreparedDesign};

impl CrossTab {
    pub(crate) fn from_dense_for_test(table: &[f64], n_rows: usize, n_cols: usize) -> Self {
        let c = CsrBlock::from_dense_table(table, n_rows, n_cols);
        Self::eager(c)
    }
}

/// Terms 0 and 1 paired on their intercept channels (plain cross-tab).
const INTERCEPT_PAIR: ChannelPair = ChannelPair {
    rows: Channel { term: 0, column: 0 },
    cols: Channel { term: 1, column: 0 },
};

fn design_of(columns: Vec<Vec<u32>>) -> PreparedDesign<'static> {
    PreparedDesign::from_levels_for_test(columns)
}

#[test]
fn test_extract_component_two_components() {
    // Two disconnected bipartite components: q/r levels {0,1} and {2,3}.
    let fa = vec![0u32, 0, 1, 1, 2, 2, 3, 3];
    let fb = vec![0u32, 1, 0, 1, 2, 3, 2, 3];
    let design = design_of(vec![fa, fb]);
    let (ct, _) = CrossTab::build_for_pair(&design, INTERCEPT_PAIR);

    let components = ct.bipartite_connected_components();
    assert_eq!(components.len(), 2, "should have 2 connected components");

    // Reusable remap buffers, reset by `extract_component` between components.
    let mut row_remap = vec![u32::MAX; ct.n_rows()];
    let mut col_remap = vec![u32::MAX; ct.n_cols()];

    // Sort components by their smallest q-index for deterministic comparison.
    let mut comps: Vec<_> = components.iter().collect();
    comps.sort_by_key(|c| c.rows[0]);

    let comp_a = comps[0];
    let comp_b = comps[1];

    assert_eq!(comp_a.rows, vec![0, 1], "component A row indices");
    assert_eq!(comp_a.cols, vec![0, 1], "component A col indices");
    assert_eq!(comp_b.rows, vec![2, 3], "component B row indices");
    assert_eq!(comp_b.cols, vec![2, 3], "component B col indices");

    // Extract component A and verify its sub-CrossTab.
    let sub_a = ct.extract_component(comp_a, &mut row_remap, &mut col_remap);
    assert_eq!(sub_a.n_rows(), 2, "component A: n_rows=2");
    assert_eq!(sub_a.n_cols(), 2, "component A: n_cols=2");

    // Column indices in sub_a.c should be 0-based (0..n_cols for component A = 0..2).
    let max_col_a = sub_a.c.indices.iter().copied().max().unwrap_or(0);
    assert!(
        (max_col_a as usize) < sub_a.n_cols(),
        "sub_a C column indices should be 0-based < n_cols={}",
        sub_a.n_cols()
    );

    // C^T of sub_a should equal the exact transpose of sub_a.c.
    let ct_t = sub_a.c.transpose();
    assert_eq!(
        ct_t.indptr,
        sub_a.ct().indptr,
        "sub_a: ct.indptr should equal transpose(c).indptr"
    );
    assert_eq!(
        ct_t.indices,
        sub_a.ct().indices,
        "sub_a: ct.indices should equal transpose(c).indices"
    );
    for (a, b) in ct_t.data.iter().zip(&sub_a.ct().data) {
        assert!(
            (a - b).abs() < 1e-12,
            "sub_a: ct.data should equal transpose(c).data"
        );
    }

    // Extract component B and verify its sub-CrossTab.
    let sub_b = ct.extract_component(comp_b, &mut row_remap, &mut col_remap);
    assert_eq!(sub_b.n_rows(), 2, "component B: n_rows=2");
    assert_eq!(sub_b.n_cols(), 2, "component B: n_cols=2");

    // Column indices in sub_b.c should be 0-based.
    let max_col_b = sub_b.c.indices.iter().copied().max().unwrap_or(0);
    assert!(
        (max_col_b as usize) < sub_b.n_cols(),
        "sub_b C column indices should be 0-based < n_cols={}",
        sub_b.n_cols()
    );

    // The two components should have the same structure (symmetric design).
    assert_eq!(
        sub_a.c.indptr, sub_b.c.indptr,
        "symmetric design: sub_a and sub_b should have same C structure"
    );
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10))]

    #[test]
    fn prop_bipartite_components_partition(
        n_rows in 2usize..=8,
        n_cols in 2usize..=8,
        n_obs in 4usize..=30,
        seed in 0u64..1000,
    ) {
        // Generate observations using a deterministic pseudo-random pattern.
        let mut fa: Vec<u32> = Vec::with_capacity(n_obs);
        let mut fb: Vec<u32> = Vec::with_capacity(n_obs);
        let mut s = seed;
        for _ in 0..n_obs {
            // LCG: x_{n+1} = (a * x_n + c) mod m
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            fa.push((s % n_rows as u64) as u32);
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            fb.push((s % n_cols as u64) as u32);
        }

        let design = design_of(vec![fa, fb]);
        let (ct, _) = CrossTab::build_for_pair(&design, INTERCEPT_PAIR);

        let components = ct.bipartite_connected_components();

        // Collect all row indices and col indices across components.
        let mut all_rows: Vec<usize> = components.iter().flat_map(|c| c.rows.iter().copied()).collect();
        let mut all_cols: Vec<usize> = components.iter().flat_map(|c| c.cols.iter().copied()).collect();
        all_rows.sort_unstable();
        all_cols.sort_unstable();

        // Union should cover every compact level position.
        let expected_rows: Vec<usize> = (0..ct.n_rows()).collect();
        let expected_cols: Vec<usize> = (0..ct.n_cols()).collect();
        prop_assert_eq!(&all_rows, &expected_rows, "row indices should cover 0..n_rows={}", ct.n_rows());
        prop_assert_eq!(&all_cols, &expected_cols, "col indices should cover 0..n_cols={}", ct.n_cols());

        // Indices within each component should be sorted.
        for (ci, comp) in components.iter().enumerate() {
            prop_assert!(
                comp.rows.windows(2).all(|w| w[0] < w[1]),
                "component {ci}: rows should be sorted"
            );
            prop_assert!(
                comp.cols.windows(2).all(|w| w[0] < w[1]),
                "component {ci}: cols should be sorted"
            );
        }

        // Index sets should be disjoint between components.
        let mut rows_seen = std::collections::HashSet::new();
        let mut r_seen = std::collections::HashSet::new();
        for (ci, comp) in components.iter().enumerate() {
            for &qi in &comp.rows {
                prop_assert!(rows_seen.insert(qi), "component {ci}: q-index {qi} appears in multiple components");
            }
            for &ri in &comp.cols {
                prop_assert!(r_seen.insert(ri), "component {ci}: r-index {ri} appears in multiple components");
            }
        }
    }
}

#[test]
fn design_contains_every_compact_level_position() {
    // Cover identity, bounded-gappy, and wide-sparse encodings.
    let design = design_of(vec![
        vec![0u32, 1, 2, 0, 1, 2],
        vec![0u32, 2, 4, 0, 2, 4],
        vec![7u32, u32::MAX, 7, u32::MAX, 7, u32::MAX],
    ]);

    for (term, meta) in design.design.terms.iter().enumerate() {
        let mut observed = vec![false; meta.n_levels()];
        for &level in meta.levels() {
            observed[level as usize] = true;
        }
        assert!(
            observed.into_iter().all(|is_observed| is_observed),
            "term {term} contains an unobserved internal position"
        );
    }
}

#[test]
fn dense_and_sparse_paths_agree_on_signed_data() {
    // Cell (f=0,g=0) crosses 0.0 mid-row and (f=0,g=1) cancels to 0.0; both paths must drop it.
    let f = [0u32, 0, 0, 0, 0, 1];
    let z = [1.0, -1.0, 2.0, 3.0, -3.0, 4.0];
    let g = [0u32, 0, 0, 1, 1, 0];
    let effects = vec![
        Effect::new(&f, true, [&z[..]]).unwrap(),
        Effect::new(&g, true, []).unwrap(),
    ];
    let design = Design::new(effects).unwrap();
    let pair = ChannelPair {
        rows: Channel { term: 0, column: 1 },
        cols: Channel { term: 1, column: 0 },
    };
    let cols = PairColumns {
        row_levels: design.terms[0].levels(),
        col_levels: design.terms[1].levels(),
        row_load: design.raw_slope(pair.rows).unwrap(),
        col_load: Unit,
        sqrt_weights: None,
    };
    let n_rows = design.terms[pair.rows.term].n_levels();
    let n_cols = design.terms[pair.cols.term].n_levels();
    let c_dense = accumulate_dense_cross_block(cols, n_rows, n_cols);
    let c_sparse = accumulate_sparse_cross_block(cols, n_rows, n_cols);

    // Bit-exact parity: identical per-cell addition order in both paths.
    assert_eq!(c_dense.indptr, c_sparse.indptr);
    assert_eq!(c_dense.indices, c_sparse.indices);
    assert_eq!(c_dense.data, c_sparse.data);

    // Row f=0 keeps only cell (0,0) = 2.0; the exact-0.0 cell (0,1) is gone.
    assert_eq!(&c_dense.indptr, &[0, 1, 2]);
    assert_eq!(c_dense.indices[0], 0);
    assert_eq!(c_dense.data[0], 2.0);
}
