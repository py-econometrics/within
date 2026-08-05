use proptest::prelude::*;

use super::accumulate::{
    accumulate_dense_cross_block, accumulate_sparse_cross_block, PairColumns, Unit,
};
use super::{build_compact_mapping, CrossTab};
use crate::channel::{Channel, ChannelPair};
use crate::csr_block::CsrBlock;
use crate::domain::find_all_active_levels;
use crate::domain::{Design, Effect};
use crate::observation::ObservationFrame;

impl CrossTab {
    pub(crate) fn from_dense_for_test(table: &[f64], n_rows: usize, n_cols: usize) -> Self {
        let c = CsrBlock::from_dense_table(table, n_rows, n_cols);
        let ct = c.transpose();
        Self { c, ct }
    }
}

/// Terms 0 and 1 paired on their intercept channels (plain cross-tab).
const INTERCEPT_PAIR: ChannelPair = ChannelPair {
    rows: Channel { term: 0, column: 0 },
    cols: Channel { term: 1, column: 0 },
};

fn design_of(columns: Vec<Vec<u32>>) -> Design<'static> {
    let frame = ObservationFrame::new(columns.into_iter().map(Into::into).collect(), Vec::new())
        .expect("valid frame");
    Design::from_frame(frame).expect("valid design")
}

#[test]
fn test_cross_tab_sparse_accumulation_path() {
    // n_rows * n_cols > 5M triggers the sparse path; few observations keep both paths equal.
    let n_obs = 200usize;
    let n_lev = 2237usize;

    let mut fa: Vec<u32> = Vec::with_capacity(n_obs);
    let mut fb: Vec<u32> = Vec::with_capacity(n_obs);
    for i in 0..n_obs {
        fa.push((i % n_lev) as u32);
        fb.push(((i * 7) % n_lev) as u32);
    }

    // Sparse path (large level counts)
    let design_sparse = design_of(vec![fa.clone(), fb.clone()]);
    let active_sparse = find_all_active_levels(&design_sparse);
    let (ct_sparse, diag_sparse, _) =
        CrossTab::build_for_pair_with_active(&design_sparse, None, INTERCEPT_PAIR, &active_sparse)
            .expect("sparse cross tab should build");

    // Dense reference: collapse levels so n_rows * n_cols <= 5M.
    let fa_small: Vec<u32> = fa.iter().map(|&x| x % 100).collect();
    let fb_small: Vec<u32> = fb.iter().map(|&x| x % 100).collect();
    let design_dense = design_of(vec![fa_small.clone(), fb_small.clone()]);
    let active_dense = find_all_active_levels(&design_dense);
    let (_ct_dense, diag_dense, _) =
        CrossTab::build_for_pair_with_active(&design_dense, None, INTERCEPT_PAIR, &active_dense)
            .expect("dense cross tab should build");

    // Each observation appears exactly once in its row/col bucket.
    assert_eq!(
        diag_sparse.rows.len(),
        ct_sparse.n_rows(),
        "row_diag length matches n_rows"
    );
    assert_eq!(
        diag_sparse.cols.len(),
        ct_sparse.n_cols(),
        "col_diag length matches n_cols"
    );

    // row_diag[i] counts observations with fa == i; every active entry must be positive.
    for &v in &diag_sparse.rows {
        assert!(v > 0.0, "all active q-diagonals must be positive");
    }
    for &v in &diag_sparse.cols {
        assert!(v > 0.0, "all active r-diagonals must be positive");
    }

    // Cross-verify: sum of sparse diagonals should equal n_obs.
    let row_diag_sum: f64 = diag_sparse.rows.iter().sum();
    assert!(
        (row_diag_sum - n_obs as f64).abs() < 1e-12,
        "row_diag sum should equal n_obs: {} vs {}",
        row_diag_sum,
        n_obs
    );

    // Same cross-check for the dense path.
    let row_diag_dense_sum: f64 = diag_dense.rows.iter().sum();
    assert!(
        (row_diag_dense_sum - n_obs as f64).abs() < 1e-12,
        "dense row_diag sum should equal n_obs: {} vs {}",
        row_diag_dense_sum,
        n_obs
    );

    // C^T must equal the transpose of C for both paths.
    let ct_t = ct_sparse.c.transpose();
    assert_eq!(
        ct_t.indptr, ct_sparse.ct.indptr,
        "sparse: C^T indptr should equal transpose(C)"
    );
    assert_eq!(
        ct_t.indices, ct_sparse.ct.indices,
        "sparse: C^T indices should equal transpose(C)"
    );
    for (a, b) in ct_t.data.iter().zip(&ct_sparse.ct.data) {
        assert!(
            (a - b).abs() < 1e-12,
            "sparse: C^T data should equal transpose(C)"
        );
    }
}

#[test]
fn test_extract_component_two_components() {
    // Two disconnected bipartite components: q/r levels {0,1} and {2,3}.
    let fa = vec![0u32, 0, 1, 1, 2, 2, 3, 3];
    let fb = vec![0u32, 1, 0, 1, 2, 3, 2, 3];
    let design = design_of(vec![fa, fb]);
    let all_active = find_all_active_levels(&design);
    let (ct, parent_diag, _) =
        CrossTab::build_for_pair_with_active(&design, None, INTERCEPT_PAIR, &all_active)
            .expect("cross tab should build");

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

    // Component A's diagonal matches the parent's at 0,1, flat as `[rows | cols]`.
    let sub_a_diag = parent_diag.extract_component(comp_a);
    for (new_i, &old_i) in comp_a.rows.iter().enumerate() {
        assert!(
            (sub_a_diag[new_i] - parent_diag.rows[old_i]).abs() < 1e-12,
            "sub_a diag row[{new_i}] should match parent diag row[{old_i}]"
        );
    }
    for (new_i, &old_i) in comp_a.cols.iter().enumerate() {
        assert!(
            (sub_a_diag[comp_a.rows.len() + new_i] - parent_diag.cols[old_i]).abs() < 1e-12,
            "sub_a diag col[{new_i}] should match parent diag col[{old_i}]"
        );
    }

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
        ct_t.indptr, sub_a.ct.indptr,
        "sub_a: ct.indptr should equal transpose(c).indptr"
    );
    assert_eq!(
        ct_t.indices, sub_a.ct.indices,
        "sub_a: ct.indices should equal transpose(c).indices"
    );
    for (a, b) in ct_t.data.iter().zip(&sub_a.ct.data) {
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
        let all_active = find_all_active_levels(&design);
        let (ct, _, _) = CrossTab::build_for_pair_with_active(&design, None, INTERCEPT_PAIR, &all_active)
            .expect("cross tab should build");

        let components = ct.bipartite_connected_components();

        // Collect all row indices and col indices across components.
        let mut all_rows: Vec<usize> = components.iter().flat_map(|c| c.rows.iter().copied()).collect();
        let mut all_cols: Vec<usize> = components.iter().flat_map(|c| c.cols.iter().copied()).collect();
        all_rows.sort_unstable();
        all_cols.sort_unstable();

        // Union should cover 0..n_rows (compact active levels).
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
fn test_find_all_active_levels_with_gaps() {
    // Factor 0 has 5 levels but only 0, 2, 4 appear; factor 1 uses all 3.
    let fa = vec![0u32, 2, 4, 0, 2, 4];
    let fb = vec![0u32, 1, 2, 0, 1, 2];
    let design = design_of(vec![fa, fb]);

    let active = find_all_active_levels(&design);

    // Factor 0: 5 levels, only 0, 2, 4 active.
    assert_eq!(active[0].len(), 5, "factor 0 should have 5 levels");
    assert_eq!(
        active[0],
        vec![true, false, true, false, true],
        "factor 0 active pattern: [true, false, true, false, true]"
    );

    // Factor 1: all 3 levels active.
    assert_eq!(active[1].len(), 3, "factor 1 should have 3 levels");
    assert_eq!(
        active[1],
        vec![true, true, true],
        "factor 1 active pattern: all true"
    );
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
    let all_active = find_all_active_levels(&design);
    let active = build_compact_mapping(
        &all_active[0],
        &all_active[1],
        design.terms[0].column_base(pair.rows.column),
        design.terms[1].column_base(pair.cols.column),
    )
    .expect("both factors have active levels");

    let cols = PairColumns {
        row_levels: design.frame.level_column(0),
        col_levels: design.frame.level_column(1),
        row_load: design.frame.loading_column(0),
        col_load: Unit,
        weights: None,
    };
    let (c_dense, dq_dense, dr_dense) = accumulate_dense_cross_block(cols, &active);
    let (c_sparse, dq_sparse, dr_sparse) = accumulate_sparse_cross_block(cols, &active);

    // Bit-exact parity: identical per-cell addition order in both paths.
    assert_eq!(c_dense.indptr, c_sparse.indptr);
    assert_eq!(c_dense.indices, c_sparse.indices);
    assert_eq!(c_dense.data, c_sparse.data);
    assert_eq!(dq_dense, dq_sparse);
    assert_eq!(dr_dense, dr_sparse);

    // Row f=0 keeps only cell (0,0) = 2.0; the exact-0.0 cell (0,1) is gone.
    assert_eq!(&c_dense.indptr, &[0, 1, 2]);
    assert_eq!(c_dense.indices[0], 0);
    assert_eq!(c_dense.data[0], 2.0);
    // Diagonals accumulate w·l²: z² on the slope side, plain counts on the intercept side.
    assert_eq!(dq_dense, vec![1.0 + 1.0 + 4.0 + 9.0 + 9.0, 16.0]);
    assert_eq!(dr_dense, vec![4.0, 2.0]);
}
