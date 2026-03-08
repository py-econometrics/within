// ---------------------------------------------------------------------------
// CrossTab tests (extracted from cross_tab.rs)
// ---------------------------------------------------------------------------

use proptest::prelude::*;

use super::CrossTab;
use super::Gramian;
use crate::domain::{FixedEffectsDesign, WeightedDesign};
use crate::observation::{FactorMajorStore, ObservationWeights};
use crate::operator::gramian::find_all_active_levels;

fn assert_cross_tabs_equal(a: &CrossTab, b: &CrossTab) {
    assert_eq!(a.n_q(), b.n_q(), "n_q mismatch");
    assert_eq!(a.n_r(), b.n_r(), "n_r mismatch");
    for (i, (av, bv)) in a.diag_q.iter().zip(&b.diag_q).enumerate() {
        assert!(
            (av - bv).abs() < 1e-12,
            "diag_q[{i}] mismatch: {av} vs {bv}"
        );
    }
    for (i, (av, bv)) in a.diag_r.iter().zip(&b.diag_r).enumerate() {
        assert!(
            (av - bv).abs() < 1e-12,
            "diag_r[{i}] mismatch: {av} vs {bv}"
        );
    }
    assert_eq!(a.c.indptr, b.c.indptr, "C indptr mismatch");
    assert_eq!(a.c.indices, b.c.indices, "C indices mismatch");
    for (i, (av, bv)) in a.c.data.iter().zip(&b.c.data).enumerate() {
        assert!(
            (av - bv).abs() < 1e-12,
            "C data[{i}] mismatch: {av} vs {bv}"
        );
    }
}

fn make_2fe_design() -> FixedEffectsDesign {
    let store = FactorMajorStore::new(
        vec![vec![0, 1, 2, 0, 1], vec![0, 1, 2, 3, 0]],
        ObservationWeights::Unit,
        5,
    )
    .expect("valid factor-major store");
    FixedEffectsDesign::from_store(store).expect("valid 2FE design")
}

fn make_3fe_design() -> FixedEffectsDesign {
    let store = FactorMajorStore::new(
        vec![
            vec![0, 1, 2, 0, 1, 2],
            vec![0, 1, 0, 1, 0, 1],
            vec![0, 0, 1, 1, 0, 1],
        ],
        ObservationWeights::Unit,
        6,
    )
    .expect("valid factor-major store");
    FixedEffectsDesign::from_store(store).expect("valid 3FE design")
}

#[test]
fn test_from_gramian_block_matches_build_for_pair_2fe() {
    let design = make_2fe_design();
    let gramian = Gramian::build(&design);

    let (ct_obs, l2g_obs) = CrossTab::build_for_pair(&design, 0, 1).unwrap();
    let (ct_gram, l2g_gram) =
        CrossTab::from_gramian_block(&gramian.matrix, &design.factors[0], &design.factors[1])
            .unwrap();

    assert_eq!(l2g_obs, l2g_gram, "local_to_global mismatch");
    assert_cross_tabs_equal(&ct_obs, &ct_gram);
}

#[test]
fn test_from_gramian_block_matches_build_for_pair_3fe() {
    let design = make_3fe_design();
    let gramian = Gramian::build(&design);

    for q in 0..3 {
        for r in (q + 1)..3 {
            let obs_result = CrossTab::build_for_pair(&design, q, r);
            let gram_result = CrossTab::from_gramian_block(
                &gramian.matrix,
                &design.factors[q],
                &design.factors[r],
            );

            match (obs_result, gram_result) {
                (Some((ct_obs, l2g_obs)), Some((ct_gram, l2g_gram))) => {
                    assert_eq!(l2g_obs, l2g_gram, "l2g mismatch for pair ({q},{r})");
                    assert_cross_tabs_equal(&ct_obs, &ct_gram);
                }
                (None, None) => {}
                _ => panic!("one returned None and the other Some for pair ({q},{r})"),
            }
        }
    }
}

#[test]
fn test_from_gramian_block_single_component() {
    // Fully connected 2FE design: single component
    let store = FactorMajorStore::new(
        vec![vec![0, 1, 0, 1], vec![0, 0, 1, 1]],
        ObservationWeights::Unit,
        4,
    )
    .expect("valid factor-major store");
    let design = FixedEffectsDesign::from_store(store).expect("valid design");
    let gramian = Gramian::build(&design);

    let (ct_gram, _) =
        CrossTab::from_gramian_block(&gramian.matrix, &design.factors[0], &design.factors[1])
            .unwrap();

    let components = ct_gram.bipartite_connected_components();
    assert_eq!(components.len(), 1, "expected single component");
}

#[test]
fn test_from_gramian_block_multiple_components() {
    // Design with two disconnected components:
    // factor 0: [0, 0, 1, 1]   factor 1: [0, 1, 2, 3]
    // levels (0,0), (0,1) form one component; (1,2), (1,3) form another
    let store = FactorMajorStore::new(
        vec![vec![0, 0, 1, 1], vec![0, 1, 2, 3]],
        ObservationWeights::Unit,
        4,
    )
    .expect("valid factor-major store");
    let design = FixedEffectsDesign::from_store(store).expect("valid design");
    let gramian = Gramian::build(&design);

    let (ct_obs, _) = CrossTab::build_for_pair(&design, 0, 1).unwrap();
    let (ct_gram, _) =
        CrossTab::from_gramian_block(&gramian.matrix, &design.factors[0], &design.factors[1])
            .unwrap();

    let comps_obs = ct_obs.bipartite_connected_components();
    let comps_gram = ct_gram.bipartite_connected_components();
    assert_eq!(comps_obs.len(), comps_gram.len());
    for (co, cg) in comps_obs.iter().zip(&comps_gram) {
        assert_eq!(co.q_indices, cg.q_indices);
        assert_eq!(co.r_indices, cg.r_indices);
    }
}

#[test]
fn test_from_gramian_block_weighted() {
    let fl = vec![vec![0u32, 1, 0, 1], vec![0, 0, 1, 1]];
    let weights = vec![1.0, 2.0, 3.0, 4.0];

    let store = FactorMajorStore::new(fl, ObservationWeights::Dense(weights), 4)
        .expect("valid weighted store");
    let design = WeightedDesign::from_store(store).expect("valid weighted design");
    let gramian = Gramian::build(&design);

    let (ct_obs, l2g_obs) = CrossTab::build_for_pair(&design, 0, 1).unwrap();
    let (ct_gram, l2g_gram) =
        CrossTab::from_gramian_block(&gramian.matrix, &design.factors[0], &design.factors[1])
            .unwrap();

    assert_eq!(l2g_obs, l2g_gram);
    assert_cross_tabs_equal(&ct_obs, &ct_gram);
}

// ---------------------------------------------------------------------------
// Test: sparse accumulation path in CrossTab
// ---------------------------------------------------------------------------

#[test]
fn test_cross_tab_sparse_accumulation_path() {
    // n_q * n_r = 2237 * 2237 > 5_000_000 triggers accumulate_sparse_cross_block.
    // We use a small number of observations so the two paths produce the same
    // logical result but exercise different code paths.
    let n_obs = 200usize;
    let n_lev = 2237usize;

    let mut fa: Vec<u32> = Vec::with_capacity(n_obs);
    let mut fb: Vec<u32> = Vec::with_capacity(n_obs);
    for i in 0..n_obs {
        fa.push((i % n_lev) as u32);
        fb.push(((i * 7) % n_lev) as u32);
    }

    // Sparse path (large level counts)
    let store_sparse = FactorMajorStore::new(
        vec![fa.clone(), fb.clone()],
        ObservationWeights::Unit,
        n_obs,
    )
    .expect("valid sparse store");
    let design_sparse = FixedEffectsDesign::from_store(store_sparse).expect("valid sparse design");
    let (ct_sparse, _) =
        CrossTab::build_for_pair(&design_sparse, 0, 1).expect("sparse cross tab should build");

    // Dense path reference: collapse levels to a small range so n_q * n_r <= 5M.
    // Map each observation to level % 100 for both factors (100*100 = 10 000 <= 5M).
    let fa_small: Vec<u32> = fa.iter().map(|&x| x % 100).collect();
    let fb_small: Vec<u32> = fb.iter().map(|&x| x % 100).collect();
    let store_dense = FactorMajorStore::new(
        vec![fa_small.clone(), fb_small.clone()],
        ObservationWeights::Unit,
        n_obs,
    )
    .expect("valid dense store");
    let design_dense = FixedEffectsDesign::from_store(store_dense).expect("valid dense design");
    let (ct_dense, _) =
        CrossTab::build_for_pair(&design_dense, 0, 1).expect("dense cross tab should build");

    // The sparse CrossTab for the large design should have identical diagonals
    // to what we'd compute by hand (each observation appears exactly once in the
    // corresponding row/col bucket).
    assert_eq!(
        ct_sparse.diag_q.len(),
        ct_sparse.n_q(),
        "diag_q length matches n_q"
    );
    assert_eq!(
        ct_sparse.diag_r.len(),
        ct_sparse.n_r(),
        "diag_r length matches n_r"
    );

    // diag_q[i] = number of observations with fa == i (those within the first n_obs % n_lev levels)
    // All active diagonal entries must be positive.
    for &v in &ct_sparse.diag_q {
        assert!(v > 0.0, "all active q-diagonals must be positive");
    }
    for &v in &ct_sparse.diag_r {
        assert!(v > 0.0, "all active r-diagonals must be positive");
    }

    // Cross-verify: sum of sparse diagonals should equal n_obs.
    let diag_q_sum: f64 = ct_sparse.diag_q.iter().sum();
    assert!(
        (diag_q_sum - n_obs as f64).abs() < 1e-12,
        "diag_q sum should equal n_obs: {} vs {}",
        diag_q_sum,
        n_obs
    );

    // Same cross-check for the dense path.
    let diag_q_dense_sum: f64 = ct_dense.diag_q.iter().sum();
    assert!(
        (diag_q_dense_sum - n_obs as f64).abs() < 1e-12,
        "dense diag_q sum should equal n_obs: {} vs {}",
        diag_q_dense_sum,
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

// ---------------------------------------------------------------------------
// Test: to_sddm structure
// ---------------------------------------------------------------------------

#[test]
fn test_to_sddm_structure() {
    // Simple fully-connected design: factor 0 has 2 levels, factor 1 has 3 levels.
    // Observations: (0,0), (0,1), (1,1), (1,2), (0,2)
    let store = FactorMajorStore::new(
        vec![vec![0, 0, 1, 1, 0], vec![0, 1, 1, 2, 2]],
        ObservationWeights::Unit,
        5,
    )
    .expect("valid store");
    let design = FixedEffectsDesign::from_store(store).expect("valid design");
    let (ct, _) = CrossTab::build_for_pair(&design, 0, 1).expect("cross tab should build");

    let n_q = ct.n_q();
    let n_r = ct.n_r();
    let n = n_q + n_r;

    let sddm = ct.to_sddm();
    assert_eq!(sddm.n(), n, "SDDM dimension should be n_q + n_r");

    let indptr = sddm.indptr();
    let indices = sddm.indices();
    let data = sddm.data();

    // Diagonal entries: for q-rows, diagonal is at column i; for r-rows, at n_q+i.
    for i in 0..n_q {
        let row_start = indptr[i] as usize;
        let row_end = indptr[i + 1] as usize;
        // First entry in q-row is always the diagonal.
        assert_eq!(
            indices[row_start] as usize, i,
            "q-row {i}: first entry should be diagonal at column {i}"
        );
        assert!(
            (data[row_start] - ct.diag_q[i]).abs() < 1e-12,
            "q-row {i}: diagonal value should match diag_q[{i}]"
        );
        // Off-diagonal entries are in columns >= n_q (r-block columns).
        for idx in (row_start + 1)..row_end {
            assert!(
                indices[idx] as usize >= n_q,
                "q-row {i}: off-diag column {} should be >= n_q={n_q}",
                indices[idx]
            );
            assert!(
                data[idx] <= 0.0,
                "q-row {i}: off-diag entry at C-column should be negated (non-positive)"
            );
        }
    }

    // r-rows: off-diagonal entries come before diagonal.
    for i in 0..n_r {
        let row = n_q + i;
        let row_start = indptr[row] as usize;
        let row_end = indptr[row + 1] as usize;
        // Last entry in r-row is always the diagonal.
        let last_idx = row_end - 1;
        assert_eq!(
            indices[last_idx] as usize,
            n_q + i,
            "r-row {i}: last entry should be diagonal at column {}",
            n_q + i
        );
        assert!(
            (data[last_idx] - ct.diag_r[i]).abs() < 1e-12,
            "r-row {i}: diagonal value should match diag_r[{i}]"
        );
        // Off-diagonal entries are in columns < n_q (q-block columns).
        for idx in row_start..last_idx {
            assert!(
                (indices[idx] as usize) < n_q,
                "r-row {i}: off-diag column {} should be < n_q={n_q}",
                indices[idx]
            );
            assert!(
                data[idx] <= 0.0,
                "r-row {i}: off-diag entry should be negated (non-positive)"
            );
        }
    }

    // Symmetry: L[i,j] == L[j,i]
    // Build a dense matrix from the SDDM sparse format.
    let mut dense = vec![vec![0.0f64; n]; n];
    for row in 0..n {
        let row_start = indptr[row] as usize;
        let row_end = indptr[row + 1] as usize;
        for idx in row_start..row_end {
            dense[row][indices[idx] as usize] = data[idx];
        }
    }
    #[allow(clippy::needless_range_loop)]
    for i in 0..n {
        for j in 0..n {
            assert!(
                (dense[i][j] - dense[j][i]).abs() < 1e-12,
                "SDDM symmetry violation at ({i},{j}): {} vs {}",
                dense[i][j],
                dense[j][i]
            );
        }
    }
}

#[test]
fn test_to_sddm_diagonal_only_when_no_cross_entries() {
    // A design where factor 0 and factor 1 are perfectly nested:
    // each q-level appears with exactly one r-level (but each observation
    // is still stored). The C block will be non-empty, but we separately
    // test that the SDDM diagonal = diag_q / diag_r.
    //
    // To get an empty C block we'd need factor 0 levels that never share
    // observations with any factor 1 level — impossible with an ObservationStore.
    // Instead we test that a single-observation design has C with exactly one entry.
    let store = FactorMajorStore::new(vec![vec![0], vec![0]], ObservationWeights::Unit, 1)
        .expect("valid single-obs store");
    let design = FixedEffectsDesign::from_store(store).expect("valid single-obs design");
    let (ct, _) = CrossTab::build_for_pair(&design, 0, 1).expect("should build");

    let sddm = ct.to_sddm();
    let n = sddm.n();
    assert_eq!(n, 2, "single-observation design: n_q=1, n_r=1, total=2");

    let indptr = sddm.indptr();
    let indices = sddm.indices();
    let data = sddm.data();

    // Row 0 (q): diagonal=1, off-diag col=1 val=-1
    assert_eq!(indptr[0], 0);
    assert_eq!(indptr[1], 2);
    assert_eq!(indices[0], 0);
    assert!((data[0] - 1.0).abs() < 1e-12, "q-diag should be 1.0");
    assert_eq!(indices[1], 1);
    assert!((data[1] + 1.0).abs() < 1e-12, "C entry should be -1.0");

    // Row 1 (r): off-diag col=0 val=-1, diagonal=1
    assert_eq!(indptr[1], 2);
    assert_eq!(indptr[2], 4);
    assert_eq!(indices[2], 0);
    assert!((data[2] + 1.0).abs() < 1e-12, "C^T entry should be -1.0");
    assert_eq!(indices[3], 1);
    assert!((data[3] - 1.0).abs() < 1e-12, "r-diag should be 1.0");
}

// ---------------------------------------------------------------------------
// Test: extract_component correctness
// ---------------------------------------------------------------------------

#[test]
fn test_extract_component_two_components() {
    // Two disconnected bipartite components:
    //   Component A: q-levels {0,1} <-> r-levels {0,1}
    //   Component B: q-levels {2,3} <-> r-levels {2,3}
    // Observations:
    //   (q=0, r=0), (q=0, r=1), (q=1, r=0), (q=1, r=1),  <- component A
    //   (q=2, r=2), (q=2, r=3), (q=3, r=2), (q=3, r=3)   <- component B
    let fa = vec![0u32, 0, 1, 1, 2, 2, 3, 3];
    let fb = vec![0u32, 1, 0, 1, 2, 3, 2, 3];
    let n_obs = 8;
    let store =
        FactorMajorStore::new(vec![fa, fb], ObservationWeights::Unit, n_obs).expect("valid store");
    let design = FixedEffectsDesign::from_store(store).expect("valid design");
    let (ct, _) = CrossTab::build_for_pair(&design, 0, 1).expect("cross tab should build");

    let components = ct.bipartite_connected_components();
    assert_eq!(components.len(), 2, "should have 2 connected components");

    // Sort components by their smallest q-index for deterministic comparison.
    let mut comps: Vec<_> = components.iter().collect();
    comps.sort_by_key(|c| c.q_indices[0]);

    let comp_a = comps[0];
    let comp_b = comps[1];

    assert_eq!(comp_a.q_indices, vec![0, 1], "component A q-indices");
    assert_eq!(comp_a.r_indices, vec![0, 1], "component A r-indices");
    assert_eq!(comp_b.q_indices, vec![2, 3], "component B q-indices");
    assert_eq!(comp_b.r_indices, vec![2, 3], "component B r-indices");

    // Extract component A and verify its sub-CrossTab.
    let sub_a = ct.extract_component(comp_a);
    assert_eq!(sub_a.n_q(), 2, "component A: n_q=2");
    assert_eq!(sub_a.n_r(), 2, "component A: n_r=2");

    // diag_q for component A should be the parent's diag_q at indices 0,1.
    for (new_i, &old_i) in comp_a.q_indices.iter().enumerate() {
        assert!(
            (sub_a.diag_q[new_i] - ct.diag_q[old_i]).abs() < 1e-12,
            "sub_a.diag_q[{new_i}] should match ct.diag_q[{old_i}]"
        );
    }
    // diag_r for component A should be the parent's diag_r at indices 0,1.
    for (new_i, &old_i) in comp_a.r_indices.iter().enumerate() {
        assert!(
            (sub_a.diag_r[new_i] - ct.diag_r[old_i]).abs() < 1e-12,
            "sub_a.diag_r[{new_i}] should match ct.diag_r[{old_i}]"
        );
    }

    // Column indices in sub_a.c should be 0-based (0..n_r for component A = 0..2).
    let max_col_a = sub_a.c.indices.iter().copied().max().unwrap_or(0);
    assert!(
        (max_col_a as usize) < sub_a.n_r(),
        "sub_a C column indices should be 0-based < n_r={}",
        sub_a.n_r()
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
    let sub_b = ct.extract_component(comp_b);
    assert_eq!(sub_b.n_q(), 2, "component B: n_q=2");
    assert_eq!(sub_b.n_r(), 2, "component B: n_r=2");

    // Column indices in sub_b.c should be 0-based.
    let max_col_b = sub_b.c.indices.iter().copied().max().unwrap_or(0);
    assert!(
        (max_col_b as usize) < sub_b.n_r(),
        "sub_b C column indices should be 0-based < n_r={}",
        sub_b.n_r()
    );

    // The two components should have the same structure (symmetric design).
    assert_eq!(
        sub_a.c.indptr, sub_b.c.indptr,
        "symmetric design: sub_a and sub_b should have same C structure"
    );
}

// ---------------------------------------------------------------------------
// Test: bipartite_connected_components partition property (proptest)
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10))]

    #[test]
    fn prop_bipartite_components_partition(
        n_q in 2usize..=8,
        n_r in 2usize..=8,
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
            fa.push((s % n_q as u64) as u32);
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            fb.push((s % n_r as u64) as u32);
        }

        let store = FactorMajorStore::new(
            vec![fa, fb],
            ObservationWeights::Unit,
            n_obs,
        ).expect("valid store");
        let design = FixedEffectsDesign::from_store(store).expect("valid design");
        let (ct, _) = CrossTab::build_for_pair(&design, 0, 1)
            .expect("cross tab should build");

        let components = ct.bipartite_connected_components();

        // Collect all q-indices and r-indices across components.
        let mut all_q: Vec<usize> = components.iter().flat_map(|c| c.q_indices.iter().copied()).collect();
        let mut all_r: Vec<usize> = components.iter().flat_map(|c| c.r_indices.iter().copied()).collect();
        all_q.sort_unstable();
        all_r.sort_unstable();

        // Union should cover 0..n_q (compact active levels).
        let expected_q: Vec<usize> = (0..ct.n_q()).collect();
        let expected_r: Vec<usize> = (0..ct.n_r()).collect();
        prop_assert_eq!(&all_q, &expected_q, "q-indices should cover 0..n_q={}", ct.n_q());
        prop_assert_eq!(&all_r, &expected_r, "r-indices should cover 0..n_r={}", ct.n_r());

        // Indices within each component should be sorted.
        for (ci, comp) in components.iter().enumerate() {
            prop_assert!(
                comp.q_indices.windows(2).all(|w| w[0] < w[1]),
                "component {ci}: q_indices should be sorted"
            );
            prop_assert!(
                comp.r_indices.windows(2).all(|w| w[0] < w[1]),
                "component {ci}: r_indices should be sorted"
            );
        }

        // Index sets should be disjoint between components.
        let mut q_seen = std::collections::HashSet::new();
        let mut r_seen = std::collections::HashSet::new();
        for (ci, comp) in components.iter().enumerate() {
            for &qi in &comp.q_indices {
                prop_assert!(q_seen.insert(qi), "component {ci}: q-index {qi} appears in multiple components");
            }
            for &ri in &comp.r_indices {
                prop_assert!(r_seen.insert(ri), "component {ci}: r-index {ri} appears in multiple components");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Test: find_all_active_levels
// ---------------------------------------------------------------------------

#[test]
fn test_find_all_active_levels_with_gaps() {
    // Factor 0 has 5 levels (max value = 4), but only levels 0, 2, 4 appear.
    // Factor 1 uses all 3 levels 0, 1, 2.
    let fa = vec![0u32, 2, 4, 0, 2, 4];
    let fb = vec![0u32, 1, 2, 0, 1, 2];
    let n_obs = 6;
    let store =
        FactorMajorStore::new(vec![fa, fb], ObservationWeights::Unit, n_obs).expect("valid store");
    let design = FixedEffectsDesign::from_store(store).expect("valid design");

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
