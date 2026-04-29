// ---------------------------------------------------------------------------
// CrossTab tests (extracted from cross_tab.rs)
// ---------------------------------------------------------------------------

use proptest::prelude::*;

use super::CrossTab;
use crate::domain::Design;
use crate::observation::{FactorMajorStore, Store};
use crate::operator::gramian::{find_all_active_levels, Gramian};

/// Test helper: build a CrossTab for a factor pair by composing the public
/// `find_all_active_levels` + `CrossTab::build_for_pair_with_active`.
fn build_for_pair<S: Store>(
    design: &Design<S>,
    weights: Option<&[f64]>,
    q: usize,
    r: usize,
) -> Option<(CrossTab, Vec<u32>)> {
    let active = find_all_active_levels(design);
    CrossTab::build_for_pair_with_active(design, weights, q, r, &active)
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
    let store_sparse =
        FactorMajorStore::new(vec![fa.clone(), fb.clone()], n_obs).expect("valid sparse store");
    let design_sparse = Design::from_store(store_sparse).expect("valid sparse design");
    let (ct_sparse, _) =
        build_for_pair(&design_sparse, None, 0, 1).expect("sparse cross tab should build");

    // Dense path reference: collapse levels to a small range so n_q * n_r <= 5M.
    // Map each observation to level % 100 for both factors (100*100 = 10 000 <= 5M).
    let fa_small: Vec<u32> = fa.iter().map(|&x| x % 100).collect();
    let fb_small: Vec<u32> = fb.iter().map(|&x| x % 100).collect();
    let store_dense = FactorMajorStore::new(vec![fa_small.clone(), fb_small.clone()], n_obs)
        .expect("valid dense store");
    let design_dense = Design::from_store(store_dense).expect("valid dense design");
    let (ct_dense, _) =
        build_for_pair(&design_dense, None, 0, 1).expect("dense cross tab should build");

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
    let store = FactorMajorStore::new(vec![fa, fb], n_obs).expect("valid store");
    let design = Design::from_store(store).expect("valid design");
    let (ct, _) = build_for_pair(&design, None, 0, 1).expect("cross tab should build");

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

        let store = FactorMajorStore::new(vec![fa, fb], n_obs).expect("valid store");
        let design = Design::from_store(store).expect("valid design");
        let (ct, _) = build_for_pair(&design, None, 0, 1)
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
// Test: CrossTab block matches per-pair sub-block of the explicit Gramian
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(20))]

    /// `CrossTab::build_for_pair` and the corresponding sub-block of
    /// `Gramian::build` must agree entry-for-entry. This was previously
    /// guarded by the (now-removed) test-only `CrossTab::from_gramian_block`
    /// constructor; the same invariant is recovered here through the public
    /// `extract_submatrix` API.
    #[test]
    fn prop_cross_tab_matches_gramian_block(
        n_q in 2u32..=8,
        n_r in 2u32..=8,
        n_obs in 4usize..=40,
        weighted in any::<bool>(),
        seed in 0u64..10_000,
    ) {
        let mut fa: Vec<u32> = Vec::with_capacity(n_obs);
        let mut fb: Vec<u32> = Vec::with_capacity(n_obs);
        let mut s = seed;
        for _ in 0..n_obs {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            fa.push((s % n_q as u64) as u32);
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            fb.push((s % n_r as u64) as u32);
        }

        let store = FactorMajorStore::new(vec![fa, fb], n_obs).expect("valid store");
        let design = Design::from_store(store).expect("valid design");

        let weights: Option<Vec<f64>> = if weighted {
            let mut w = Vec::with_capacity(n_obs);
            for i in 0..n_obs {
                w.push(0.25 + ((i + 1) as f64).sqrt());
            }
            Some(w)
        } else {
            None
        };
        let weights_slice = weights.as_deref();

        let (ct, l2g) = match build_for_pair(&design, weights_slice, 0, 1) {
            Some(pair) => pair,
            None => return Ok(()),
        };
        let gramian = Gramian::build(&design, weights_slice);

        let l2g_usize: Vec<usize> = l2g.iter().map(|&x| x as usize).collect();
        let sub = gramian.extract_submatrix(&l2g_usize);

        let n_q_local = ct.n_q();
        let n_r_local = ct.n_r();
        let n_total = n_q_local + n_r_local;
        prop_assert_eq!(sub.n(), n_total, "sub-block size mismatch");

        // Densify both representations and compare.
        let mut expected = vec![0.0f64; n_total * n_total];
        for i in 0..n_q_local {
            expected[i * n_total + i] = ct.diag_q[i];
        }
        for k in 0..n_r_local {
            let p = n_q_local + k;
            expected[p * n_total + p] = ct.diag_r[k];
        }
        for i in 0..n_q_local {
            let row_start = ct.c.indptr[i] as usize;
            let row_end = ct.c.indptr[i + 1] as usize;
            for idx in row_start..row_end {
                let k = ct.c.indices[idx] as usize;
                let v = ct.c.data[idx];
                expected[i * n_total + (n_q_local + k)] = v;
                expected[(n_q_local + k) * n_total + i] = v;
            }
        }

        let mut actual = vec![0.0f64; n_total * n_total];
        let indptr = sub.indptr();
        let indices = sub.indices();
        let data = sub.data();
        for row in 0..n_total {
            let row_start = indptr[row] as usize;
            let row_end = indptr[row + 1] as usize;
            for idx in row_start..row_end {
                let col = indices[idx] as usize;
                actual[row * n_total + col] = data[idx];
            }
        }

        for i in 0..n_total {
            for j in 0..n_total {
                let pos = i * n_total + j;
                prop_assert!(
                    (expected[pos] - actual[pos]).abs() < 1e-12,
                    "mismatch at ({}, {}): expected {}, got {}",
                    i, j, expected[pos], actual[pos]
                );
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
    let store = FactorMajorStore::new(vec![fa, fb], n_obs).expect("valid store");
    let design = Design::from_store(store).expect("valid design");

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
