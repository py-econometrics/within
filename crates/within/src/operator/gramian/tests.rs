// ---------------------------------------------------------------------------
// CrossTab tests (extracted from cross_tab.rs)
// ---------------------------------------------------------------------------

use super::CrossTab;
use super::Gramian;
use crate::domain::{FixedEffectsDesign, WeightedDesign};
use crate::observation::{FactorMajorStore, ObservationWeights};

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
