use super::*;
use crate::observation::{FactorMajorStore, ObservationWeights};

fn make_test_design() -> FixedEffectsDesign {
    let categories = vec![vec![0, 1, 2, 0, 1], vec![0, 1, 2, 3, 0]];
    let n_levels = vec![3, 4];
    let store = FactorMajorStore::new(categories, ObservationWeights::Unit, 5)
        .expect("valid factor-major store");
    FixedEffectsDesign::from_store(store, &n_levels).expect("valid test design")
}

fn make_weighted_design(weights: Vec<f64>) -> FixedEffectsDesign {
    let categories = vec![vec![0, 1, 0, 1], vec![0, 0, 1, 1]];
    let n_levels = vec![2, 2];
    let store = FactorMajorStore::new(categories, ObservationWeights::Dense(weights), 4)
        .expect("valid weighted factor-major store");
    FixedEffectsDesign::from_store(store, &n_levels).expect("valid weighted design")
}

#[test]
fn test_construction() {
    let dm = make_test_design();
    assert_eq!(dm.n_factors(), 2);
    assert_eq!(dm.n_dofs, 7);
    assert_eq!(dm.n_rows, 5);
    assert_eq!(dm.factors[0].offset, 0);
    assert_eq!(dm.factors[1].offset, 3);
    assert_eq!(dm.block_offsets(), vec![0, 3, 7]);
}

#[test]
fn test_factor_meta() {
    let dm = make_test_design();
    assert_eq!(dm.factors[0].n_levels, 3);
    assert_eq!(dm.factors[1].n_levels, 4);
    // Categories are in the store, not in FactorMeta
    assert_eq!(dm.level(0, 0), 0);
    assert_eq!(dm.level(1, 0), 1);
    assert_eq!(dm.level(2, 0), 2);
    assert_eq!(dm.level(3, 0), 0);
    assert_eq!(dm.level(4, 0), 1);
    assert_eq!(dm.level(0, 1), 0);
    assert_eq!(dm.level(1, 1), 1);
    assert_eq!(dm.level(4, 1), 0);
}

#[test]
fn test_matvec_d() {
    let dm = make_test_design();
    let x = vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 40.0];
    let mut y = vec![0.0; 5];
    dm.matvec_d(&x, &mut y);
    assert_eq!(y, vec![11.0, 22.0, 33.0, 41.0, 12.0]);
}

#[test]
fn test_rmatvec_dt() {
    let dm = make_test_design();
    let r = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let mut x = vec![0.0; 7];
    dm.rmatvec_dt(&r, &mut x);
    assert_eq!(x, vec![5.0, 7.0, 3.0, 6.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_rmatvec_wdt_unweighted() {
    let dm = make_test_design();
    let r = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let mut x_dt = vec![0.0; 7];
    let mut x_wdt = vec![0.0; 7];
    dm.rmatvec_dt(&r, &mut x_dt);
    dm.rmatvec_wdt(&r, &mut x_wdt);
    assert_eq!(x_dt, x_wdt);
}

#[test]
fn test_rmatvec_wdt_weighted() {
    let dm = make_weighted_design(vec![1.0, 2.0, 3.0, 4.0]);
    let r = vec![1.0, 1.0, 1.0, 1.0];
    let mut x = vec![0.0; 4];
    dm.rmatvec_wdt(&r, &mut x);
    // factor 0: level 0 has obs 0(w=1)+2(w=3)=4, level 1 has obs 1(w=2)+3(w=4)=6
    // factor 1: level 0 has obs 0(w=1)+1(w=2)=3, level 1 has obs 2(w=3)+3(w=4)=7
    assert_eq!(x, vec![4.0, 6.0, 3.0, 7.0]);
}

#[test]
fn test_gramian_diagonal_unweighted() {
    let dm = make_test_design();
    let diag = dm.gramian_diagonal();
    // factor 0: levels [0,1,2,0,1] -> counts [2,2,1]
    // factor 1: levels [0,1,2,3,0] -> counts [2,1,1,1]
    assert_eq!(diag, vec![2.0, 2.0, 1.0, 2.0, 1.0, 1.0, 1.0]);
}

#[test]
fn test_gramian_diagonal_weighted() {
    let dm = make_weighted_design(vec![1.0, 2.0, 3.0, 4.0]);
    let diag = dm.gramian_diagonal();
    // factor 0: level 0 -> w=1+3=4, level 1 -> w=2+4=6
    // factor 1: level 0 -> w=1+2=3, level 1 -> w=3+4=7
    assert_eq!(diag, vec![4.0, 6.0, 3.0, 7.0]);
}
