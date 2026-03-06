use within::{
    solve_least_squares, LocalSolverConfig, OperatorRepr, SolverMethod, SolverParams,
    WeightedDesign,
};

#[test]
fn test_solve_row_major_matches_factor_major() {
    use within::observation::{FactorMajorStore, ObservationWeights, RowMajorStore};

    let fl = vec![vec![0u32, 1, 0, 1, 2], vec![0, 0, 1, 1, 0]];
    let n_levels = vec![3, 2];
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let params = SolverParams::default();

    let fm = FactorMajorStore::new(fl.clone(), ObservationWeights::Unit, 5)
        .expect("valid factor-major store");
    let dm_fm = WeightedDesign::from_store(fm, &n_levels).expect("valid factor-major design");
    let r_fm = solve_least_squares(&dm_fm, &y, &params).expect("fm solve");

    let rm = RowMajorStore::from_factor_major(fl, ObservationWeights::Unit, 5);
    let dm_rm = WeightedDesign::from_store(rm, &n_levels).expect("valid row-major design");
    let r_rm = solve_least_squares(&dm_rm, &y, &params).expect("rm solve");

    assert!(r_fm.converged && r_rm.converged, "Both must converge");
    for (a, b) in r_fm.x.iter().zip(r_rm.x.iter()) {
        assert!((a - b).abs() < 1e-6, "Solution mismatch: {a} vs {b}");
    }
}

#[test]
fn test_solve_compressed_matches_factor_major() {
    use within::observation::{CompressedStore, FactorMajorStore, ObservationWeights};

    let fl = vec![vec![0u32, 1, 0, 1, 2], vec![0, 0, 1, 1, 0]];
    let n_levels = vec![3, 2];
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let params = SolverParams::default();

    let fm = FactorMajorStore::new(fl.clone(), ObservationWeights::Unit, 5)
        .expect("valid factor-major store");
    let dm_fm = WeightedDesign::from_store(fm, &n_levels).expect("valid factor-major design");
    let r_fm = solve_least_squares(&dm_fm, &y, &params).expect("fm solve");

    let cs = CompressedStore::from_factor_major(fl, ObservationWeights::Unit, 5);
    let dm_cs = WeightedDesign::from_store(cs, &n_levels).expect("valid compressed design");
    let r_cs = solve_least_squares(&dm_cs, &y, &params).expect("compressed solve");

    assert!(r_fm.converged && r_cs.converged, "Both must converge");
    for (a, b) in r_fm.x.iter().zip(r_cs.x.iter()) {
        assert!((a - b).abs() < 1e-6, "Solution mismatch: {a} vs {b}");
    }
}

#[test]
fn test_solve_cg_preconditioned_all_backends() {
    use within::observation::{
        CompressedStore, FactorMajorStore, ObservationWeights, RowMajorStore,
    };

    let fl = vec![vec![0u32, 1, 0, 1, 2], vec![0, 0, 1, 1, 0]];
    let n_levels = vec![3, 2];
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let params = SolverParams {
        method: SolverMethod::Cg {
            preconditioner: Some(LocalSolverConfig::cg_default()),
            operator: OperatorRepr::Implicit,
        },
        tol: 1e-8,
        maxiter: 1000,
    };

    let fm = FactorMajorStore::new(fl.clone(), ObservationWeights::Unit, 5)
        .expect("valid factor-major store");
    let dm_fm = WeightedDesign::from_store(fm, &n_levels).expect("valid factor-major design");
    let r_fm = solve_least_squares(&dm_fm, &y, &params).expect("fm solve");

    let rm = RowMajorStore::from_factor_major(fl.clone(), ObservationWeights::Unit, 5);
    let dm_rm = WeightedDesign::from_store(rm, &n_levels).expect("valid row-major design");
    let r_rm = solve_least_squares(&dm_rm, &y, &params).expect("rm solve");

    let cs = CompressedStore::from_factor_major(fl, ObservationWeights::Unit, 5);
    let dm_cs = WeightedDesign::from_store(cs, &n_levels).expect("valid compressed design");
    let r_cs = solve_least_squares(&dm_cs, &y, &params).expect("compressed solve");

    assert!(r_fm.converged, "FM must converge");
    assert!(r_rm.converged, "RM must converge");
    assert!(r_cs.converged, "CS must converge");

    for (a, b) in r_fm.x.iter().zip(r_rm.x.iter()) {
        assert!(
            (a - b).abs() < 1e-6,
            "FM vs RM solution mismatch: {a} vs {b}"
        );
    }
    for (a, b) in r_fm.x.iter().zip(r_cs.x.iter()) {
        assert!(
            (a - b).abs() < 1e-6,
            "FM vs CS solution mismatch: {a} vs {b}"
        );
    }
}
