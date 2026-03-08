use ndarray::Array2;
use within::observation::{FactorMajorStore, ObservationWeights};
use within::{solve, LocalSolverConfig, Preconditioner, SolverParams, WeightedDesign, WithinError};

#[test]
fn test_empty_observations_error() {
    // FactorMajorStore::new allows 0 rows; EmptyObservations is raised by WeightedDesign::from_store
    let store =
        FactorMajorStore::new(vec![vec![], vec![]], ObservationWeights::Unit, 0).expect("store ok");
    let result = WeightedDesign::from_store(store);
    assert!(result.is_err());
    match result.unwrap_err() {
        WithinError::EmptyObservations => {}
        other => panic!("Expected EmptyObservations, got: {:?}", other),
    }
}

#[test]
fn test_observation_count_mismatch_error() {
    // Factor columns have different lengths
    let result =
        FactorMajorStore::new(vec![vec![0, 1, 2], vec![0, 1]], ObservationWeights::Unit, 3);
    assert!(result.is_err());
    match result.unwrap_err() {
        WithinError::ObservationCountMismatch { .. } => {}
        other => panic!("Expected ObservationCountMismatch, got: {:?}", other),
    }
}

#[test]
fn test_weight_count_mismatch_error() {
    let result = FactorMajorStore::new(
        vec![vec![0, 1, 2], vec![0, 1, 0]],
        ObservationWeights::Dense(vec![1.0, 2.0]), // wrong length
        3,
    );
    assert!(result.is_err());
    match result.unwrap_err() {
        WithinError::WeightCountMismatch { .. } => {}
        other => panic!("Expected WeightCountMismatch, got: {:?}", other),
    }
}

#[test]
fn test_empty_categories_via_solve() {
    let cats = Array2::<u32>::zeros((0, 2));
    let y: Vec<f64> = vec![];
    let params = SolverParams::default();
    let precond = Preconditioner::Additive(LocalSolverConfig::solver_default());
    let result = solve(cats.view(), &y, None, &params, Some(&precond));
    assert!(result.is_err());
}
