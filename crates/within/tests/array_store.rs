use ndarray::{array, s, Array2, ShapeBuilder};
use within::observation::{ArrayStore, FactorMajorStore, Store};
use within::{solve, Design, LsmrOptions, PreconditionerConfig};

#[path = "common/orchestrate_helpers.rs"]
mod common;

fn default_params() -> LsmrOptions {
    LsmrOptions::default()
}

fn additive_precond() -> PreconditionerConfig {
    PreconditionerConfig::default()
}

/// Build a larger problem for more meaningful convergence tests.
fn larger_problem() -> (Array2<u32>, Vec<f64>) {
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    let mut rng = SmallRng::seed_from_u64(42);
    let n_obs = 500;
    let n_lev = [20u32, 30];
    let mut cats = Array2::<u32>::zeros((n_obs, 2));
    for i in 0..n_obs {
        cats[[i, 0]] = rng.random_range(0..n_lev[0]);
        cats[[i, 1]] = rng.random_range(0..n_lev[1]);
    }
    let y: Vec<f64> = (0..n_obs).map(|_| rng.random::<f64>()).collect();
    (cats, y)
}

#[test]
fn test_array_store_f_contiguous_matches_factor_major() {
    // Compare ArrayStore (F-contiguous) with FactorMajorStore on same data
    let (cats, y) = larger_problem();
    let cats_f = {
        let mut f = Array2::<u32>::zeros(cats.dim().f());
        f.assign(&cats);
        f
    };

    // Solve via ArrayStore
    let result_array = solve(
        cats_f.view(),
        &y,
        None,
        &default_params(),
        additive_precond(),
    )
    .expect("ArrayStore solve");

    // Solve via FactorMajorStore (using the same categories as Vec<Vec<u32>>)
    let factor_cols: Vec<Vec<u32>> = (0..2)
        .map(|f| cats.column(f).iter().copied().collect())
        .collect();
    let store = FactorMajorStore::new(factor_cols, cats.nrows()).expect("valid FactorMajorStore");
    let design = Design::from_store(store).expect("valid design");
    let solver = within::Solver::new(design, None, additive_precond()).expect("solver");
    let result_fms = solver
        .solve(&y, &default_params())
        .expect("FactorMajorStore solve");

    assert!(result_array.converged);
    assert!(result_fms.converged);
    for (a, b) in result_array.x.iter().zip(result_fms.x.iter()) {
        assert!(
            (a - b).abs() < 1e-8,
            "ArrayStore vs FactorMajorStore mismatch: {} vs {}",
            a,
            b
        );
    }
}

#[test]
fn test_array_store_c_contiguous_solves() {
    // C-contiguous array should also solve (just slower)
    let (cats, y) = larger_problem();
    assert!(cats.is_standard_layout()); // C-contiguous by default

    let result = solve(cats.view(), &y, None, &default_params(), additive_precond())
        .expect("C-contiguous ArrayStore solve");

    assert!(result.converged);
    common::assert_solution_finite(&result);
}

#[test]
fn test_array_store_factor_column_f_order() {
    // F-contiguous array should return Some from factor_column()
    let cats_f = {
        let cats = array![[0u32, 0], [1, 0], [0, 1], [1, 1]];
        let mut f = Array2::<u32>::zeros(cats.dim().f());
        f.assign(&cats);
        f
    };
    let store = ArrayStore::new(cats_f.view()).expect("valid store");
    assert!(store.factor_column(0).is_some());
    assert!(store.factor_column(1).is_some());
}

#[test]
fn test_array_store_factor_column_c_order() {
    // C-contiguous array should return None from factor_column()
    let cats = array![[0u32, 0], [1, 0], [0, 1], [1, 1]];
    assert!(cats.is_standard_layout()); // C-contiguous
    let store = ArrayStore::new(cats.view()).expect("valid store");
    assert!(store.factor_column(0).is_none());
    assert!(store.factor_column(1).is_none());
}

#[test]
fn test_array_store_factor_column_negative_col_stride_falls_back() {
    // A column-reversed view of an F-order array keeps strides[0] == 1 (row
    // stride) but has strides[1] < 1 (negative column stride). The unsafe
    // contiguous fast-path would derive a huge usize col_stride and read out
    // of bounds, so factor_column() must reject it and fall back to the safe
    // per-element level() path (returning None).
    let cats_f = {
        let cats = array![[0u32, 10], [1, 11], [2, 12], [3, 13]];
        let mut f = Array2::<u32>::zeros(cats.dim().f());
        f.assign(&cats);
        f
    };
    // Reverse along the column axis: strides[0] stays 1, strides[1] becomes negative.
    let reversed = cats_f.slice(s![.., ..;-1]);
    let strides = reversed.strides();
    assert_eq!(strides[0], 1, "row stride should remain 1 (F-order)");
    assert!(
        strides[1] < 1,
        "column stride should be negative after reversal"
    );

    let store = ArrayStore::new(reversed).expect("valid store");
    // Must fall back to the safe path rather than perform an OOB unsafe read.
    assert!(
        store.factor_column(0).is_none(),
        "negative column stride must force the safe fallback"
    );
    assert!(store.factor_column(1).is_none());

    // The safe level() path must still return correct, in-bounds values. The
    // reversed view swaps the two columns, so column 0 now holds the original
    // factor-1 values and vice versa.
    assert_eq!(store.level(0, 0), 10);
    assert_eq!(store.level(0, 1), 0);
    assert_eq!(store.level(3, 0), 13);
    assert_eq!(store.level(3, 1), 3);
}

#[test]
fn test_array_store_weighted() {
    let (cats, y) = larger_problem();
    let weights: Vec<f64> = (0..cats.nrows()).map(|i| 1.0 + (i as f64) * 0.01).collect();

    let result = solve(
        cats.view(),
        &y,
        Some(&weights),
        &default_params(),
        additive_precond(),
    )
    .expect("weighted ArrayStore solve");

    assert!(result.converged);
    common::assert_solution_finite(&result);
}
