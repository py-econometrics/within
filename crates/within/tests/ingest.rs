//! Ingest boundary: `ArrayView2<u32>` categories arrays of any layout
//! normalize to the same contiguous-column frame.

use ndarray::{array, s, Array2, ShapeBuilder};
use within::{solve, LsmrOptions, PreconditionerConfig};

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
fn f_order_view_matches_owned_columns() {
    let (cats, y) = larger_problem();
    let cats_f = {
        let mut f = Array2::<u32>::zeros(cats.dim().f());
        f.assign(&cats);
        f
    };

    let result_view = solve(
        cats_f.view(),
        &y,
        None,
        &default_params(),
        additive_precond(),
    )
    .expect("view solve");

    let factor_cols: Vec<Vec<u32>> = (0..2)
        .map(|f| cats.column(f).iter().copied().collect())
        .collect();
    let design = common::make_design(factor_cols).expect("valid design");
    let solver = within::Solver::new(design, None, additive_precond()).expect("solver");
    let result_owned = solver.solve(&y, &default_params()).expect("owned solve");

    assert!(result_view.converged);
    assert!(result_owned.converged);
    assert_eq!(result_view.x, result_owned.x, "must be bit-identical");
}

#[test]
fn c_order_view_matches_f_order_bitwise() {
    // Both layouts normalize to identical contiguous columns at ingest, so
    // the solves are bit-identical — layout affects ingest cost only.
    let (cats, y) = larger_problem();
    assert!(cats.is_standard_layout()); // C-contiguous by default
    let cats_f = {
        let mut f = Array2::<u32>::zeros(cats.dim().f());
        f.assign(&cats);
        f
    };

    let result_c =
        solve(cats.view(), &y, None, &default_params(), additive_precond()).expect("C-order solve");
    let result_f = solve(
        cats_f.view(),
        &y,
        None,
        &default_params(),
        additive_precond(),
    )
    .expect("F-order solve");

    assert!(result_c.converged);
    assert_eq!(result_c.x, result_f.x, "must be bit-identical");
    common::assert_solution_finite(&result_c);
}

#[test]
fn column_reversed_view_ingests_in_logical_order() {
    // A column-reversed view (negative column stride) keeps each column
    // contiguous, so ingest borrows it zero-copy in logical order, swapping
    // the two factors relative to the unreversed array.
    let cats_f = {
        let cats = array![[0u32, 1], [1, 0], [0, 1], [1, 0], [2, 2]];
        let mut f = Array2::<u32>::zeros(cats.dim().f());
        f.assign(&cats);
        f
    };
    let reversed = cats_f.slice(s![.., ..;-1]);
    assert!(reversed.strides()[1] < 1, "column stride must be negative");

    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let result_rev =
        solve(reversed, &y, None, &default_params(), additive_precond()).expect("reversed solve");

    // Oracle: the same columns handed over owned, in swapped order.
    let design =
        common::make_design(vec![vec![1, 0, 1, 0, 2], vec![0, 1, 0, 1, 2]]).expect("valid design");
    let solver = within::Solver::new(design, None, additive_precond()).expect("solver");
    let result_owned = solver.solve(&y, &default_params()).expect("owned solve");

    assert_eq!(result_rev.x, result_owned.x, "must be bit-identical");
}

#[test]
fn weighted_view_solve_converges() {
    let (cats, y) = larger_problem();
    let weights: Vec<f64> = (0..cats.nrows()).map(|i| 1.0 + (i as f64) * 0.01).collect();

    let result = solve(
        cats.view(),
        &y,
        Some(&weights),
        &default_params(),
        additive_precond(),
    )
    .expect("weighted view solve");

    assert!(result.converged);
    common::assert_solution_finite(&result);
}
