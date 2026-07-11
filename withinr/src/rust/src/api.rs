//! Solve entry points and the persistent solver handle.
//!
//! Mirrors `crates/within-py/src/api.rs`: the one-shot `solve`/`solve_batch`
//! bindings plus the persistent `Solver` surface.

use extendr_api::prelude::*;

use within::observation::FactorMajorStore;
use within::{
    solve as solve_native, solve_batch as solve_batch_native, Design, Preconditioner,
    Solver as NativeSolver,
};

use crate::config::{parse_lsmr_options, parse_preconditioner, PreconditionerArg};
use crate::convert::{
    cast_categories, categories_view, err, extract_weights, factor_major_store, or_throw,
    usize_to_i32,
};
use crate::results::{batch_result_to_list, result_to_list};

/// Owned-store solver held behind the R external pointer.
pub(crate) type SolverHandle = NativeSolver<FactorMajorStore>;

// ---------------------------------------------------------------------------
// One-shot solve API
// ---------------------------------------------------------------------------

// Solve fixed-effects normal equations for a single response vector.
#[extendr]
fn solve_impl(
    categories: RMatrix<i32>,
    y: &[f64],
    options: Robj,
    weights: Robj,
    preconditioner: Robj,
) -> List {
    or_throw((|| -> Result<List> {
        let cats_u32 = cast_categories(categories.data())?;
        let cats = categories_view(&categories, &cats_u32)?;
        let lsmr = parse_lsmr_options(&options)?;
        let weights = extract_weights(weights)?;

        match parse_preconditioner(preconditioner)? {
            PreconditionerArg::Config(config) => {
                solve_native(cats, y, weights.as_deref(), &lsmr, config.as_ref())
                    .map_err(|e| err(e.to_string()))
                    .and_then(result_to_list)
            }
            PreconditionerArg::Built(built) => {
                solve_native(cats, y, weights.as_deref(), &lsmr, built)
                    .map_err(|e| err(e.to_string()))
                    .and_then(result_to_list)
            }
        }
    })())
}

// Solve fixed-effects normal equations for multiple response vectors.
#[extendr]
fn solve_batch_impl(
    categories: RMatrix<i32>,
    y_matrix: RMatrix<f64>,
    options: Robj,
    weights: Robj,
    preconditioner: Robj,
) -> List {
    or_throw((|| -> Result<List> {
        if y_matrix.nrows() != categories.nrows() {
            return Err(err(format!(
                "Y has {} rows but categories has {} observations",
                y_matrix.nrows(),
                categories.nrows()
            )));
        }

        let cats_u32 = cast_categories(categories.data())?;
        let cats = categories_view(&categories, &cats_u32)?;
        let lsmr = parse_lsmr_options(&options)?;
        let weights = extract_weights(weights)?;

        let y_data = y_matrix.data();
        let y_nrow = y_matrix.nrows();
        let y_ncol = y_matrix.ncols();
        let columns: Vec<Vec<f64>> = (0..y_ncol)
            .map(|j| y_data[j * y_nrow..(j + 1) * y_nrow].to_vec())
            .collect();
        let column_refs: Vec<&[f64]> = columns.iter().map(Vec::as_slice).collect();

        match parse_preconditioner(preconditioner)? {
            PreconditionerArg::Config(config) => solve_batch_native(
                cats,
                &column_refs,
                weights.as_deref(),
                &lsmr,
                config.as_ref(),
            )
            .map_err(|e| err(e.to_string()))
            .and_then(batch_result_to_list),
            PreconditionerArg::Built(built) => {
                solve_batch_native(cats, &column_refs, weights.as_deref(), &lsmr, built)
                    .map_err(|e| err(e.to_string()))
                    .and_then(batch_result_to_list)
            }
        }
    })())
}

// ---------------------------------------------------------------------------
// Persistent solver API
// ---------------------------------------------------------------------------

// Build a persistent solver that can be reused across multiple solves.
#[extendr]
fn solver_new_impl(
    categories: RMatrix<i32>,
    weights: Robj,
    preconditioner: Robj,
) -> ExternalPtr<SolverHandle> {
    or_throw((|| -> Result<ExternalPtr<SolverHandle>> {
        let weights = extract_weights(weights)?;
        let store = factor_major_store(&categories)?;
        let design = Design::from_store(store).map_err(|e| err(e.to_string()))?;

        let solver = match parse_preconditioner(preconditioner)? {
            PreconditionerArg::Config(config) => {
                NativeSolver::new(design, weights, config.as_ref())
            }
            PreconditionerArg::Built(built) => NativeSolver::new(design, weights, built),
        }
        .map_err(|e| err(e.to_string()))?;

        Ok(ExternalPtr::new(solver))
    })())
}

// Solve one response vector with a persistent solver.
#[extendr]
fn solver_solve_impl(solver: ExternalPtr<SolverHandle>, y: &[f64], options: Robj) -> List {
    or_throw((|| -> Result<List> {
        let lsmr = parse_lsmr_options(&options)?;
        solver
            .try_addr()?
            .solve(y, &lsmr)
            .map_err(|e| err(e.to_string()))
            .and_then(result_to_list)
    })())
}

// Solve multiple response vectors with a persistent solver.
#[extendr]
fn solver_solve_batch_impl(
    solver: ExternalPtr<SolverHandle>,
    y_matrix: RMatrix<f64>,
    options: Robj,
) -> List {
    or_throw((|| -> Result<List> {
        let lsmr = parse_lsmr_options(&options)?;
        let handle = solver.try_addr()?;

        if y_matrix.nrows() != handle.n_obs() {
            return Err(err(format!(
                "Y has {} rows but solver has {} observations",
                y_matrix.nrows(),
                handle.n_obs()
            )));
        }

        let y_data = y_matrix.data();
        let y_nrow = y_matrix.nrows();
        let y_ncol = y_matrix.ncols();
        let columns: Vec<Vec<f64>> = (0..y_ncol)
            .map(|j| y_data[j * y_nrow..(j + 1) * y_nrow].to_vec())
            .collect();
        let column_refs: Vec<&[f64]> = columns.iter().map(Vec::as_slice).collect();

        handle
            .solve_batch(&column_refs, &lsmr)
            .map_err(|e| err(e.to_string()))
            .and_then(batch_result_to_list)
    })())
}

// Return the built preconditioner from a persistent solver, or NULL.
#[extendr]
fn solver_preconditioner_impl(
    solver: ExternalPtr<SolverHandle>,
) -> Option<ExternalPtr<Preconditioner>> {
    or_throw((|| -> Result<Option<ExternalPtr<Preconditioner>>> {
        Ok(solver
            .try_addr()?
            .preconditioner()
            .map(|preconditioner| ExternalPtr::new(preconditioner.clone())))
    })())
}

// Number of DOFs (coefficients) in the persistent solver.
#[extendr]
fn solver_n_dofs_impl(solver: ExternalPtr<SolverHandle>) -> i32 {
    or_throw((|| -> Result<i32> {
        usize_to_i32(solver.try_addr()?.n_dofs(), "n_dofs")
    })())
}

// Number of observations in the persistent solver.
#[extendr]
fn solver_n_obs_impl(solver: ExternalPtr<SolverHandle>) -> i32 {
    or_throw((|| -> Result<i32> {
        usize_to_i32(solver.try_addr()?.n_obs(), "n_obs")
    })())
}

extendr_module! {
    mod api;
    fn solve_impl;
    fn solve_batch_impl;
    fn solver_new_impl;
    fn solver_solve_impl;
    fn solver_solve_batch_impl;
    fn solver_preconditioner_impl;
    fn solver_n_dofs_impl;
    fn solver_n_obs_impl;
}
