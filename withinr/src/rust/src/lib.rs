//! Extendr bridge exposing the [`within`] Rust crate to R as the `withinr` package.
//!
//! This crate is intentionally minimal: it converts between R/extendr types
//! and the native Rust API, then delegates all computation to [`within`].
//!
//! # Index convention
//!
//! R users pass **1-based** integer category matrices. This bridge validates
//! that all entries are >= 1 and contain no `NA`, then subtracts 1 to produce
//! the **0-based** `u32` indices expected by the Rust solver.

use extendr_api::prelude::*;
use ndarray::Array2;

use within::config::{
    KrylovMethod, LocalSolverConfig, Preconditioner as WithinPreconditioner, ReductionStrategy,
    SolverParams,
};
use within::{solve as solve_native, solve_batch as solve_batch_native};

// ---------------------------------------------------------------------------
// Conversion helpers
// ---------------------------------------------------------------------------

/// Convert an R integer matrix (1-based categories) to a 0-based `u32` ndarray.
///
/// R stores matrices column-major: `data[j * nrow + i]` is element `(i, j)`.
/// The output is a standard C-order `Array2<u32>` of shape `(n_obs, n_factors)`.
fn categories_to_array(categories: RMatrix<i32>) -> std::result::Result<Array2<u32>, String> {
    let nrow = categories.nrows();
    let ncol = categories.ncols();
    let data = categories.data();

    let mut buf = Vec::with_capacity(nrow * ncol);
    for i in 0..nrow {
        for j in 0..ncol {
            let val = data[j * nrow + i];
            if val == i32::MIN {
                return Err(format!(
                    "categories must not contain NA values (found at row {}, col {})",
                    i + 1,
                    j + 1
                ));
            }
            if val < 1 {
                return Err(format!(
                    "categories must be >= 1 (1-based); found {} at row {}, col {}",
                    val,
                    i + 1,
                    j + 1
                ));
            }
            buf.push((val - 1) as u32);
        }
    }
    Array2::from_shape_vec((nrow, ncol), buf)
        .map_err(|e| format!("internal error building categories array: {e}"))
}

/// Parse the `method` string into a [`KrylovMethod`].
fn parse_krylov(method: &str, restart: i32) -> std::result::Result<KrylovMethod, String> {
    match method {
        "cg" => Ok(KrylovMethod::Cg),
        "gmres" => {
            if restart < 1 {
                return Err(format!(
                    "restart must be >= 1 for gmres; got {}",
                    restart
                ));
            }
            Ok(KrylovMethod::Gmres {
                restart: restart as usize,
            })
        }
        _ => Err(format!(
            "unknown method '{method}': expected 'cg' or 'gmres'"
        )),
    }
}

/// Parse the `preconditioner` string into an `Option<Preconditioner>`.
fn parse_preconditioner(
    preconditioner: &str,
) -> std::result::Result<Option<WithinPreconditioner>, String> {
    match preconditioner {
        "additive" => Ok(Some(WithinPreconditioner::Additive(
            LocalSolverConfig::solver_default(),
            ReductionStrategy::Auto,
        ))),
        "multiplicative" => Ok(Some(WithinPreconditioner::Multiplicative(
            LocalSolverConfig::solver_default(),
        ))),
        "off" => Ok(None),
        _ => Err(format!(
            "unknown preconditioner '{preconditioner}': expected 'additive', 'multiplicative', or 'off'"
        )),
    }
}

/// Build [`SolverParams`] from R arguments.
fn build_params(
    method: &str,
    tol: f64,
    maxiter: i32,
    restart: i32,
) -> std::result::Result<SolverParams, String> {
    if !tol.is_finite() || tol <= 0.0 {
        return Err(format!(
            "tol must be a finite positive number; got {}",
            tol
        ));
    }
    if maxiter < 1 {
        return Err(format!("maxiter must be >= 1; got {}", maxiter));
    }

    Ok(SolverParams {
        krylov: parse_krylov(method, restart)?,
        tol,
        maxiter: maxiter as usize,
        ..SolverParams::default()
    })
}

/// Validate that CG is not paired with a non-symmetric preconditioner.
fn validate_cg_preconditioner(
    params: &SolverParams,
    precond: &Option<WithinPreconditioner>,
) -> std::result::Result<(), String> {
    if matches!(params.krylov, KrylovMethod::Cg)
        && matches!(precond, Some(WithinPreconditioner::Multiplicative(_)))
    {
        return Err(
            "CG requires a symmetric preconditioner; use preconditioner='additive' or method='gmres'"
                .to_string(),
        );
    }
    Ok(())
}

/// Extract an optional weight vector from an R object (NULL → None).
fn extract_weights(weights: Robj) -> std::result::Result<Option<Vec<f64>>, String> {
    if weights.is_null() {
        return Ok(None);
    }
    weights
        .as_real_vector()
        .map(Some)
        .ok_or_else(|| "weights must be a numeric vector or NULL".to_string())
}

// ---------------------------------------------------------------------------
// Exported functions
// ---------------------------------------------------------------------------

/// Solve fixed-effects normal equations for a single response vector.
///
/// Called from the R wrapper `withinr::solve()`. Categories are 1-based
/// integer matrices; conversion to 0-based is handled here.
/// @export
#[extendr]
fn solve_impl(
    categories: RMatrix<i32>,
    y: &[f64],
    weights: Robj,
    method: &str,
    tol: f64,
    maxiter: i32,
    restart: i32,
    preconditioner: &str,
) -> extendr_api::Result<Robj> {
    let cats = categories_to_array(categories)?;
    let params = build_params(method, tol, maxiter, restart)?;
    let precond = parse_preconditioner(preconditioner)?;
    validate_cg_preconditioner(&params, &precond)?;
    let w_vec = extract_weights(weights)?;

    let result = solve_native(cats.view(), y, w_vec.as_deref(), &params, precond.as_ref())
        .map_err(|e| e.to_string())?;

    Ok(list!(
        coefficients = result.x,
        demeaned = result.demeaned,
        converged = result.converged,
        iterations = result.iterations as i32,
        residual = result.final_residual,
        time_total = result.time_total,
        time_setup = result.time_setup,
        time_solve = result.time_solve
    )
    .into())
}

/// Solve fixed-effects normal equations for multiple response vectors.
///
/// Called from the R wrapper `withinr::solve_batch()`. The response matrix Y
/// has shape (n_obs, k) in R's column-major layout; each column is one RHS.
/// @export
#[extendr]
fn solve_batch_impl(
    categories: RMatrix<i32>,
    y_matrix: RMatrix<f64>,
    weights: Robj,
    method: &str,
    tol: f64,
    maxiter: i32,
    restart: i32,
    preconditioner: &str,
) -> extendr_api::Result<Robj> {
    let cats = categories_to_array(categories)?;
    let params = build_params(method, tol, maxiter, restart)?;
    let precond = parse_preconditioner(preconditioner)?;
    validate_cg_preconditioner(&params, &precond)?;
    let w_vec = extract_weights(weights)?;

    let nrow = y_matrix.nrows();
    let ncol = y_matrix.ncols();
    let y_data = y_matrix.data();

    // Extract columns from R's column-major storage.
    // Kept as owned buffers for now; we'll revisit borrowing optimization separately.
    let columns: Vec<Vec<f64>> = (0..ncol)
        .map(|j| y_data[j * nrow..(j + 1) * nrow].to_vec())
        .collect();
    let col_refs: Vec<&[f64]> = columns.iter().map(|c| c.as_slice()).collect();

    let result = solve_batch_native(cats.view(), &col_refs, w_vec.as_deref(), &params, precond.as_ref())
        .map_err(|e| e.to_string())?;

    let n_rhs = result.n_rhs();
    let n_dofs = if n_rhs > 0 {
        result.x_all().len() / n_rhs
    } else {
        0
    };
    let n_obs = if n_rhs > 0 {
        result.demeaned_all().len() / n_rhs
    } else {
        0
    };

    // Return flat vectors; the R wrapper reshapes into matrices.
    let iterations: Vec<i32> = result.iterations().iter().map(|&i| i as i32).collect();

    Ok(list!(
        coefficients = result.x_all().to_vec(),
        demeaned = result.demeaned_all().to_vec(),
        converged = result.converged().to_vec(),
        iterations = iterations,
        residual = result.final_residual().to_vec(),
        time_total = result.time_total(),
        time_solve = result.time_solve().to_vec(),
        n_dofs = n_dofs as i32,
        n_obs = n_obs as i32,
        n_rhs = n_rhs as i32
    )
    .into())
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

extendr_module! {
    mod withinr;
    fn solve_impl;
    fn solve_batch_impl;
}
