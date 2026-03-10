//! R API for the `within` fixed-effects solver via extendr.

use extendr_api::prelude::*;
use ndarray::{ArrayView2, ShapeBuilder};

use within::config::{KrylovMethod, LocalSolverConfig, Preconditioner, SolverParams};
use within::{solve as solve_native, solve_batch as solve_batch_native, SolveResult};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Parse string method name into SolverParams.
fn build_params(
    method: Option<&str>,
    tol: Option<f64>,
    maxiter: Option<i32>,
    restart: Option<i32>,
) -> Result<SolverParams> {
    let tol = tol.unwrap_or(1e-8);
    let maxiter = maxiter.unwrap_or(1000) as usize;
    let krylov = match method.unwrap_or("cg") {
        "cg" => KrylovMethod::Cg,
        "gmres" => KrylovMethod::Gmres {
            restart: restart.unwrap_or(30) as usize,
        },
        other => {
            return Err(Error::Other(format!(
                "Unknown method '{}'. Use 'cg' or 'gmres'.",
                other,
            )))
        }
    };
    Ok(SolverParams {
        krylov,
        tol,
        maxiter,
        ..Default::default()
    })
}

/// Parse string preconditioner name into Option<Preconditioner>.
fn build_preconditioner(preconditioner: Option<&str>) -> Result<Option<Preconditioner>> {
    match preconditioner.unwrap_or("additive") {
        "additive" => Ok(Some(Preconditioner::Additive(
            LocalSolverConfig::solver_default(),
        ))),
        "multiplicative" => Ok(Some(Preconditioner::Multiplicative(
            LocalSolverConfig::solver_default(),
        ))),
        "off" => Ok(None),
        other => Err(Error::Other(format!(
            "Unknown preconditioner '{}'. Use 'additive', 'multiplicative', or 'off'.",
            other,
        ))),
    }
}

/// Validate that CG is not paired with a multiplicative preconditioner.
fn validate_cg_preconditioner(
    params: &SolverParams,
    preconditioner: &Option<Preconditioner>,
) -> Result<()> {
    if matches!(params.krylov, KrylovMethod::Cg)
        && matches!(preconditioner, Some(Preconditioner::Multiplicative(_)))
    {
        return Err(Error::Other(
            "CG requires a symmetric preconditioner; use 'additive' or switch to method='gmres'"
                .to_string(),
        ));
    }
    Ok(())
}

/// Convert a SolveResult into a named R list.
fn result_to_list(r: SolveResult) -> List {
    list!(
        x = r.x,
        demeaned = r.demeaned,
        converged = r.converged,
        iterations = r.iterations as i32,
        residual = r.final_residual,
        time_total = r.time_total,
        time_setup = r.time_setup,
        time_solve = r.time_solve
    )
}

/// Cast an i32 slice to a Vec<u32>, checking for negative values.
fn cast_categories(data: &[i32]) -> Result<Vec<u32>> {
    if let Some(pos) = data.iter().position(|&v| v < 0) {
        return Err(Error::Other(format!(
            "categories must be non-negative integers, found {} at position {}",
            data[pos], pos,
        )));
    }
    Ok(data.iter().map(|&v| v as u32).collect())
}

/// Extract weights from Nullable<Doubles> as Option<Vec<f64>>.
fn extract_weights(weights: Nullable<Doubles>) -> Option<Vec<f64>> {
    match weights {
        Nullable::NotNull(d) => Some(d.iter().map(|v| v.inner()).collect()),
        Nullable::Null => None,
    }
}

// ---------------------------------------------------------------------------
// Exported functions
// ---------------------------------------------------------------------------

/// Solve fixed-effects demeaning for a single response vector.
///
/// @param categories Integer matrix (n_obs x n_factors) of factor levels (0-based).
/// @param y Numeric vector of length n_obs.
/// @param method Solver method: "cg" (default) or "gmres".
/// @param tol Convergence tolerance (default 1e-8).
/// @param maxiter Maximum iterations (default 1000).
/// @param restart GMRES restart parameter (default 30, ignored for CG).
/// @param preconditioner Preconditioner: "additive" (default), "multiplicative", or "off".
/// @param weights Optional observation weights (length n_obs).
/// @return A named list with components x, demeaned, converged, iterations,
///         residual, time_total, time_setup, time_solve.
#[extendr]
fn solve(
    categories: RMatrix<i32>,
    y: &[f64],
    method: Nullable<&str>,
    tol: Nullable<f64>,
    maxiter: Nullable<i32>,
    restart: Nullable<i32>,
    preconditioner: Nullable<&str>,
    weights: Nullable<Doubles>,
) -> Result<List> {
    let method: Option<&str> = method.into();
    let tol: Option<f64> = tol.into();
    let maxiter: Option<i32> = maxiter.into();
    let restart: Option<i32> = restart.into();
    let preconditioner: Option<&str> = preconditioner.into();

    let nrow = categories.nrows();
    let ncol = categories.ncols();

    // Cast i32 → u32 (R has no unsigned integers)
    let cats_u32 = cast_categories(categories.data())?;

    // R matrices are column-major, so use Fortran order
    let cats_view = ArrayView2::from_shape((nrow, ncol).f(), &cats_u32)
        .map_err(|e| Error::Other(e.to_string()))?;

    let params = build_params(method, tol, maxiter, restart)?;
    let precond = build_preconditioner(preconditioner)?;
    validate_cg_preconditioner(&params, &precond)?;

    let w_vec = extract_weights(weights);
    let w_ref = w_vec.as_deref();

    let result = solve_native(cats_view, y, w_ref, &params, precond.as_ref())
        .map_err(|e| Error::Other(e.to_string()))?;

    Ok(result_to_list(result))
}

/// Solve fixed-effects demeaning for multiple response vectors (batch).
///
/// @param categories Integer matrix (n_obs x n_factors) of factor levels (0-based).
/// @param Y Numeric matrix (n_obs x k) where each column is a response vector.
/// @param method Solver method: "cg" (default) or "gmres".
/// @param tol Convergence tolerance (default 1e-8).
/// @param maxiter Maximum iterations (default 1000).
/// @param restart GMRES restart parameter (default 30, ignored for CG).
/// @param preconditioner Preconditioner: "additive" (default), "multiplicative", or "off".
/// @param weights Optional observation weights (length n_obs).
/// @return A named list with components x, demeaned (matrices), converged,
///         iterations, residual, time_solve (vectors), and time_total (scalar).
#[extendr]
fn solve_batch(
    categories: RMatrix<i32>,
    y: RMatrix<f64>,
    method: Nullable<&str>,
    tol: Nullable<f64>,
    maxiter: Nullable<i32>,
    restart: Nullable<i32>,
    preconditioner: Nullable<&str>,
    weights: Nullable<Doubles>,
) -> Result<List> {
    let method: Option<&str> = method.into();
    let tol: Option<f64> = tol.into();
    let maxiter: Option<i32> = maxiter.into();
    let restart: Option<i32> = restart.into();
    let preconditioner: Option<&str> = preconditioner.into();

    let nrow = categories.nrows();
    let ncol = categories.ncols();

    let cats_u32 = cast_categories(categories.data())?;
    let cats_view = ArrayView2::from_shape((nrow, ncol).f(), &cats_u32)
        .map_err(|e| Error::Other(e.to_string()))?;

    let params = build_params(method, tol, maxiter, restart)?;
    let precond = build_preconditioner(preconditioner)?;
    validate_cg_preconditioner(&params, &precond)?;

    let w_vec = extract_weights(weights);
    let w_ref = w_vec.as_deref();

    // Y is column-major from R; extract each column as a contiguous Vec
    let y_data = y.data();
    let y_nrow = y.nrows();
    let y_ncol = y.ncols();

    let columns: Vec<Vec<f64>> = (0..y_ncol)
        .map(|j| y_data[j * y_nrow..(j + 1) * y_nrow].to_vec())
        .collect();
    let column_refs: Vec<&[f64]> = columns.iter().map(Vec::as_slice).collect();

    let result = solve_batch_native(cats_view, &column_refs, w_ref, &params, precond.as_ref())
        .map_err(|e| Error::Other(e.to_string()))?;

    // Build result matrices: x is (n_dofs x k), demeaned is (n_obs x k)
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

    // Build column-major R matrices
    let mut x_mat = RMatrix::new(n_dofs, n_rhs);
    let x_data = x_mat.data_mut();
    x_data.copy_from_slice(result.x_all());

    let mut dem_mat = RMatrix::new(n_obs, n_rhs);
    let dem_data = dem_mat.data_mut();
    dem_data.copy_from_slice(result.demeaned_all());

    let converged: Vec<bool> = result.converged().to_vec();
    let iterations: Vec<i32> = result.iterations().iter().map(|&v| v as i32).collect();
    let residual: Vec<f64> = result.final_residual().to_vec();
    let time_solve: Vec<f64> = result.time_solve().to_vec();

    Ok(list!(
        x = x_mat,
        demeaned = dem_mat,
        converged = converged,
        iterations = iterations,
        residual = residual,
        time_solve = time_solve,
        time_total = result.time_total()
    ))
}

extendr_module! {
    mod within;
    fn solve;
    fn solve_batch;
}
