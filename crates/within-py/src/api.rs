//! Free `#[pyfunction]`s exposed via `within._within`: [`solve`],
//! [`solve_batch`], and one-shot approximate parallel variants.

use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use within::SolveResult;
use within::{
    solve as solve_native, solve_approx_parallel as solve_approx_parallel_native,
    solve_approx_parallel_batch as solve_approx_parallel_batch_native,
    solve_batch as solve_batch_native, Solver, WithinError,
};

use crate::config::{extract_preconditioner_config, resolve_lsmr_config};
use crate::convert::{
    coerce_to_slice, column_refs, extract_columns, extract_weight_vec, value_err, warn_c_contiguous,
};
use crate::objects::extract_prebuilt;
use crate::results::{into_py_batch_result, into_py_result, PyBatchSolveResult, PySolveResult};

// ---------------------------------------------------------------------------
// Public solve functions
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (categories, y, options=None, weights=None, preconditioner=None))]
pub fn solve<'py>(
    py: Python<'py>,
    categories: PyReadonlyArray2<'py, u32>,
    y: PyReadonlyArray1<'py, f64>,
    options: Option<&Bound<'py, PyAny>>,
    weights: Option<PyReadonlyArray1<'py, f64>>,
    preconditioner: Option<&Bound<'py, PyAny>>,
) -> PyResult<PySolveResult> {
    let cats = categories.as_array();
    warn_c_contiguous(py, cats.strides())?;

    // Views and the (single-column) weight copy are cheap and require the GIL
    // token (`PyReadonlyArray`), so they stay GIL-held. `solve` performs no
    // large GIL-held copy: the response is borrowed and categories are a view.
    let y_arr = y.as_array();
    let w_vec = extract_weight_vec(&weights);
    let params = resolve_lsmr_config(options)?;

    let result = if let Some(built) = extract_prebuilt(preconditioner) {
        py.allow_threads(|| -> Result<SolveResult, WithinError> {
            let y_cow = coerce_to_slice(&y_arr);
            let solver = Solver::new(cats, w_vec, built)?;
            Ok(solver.solve(&y_cow, &params)?)
        })
        .map_err(value_err)?
    } else {
        let precond = extract_preconditioner_config(py, preconditioner)?;
        let w_ref = w_vec.as_deref();
        py.allow_threads(|| {
            let y_cow = coerce_to_slice(&y_arr);
            solve_native(cats, &y_cow, w_ref, &params, precond.as_ref())
        })
        .map_err(value_err)?
    };

    Ok(into_py_result(py, result))
}

#[pyfunction]
#[pyo3(signature = (categories, y, options=None, weights=None, preconditioner=None))]
pub fn solve_approx_parallel<'py>(
    py: Python<'py>,
    categories: PyReadonlyArray2<'py, u32>,
    y: PyReadonlyArray1<'py, f64>,
    options: Option<&Bound<'py, PyAny>>,
    weights: Option<PyReadonlyArray1<'py, f64>>,
    preconditioner: Option<&Bound<'py, PyAny>>,
) -> PyResult<PySolveResult> {
    let cats = categories.as_array();
    warn_c_contiguous(py, cats.strides())?;

    let y_arr = y.as_array();
    let w_vec = extract_weight_vec(&weights);
    let params = resolve_lsmr_config(options)?;

    let result = if let Some(built) = extract_prebuilt(preconditioner) {
        py.allow_threads(|| -> Result<SolveResult, WithinError> {
            let y_cow = coerce_to_slice(&y_arr);
            let solver = Solver::new(cats, w_vec, built)?;
            Ok(solver.solve_approx_parallel(&y_cow, params.tol)?)
        })
        .map_err(value_err)?
    } else {
        let precond = extract_preconditioner_config(py, preconditioner)?;
        let w_ref = w_vec.as_deref();
        py.allow_threads(|| {
            let y_cow = coerce_to_slice(&y_arr);
            solve_approx_parallel_native(cats, &y_cow, w_ref, &params, precond.as_ref())
        })
        .map_err(value_err)?
    };

    Ok(into_py_result(py, result))
}

#[pyfunction]
#[pyo3(signature = (categories, Y, options=None, weights=None, preconditioner=None))]
pub fn solve_batch<'py>(
    py: Python<'py>,
    categories: PyReadonlyArray2<'py, u32>,
    #[allow(non_snake_case)] Y: PyReadonlyArray2<'py, f64>,
    options: Option<&Bound<'py, PyAny>>,
    weights: Option<PyReadonlyArray1<'py, f64>>,
    preconditioner: Option<&Bound<'py, PyAny>>,
) -> PyResult<PyBatchSolveResult> {
    let cats = categories.as_array();
    warn_c_contiguous(py, cats.strides())?;

    let y_arr = Y.as_array();

    // Validate Y row count against the design up front. Without this, an empty
    // batch (Y.shape[1] == 0) would silently skip the per-column length check
    // inside `Solver::solve`.
    if y_arr.nrows() != cats.nrows() {
        return Err(value_err(format!(
            "Y has {} rows but categories has {} observations",
            y_arr.nrows(),
            cats.nrows()
        )));
    }

    // The (single-column) weight copy requires the GIL token, so it stays
    // GIL-held; the large O(n_obs·k) column copy (`extract_columns`) is
    // deferred into the GIL-released closure below.
    let w_vec = extract_weight_vec(&weights);
    let params = resolve_lsmr_config(options)?;

    let result = if let Some(built) = extract_prebuilt(preconditioner) {
        py.allow_threads(|| -> Result<_, WithinError> {
            let columns = extract_columns(&y_arr);
            let col_refs = column_refs(&columns);
            let solver = Solver::new(cats, w_vec, built)?;
            Ok(solver.solve_batch(&col_refs, &params)?)
        })
        .map_err(value_err)?
    } else {
        let precond = extract_preconditioner_config(py, preconditioner)?;
        let w_ref = w_vec.as_deref();
        py.allow_threads(|| {
            let columns = extract_columns(&y_arr);
            let col_refs = column_refs(&columns);
            solve_batch_native(cats, &col_refs, w_ref, &params, precond.as_ref())
        })
        .map_err(value_err)?
    };

    // Use the design dimensions carried by the result rather than inferring
    // them from output lengths — that keeps empty batches well-shaped at
    // (n_dofs, 0) / (n_obs, 0).
    let n_dofs = result.n_dofs;
    let n_obs = result.n_obs;
    into_py_batch_result(py, result, n_dofs, n_obs)
}

#[pyfunction]
#[pyo3(signature = (categories, Y, options=None, weights=None, preconditioner=None))]
pub fn solve_approx_parallel_batch<'py>(
    py: Python<'py>,
    categories: PyReadonlyArray2<'py, u32>,
    #[allow(non_snake_case)] Y: PyReadonlyArray2<'py, f64>,
    options: Option<&Bound<'py, PyAny>>,
    weights: Option<PyReadonlyArray1<'py, f64>>,
    preconditioner: Option<&Bound<'py, PyAny>>,
) -> PyResult<PyBatchSolveResult> {
    let cats = categories.as_array();
    warn_c_contiguous(py, cats.strides())?;

    let y_arr = Y.as_array();
    if y_arr.nrows() != cats.nrows() {
        return Err(value_err(format!(
            "Y has {} rows but categories has {} observations",
            y_arr.nrows(),
            cats.nrows()
        )));
    }

    let w_vec = extract_weight_vec(&weights);
    let params = resolve_lsmr_config(options)?;

    let result = if let Some(built) = extract_prebuilt(preconditioner) {
        py.allow_threads(|| -> Result<_, WithinError> {
            let columns = extract_columns(&y_arr);
            let col_refs = column_refs(&columns);
            let solver = Solver::new(cats, w_vec, built)?;
            Ok(solver.solve_approx_parallel_batch(&col_refs, params.tol)?)
        })
        .map_err(value_err)?
    } else {
        let precond = extract_preconditioner_config(py, preconditioner)?;
        let w_ref = w_vec.as_deref();
        py.allow_threads(|| {
            let columns = extract_columns(&y_arr);
            let col_refs = column_refs(&columns);
            solve_approx_parallel_batch_native(cats, &col_refs, w_ref, &params, precond.as_ref())
        })
        .map_err(value_err)?
    };

    let n_dofs = result.n_dofs;
    let n_obs = result.n_obs;
    into_py_batch_result(py, result, n_dofs, n_obs)
}
