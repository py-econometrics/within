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

use crate::config::{resolve_lsmr_config, resolve_precond_input, PrecondInput};
use crate::convert::{
    coerce_to_slice, column_refs, extract_columns, run_batch, run_solve, value_err,
    warn_c_contiguous,
};
use crate::results::{PyBatchSolveResult, PySolveResult};

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
    warn_c_contiguous(py, &cats)?;

    // Borrow the array views while the GIL is held (`PyReadonlyArray` needs the
    // token), but defer the slice coercion -- and any copy of strided input --
    // into the GIL-released closures below. The common F-contiguous path then
    // borrows both `y` and `weights` with no copy at all.
    let y_arr = y.as_array();
    let w_view = weights.as_ref().map(|w| w.as_array());
    let params = resolve_lsmr_config(options)?;

    match resolve_precond_input(py, preconditioner)? {
        PrecondInput::Prebuilt(built) => run_solve(py, || -> Result<SolveResult, WithinError> {
            let y_cow = coerce_to_slice(&y_arr);
            let w_vec = w_view.as_ref().map(|v| coerce_to_slice(v).into_owned());
            let solver = Solver::new(cats, w_vec, built)?;
            Ok(solver.solve(&y_cow, &params)?)
        }),
        PrecondInput::Config(precond) => run_solve(py, || {
            let y_cow = coerce_to_slice(&y_arr);
            let w_cow = w_view.as_ref().map(coerce_to_slice);
            solve_native(cats, &y_cow, w_cow.as_deref(), &params, precond.as_ref())
        }),
    }
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
    warn_c_contiguous(py, &cats)?;

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

    // Defer column extraction and weight coercion into the GIL-released closures
    // below: F-contiguous columns and contiguous weights are borrowed directly,
    // only strided input is copied (off-GIL).
    let w_view = weights.as_ref().map(|w| w.as_array());
    let params = resolve_lsmr_config(options)?;

    match resolve_precond_input(py, preconditioner)? {
        PrecondInput::Prebuilt(built) => run_batch(py, || -> Result<_, WithinError> {
            let columns = extract_columns(&y_arr);
            let col_refs = column_refs(&columns);
            let w_vec = w_view.as_ref().map(|v| coerce_to_slice(v).into_owned());
            let solver = Solver::new(cats, w_vec, built)?;
            Ok(solver.solve_batch(&col_refs, &params)?)
        }),
        PrecondInput::Config(precond) => run_batch(py, || {
            let columns = extract_columns(&y_arr);
            let col_refs = column_refs(&columns);
            let w_cow = w_view.as_ref().map(coerce_to_slice);
            solve_batch_native(cats, &col_refs, w_cow.as_deref(), &params, precond.as_ref())
        }),
    }
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
