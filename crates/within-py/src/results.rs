//! PyO3 result wrapper classes and native-to-Python result conversions.

use numpy::ndarray::{Array2, ShapeBuilder};
use numpy::IntoPyArray;
use pyo3::prelude::*;

use within::{BatchSolveResult, SolveResult};

use crate::convert::value_err;

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

#[pyclass(module = "within._within")]
#[pyo3(name = "SolveResult")]
pub struct PySolveResult {
    #[pyo3(get)]
    pub x: Py<numpy::PyArray1<f64>>,
    #[pyo3(get)]
    pub demeaned: Py<numpy::PyArray1<f64>>,
    #[pyo3(get)]
    pub converged: bool,
    #[pyo3(get)]
    pub iterations: usize,
    #[pyo3(get)]
    pub residual: f64,
    #[pyo3(get)]
    pub time_total: f64,
    #[pyo3(get)]
    pub time_setup: f64,
    #[pyo3(get)]
    pub time_solve: f64,
}

#[pyclass(module = "within._within")]
#[pyo3(name = "BatchSolveResult")]
pub struct PyBatchSolveResult {
    #[pyo3(get)]
    pub x: Py<numpy::PyArray2<f64>>,
    #[pyo3(get)]
    pub demeaned: Py<numpy::PyArray2<f64>>,
    #[pyo3(get)]
    pub converged: Vec<bool>,
    #[pyo3(get)]
    pub iterations: Vec<usize>,
    #[pyo3(get)]
    pub residual: Vec<f64>,
    #[pyo3(get)]
    pub time_solve: Vec<f64>,
    #[pyo3(get)]
    pub time_total: f64,
}

// ---------------------------------------------------------------------------
// Result conversion helpers
// ---------------------------------------------------------------------------

pub(crate) fn into_py_result(py: Python<'_>, result: SolveResult) -> PySolveResult {
    PySolveResult {
        x: result.x.into_pyarray(py).unbind(),
        demeaned: result.demeaned.into_pyarray(py).unbind(),
        converged: result.converged,
        iterations: result.iterations,
        residual: result.residual,
        time_total: result.time_total,
        time_setup: result.time_setup,
        time_solve: result.time_solve,
    }
}

pub(crate) fn into_py_batch_result(
    py: Python<'_>,
    result: within::BatchSolveResult,
) -> PyResult<PyBatchSolveResult> {
    let n_rhs = result.converged.len();

    // Source dimensions from the result (not output lengths) so empty batches
    // stay well-shaped at (n_dofs, 0) / (n_obs, 0).
    let x = Array2::from_shape_vec((result.n_dofs, n_rhs).f(), result.x).map_err(value_err)?;
    let demeaned =
        Array2::from_shape_vec((result.n_obs, n_rhs).f(), result.demeaned).map_err(value_err)?;

    Ok(PyBatchSolveResult {
        x: x.into_pyarray(py).unbind(),
        demeaned: demeaned.into_pyarray(py).unbind(),
        converged: result.converged,
        iterations: result.iterations,
        residual: result.residual,
        time_solve: result.time_solve,
        time_total: result.time_total,
    })
}

// ---------------------------------------------------------------------------
// Off-GIL solve orchestration
// ---------------------------------------------------------------------------

/// Run a native single-response solve with the GIL released, then convert.
///
/// The closure produces a [`SolveResult`] off-GIL (`allow_threads`); its native
/// error is mapped to a `PyValueError` and the result to its Python wrapper.
/// Shared by the free `solve` function and the persistent `Solver.solve`.
pub(crate) fn run_solve<E, F>(py: Python<'_>, solve: F) -> PyResult<PySolveResult>
where
    E: std::fmt::Display + Send,
    F: Send + FnOnce() -> Result<SolveResult, E>,
{
    let result = py.allow_threads(solve).map_err(value_err)?;
    Ok(into_py_result(py, result))
}

/// Run a native batch solve with the GIL released, then convert.
///
/// Batch counterpart to [`run_solve`]; the conversion itself is fallible
/// (re-shaping the flat column buffers into 2-D arrays).
pub(crate) fn run_batch<E, F>(py: Python<'_>, solve: F) -> PyResult<PyBatchSolveResult>
where
    E: std::fmt::Display + Send,
    F: Send + FnOnce() -> Result<BatchSolveResult, E>,
{
    let result = py.allow_threads(solve).map_err(value_err)?;
    into_py_batch_result(py, result)
}
