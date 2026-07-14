//! PyO3 result wrapper classes and native-to-Python result conversions.

use std::ffi::CString;

use numpy::ndarray::{Array2, ShapeBuilder};
use numpy::IntoPyArray;
use pyo3::exceptions::{PyIndexError, PyUserWarning};
use pyo3::prelude::*;

use within::{BatchSolveResult, BuildWarning, CoefficientLayout, SolveResult};

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
    pub unidentified: Vec<PyUnidentifiedDirection>,
    #[pyo3(get)]
    pub layout: PyCoefficientLayout,
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
    pub unidentified: Vec<PyUnidentifiedDirection>,
    #[pyo3(get)]
    pub layout: PyCoefficientLayout,
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

/// A per-level design direction the data cannot identify.
#[pyclass(frozen, eq, hash, module = "within._within")]
#[pyo3(name = "UnidentifiedDirection")]
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct PyUnidentifiedDirection {
    #[pyo3(get)]
    pub term: usize,
    #[pyo3(get)]
    pub level: usize,
    #[pyo3(get)]
    pub column: usize,
}

#[pymethods]
impl PyUnidentifiedDirection {
    fn __repr__(&self) -> String {
        format!(
            "UnidentifiedDirection(term={}, level={}, column={})",
            self.term, self.level, self.column
        )
    }
}

/// Translates a ``(term, level, column)`` coefficient address to its flat
/// index in ``SolveResult.x`` and back.
#[pyclass(frozen, module = "within._within")]
#[pyo3(name = "CoefficientLayout")]
#[derive(Clone)]
pub struct PyCoefficientLayout {
    inner: CoefficientLayout,
}

#[pymethods]
impl PyCoefficientLayout {
    fn n_dofs(&self) -> usize {
        self.inner.n_dofs()
    }

    fn n_terms(&self) -> usize {
        self.inner.n_terms()
    }

    fn n_levels(&self, term: usize) -> PyResult<usize> {
        self.inner.n_levels(term).ok_or_else(|| self.term_oob(term))
    }

    fn n_columns(&self, term: usize) -> PyResult<usize> {
        self.inner
            .n_columns(term)
            .ok_or_else(|| self.term_oob(term))
    }

    fn index(&self, term: usize, level: usize, column: usize) -> PyResult<usize> {
        self.inner.index(term, level, column).ok_or_else(|| {
            PyIndexError::new_err(format!(
                "coefficient address (term={term}, level={level}, column={column}) out of range"
            ))
        })
    }

    fn address(&self, index: usize) -> PyResult<(usize, usize, usize)> {
        self.inner.address(index).ok_or_else(|| {
            PyIndexError::new_err(format!(
                "x index {index} out of range (n_dofs={})",
                self.inner.n_dofs()
            ))
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "CoefficientLayout(n_terms={}, n_dofs={})",
            self.inner.n_terms(),
            self.inner.n_dofs()
        )
    }
}

impl PyCoefficientLayout {
    fn term_oob(&self, term: usize) -> PyErr {
        PyIndexError::new_err(format!(
            "term {term} out of range (n_terms={})",
            self.inner.n_terms()
        ))
    }
}

// ---------------------------------------------------------------------------
// Result conversion helpers
// ---------------------------------------------------------------------------

pub(crate) fn into_py_result(py: Python<'_>, result: SolveResult) -> PySolveResult {
    PySolveResult {
        x: result.x.into_pyarray(py).unbind(),
        unidentified: result
            .unidentified
            .iter()
            .map(|u| PyUnidentifiedDirection {
                term: u.term,
                level: u.level,
                column: u.column,
            })
            .collect(),
        layout: PyCoefficientLayout {
            inner: result.layout,
        },
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
        unidentified: result
            .unidentified
            .iter()
            .map(|u| PyUnidentifiedDirection {
                term: u.term,
                level: u.level,
                column: u.column,
            })
            .collect(),
        layout: PyCoefficientLayout {
            inner: result.layout,
        },
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

/// Re-emit build-time warnings as Python `UserWarning`s. Shared by the
/// persistent `Solver` (at construction) and the one-shot `solve` path.
pub(crate) fn emit_build_warnings(py: Python<'_>, warnings: &[BuildWarning]) -> PyResult<()> {
    for warning in warnings {
        let message =
            CString::new(warning.to_string()).expect("warning messages contain no NUL bytes");
        PyErr::warn(py, &py.get_type::<PyUserWarning>(), &message, 1)?;
    }
    Ok(())
}

/// [`run_solve`] for the one-shot path: the off-GIL closure also returns the
/// build warnings collected during construction, which are re-emitted on-GIL.
pub(crate) fn run_solve_with_warnings<E, F>(py: Python<'_>, solve: F) -> PyResult<PySolveResult>
where
    E: std::fmt::Display + Send,
    F: Send + FnOnce() -> Result<(SolveResult, Vec<BuildWarning>), E>,
{
    let (result, warnings) = py.allow_threads(solve).map_err(value_err)?;
    emit_build_warnings(py, &warnings)?;
    Ok(into_py_result(py, result))
}

/// Batch counterpart to [`run_solve_with_warnings`].
pub(crate) fn run_batch_with_warnings<E, F>(
    py: Python<'_>,
    solve: F,
) -> PyResult<PyBatchSolveResult>
where
    E: std::fmt::Display + Send,
    F: Send + FnOnce() -> Result<(BatchSolveResult, Vec<BuildWarning>), E>,
{
    let (result, warnings) = py.allow_threads(solve).map_err(value_err)?;
    emit_build_warnings(py, &warnings)?;
    into_py_batch_result(py, result)
}
