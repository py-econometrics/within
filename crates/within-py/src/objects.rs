//! PyO3 object wrappers that hold native solver state: the picklable built
//! [`PyPreconditioner`] and the persistent [`PySolver`].

use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use within::observation::FactorMajorStore;
use within::{Design, Preconditioner, Solver};

use crate::config::{extract_preconditioner_config, resolve_lsmr_config};
use crate::convert::{coerce_to_slice, column_refs, extract_columns, value_err, warn_c_contiguous};
use crate::results::{into_py_batch_result, into_py_result, PyBatchSolveResult, PySolveResult};

// ---------------------------------------------------------------------------
// Built preconditioner (returned by Solver, picklable)
// ---------------------------------------------------------------------------

/// A pre-built preconditioner that can be pickled and reused.
///
/// Obtained via ``Solver.preconditioner``. Pass it back to a new
/// ``Solver(…, preconditioner=p)`` to skip the expensive factorisation.
#[pyclass(frozen, module = "within._within")]
#[pyo3(name = "Preconditioner")]
pub struct PyPreconditioner {
    pub(crate) inner: Preconditioner,
}

#[pymethods]
impl PyPreconditioner {
    /// Apply the preconditioner: ``y = M⁻¹ x``.
    fn apply<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
        let x_slice = x
            .as_slice()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("x must be contiguous"))?;
        if x_slice.len() != self.inner.ncols() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "x has length {} but preconditioner expects {}",
                x_slice.len(),
                self.inner.ncols()
            )));
        }
        let mut y = vec![0.0; self.inner.nrows()];
        self.inner
            .apply(x_slice, &mut y)
            .map_err(|e: within::SolveError| {
                pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
            })?;
        Ok(numpy::PyArray1::from_vec(py, y))
    }
    /// Number of rows (DOFs).
    #[getter]
    fn nrows(&self) -> usize {
        self.inner.nrows()
    }

    /// Number of columns (DOFs).
    #[getter]
    fn ncols(&self) -> usize {
        self.inner.ncols()
    }

    fn __repr__(&self) -> String {
        format!("Preconditioner(Additive, n={})", self.inner.nrows())
    }

    /// Pickle support: serialize to ``(bytes,)`` constructor arg.
    fn __reduce__<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyAny>, (Bound<'py, pyo3::types::PyBytes>,))> {
        let bytes = postcard::to_stdvec(&self.inner)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let cls = py.get_type::<Self>();
        let py_bytes = pyo3::types::PyBytes::new(py, &bytes);
        Ok((cls.into_any(), (py_bytes,)))
    }

    /// Construct from serialised bytes (used by pickle and for manual persistence).
    #[new]
    fn new(data: &[u8]) -> PyResult<Self> {
        let inner: Preconditioner = postcard::from_bytes(data).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "failed to deserialize preconditioner: {}",
                e
            ))
        })?;
        Ok(Self { inner })
    }
}

/// If the Python preconditioner argument is a pre-built `Preconditioner`,
/// return a clone of the inner native value. Otherwise `None`.
pub(crate) fn extract_prebuilt(
    preconditioner: Option<&Bound<'_, PyAny>>,
) -> Option<Preconditioner> {
    let obj = preconditioner?;
    obj.downcast::<PyPreconditioner>()
        .ok()
        .map(|b| b.get().inner.clone())
}

// ---------------------------------------------------------------------------
// Persistent Solver
// ---------------------------------------------------------------------------

/// Persistent solver that reuses preconditioners across multiple solves.
///
/// Build once with `Solver(categories, ...)`, then call `solve()` or
/// `solve_batch()` repeatedly. The expensive preconditioner factorization
/// happens only at construction time.
#[pyclass(frozen, module = "within._within")]
#[pyo3(name = "Solver")]
pub struct PySolver {
    solver: Solver<FactorMajorStore>,
}

#[pymethods]
impl PySolver {
    #[new]
    #[pyo3(signature = (categories, weights=None, preconditioner=None))]
    fn new<'py>(
        py: Python<'py>,
        categories: PyReadonlyArray2<'py, u32>,
        weights: Option<PyReadonlyArray1<'py, f64>>,
        preconditioner: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Self> {
        let cats = categories.as_array();
        warn_c_contiguous(py, cats.strides())?;

        // Build owned factor-major store from numpy array
        let n_obs = cats.nrows();
        let n_factors = cats.ncols();
        let factor_levels: Vec<Vec<u32>> = (0..n_factors)
            .map(|f| cats.column(f).iter().copied().collect())
            .collect();
        let store = FactorMajorStore::new(factor_levels, n_obs).map_err(value_err)?;
        let design = Design::from_store(store).map_err(value_err)?;
        let weights_vec: Option<Vec<f64>> = weights
            .as_ref()
            .map(|w| w.as_array().iter().copied().collect());

        // Pre-built Preconditioner uses the reuse path;
        // all other variants go through extract_preconditioner_config.
        let solver = if let Some(built) = extract_prebuilt(preconditioner) {
            py.allow_threads(|| Solver::new(design, weights_vec, built))
                .map_err(value_err)?
        } else {
            let precond = extract_preconditioner_config(py, preconditioner)?;
            py.allow_threads(|| Solver::new(design, weights_vec, precond.as_ref()))
                .map_err(value_err)?
        };

        Ok(Self { solver })
    }

    /// Solve for a single response vector with the given LSMR tuning.
    #[pyo3(name = "solve", signature = (y, options=None))]
    fn solve_py<'py>(
        &self,
        py: Python<'py>,
        y: PyReadonlyArray1<'py, f64>,
        options: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<PySolveResult> {
        let y_arr = y.as_array();
        let y_cow = coerce_to_slice(&y_arr);
        let params = resolve_lsmr_config(options)?;

        let result = py
            .allow_threads(|| self.solver.solve(&y_cow, &params))
            .map_err(value_err)?;

        Ok(into_py_result(py, result))
    }

    /// Solve for multiple response vectors in parallel.
    ///
    /// `Y` is a 2-D array of shape `(n_obs, k)` where each column is a
    /// separate response vector.
    #[pyo3(name = "solve_batch", signature = (Y, options=None))]
    fn solve_batch_py<'py>(
        &self,
        py: Python<'py>,
        #[allow(non_snake_case)] Y: PyReadonlyArray2<'py, f64>,
        options: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<PyBatchSolveResult> {
        let y_arr = Y.as_array();

        let n_obs = self.solver.n_obs();
        if y_arr.nrows() != n_obs {
            return Err(value_err(format!(
                "Y has {} rows but solver has {} observations",
                y_arr.nrows(),
                n_obs
            )));
        }

        let columns = extract_columns(&y_arr);
        let col_refs = column_refs(&columns);

        let n_dofs = self.solver.n_dofs();
        let params = resolve_lsmr_config(options)?;

        let result = py
            .allow_threads(|| self.solver.solve_batch(&col_refs, &params))
            .map_err(value_err)?;

        into_py_batch_result(py, result, n_dofs, n_obs)
    }

    /// Compute a one-shot approximate solution by applying the cached
    /// Schwarz preconditioner once, without LSMR correction iterations.
    #[pyo3(name = "solve_approx_parallel", signature = (y, options=None))]
    fn solve_approx_parallel_py<'py>(
        &self,
        py: Python<'py>,
        y: PyReadonlyArray1<'py, f64>,
        options: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<PySolveResult> {
        let y_arr = y.as_array();
        let y_cow = coerce_to_slice(&y_arr);
        let params = resolve_lsmr_config(options)?;

        let result = py
            .allow_threads(|| self.solver.solve_approx_parallel(&y_cow, params.tol))
            .map_err(value_err)?;

        Ok(into_py_result(py, result))
    }

    /// Compute one-shot approximate solutions for multiple response vectors.
    #[pyo3(name = "solve_approx_parallel_batch", signature = (Y, options=None))]
    fn solve_approx_parallel_batch_py<'py>(
        &self,
        py: Python<'py>,
        #[allow(non_snake_case)] Y: PyReadonlyArray2<'py, f64>,
        options: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<PyBatchSolveResult> {
        let y_arr = Y.as_array();

        let n_obs = self.solver.n_obs();
        if y_arr.nrows() != n_obs {
            return Err(value_err(format!(
                "Y has {} rows but solver has {} observations",
                y_arr.nrows(),
                n_obs
            )));
        }

        let columns = extract_columns(&y_arr);
        let col_refs = column_refs(&columns);

        let n_dofs = self.solver.n_dofs();
        let params = resolve_lsmr_config(options)?;

        let result = py
            .allow_threads(|| {
                self.solver
                    .solve_approx_parallel_batch(&col_refs, params.tol)
            })
            .map_err(value_err)?;

        into_py_batch_result(py, result, n_dofs, n_obs)
    }

    /// Return the built preconditioner, or ``None`` if unconfigured.
    ///
    /// The returned object is picklable and can be passed to a new
    /// ``Solver(…, preconditioner=p)`` to skip the expensive build step.
    #[getter]
    #[pyo3(name = "preconditioner")]
    fn preconditioner_py(&self) -> PyResult<Option<PyPreconditioner>> {
        match self.solver.preconditioner() {
            None => Ok(None),
            Some(p) => Ok(Some(PyPreconditioner { inner: p.clone() })),
        }
    }

    /// Number of DOFs (coefficients) in the model.
    #[getter]
    fn n_dofs(&self) -> usize {
        self.solver.n_dofs()
    }

    /// Number of observations.
    #[getter]
    fn n_obs(&self) -> usize {
        self.solver.n_obs()
    }
}
