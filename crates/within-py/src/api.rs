//! Free `#[pyfunction]`s exposed via `within._within`: [`solve`] and
//! [`solve_batch`].

use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use within::observation::FactorMajorStore;
use within::{
    solve as solve_native, solve_batch as solve_batch_native, BuildError, Design, SolveResult,
    Solver, WithinError,
};

use crate::config::{resolve_lsmr_config, resolve_precond_input, PrecondInput, PyPreconditioner};
use crate::convert::{coerce_to_slice, column_refs, extract_columns, value_err, warn_c_contiguous};
use crate::results::{run_batch, run_solve, PyBatchSolveResult, PySolveResult};

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
        warn_c_contiguous(py, &cats)?;

        let n_obs = cats.nrows();
        let n_factors = cats.ncols();
        let weights_vec: Option<Vec<f64>> = weights
            .as_ref()
            .map(|w| w.as_array().iter().copied().collect());

        // Resolve the preconditioner argument while the GIL is held (it inspects
        // Python objects); the result carries only native data.
        let precond = resolve_precond_input(py, preconditioner)?;

        // Release the GIL for the CPU-heavy work: the factor-major copy out of
        // the numpy array, the store/design construction, and the preconditioner
        // factorisation. The `BuildError` carries no GIL types, so it is mapped
        // to a Python exception only after the GIL is reacquired.
        let solver = py
            .allow_threads(move || -> Result<Solver<FactorMajorStore>, BuildError> {
                let factor_levels: Vec<Vec<u32>> = (0..n_factors)
                    .map(|f| cats.column(f).iter().copied().collect())
                    .collect();
                let store = FactorMajorStore::new(factor_levels, n_obs)?;
                let design = Design::from_store(store)?;
                match precond {
                    PrecondInput::Prebuilt(built) => Solver::new(design, weights_vec, built),
                    PrecondInput::Config(config) => {
                        Solver::new(design, weights_vec, config.as_ref())
                    }
                }
            })
            .map_err(value_err)?;

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

        run_solve(py, || self.solver.solve(&y_cow, &params))
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

        let params = resolve_lsmr_config(options)?;

        run_batch(py, || self.solver.solve_batch(&col_refs, &params))
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
