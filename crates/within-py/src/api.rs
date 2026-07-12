//! Free `#[pyfunction]`s exposed via `within._within`: [`solve`] and
//! [`solve_batch`].

use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArray};
use pyo3::prelude::*;

use within::{
    solve as solve_native, solve_batch as solve_batch_native, BuildError, Design, Effect,
    IntoDesign, SolveResult, Solver, WithinError,
};

use crate::config::{resolve_lsmr_config, resolve_precond_input, PyPreconditioner};
use crate::convert::{coerce_to_slice, column_refs, extract_columns, value_err, warn_c_contiguous};
use crate::results::{run_batch, run_solve, PyBatchSolveResult, PySolveResult};

// ---------------------------------------------------------------------------
// Public solve functions
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (design, y, options=None, weights=None, preconditioner=None))]
pub fn solve<'py>(
    py: Python<'py>,
    design: &Bound<'py, PyAny>,
    y: PyReadonlyArray1<'py, f64>,
    options: Option<&Bound<'py, PyAny>>,
    weights: Option<PyReadonlyArray1<'py, f64>>,
    preconditioner: Option<&Bound<'py, PyAny>>,
) -> PyResult<PySolveResult> {
    let params = resolve_lsmr_config(options)?;
    let precond = resolve_precond_input(py, preconditioner)?;

    // Borrow the array views while the GIL is held (`PyReadonlyArray` needs the
    // token), but defer slice coercion -- and any copy of strided input -- into
    // the GIL-released closures below, so the F-contiguous path copies nothing.
    let y_arr = y.as_array();
    let w_view = weights.as_ref().map(|w| w.as_array());

    match extract_design(py, design)? {
        DesignSource::Categories(categories) => {
            let cats = categories.as_array();
            run_solve(py, move || -> Result<SolveResult, WithinError> {
                let y_cow = coerce_to_slice(&y_arr);
                let w_cow = w_view.as_ref().map(coerce_to_slice);
                solve_native(cats, &y_cow, w_cow.as_deref(), &params, precond)
            })
        }
        DesignSource::Effects(terms) => {
            run_solve(py, move || -> Result<SolveResult, WithinError> {
                let effects: Vec<_> = terms.iter().map(PyEffect::as_effect).collect();
                let y_cow = coerce_to_slice(&y_arr);
                let w_cow = w_view.as_ref().map(coerce_to_slice);
                solve_native(effects, &y_cow, w_cow.as_deref(), &params, precond)
            })
        }
    }
}

#[pyfunction]
#[pyo3(signature = (design, Y, options=None, weights=None, preconditioner=None))]
pub fn solve_batch<'py>(
    py: Python<'py>,
    design: &Bound<'py, PyAny>,
    #[allow(non_snake_case)] Y: PyReadonlyArray2<'py, f64>,
    options: Option<&Bound<'py, PyAny>>,
    weights: Option<PyReadonlyArray1<'py, f64>>,
    preconditioner: Option<&Bound<'py, PyAny>>,
) -> PyResult<PyBatchSolveResult> {
    let params = resolve_lsmr_config(options)?;
    let precond = resolve_precond_input(py, preconditioner)?;

    let y_arr = Y.as_array();
    let w_view = weights.as_ref().map(|w| w.as_array());

    match extract_design(py, design)? {
        DesignSource::Categories(categories) => {
            let cats = categories.as_array();
            warn_c_contiguous(py, &cats)?;
            validate_batch_rows(y_arr.nrows(), cats.nrows())?;
            run_batch(py, move || -> Result<_, WithinError> {
                let columns = extract_columns(&y_arr);
                let col_refs = column_refs(&columns);
                let w_cow = w_view.as_ref().map(coerce_to_slice);
                solve_batch_native(cats, &col_refs, w_cow.as_deref(), &params, precond)
            })
        }
        DesignSource::Effects(terms) => {
            if let Some(first) = terms.first() {
                validate_batch_rows(y_arr.nrows(), first.levels.len())?;
            }
            run_batch(py, move || -> Result<_, WithinError> {
                let effects: Vec<_> = terms.iter().map(PyEffect::as_effect).collect();
                let columns = extract_columns(&y_arr);
                let col_refs = column_refs(&columns);
                let w_cow = w_view.as_ref().map(coerce_to_slice);
                solve_batch_native(effects, &col_refs, w_cow.as_deref(), &params, precond)
            })
        }
    }
}

/// Validate `Y`'s row count up front: an empty batch (`Y.shape[1] == 0`) would
/// otherwise silently skip the per-column length check inside `Solver::solve`.
fn validate_batch_rows(y_rows: usize, n_obs: usize) -> PyResult<()> {
    if y_rows != n_obs {
        return Err(value_err(format!(
            "Y has {y_rows} rows but the design has {n_obs} observations"
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Effect-term design
// ---------------------------------------------------------------------------

/// One factor's effect: level codes, an optional intercept, and slope covariates.
///
/// Holds its columns natively (copied out of numpy) so the borrowed [`Effect`]
/// it lowers to can be rebuilt off-GIL, where `Py` handles can't reach.
#[pyclass(frozen, module = "within._within")]
#[pyo3(name = "Effect")]
#[derive(Clone)]
pub struct PyEffect {
    levels: Vec<u32>,
    intercept: bool,
    slopes: Vec<Vec<f64>>,
}

#[pymethods]
impl PyEffect {
    #[new]
    #[pyo3(signature = (levels, intercept, slopes=None))]
    fn new<'py>(
        levels: PyReadonlyArray1<'py, u32>,
        intercept: bool,
        slopes: Option<Vec<PyReadonlyArray1<'py, f64>>>,
    ) -> PyResult<Self> {
        let levels = levels.as_array().to_vec();
        let slopes: Vec<Vec<f64>> = slopes
            .unwrap_or_default()
            .iter()
            .map(|s| s.as_array().to_vec())
            .collect();
        // Validate through the native constructor so the rules live in one place.
        Effect::new(&levels, intercept, slopes.iter().map(Vec::as_slice)).map_err(value_err)?;
        Ok(Self {
            levels,
            intercept,
            slopes,
        })
    }
}

impl PyEffect {
    /// Rebuild the borrowed native [`Effect`]. Infallible: `PyEffect::new`
    /// already validated these exact columns through `Effect::new`.
    fn as_effect(&self) -> Effect<'_> {
        Effect::new(
            &self.levels,
            self.intercept,
            self.slopes.iter().map(Vec::as_slice),
        )
        .expect("PyEffect columns were validated at construction")
    }
}

/// A solve's design, as interpreted from the Python `design` argument.
enum DesignSource<'py> {
    /// An `(n_obs, n_factors)` categories matrix, borrowed from numpy.
    Categories(PyReadonlyArray2<'py, u32>),
    /// Effect terms, cloned out of Python so they can be rebuilt off-GIL.
    Effects(Vec<PyEffect>),
}

/// Interpret the Python `design` argument: a 2-D `uint32` categories matrix
/// (borrowed) or a list of [`Effect`] terms (cloned out of Python).
fn extract_design<'py>(py: Python<'_>, design: &Bound<'py, PyAny>) -> PyResult<DesignSource<'py>> {
    if design.downcast::<PyUntypedArray>().is_ok() {
        let categories = design.extract::<PyReadonlyArray2<u32>>()?;
        warn_c_contiguous(py, &categories.as_array())?;
        return Ok(DesignSource::Categories(categories));
    }
    let effects: Vec<Py<PyEffect>> = design
        .extract()
        .map_err(|_| value_err("design must be a 2-D uint32 array or a list of Effect"))?;
    Ok(DesignSource::Effects(
        effects
            .iter()
            .map(|e| e.bind(py).borrow().clone())
            .collect(),
    ))
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
    solver: Solver<'static>,
}

#[pymethods]
impl PySolver {
    #[new]
    #[pyo3(signature = (design, weights=None, preconditioner=None))]
    fn new<'py>(
        py: Python<'py>,
        design: &Bound<'py, PyAny>,
        weights: Option<PyReadonlyArray1<'py, f64>>,
        preconditioner: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Self> {
        let weights_vec: Option<Vec<f64>> = weights.as_ref().map(|w| w.as_array().to_vec());
        let precond = resolve_precond_input(py, preconditioner)?;

        // Release the GIL for the design copy/build and preconditioner
        // factorisation; `BuildError` carries no Python types, so it is mapped
        // to a Python exception only once the GIL is reacquired.
        let solver = match extract_design(py, design)? {
            DesignSource::Categories(categories) => {
                let cats = categories.as_array();
                py.allow_threads(move || -> Result<Solver<'static>, BuildError> {
                    Solver::new(cats.into_design()?.into_owned(), weights_vec, precond)
                })
            }
            DesignSource::Effects(terms) => {
                py.allow_threads(move || -> Result<Solver<'static>, BuildError> {
                    let effects: Vec<_> = terms.iter().map(PyEffect::as_effect).collect();
                    // The design borrows the terms' buffers; the solver outlives
                    // them, so lower to owned columns first.
                    let design = Design::new(effects)?.into_owned();
                    Solver::new(design, weights_vec, precond)
                })
            }
        }
        .map_err(value_err)?;

        for warning in solver.warnings() {
            let message = std::ffi::CString::new(warning.to_string())
                .expect("warning messages contain no NUL bytes");
            PyErr::warn(
                py,
                &py.get_type::<pyo3::exceptions::PyUserWarning>(),
                &message,
                1,
            )?;
        }
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
