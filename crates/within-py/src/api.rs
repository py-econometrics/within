//! Free `#[pyfunction]`s exposed via `within._within`: [`solve`] and
//! [`solve_batch`].

use std::time::Instant;

use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArray};
use pyo3::prelude::*;

use within::{
    BatchSolveResult, BuildError, BuildWarning, Design, Effect, IntoDesign, LsmrOptions,
    PreconditionerInput, SolveResult, Solver, WithinError,
};

use crate::config::{resolve_lsmr_config, resolve_precond_input, PyPreconditioner};
use crate::convert::{
    coerce_to_slice, column_refs, extract_columns, readonly_f64_1d, readonly_f64_2d,
    readonly_u32_2d, value_err, warn_c_contiguous,
};
use crate::results::{
    emit_build_warnings, run_batch, run_batch_with_warnings, run_solve, run_solve_with_warnings,
    PyBatchSolveResult, PySolveResult,
};

/// Build a one-shot solver, run the solve (mirroring [`within::solve`]'s
/// timing), and hand back the build warnings so the caller can re-emit them.
fn build_and_solve<'a>(
    design: impl IntoDesign<'a>,
    y: &[f64],
    weights: Option<&[f64]>,
    lsmr: &LsmrOptions,
    precond: impl Into<PreconditionerInput>,
) -> Result<(SolveResult, Vec<BuildWarning>), WithinError> {
    let t_start = Instant::now();
    let solver = Solver::new(design, weights.map(|w| w.to_vec()), precond)?;
    let time_setup = t_start.elapsed().as_secs_f64();
    let mut result = solver.solve(y, lsmr)?;
    result.time_setup += time_setup;
    result.time_total = t_start.elapsed().as_secs_f64();
    Ok((result, solver.warnings().to_vec()))
}

/// Batch counterpart to [`build_and_solve`], mirroring [`within::solve_batch`].
fn build_and_solve_batch<'a>(
    design: impl IntoDesign<'a>,
    ys: &[&[f64]],
    weights: Option<&[f64]>,
    lsmr: &LsmrOptions,
    precond: impl Into<PreconditionerInput>,
) -> Result<(BatchSolveResult, Vec<BuildWarning>), WithinError> {
    let t_start = Instant::now();
    let solver = Solver::new(design, weights.map(|w| w.to_vec()), precond)?;
    let mut result = solver.solve_batch(ys, lsmr)?;
    result.time_total = t_start.elapsed().as_secs_f64();
    Ok((result, solver.warnings().to_vec()))
}

// ---------------------------------------------------------------------------
// Public solve functions
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (design, y, weights=None, options=None, preconditioner=None))]
pub fn solve<'py>(
    py: Python<'py>,
    design: &Bound<'py, PyAny>,
    y: &Bound<'py, PyAny>,
    weights: Option<&Bound<'py, PyAny>>,
    options: Option<&Bound<'py, PyAny>>,
    preconditioner: Option<&Bound<'py, PyAny>>,
) -> PyResult<PySolveResult> {
    let params = resolve_lsmr_config(options)?;
    let precond = resolve_precond_input(py, preconditioner)?;
    let y = readonly_f64_1d("y", y)?;
    let weights = weights.map(|w| readonly_f64_1d("weights", w)).transpose()?;

    // Borrow the array views while the GIL is held (`PyReadonlyArray` needs the
    // token), but defer slice coercion -- and any copy of strided input -- into
    // the GIL-released closures below, so the F-contiguous path copies nothing.
    let y_arr = y.as_array();
    let w_view = weights.as_ref().map(|w| w.as_array());

    match extract_design(py, design)? {
        DesignSource::Categories(categories) => {
            let cats = categories.as_array();
            run_solve_with_warnings(py, move || {
                let y_cow = coerce_to_slice(&y_arr);
                let w_cow = w_view.as_ref().map(coerce_to_slice);
                build_and_solve(cats, &y_cow, w_cow.as_deref(), &params, precond)
            })
        }
        DesignSource::Effects(terms) => run_solve_with_warnings(py, move || {
            let effects: Vec<_> = terms.iter().map(PyEffect::as_effect).collect();
            let y_cow = coerce_to_slice(&y_arr);
            let w_cow = w_view.as_ref().map(coerce_to_slice);
            build_and_solve(effects, &y_cow, w_cow.as_deref(), &params, precond)
        }),
    }
}

#[pyfunction]
#[pyo3(signature = (design, Y, weights=None, options=None, preconditioner=None))]
pub fn solve_batch<'py>(
    py: Python<'py>,
    design: &Bound<'py, PyAny>,
    #[allow(non_snake_case)] Y: &Bound<'py, PyAny>,
    weights: Option<&Bound<'py, PyAny>>,
    options: Option<&Bound<'py, PyAny>>,
    preconditioner: Option<&Bound<'py, PyAny>>,
) -> PyResult<PyBatchSolveResult> {
    let params = resolve_lsmr_config(options)?;
    let precond = resolve_precond_input(py, preconditioner)?;
    let y = readonly_f64_2d("Y", Y)?;
    let weights = weights.map(|w| readonly_f64_1d("weights", w)).transpose()?;

    let y_arr = y.as_array();
    let w_view = weights.as_ref().map(|w| w.as_array());

    match extract_design(py, design)? {
        DesignSource::Categories(categories) => {
            let cats = categories.as_array();
            warn_c_contiguous(py, &cats)?;
            validate_batch_rows(y_arr.nrows(), cats.nrows())?;
            run_batch_with_warnings(py, move || {
                let columns = extract_columns(&y_arr);
                let col_refs = column_refs(&columns);
                let w_cow = w_view.as_ref().map(coerce_to_slice);
                build_and_solve_batch(cats, &col_refs, w_cow.as_deref(), &params, precond)
            })
        }
        DesignSource::Effects(terms) => {
            if let Some(first) = terms.first() {
                validate_batch_rows(y_arr.nrows(), first.levels.len())?;
            }
            run_batch_with_warnings(py, move || {
                let effects: Vec<_> = terms.iter().map(PyEffect::as_effect).collect();
                let columns = extract_columns(&y_arr);
                let col_refs = column_refs(&columns);
                let w_cow = w_view.as_ref().map(coerce_to_slice);
                build_and_solve_batch(effects, &col_refs, w_cow.as_deref(), &params, precond)
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
        let categories = readonly_u32_2d("design", design)?;
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
        weights: Option<&Bound<'py, PyAny>>,
        preconditioner: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Self> {
        let weights = weights.map(|w| readonly_f64_1d("weights", w)).transpose()?;
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

        emit_build_warnings(py, solver.warnings())?;
        Ok(Self { solver })
    }

    /// Solve for a single response vector with the given LSMR tuning.
    #[pyo3(name = "solve", signature = (y, options=None))]
    fn solve_py<'py>(
        &self,
        py: Python<'py>,
        y: &Bound<'py, PyAny>,
        options: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<PySolveResult> {
        let y = readonly_f64_1d("y", y)?;
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
        #[allow(non_snake_case)] Y: &Bound<'py, PyAny>,
        options: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<PyBatchSolveResult> {
        let y = readonly_f64_2d("Y", Y)?;
        let y_arr = y.as_array();

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
