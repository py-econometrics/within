//! Python API: typed config classes and solve entrypoint.

use numpy::ndarray::{Array1, ArrayView2};
use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use within::config::{
    ApproxSchurConfig, GmresPrecond, LocalSolverConfig, OperatorRepr, SolverMethod, SolverParams,
};
use within::{solve as solve_native, SolveResult};

#[pyclass(frozen)]
#[pyo3(name = "ApproxCholConfig")]
pub struct PyApproxCholConfig {
    #[pyo3(get)]
    pub seed: u64,
    #[pyo3(get)]
    pub split: u32,
}

#[pymethods]
impl PyApproxCholConfig {
    #[new]
    #[pyo3(signature = (seed=0, split=1))]
    fn new(seed: u64, split: u32) -> PyResult<Self> {
        if split == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "split must be >= 1",
            ));
        }
        Ok(Self { seed, split })
    }
}

impl PyApproxCholConfig {
    fn to_native(&self) -> approx_chol::Config {
        let split_merge = if self.split > 1 {
            Some(self.split)
        } else {
            None
        };
        approx_chol::Config {
            seed: self.seed,
            split_merge,
        }
    }
}

#[pyclass(frozen)]
#[pyo3(name = "ApproxSchurConfig")]
pub struct PyApproxSchurConfig {
    #[pyo3(get)]
    pub seed: u64,
}

#[pymethods]
impl PyApproxSchurConfig {
    #[new]
    #[pyo3(signature = (seed=0))]
    fn new(seed: u64) -> Self {
        Self { seed }
    }
}

impl PyApproxSchurConfig {
    fn to_native(&self) -> ApproxSchurConfig {
        ApproxSchurConfig { seed: self.seed }
    }
}

// ---------------------------------------------------------------------------
// OperatorRepr enum
// ---------------------------------------------------------------------------

#[pyclass(frozen, eq, eq_int)]
#[pyo3(name = "OperatorRepr")]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyOperatorRepr {
    Implicit = 0,
    Explicit = 1,
}

impl PyOperatorRepr {
    fn to_native(self) -> OperatorRepr {
        match self {
            PyOperatorRepr::Implicit => OperatorRepr::Implicit,
            PyOperatorRepr::Explicit => OperatorRepr::Explicit,
        }
    }
}

// ---------------------------------------------------------------------------
// Local solver config classes
// ---------------------------------------------------------------------------

#[pyclass(frozen)]
#[pyo3(name = "SchurComplement")]
pub struct PySchurComplement {
    #[pyo3(get)]
    pub approx_chol: Option<Py<PyApproxCholConfig>>,
    #[pyo3(get)]
    pub approx_schur: Option<Py<PyApproxSchurConfig>>,
    #[pyo3(get)]
    pub dense_threshold: usize,
}

#[pymethods]
impl PySchurComplement {
    #[new]
    #[pyo3(signature = (approx_chol=None, approx_schur=None, dense_threshold=24))]
    fn new(
        approx_chol: Option<Py<PyApproxCholConfig>>,
        approx_schur: Option<Py<PyApproxSchurConfig>>,
        dense_threshold: usize,
    ) -> Self {
        Self {
            approx_chol,
            approx_schur,
            dense_threshold,
        }
    }
}

#[pyclass(frozen)]
#[pyo3(name = "FullSddm")]
pub struct PyFullSddm {
    #[pyo3(get)]
    pub approx_chol: Option<Py<PyApproxCholConfig>>,
}

#[pymethods]
impl PyFullSddm {
    #[new]
    #[pyo3(signature = (approx_chol=None))]
    fn new(approx_chol: Option<Py<PyApproxCholConfig>>) -> Self {
        Self { approx_chol }
    }
}

// ---------------------------------------------------------------------------
// Schwarz preconditioner classes
// ---------------------------------------------------------------------------

#[pyclass(frozen)]
#[pyo3(name = "AdditiveSchwarz")]
pub struct PyAdditiveSchwarz {
    #[pyo3(get)]
    pub local_solver: Option<PyObject>,
}

#[pymethods]
impl PyAdditiveSchwarz {
    #[new]
    #[pyo3(signature = (local_solver=None))]
    fn new(local_solver: Option<PyObject>) -> Self {
        Self { local_solver }
    }
}

#[pyclass(frozen)]
#[pyo3(name = "MultiplicativeSchwarz")]
pub struct PyMultiplicativeSchwarz {
    #[pyo3(get)]
    pub local_solver: Option<PyObject>,
}

#[pymethods]
impl PyMultiplicativeSchwarz {
    #[new]
    #[pyo3(signature = (local_solver=None))]
    fn new(local_solver: Option<PyObject>) -> Self {
        Self { local_solver }
    }
}

// ---------------------------------------------------------------------------
// Solver config classes
// ---------------------------------------------------------------------------

#[pyclass(frozen)]
#[pyo3(name = "CG")]
pub struct PyCG {
    #[pyo3(get)]
    pub tol: f64,
    #[pyo3(get)]
    pub maxiter: usize,
    #[pyo3(get)]
    pub preconditioner: Option<PyObject>,
    #[pyo3(get)]
    pub operator: PyOperatorRepr,
}

#[pymethods]
impl PyCG {
    #[new]
    #[pyo3(signature = (tol=1e-8, maxiter=1000, preconditioner=None, operator=PyOperatorRepr::Implicit))]
    fn new(
        tol: f64,
        maxiter: usize,
        preconditioner: Option<PyObject>,
        operator: PyOperatorRepr,
    ) -> Self {
        Self {
            tol,
            maxiter,
            preconditioner,
            operator,
        }
    }
}

#[pyclass(frozen)]
#[pyo3(name = "GMRES")]
pub struct PyGMRES {
    #[pyo3(get)]
    pub tol: f64,
    #[pyo3(get)]
    pub maxiter: usize,
    #[pyo3(get)]
    pub restart: usize,
    #[pyo3(get)]
    pub preconditioner: Option<PyObject>,
    #[pyo3(get)]
    pub operator: PyOperatorRepr,
}

#[pymethods]
impl PyGMRES {
    #[new]
    #[pyo3(signature = (tol=1e-8, maxiter=1000, restart=30, preconditioner=None, operator=PyOperatorRepr::Implicit))]
    fn new(
        tol: f64,
        maxiter: usize,
        restart: usize,
        preconditioner: Option<PyObject>,
        operator: PyOperatorRepr,
    ) -> Self {
        Self {
            tol,
            maxiter,
            restart,
            preconditioner,
            operator,
        }
    }
}

#[pyclass]
#[pyo3(name = "SolveResult")]
pub struct PySolveResult {
    #[pyo3(get)]
    pub x: Py<numpy::PyArray1<f64>>,
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

// ---------------------------------------------------------------------------
// Extraction helpers
// ---------------------------------------------------------------------------

/// Extract a `LocalSolverConfig` from a Python `SchurComplement` or `FullSddm` object.
fn extract_local_solver_config(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
) -> PyResult<LocalSolverConfig> {
    if let Ok(sc) = obj.downcast::<PySchurComplement>() {
        let sc = sc.get();
        let approx_chol = sc
            .approx_chol
            .as_ref()
            .map(|c| c.bind(py).get().to_native())
            .unwrap_or_default();
        let approx_schur = sc
            .approx_schur
            .as_ref()
            .map(|c| c.bind(py).get().to_native());
        Ok(LocalSolverConfig::SchurComplement {
            approx_chol,
            approx_schur,
            dense_threshold: sc.dense_threshold,
        })
    } else if let Ok(fd) = obj.downcast::<PyFullSddm>() {
        let fd = fd.get();
        let approx_chol = fd
            .approx_chol
            .as_ref()
            .map(|c| c.bind(py).get().to_native())
            .unwrap_or_default();
        Ok(LocalSolverConfig::FullSddm { approx_chol })
    } else {
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "local_solver must be SchurComplement, FullSddm, or None",
        ))
    }
}

/// Extract `LocalSolverConfig` from the `local_solver` field of a Schwarz class,
/// falling back to the given default when the field is `None`.
fn extract_local_solver_or_default(
    py: Python<'_>,
    local_solver: &Option<PyObject>,
    default: LocalSolverConfig,
) -> PyResult<LocalSolverConfig> {
    match local_solver {
        None => Ok(default),
        Some(obj) => extract_local_solver_config(py, obj.bind(py)),
    }
}

fn extract_additive_schwarz(
    py: Python<'_>,
    preconditioner: &Option<PyObject>,
    default: LocalSolverConfig,
    context: &str,
) -> PyResult<Option<LocalSolverConfig>> {
    match preconditioner {
        None => Ok(None),
        Some(obj) => {
            let bound = obj.bind(py);
            let schwarz = bound.downcast::<PyAdditiveSchwarz>().map_err(|_| {
                PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                    "{context} preconditioner must be AdditiveSchwarz or None"
                ))
            })?;
            let cfg = extract_local_solver_or_default(py, &schwarz.get().local_solver, default)?;
            Ok(Some(cfg))
        }
    }
}

fn extract_gmres_preconditioner(
    py: Python<'_>,
    preconditioner: &Option<PyObject>,
) -> PyResult<Option<GmresPrecond>> {
    match preconditioner {
        None => Ok(None),
        Some(obj) => {
            let bound = obj.bind(py);
            if let Ok(schwarz) = bound.downcast::<PyAdditiveSchwarz>() {
                let cfg = extract_local_solver_or_default(
                    py,
                    &schwarz.get().local_solver,
                    LocalSolverConfig::gmres_default(),
                )?;
                Ok(Some(GmresPrecond::Additive(cfg)))
            } else if let Ok(schwarz) = bound.downcast::<PyMultiplicativeSchwarz>() {
                let cfg = extract_local_solver_or_default(
                    py,
                    &schwarz.get().local_solver,
                    LocalSolverConfig::gmres_default(),
                )?;
                Ok(Some(GmresPrecond::Multiplicative(cfg)))
            } else {
                Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "GMRES preconditioner must be AdditiveSchwarz, MultiplicativeSchwarz, or None",
                ))
            }
        }
    }
}

/// Extract solver parameters from a Python config object (CG or GMRES).
fn extract_solver_params(py: Python<'_>, config: &Bound<'_, PyAny>) -> PyResult<SolverParams> {
    if let Ok(cg) = config.downcast::<PyCG>() {
        let cg = cg.get();
        let preconditioner = extract_additive_schwarz(
            py,
            &cg.preconditioner,
            LocalSolverConfig::cg_default(),
            "CG",
        )?;
        return Ok(SolverParams {
            method: SolverMethod::Cg {
                preconditioner,
                operator: cg.operator.to_native(),
            },
            tol: cg.tol,
            maxiter: cg.maxiter,
        });
    }

    if let Ok(gmres) = config.downcast::<PyGMRES>() {
        let gmres = gmres.get();
        let preconditioner = extract_gmres_preconditioner(py, &gmres.preconditioner)?;
        return Ok(SolverParams {
            method: SolverMethod::Gmres {
                preconditioner,
                operator: gmres.operator.to_native(),
                restart: gmres.restart,
            },
            tol: gmres.tol,
            maxiter: gmres.maxiter,
        });
    }

    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
        "config must be CG or GMRES",
    ))
}

fn categories_to_factor_major(categories: ArrayView2<'_, usize>) -> PyResult<Vec<Vec<u32>>> {
    let n_rows = categories.nrows();
    let mut factor_major = Vec::with_capacity(categories.ncols());
    for factor in 0..categories.ncols() {
        let mut levels = Vec::with_capacity(n_rows);
        for (observation, &level) in categories.column(factor).iter().enumerate() {
            let level = u32::try_from(level).map_err(|_| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "category out of range for u32 at factor {} observation {}",
                    factor, observation
                ))
            })?;
            levels.push(level);
        }
        factor_major.push(levels);
    }
    Ok(factor_major)
}

#[pyfunction]
#[pyo3(signature = (categories, y, config=None, n_levels=None, weights=None))]
pub fn solve<'py>(
    py: Python<'py>,
    categories: PyReadonlyArray2<'py, usize>,
    y: PyReadonlyArray1<'py, f64>,
    config: Option<&Bound<'py, PyAny>>,
    n_levels: Option<Vec<usize>>,
    weights: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<PySolveResult> {
    let cats = categories.as_array();
    let y_vec: Vec<f64> = y.as_array().to_vec();
    let w_vec: Option<Vec<f64>> = weights.map(|w| w.as_array().to_vec());
    let factor_levels = categories_to_factor_major(cats)?;

    let n_levels = match n_levels {
        Some(nl) => nl,
        None => {
            let n_factors = cats.ncols();
            (0..n_factors)
                .map(|col| cats.column(col).iter().copied().max().unwrap_or(0) + 1)
                .collect()
        }
    };

    let params = match config {
        Some(config) => extract_solver_params(py, config)?,
        None => SolverParams::default(),
    };

    let result = py
        .allow_threads(|| {
            solve_native(&factor_levels, &n_levels, &y_vec, w_vec.as_deref(), &params)
        })
        .map_err(|err| PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string()))?;

    Ok(into_py_result(py, result))
}

fn into_py_result(py: Python<'_>, result: SolveResult) -> PySolveResult {
    PySolveResult {
        x: Array1::from_vec(result.x).into_pyarray(py).unbind(),
        converged: result.converged,
        iterations: result.iterations,
        residual: result.final_residual,
        time_total: result.time_total,
        time_setup: result.time_setup,
        time_solve: result.time_solve,
    }
}
