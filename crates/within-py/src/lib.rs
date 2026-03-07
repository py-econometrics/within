//! Python API: typed config classes and solve entrypoint.

use numpy::ndarray::{Array1, ArrayView2};
use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use within::config::{
    ApproxSchurConfig, KrylovMethod, LocalSolverConfig, OperatorRepr, Preconditioner, SolverParams,
    DEFAULT_DENSE_SCHUR_THRESHOLD,
};
use within::{solve as solve_native, SolveResult};

// ---------------------------------------------------------------------------
// Low-level config classes (available via `_within` for benchmarks)
// ---------------------------------------------------------------------------

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
    #[pyo3(get)]
    pub split: u32,
}

#[pymethods]
impl PyApproxSchurConfig {
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

impl PyApproxSchurConfig {
    fn to_native(&self) -> ApproxSchurConfig {
        ApproxSchurConfig {
            seed: self.seed,
            split: self.split,
        }
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
// Preconditioner enum (public API)
// ---------------------------------------------------------------------------

/// Preconditioner selection for CG / GMRES solvers.
///
/// - ``Preconditioner.Additive`` — additive Schwarz (default, symmetric)
/// - ``Preconditioner.Multiplicative`` — multiplicative Schwarz (GMRES only)
/// - ``Preconditioner.Off`` — no preconditioner
#[pyclass(frozen, eq, eq_int)]
#[pyo3(name = "Preconditioner")]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyPreconditioner {
    Additive = 0,
    Multiplicative = 1,
    Off = 2,
}

// ---------------------------------------------------------------------------
// Local solver config classes (available via `_within` for benchmarks)
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
    #[pyo3(signature = (approx_chol=None, approx_schur=None, dense_threshold=None))]
    fn new(
        approx_chol: Option<Py<PyApproxCholConfig>>,
        approx_schur: Option<Py<PyApproxSchurConfig>>,
        dense_threshold: Option<usize>,
    ) -> Self {
        Self {
            approx_chol,
            approx_schur,
            dense_threshold: dense_threshold.unwrap_or(DEFAULT_DENSE_SCHUR_THRESHOLD),
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
// Schwarz preconditioner classes (available via `_within` for benchmarks)
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
    /// Create a CG solver configuration.
    ///
    /// `preconditioner` accepts:
    /// - `None` (default) — additive Schwarz with default local solver
    /// - `Preconditioner.Additive` — same as None (explicit)
    /// - `Preconditioner.Off` — unpreconditioned CG
    /// - `AdditiveSchwarz(...)` — advanced: fine-grained local solver config
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
    /// Create a GMRES solver configuration.
    ///
    /// `preconditioner` accepts:
    /// - `None` (default) — additive Schwarz with default local solver
    /// - `Preconditioner.Additive` — same as None (explicit)
    /// - `Preconditioner.Multiplicative` — multiplicative Schwarz with default local solver
    /// - `Preconditioner.Off` — unpreconditioned GMRES
    /// - `AdditiveSchwarz(...)` / `MultiplicativeSchwarz(...)` — advanced config
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

/// Extract a `Preconditioner` from a Python object.
///
/// Accepts:
/// - `PyPreconditioner` enum variant (Additive / Multiplicative)
/// - `AdditiveSchwarz(...)` / `MultiplicativeSchwarz(...)` — advanced config
fn extract_preconditioner(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Preconditioner> {
    // Enum shorthand
    if let Ok(p) = obj.extract::<PyPreconditioner>() {
        return match p {
            PyPreconditioner::Additive => {
                Ok(Preconditioner::Additive(LocalSolverConfig::solver_default()))
            }
            PyPreconditioner::Multiplicative => Ok(Preconditioner::Multiplicative(
                LocalSolverConfig::solver_default(),
            )),
            PyPreconditioner::Off => {
                // Shouldn't reach here (handled by caller), but be safe
                Ok(Preconditioner::Additive(LocalSolverConfig::solver_default()))
            }
        };
    }
    // Advanced: AdditiveSchwarz / MultiplicativeSchwarz objects
    if let Ok(schwarz) = obj.downcast::<PyAdditiveSchwarz>() {
        let cfg = extract_local_solver_or_default(
            py,
            &schwarz.get().local_solver,
            LocalSolverConfig::solver_default(),
        )?;
        return Ok(Preconditioner::Additive(cfg));
    }
    if let Ok(schwarz) = obj.downcast::<PyMultiplicativeSchwarz>() {
        let cfg = extract_local_solver_or_default(
            py,
            &schwarz.get().local_solver,
            LocalSolverConfig::solver_default(),
        )?;
        return Ok(Preconditioner::Multiplicative(cfg));
    }
    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
        "preconditioner must be Preconditioner.Additive, Preconditioner.Multiplicative, \
         Preconditioner.Off, AdditiveSchwarz(...), MultiplicativeSchwarz(...), or None",
    ))
}

/// Extract `Option<Preconditioner>` from the `preconditioner` field of CG/GMRES.
///
/// - `None` → additive Schwarz with default local solver
/// - `Preconditioner.Off` → unpreconditioned (returns `Ok(None)`)
/// - `Preconditioner.Additive` → additive Schwarz with default local solver
/// - `Preconditioner.Multiplicative` → multiplicative Schwarz with default local solver
/// - `AdditiveSchwarz(...)` / `MultiplicativeSchwarz(...)` → advanced config
fn extract_optional_preconditioner(
    py: Python<'_>,
    field: &Option<PyObject>,
) -> PyResult<Option<Preconditioner>> {
    match field {
        // None → default: additive Schwarz
        None => Ok(Some(Preconditioner::Additive(
            LocalSolverConfig::solver_default(),
        ))),
        Some(obj) => {
            let bound = obj.bind(py);
            // Preconditioner.Off → unpreconditioned
            if matches!(
                bound.extract::<PyPreconditioner>(),
                Ok(PyPreconditioner::Off)
            ) {
                return Ok(None);
            }
            extract_preconditioner(py, bound).map(Some)
        }
    }
}

/// Extract solver parameters from a Python config object (CG or GMRES).
fn extract_solver_params(py: Python<'_>, config: &Bound<'_, PyAny>) -> PyResult<SolverParams> {
    if let Ok(cg) = config.downcast::<PyCG>() {
        let cg = cg.get();
        let preconditioner = extract_optional_preconditioner(py, &cg.preconditioner)?;
        if matches!(preconditioner, Some(Preconditioner::Multiplicative(_))) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "CG requires a symmetric preconditioner; use 'additive' or switch to GMRES",
            ));
        }
        return Ok(SolverParams {
            krylov: KrylovMethod::Cg,
            operator: cg.operator.to_native(),
            preconditioner,
            tol: cg.tol,
            maxiter: cg.maxiter,
        });
    }

    if let Ok(gmres) = config.downcast::<PyGMRES>() {
        let gmres = gmres.get();
        let preconditioner = extract_optional_preconditioner(py, &gmres.preconditioner)?;
        return Ok(SolverParams {
            krylov: KrylovMethod::Gmres {
                restart: gmres.restart,
            },
            operator: gmres.operator.to_native(),
            preconditioner,
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

#[pymodule]
fn _within(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySolveResult>()?;
    m.add_class::<PyCG>()?;
    m.add_class::<PyGMRES>()?;
    m.add_class::<PyAdditiveSchwarz>()?;
    m.add_class::<PyMultiplicativeSchwarz>()?;
    m.add_class::<PyOperatorRepr>()?;
    m.add_class::<PyPreconditioner>()?;
    m.add_class::<PyApproxCholConfig>()?;
    m.add_class::<PyApproxSchurConfig>()?;
    m.add_class::<PySchurComplement>()?;
    m.add_class::<PyFullSddm>()?;
    m.add_function(wrap_pyfunction!(solve, m)?)?;
    Ok(())
}
