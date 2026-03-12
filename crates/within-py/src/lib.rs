//! Python API: typed config classes, solve entrypoint, and persistent Solver.

use numpy::ndarray::{Array1, Array2, ShapeBuilder};
use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use within::config::{
    ApproxSchurConfig, KrylovMethod, LocalSolverConfig, OperatorRepr, Preconditioner,
    ReductionStrategy, SolverParams, DEFAULT_DENSE_SCHUR_THRESHOLD,
};
use within::domain::WeightedDesign;
use within::observation::{FactorMajorStore, ObservationWeights};
use within::operator::preconditioner::{
    additive_reduction_strategy, additive_schwarz_diagnostics, resolved_additive_reduction_strategy,
};
use within::{
    solve as solve_native, solve_batch as solve_batch_native, FePreconditioner, Operator,
    SolveResult, Solver,
};

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

#[pyclass(frozen, eq, eq_int)]
#[pyo3(name = "ReductionStrategy")]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyReductionStrategy {
    Auto = 0,
    AtomicScatter = 1,
    ParallelReduction = 2,
}

impl PyReductionStrategy {
    fn to_native(self) -> ReductionStrategy {
        match self {
            Self::Auto => ReductionStrategy::Auto,
            Self::AtomicScatter => ReductionStrategy::AtomicScatter,
            Self::ParallelReduction => ReductionStrategy::ParallelReduction,
        }
    }

    fn from_native(strategy: ReductionStrategy) -> Self {
        match strategy {
            ReductionStrategy::Auto => Self::Auto,
            ReductionStrategy::AtomicScatter => Self::AtomicScatter,
            ReductionStrategy::ParallelReduction => Self::ParallelReduction,
        }
    }
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
    #[pyo3(get)]
    pub reduction: PyReductionStrategy,
}

#[pymethods]
impl PyAdditiveSchwarz {
    #[new]
    #[pyo3(signature = (local_solver=None, reduction=PyReductionStrategy::Auto))]
    fn new(local_solver: Option<PyObject>, reduction: PyReductionStrategy) -> Self {
        Self {
            local_solver,
            reduction,
        }
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
    pub operator: PyOperatorRepr,
    #[pyo3(get)]
    pub max_refinements: usize,
}

#[pymethods]
impl PyCG {
    #[new]
    #[pyo3(signature = (tol=1e-8, maxiter=1000, operator=PyOperatorRepr::Implicit, max_refinements=2))]
    fn new(tol: f64, maxiter: usize, operator: PyOperatorRepr, max_refinements: usize) -> Self {
        Self {
            tol,
            maxiter,
            operator,
            max_refinements,
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
    pub operator: PyOperatorRepr,
    #[pyo3(get)]
    pub max_refinements: usize,
}

#[pymethods]
impl PyGMRES {
    #[new]
    #[pyo3(signature = (tol=1e-8, maxiter=1000, restart=30, operator=PyOperatorRepr::Implicit, max_refinements=2))]
    fn new(
        tol: f64,
        maxiter: usize,
        restart: usize,
        operator: PyOperatorRepr,
        max_refinements: usize,
    ) -> Self {
        Self {
            tol,
            maxiter,
            restart,
            operator,
            max_refinements,
        }
    }
}

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

#[pyclass]
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

#[pyclass]
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

/// Extract `Option<Preconditioner>` from a Python preconditioner argument.
///
/// - `None` → additive Schwarz with default local solver
/// - `Preconditioner.Off` → unpreconditioned (returns `Ok(None)`)
/// - `Preconditioner.Additive` → additive Schwarz with default local solver
/// - `Preconditioner.Multiplicative` → multiplicative Schwarz with default local solver
/// - `AdditiveSchwarz(...)` / `MultiplicativeSchwarz(...)` → advanced config
fn extract_preconditioner_config(
    py: Python<'_>,
    preconditioner: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<Preconditioner>> {
    let obj = match preconditioner {
        None => {
            return Ok(Some(Preconditioner::Additive(
                LocalSolverConfig::solver_default(),
                ReductionStrategy::Auto,
            )));
        }
        Some(obj) => obj,
    };

    // Enum shorthand
    if let Ok(p) = obj.extract::<PyPreconditioner>() {
        return match p {
            PyPreconditioner::Off => Ok(None),
            PyPreconditioner::Additive => Ok(Some(Preconditioner::Additive(
                LocalSolverConfig::solver_default(),
                ReductionStrategy::Auto,
            ))),
            PyPreconditioner::Multiplicative => Ok(Some(Preconditioner::Multiplicative(
                LocalSolverConfig::solver_default(),
            ))),
        };
    }

    // Advanced: AdditiveSchwarz / MultiplicativeSchwarz objects
    if let Ok(schwarz) = obj.downcast::<PyAdditiveSchwarz>() {
        let s = schwarz.get();
        let cfg = extract_local_solver_or_default(
            py,
            &s.local_solver,
            LocalSolverConfig::solver_default(),
        )?;
        let strategy = s.reduction.to_native();
        return Ok(Some(Preconditioner::Additive(cfg, strategy)));
    }
    if let Ok(schwarz) = obj.downcast::<PyMultiplicativeSchwarz>() {
        let cfg = extract_local_solver_or_default(
            py,
            &schwarz.get().local_solver,
            LocalSolverConfig::solver_default(),
        )?;
        return Ok(Some(Preconditioner::Multiplicative(cfg)));
    }

    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
        "preconditioner must be Preconditioner.Additive, Preconditioner.Multiplicative, \
         Preconditioner.Off, AdditiveSchwarz(...), MultiplicativeSchwarz(...), or None",
    ))
}
/// Extract solver parameters from a Python config object (CG or GMRES).
fn extract_solver_params(config: &Bound<'_, PyAny>) -> PyResult<SolverParams> {
    if let Ok(cg) = config.downcast::<PyCG>() {
        let cg = cg.get();
        return Ok(SolverParams {
            krylov: KrylovMethod::Cg,
            operator: cg.operator.to_native(),
            tol: cg.tol,
            maxiter: cg.maxiter,
            max_refinements: cg.max_refinements,
        });
    }

    if let Ok(gmres) = config.downcast::<PyGMRES>() {
        let gmres = gmres.get();
        return Ok(SolverParams {
            krylov: KrylovMethod::Gmres {
                restart: gmres.restart,
            },
            operator: gmres.operator.to_native(),
            tol: gmres.tol,
            maxiter: gmres.maxiter,
            max_refinements: gmres.max_refinements,
        });
    }

    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
        "config must be CG or GMRES",
    ))
}

/// Validate that CG is not paired with a multiplicative preconditioner.
fn validate_cg_preconditioner(
    params: &SolverParams,
    preconditioner: &Option<Preconditioner>,
) -> PyResult<()> {
    if matches!(params.krylov, KrylovMethod::Cg)
        && matches!(preconditioner, Some(Preconditioner::Multiplicative(_)))
    {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "CG requires a symmetric preconditioner; use 'additive' or switch to GMRES",
        ));
    }
    Ok(())
}

/// Extract solver config and preconditioner, applying CG/multiplicative validation.
fn extract_and_validate_config(
    py: Python<'_>,
    config: Option<&Bound<'_, PyAny>>,
    preconditioner: Option<&Bound<'_, PyAny>>,
) -> PyResult<(SolverParams, Option<Preconditioner>)> {
    let params = match config {
        Some(c) => extract_solver_params(c)?,
        None => SolverParams::default(),
    };
    let precond = extract_preconditioner_config(py, preconditioner)?;
    validate_cg_preconditioner(&params, &precond)?;
    Ok((params, precond))
}

// ---------------------------------------------------------------------------
// Helpers: numpy ↔ Rust conversions
// ---------------------------------------------------------------------------

fn into_py_result(py: Python<'_>, result: SolveResult) -> PySolveResult {
    PySolveResult {
        x: Array1::from_vec(result.x).into_pyarray(py).unbind(),
        demeaned: Array1::from_vec(result.demeaned).into_pyarray(py).unbind(),
        converged: result.converged,
        iterations: result.iterations,
        residual: result.final_residual,
        time_total: result.time_total,
        time_setup: result.time_setup,
        time_solve: result.time_solve,
    }
}

fn into_py_batch_result(
    py: Python<'_>,
    result: within::BatchSolveResult,
    n_dofs: usize,
    n_obs: usize,
) -> PyBatchSolveResult {
    let n_rhs = result.n_rhs();

    let x_flat = result.x_all().to_vec();
    let x_arr = Array2::from_shape_vec((n_dofs, n_rhs).f(), x_flat).expect("x shape mismatch");

    let demeaned_flat = result.demeaned_all().to_vec();
    let demeaned_arr =
        Array2::from_shape_vec((n_obs, n_rhs).f(), demeaned_flat).expect("demeaned shape mismatch");

    PyBatchSolveResult {
        x: x_arr.into_pyarray(py).unbind(),
        demeaned: demeaned_arr.into_pyarray(py).unbind(),
        converged: result.converged().to_vec(),
        iterations: result.iterations().to_vec(),
        residual: result.final_residual().to_vec(),
        time_solve: result.time_solve().to_vec(),
        time_total: result.time_total(),
    }
}

/// Extract columns from a 2-D array as owned vectors.
///
/// Columns may not be contiguous in memory, so we always copy.
fn extract_columns(arr: &numpy::ndarray::ArrayView2<'_, f64>) -> Vec<Vec<f64>> {
    (0..arr.ncols())
        .map(|j| arr.column(j).iter().copied().collect())
        .collect()
}

/// Extract an optional weight vector from a numpy array, returning owned data.
fn extract_weight_vec(weights: &Option<PyReadonlyArray1<'_, f64>>) -> Option<Vec<f64>> {
    weights
        .as_ref()
        .map(|w| w.as_array().iter().copied().collect())
}

fn warn_c_contiguous(py: Python<'_>, strides: &[isize]) -> PyResult<()> {
    if strides.len() >= 2 && strides[0] != 1 {
        PyErr::warn(
            py,
            &py.get_type::<pyo3::exceptions::PyUserWarning>(),
            c"categories array is not F-contiguous (column-major). \
             Use np.asfortranarray(categories) for faster solves.",
            1,
        )?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// One-shot solve
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (categories, y, config=None, weights=None, preconditioner=None))]
pub fn solve<'py>(
    py: Python<'py>,
    categories: PyReadonlyArray2<'py, u32>,
    y: PyReadonlyArray1<'py, f64>,
    config: Option<&Bound<'py, PyAny>>,
    weights: Option<PyReadonlyArray1<'py, f64>>,
    preconditioner: Option<&Bound<'py, PyAny>>,
) -> PyResult<PySolveResult> {
    let cats = categories.as_array();
    warn_c_contiguous(py, cats.strides())?;

    let y_slice = y.as_array();
    let y_vec;
    let y_ref: &[f64] = match y_slice.as_slice() {
        Some(s) => s,
        None => {
            y_vec = y_slice.to_vec();
            &y_vec
        }
    };
    let w_vec = extract_weight_vec(&weights);
    let w_ref = w_vec.as_deref();

    let (params, precond) = extract_and_validate_config(py, config, preconditioner)?;

    let result = py
        .allow_threads(|| solve_native(cats, y_ref, w_ref, &params, precond.as_ref()))
        .map_err(|err| PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string()))?;

    Ok(into_py_result(py, result))
}

// ---------------------------------------------------------------------------
// One-shot batch solve
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (categories, Y, config=None, weights=None, preconditioner=None))]
pub fn solve_batch<'py>(
    py: Python<'py>,
    categories: PyReadonlyArray2<'py, u32>,
    #[allow(non_snake_case)] Y: PyReadonlyArray2<'py, f64>,
    config: Option<&Bound<'py, PyAny>>,
    weights: Option<PyReadonlyArray1<'py, f64>>,
    preconditioner: Option<&Bound<'py, PyAny>>,
) -> PyResult<PyBatchSolveResult> {
    let cats = categories.as_array();
    warn_c_contiguous(py, cats.strides())?;

    let y_arr = Y.as_array();
    let n_obs = y_arr.nrows();

    let columns = extract_columns(&y_arr);
    let column_refs: Vec<&[f64]> = columns.iter().map(|c| c.as_slice()).collect();

    let w_vec = extract_weight_vec(&weights);
    let w_ref = w_vec.as_deref();

    let (params, precond) = extract_and_validate_config(py, config, preconditioner)?;

    let result = py
        .allow_threads(|| solve_batch_native(cats, &column_refs, w_ref, &params, precond.as_ref()))
        .map_err(|err| PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string()))?;

    let n_dofs = if result.n_rhs() > 0 {
        result.x_all().len() / result.n_rhs()
    } else {
        0
    };
    Ok(into_py_batch_result(py, result, n_dofs, n_obs))
}

// ---------------------------------------------------------------------------
// Built preconditioner (returned by Solver, picklable)
// ---------------------------------------------------------------------------

/// A pre-built preconditioner that can be pickled and reused.
///
/// Obtained via ``Solver.preconditioner()``.  Pass it back to a new
/// ``Solver(…, preconditioner=p)`` to skip the expensive factorisation.
#[pyclass(frozen, module = "within._within")]
#[pyo3(name = "FePreconditioner")]
pub struct PyFePreconditioner {
    inner: FePreconditioner,
}

#[pymethods]
impl PyFePreconditioner {
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
        self.inner.apply(x_slice, &mut y);
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

    /// Number of Schwarz subdomains in the built preconditioner.
    #[getter]
    fn n_subdomains(&self) -> usize {
        self.inner.n_subdomains()
    }

    /// Estimated nested-parallel work per subdomain.
    #[getter]
    fn subdomain_inner_parallel_work(&self) -> Vec<usize> {
        self.inner.subdomain_inner_parallel_work()
    }

    /// Configured additive reduction strategy, if this is an additive preconditioner.
    #[getter]
    fn reduction_strategy(&self) -> Option<PyReductionStrategy> {
        additive_reduction_strategy(&self.inner).map(PyReductionStrategy::from_native)
    }

    /// Concrete additive backend selected for the current Rayon width.
    #[getter]
    fn resolved_reduction_strategy(&self) -> Option<PyReductionStrategy> {
        resolved_additive_reduction_strategy(&self.inner).map(PyReductionStrategy::from_native)
    }

    /// Total additive inner-parallel work estimate.
    #[getter]
    fn total_inner_parallel_work(&self) -> Option<usize> {
        additive_schwarz_diagnostics(&self.inner).map(|diag| diag.total_inner_parallel_work())
    }

    /// Largest single-subdomain inner-parallel work estimate.
    #[getter]
    fn max_inner_parallel_work(&self) -> Option<usize> {
        additive_schwarz_diagnostics(&self.inner).map(|diag| diag.max_inner_parallel_work())
    }

    /// Total additive scatter entries across all subdomains.
    #[getter]
    fn total_scatter_dofs(&self) -> Option<usize> {
        additive_schwarz_diagnostics(&self.inner).map(|diag| diag.total_scatter_dofs())
    }

    /// Estimated number of heavy subdomains available to outer parallelism.
    #[getter]
    fn outer_parallel_capacity(&self) -> Option<f64> {
        additive_schwarz_diagnostics(&self.inner).map(|diag| diag.outer_parallel_capacity())
    }

    /// Average overlap multiplicity of the additive scatter.
    #[getter]
    fn scatter_overlap(&self) -> Option<f64> {
        additive_schwarz_diagnostics(&self.inner).map(|diag| diag.scatter_overlap())
    }

    fn __repr__(&self) -> String {
        let variant = match &self.inner {
            FePreconditioner::Additive(_) => "Additive",
            FePreconditioner::Multiplicative(_) => "Multiplicative",
        };
        format!("FePreconditioner({}, n={})", variant, self.inner.nrows())
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
        let inner: FePreconditioner = postcard::from_bytes(data).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "failed to deserialize preconditioner: {}",
                e
            ))
        })?;
        Ok(Self { inner })
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
#[pyclass(frozen)]
#[pyo3(name = "Solver")]
pub struct PySolver {
    solver: Solver<FactorMajorStore>,
}

#[pymethods]
impl PySolver {
    #[new]
    #[pyo3(signature = (categories, config=None, weights=None, preconditioner=None))]
    fn new<'py>(
        py: Python<'py>,
        categories: PyReadonlyArray2<'py, u32>,
        config: Option<&Bound<'py, PyAny>>,
        weights: Option<PyReadonlyArray1<'py, f64>>,
        preconditioner: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Self> {
        let cats = categories.as_array();
        warn_c_contiguous(py, cats.strides())?;

        let params = match config {
            Some(c) => extract_solver_params(c)?,
            None => SolverParams::default(),
        };

        // Build owned factor-major store from numpy array
        let n_obs = cats.nrows();
        let n_factors = cats.ncols();
        let factor_levels: Vec<Vec<u32>> = (0..n_factors)
            .map(|f| cats.column(f).iter().copied().collect())
            .collect();
        let w = match &weights {
            Some(w) => ObservationWeights::Dense(w.as_array().iter().copied().collect()),
            None => ObservationWeights::Unit,
        };
        let store = FactorMajorStore::new(factor_levels, w, n_obs)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        let design = WeightedDesign::from_store(store)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        // Handle pre-built FePreconditioner separately (uses a different constructor);
        // all other variants go through extract_preconditioner_config.
        let solver =
            if let Some(Ok(fe)) = preconditioner.map(|o| o.downcast::<PyFePreconditioner>()) {
                let fe_precond = fe.get().inner.clone();
                py.allow_threads(|| {
                    Solver::from_design_with_preconditioner(design, &params, fe_precond)
                })
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?
            } else {
                let precond = extract_preconditioner_config(py, preconditioner)?;
                validate_cg_preconditioner(&params, &precond)?;
                py.allow_threads(|| Solver::from_design(design, &params, precond.as_ref()))
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?
            };

        Ok(Self { solver })
    }

    /// Solve for a single response vector.
    #[pyo3(name = "solve")]
    fn solve_py<'py>(
        &self,
        py: Python<'py>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<PySolveResult> {
        let y_slice = y.as_array();
        let y_vec;
        let y_ref: &[f64] = match y_slice.as_slice() {
            Some(s) => s,
            None => {
                y_vec = y_slice.to_vec();
                &y_vec
            }
        };

        let result = py
            .allow_threads(|| self.solver.solve(y_ref))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        Ok(into_py_result(py, result))
    }

    /// Solve for multiple response vectors in parallel.
    ///
    /// `Y` is a 2-D array of shape `(n_obs, k)` where each column is a
    /// separate response vector.
    #[pyo3(name = "solve_batch")]
    fn solve_batch_py<'py>(
        &self,
        py: Python<'py>,
        y_matrix: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<PyBatchSolveResult> {
        let y_arr = y_matrix.as_array();

        let columns = extract_columns(&y_arr);
        let column_refs: Vec<&[f64]> = columns.iter().map(|c| c.as_slice()).collect();

        let n_dofs = self.solver.n_dofs();
        let n_obs = self.solver.n_obs();

        let result = py
            .allow_threads(|| self.solver.solve_batch(&column_refs))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        Ok(into_py_batch_result(py, result, n_dofs, n_obs))
    }

    /// Return the built preconditioner, or ``None`` if unconfigured.
    ///
    /// The returned object is picklable and can be passed to a new
    /// ``Solver(…, preconditioner=p)`` to skip the expensive build step.
    #[pyo3(name = "preconditioner")]
    fn preconditioner_py(&self) -> PyResult<Option<PyFePreconditioner>> {
        match self.solver.preconditioner() {
            None => Ok(None),
            Some(p) => Ok(Some(PyFePreconditioner { inner: p.clone() })),
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

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

#[pymodule]
fn _within(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySolveResult>()?;
    m.add_class::<PyBatchSolveResult>()?;
    m.add_class::<PyCG>()?;
    m.add_class::<PyGMRES>()?;
    m.add_class::<PyAdditiveSchwarz>()?;
    m.add_class::<PyMultiplicativeSchwarz>()?;
    m.add_class::<PyReductionStrategy>()?;
    m.add_class::<PyOperatorRepr>()?;
    m.add_class::<PyPreconditioner>()?;
    m.add_class::<PyApproxCholConfig>()?;
    m.add_class::<PyApproxSchurConfig>()?;
    m.add_class::<PySchurComplement>()?;
    m.add_class::<PyFullSddm>()?;
    m.add_class::<PyFePreconditioner>()?;
    m.add_class::<PySolver>()?;
    m.add_function(wrap_pyfunction!(solve, m)?)?;
    m.add_function(wrap_pyfunction!(solve_batch, m)?)?;
    Ok(())
}
