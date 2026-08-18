//! PyO3 config wrapper classes exposed via `within._within`.
//!
//! These mirror the native [`within::config`] types and host the
//! Python→native config conversions (`to_native`, `resolve_precond_input`,
//! `resolve_lsmr_config`). The low-level classes are exposed for benchmark tuning.

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

use within::config::{
    ApproxCholConfig, ApproxSchurConfig, LocalSolverConfig, LsmrOptions, PreconditionerConfig,
    ReductionStrategy, ScalingConfig, ScalingFailure, SchurMode,
};
use within::{Preconditioner, PreconditionerInput};

use crate::convert::IntoPyErr;

#[pyclass(frozen, module = "within._within")]
#[pyo3(name = "ApproxCholConfig")]
pub struct PyApproxCholConfig {
    #[pyo3(get)]
    pub seed: u64,
    #[pyo3(get)]
    pub split_merge: Option<u32>,
}

#[pymethods]
impl PyApproxCholConfig {
    #[new]
    #[pyo3(signature = (seed=0, split_merge=None))]
    fn new(seed: u64, split_merge: Option<u32>) -> Self {
        Self { seed, split_merge }
    }
}

impl PyApproxCholConfig {
    pub(crate) fn from_native(config: &ApproxCholConfig) -> Self {
        Self {
            seed: config.seed,
            split_merge: config.split_merge,
        }
    }

    pub(crate) fn to_native(&self) -> ApproxCholConfig {
        ApproxCholConfig {
            seed: self.seed,
            split_merge: self.split_merge,
        }
    }
}

#[pyclass(frozen, module = "within._within")]
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
    pub(crate) fn to_native(&self) -> ApproxSchurConfig {
        ApproxSchurConfig {
            seed: self.seed,
            split: self.split,
        }
    }
}

/// Schur reduction mode: approximate (the library default) or exact.
#[pyclass(frozen, skip_from_py_object, module = "within._within")]
#[pyo3(name = "Schur")]
#[derive(Clone)]
pub struct PySchur {
    inner: SchurMode,
}

#[pymethods]
impl PySchur {
    /// Approximate Schur via clique-tree sampling; `config` tunes the sampler.
    #[staticmethod]
    #[pyo3(signature = (config=None))]
    fn approximate(py: Python<'_>, config: Option<Py<PyApproxSchurConfig>>) -> Self {
        let cfg = config
            .map(|c| c.bind(py).get().to_native())
            .unwrap_or_default();
        Self {
            inner: SchurMode::Approximate(cfg),
        }
    }

    /// Exact Schur complement (higher fidelity, slower per subdomain).
    #[staticmethod]
    fn exact() -> Self {
        Self {
            inner: SchurMode::Exact,
        }
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            SchurMode::Approximate(cfg) => {
                format!("Schur.approximate(seed={}, split={})", cfg.seed, cfg.split)
            }
            SchurMode::Exact => "Schur.exact()".to_string(),
        }
    }
}

impl PySchur {
    pub(crate) fn from_native(config: &SchurMode) -> Self {
        Self {
            inner: config.clone(),
        }
    }

    pub(crate) fn to_native(&self) -> SchurMode {
        self.inner.clone()
    }
}

/// Certification policy for the diagonal scaling of signed components.
#[pyclass(frozen, module = "within._within")]
#[pyo3(name = "ScalingConfig")]
pub struct PyScalingConfig {
    #[pyo3(get)]
    pub tolerance: f64,
    #[pyo3(get)]
    pub max_iterations: usize,
    pub on_failure: ScalingFailure,
}

#[pymethods]
impl PyScalingConfig {
    #[new]
    #[pyo3(signature = (tolerance=None, max_iterations=None, on_failure=None))]
    fn new(
        tolerance: Option<f64>,
        max_iterations: Option<usize>,
        on_failure: Option<&str>,
    ) -> PyResult<Self> {
        let default = ScalingConfig::default();
        let on_failure = match on_failure {
            None => default.on_failure,
            Some("warn") => ScalingFailure::Warn,
            Some("error") => ScalingFailure::Error,
            Some(other) => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "on_failure must be 'warn' or 'error', got {other:?}"
                )))
            }
        };
        Ok(Self {
            tolerance: tolerance.unwrap_or(default.tolerance),
            max_iterations: max_iterations.unwrap_or(default.max_iterations),
            on_failure,
        })
    }

    #[getter(on_failure)]
    fn on_failure_str(&self) -> &'static str {
        match self.on_failure {
            ScalingFailure::Warn => "warn",
            ScalingFailure::Error => "error",
        }
    }
}

impl PyScalingConfig {
    pub(crate) fn from_native(config: &ScalingConfig) -> Self {
        Self {
            tolerance: config.tolerance,
            max_iterations: config.max_iterations,
            on_failure: config.on_failure,
        }
    }

    pub(crate) fn to_native(&self) -> ScalingConfig {
        ScalingConfig {
            tolerance: self.tolerance,
            max_iterations: self.max_iterations,
            on_failure: self.on_failure,
        }
    }
}

/// Construction configuration for a preconditioner, mirroring the native enum.
///
/// Complex enum: each variant is a Python-visible subclass supporting ``match``
/// and per-field getters. Unit variants are spelled ``Off()`` / ``Diagonal()``.
#[pyclass(frozen, eq, module = "within._within")]
#[pyo3(name = "PreconditionerConfig")]
#[derive(PartialEq)]
pub enum PyPreconditionerConfig {
    Off(),
    #[pyo3(constructor = (
        local_solver=PyLocalSolverConfig::default(),
        reduction=PyReductionStrategy::Auto
    ))]
    Additive {
        local_solver: PyLocalSolverConfig,
        reduction: PyReductionStrategy,
    },
    Diagonal(),
}

impl PyPreconditionerConfig {
    pub(crate) fn to_native(&self) -> PreconditionerConfig {
        match self {
            Self::Off() => PreconditionerConfig::Off,
            Self::Diagonal() => PreconditionerConfig::Diagonal,
            Self::Additive {
                local_solver,
                reduction,
            } => PreconditionerConfig::Additive {
                local_solver: local_solver.to_native(),
                reduction: reduction.to_native(),
            },
        }
    }

    pub(crate) fn from_native(config: &PreconditionerConfig) -> PyResult<Self> {
        Ok(match config {
            PreconditionerConfig::Off => Self::Off(),
            PreconditionerConfig::Diagonal => Self::Diagonal(),
            PreconditionerConfig::Additive {
                local_solver,
                reduction,
            } => Self::Additive {
                local_solver: PyLocalSolverConfig::from_native(local_solver),
                reduction: PyReductionStrategy::from_native(*reduction),
            },
            _ => {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "unsupported preconditioner configuration",
                ))
            }
        })
    }
}

#[pymethods]
impl PyPreconditionerConfig {
    /// Complex enums have no default pickle support; round-trip via native serde.
    fn __reduce__<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyAny>, (Bound<'py, pyo3::types::PyBytes>,))> {
        let bytes = postcard::to_stdvec(&self.to_native())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let ctor = py
            .import("within._within")?
            .getattr("_preconditioner_config_from_postcard")?;
        Ok((ctor, (pyo3::types::PyBytes::new(py, &bytes),)))
    }
}

#[pyfunction]
pub(crate) fn _preconditioner_config_from_postcard(
    data: &[u8],
) -> PyResult<PyPreconditionerConfig> {
    let native: PreconditionerConfig = postcard::from_bytes(data).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "failed to deserialize preconditioner config: {e}"
        ))
    })?;
    PyPreconditionerConfig::from_native(&native)
}

#[pyclass(frozen, eq, eq_int, from_py_object, module = "within._within")]
#[pyo3(name = "ReductionStrategy")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyReductionStrategy {
    Auto = 0,
    AtomicScatter = 1,
    ParallelReduction = 2,
}

impl PyReductionStrategy {
    pub(crate) fn from_native(strategy: ReductionStrategy) -> Self {
        match strategy {
            ReductionStrategy::Auto => Self::Auto,
            ReductionStrategy::AtomicScatter => Self::AtomicScatter,
            ReductionStrategy::ParallelReduction => Self::ParallelReduction,
        }
    }

    pub(crate) fn to_native(self) -> ReductionStrategy {
        match self {
            Self::Auto => ReductionStrategy::Auto,
            Self::AtomicScatter => ReductionStrategy::AtomicScatter,
            Self::ParallelReduction => ReductionStrategy::ParallelReduction,
        }
    }
}

#[pyclass(frozen, eq, from_py_object, module = "within._within")]
#[pyo3(name = "LocalSolverConfig")]
#[derive(Clone, Default, PartialEq)]
pub struct PyLocalSolverConfig {
    inner: LocalSolverConfig,
}

#[pymethods]
impl PyLocalSolverConfig {
    #[new]
    #[pyo3(signature = (approx_chol=None, schur=None, dense_threshold=None, scaling=None))]
    fn new(
        py: Python<'_>,
        approx_chol: Option<Py<PyApproxCholConfig>>,
        schur: Option<Py<PySchur>>,
        dense_threshold: Option<usize>,
        scaling: Option<Py<PyScalingConfig>>,
    ) -> Self {
        let defaults = LocalSolverConfig::default();
        Self {
            inner: LocalSolverConfig {
                approx_chol: approx_chol
                    .map(|config| config.bind(py).get().to_native())
                    .unwrap_or(defaults.approx_chol),
                schur: schur
                    .map(|config| config.bind(py).get().to_native())
                    .unwrap_or(defaults.schur),
                dense_threshold: dense_threshold.unwrap_or(defaults.dense_threshold),
                scaling: scaling
                    .map(|config| config.bind(py).get().to_native())
                    .unwrap_or(defaults.scaling),
            },
        }
    }

    #[getter]
    fn approx_chol(&self) -> PyApproxCholConfig {
        PyApproxCholConfig::from_native(&self.inner.approx_chol)
    }

    #[getter]
    fn schur(&self) -> PySchur {
        PySchur::from_native(&self.inner.schur)
    }

    #[getter]
    fn dense_threshold(&self) -> usize {
        self.inner.dense_threshold
    }

    #[getter]
    fn scaling(&self) -> PyScalingConfig {
        PyScalingConfig::from_native(&self.inner.scaling)
    }
}

impl PyLocalSolverConfig {
    pub(crate) fn from_native(config: &LocalSolverConfig) -> Self {
        Self {
            inner: config.clone(),
        }
    }

    pub(crate) fn to_native(&self) -> LocalSolverConfig {
        self.inner.clone()
    }
}

#[pyclass(frozen, module = "within._within")]
#[pyo3(name = "LsmrOptions")]
pub struct PyLsmrOptions {
    #[pyo3(get)]
    pub tol: f64,
    #[pyo3(get)]
    pub maxiter: usize,
    #[pyo3(get)]
    pub local_size: Option<usize>,
}

#[pymethods]
impl PyLsmrOptions {
    #[new]
    #[pyo3(signature = (tol=1e-8, maxiter=1000, local_size=None))]
    fn new(tol: f64, maxiter: usize, local_size: Option<usize>) -> Self {
        Self {
            tol,
            maxiter,
            local_size,
        }
    }
}

/// A pre-built preconditioner; pass it back to skip the factorisation.
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
        let y = py
            .detach(|| {
                let mut y = vec![0.0; self.inner.nrows()];
                self.inner.apply(x_slice, &mut y).map(|()| y)
            })
            .map_err(IntoPyErr::into_py_err)?;
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
        format!(
            "Preconditioner({}, n={})",
            self.inner.variant_name(),
            self.inner.nrows()
        )
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

    /// Complete normalized configuration used to build this preconditioner.
    #[getter]
    fn config(&self) -> PyResult<PyPreconditionerConfig> {
        PyPreconditionerConfig::from_native(self.inner.config())
    }
}

/// The result is all-native, so it can move into a released closure.
pub(crate) fn resolve_precond_input(
    preconditioner: Option<&Bound<'_, PyAny>>,
) -> PyResult<PreconditionerInput> {
    if let Some(obj) = preconditioner {
        if let Ok(built) = obj.cast::<PyPreconditioner>() {
            return Ok(PreconditionerInput::Prebuilt(built.get().inner.clone()));
        }
    }
    Ok(extract_preconditioner_config(preconditioner)?
        .map_or(PreconditionerInput::Default, PreconditionerInput::Config))
}

fn extract_preconditioner_config(
    preconditioner: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<PreconditionerConfig>> {
    let Some(obj) = preconditioner else {
        return Ok(None);
    };

    if let Ok(config) = obj.cast::<PyPreconditionerConfig>() {
        return Ok(Some(config.get().to_native()));
    }

    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
        "preconditioner must be PreconditionerConfig.Off(), PreconditionerConfig.Diagonal(), \
         PreconditionerConfig.Additive(...), a pre-built Preconditioner, or None",
    ))
}

pub(crate) fn resolve_lsmr_config(config: Option<&Bound<'_, PyAny>>) -> PyResult<LsmrOptions> {
    let Some(c) = config else {
        return Ok(LsmrOptions::default());
    };
    if let Ok(lsmr) = c.cast::<PyLsmrOptions>() {
        let lsmr = lsmr.get();
        return Ok(LsmrOptions {
            tol: lsmr.tol,
            maxiter: lsmr.maxiter,
            local_size: lsmr.local_size,
        });
    }
    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
        "options must be LsmrOptions",
    ))
}
