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
    pub max_sweeps: usize,
    pub on_failure: ScalingFailure,
}

#[pymethods]
impl PyScalingConfig {
    #[new]
    #[pyo3(signature = (tolerance=None, max_sweeps=None, on_failure=None))]
    fn new(
        tolerance: Option<f64>,
        max_sweeps: Option<usize>,
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
            max_sweeps: max_sweeps.unwrap_or(default.max_sweeps),
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
    pub(crate) fn to_native(&self) -> ScalingConfig {
        ScalingConfig {
            tolerance: self.tolerance,
            max_sweeps: self.max_sweeps,
            on_failure: self.on_failure,
        }
    }
}

/// Preconditioner shortcut: ``Additive`` (default), ``Off``, or ``Diagonal``.
#[pyclass(frozen, eq, eq_int, from_py_object, module = "within._within")]
#[pyo3(name = "PreconditionerConfig")]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyPreconditionerConfig {
    Additive = 0,
    Off = 1,
    Diagonal = 2,
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
    pub(crate) fn to_native(self) -> ReductionStrategy {
        match self {
            Self::Auto => ReductionStrategy::Auto,
            Self::AtomicScatter => ReductionStrategy::AtomicScatter,
            Self::ParallelReduction => ReductionStrategy::ParallelReduction,
        }
    }
}

#[pyclass(frozen, module = "within._within")]
#[pyo3(name = "LocalSolverConfig")]
pub struct PyLocalSolverConfig {
    #[pyo3(get)]
    pub approx_chol: Option<Py<PyApproxCholConfig>>,
    #[pyo3(get)]
    pub schur: Option<Py<PySchur>>,
    #[pyo3(get)]
    pub dense_threshold: usize,
    #[pyo3(get)]
    pub scaling: Option<Py<PyScalingConfig>>,
}

#[pymethods]
impl PyLocalSolverConfig {
    #[new]
    #[pyo3(signature = (approx_chol=None, schur=None, dense_threshold=None, scaling=None))]
    fn new(
        approx_chol: Option<Py<PyApproxCholConfig>>,
        schur: Option<Py<PySchur>>,
        dense_threshold: Option<usize>,
        scaling: Option<Py<PyScalingConfig>>,
    ) -> Self {
        Self {
            approx_chol,
            schur,
            dense_threshold: dense_threshold
                .unwrap_or_else(|| LocalSolverConfig::default().dense_threshold),
            scaling,
        }
    }
}

impl PyLocalSolverConfig {
    pub(crate) fn to_native(&self, py: Python<'_>) -> LocalSolverConfig {
        let approx_chol = self
            .approx_chol
            .as_ref()
            .map(|c| c.bind(py).get().to_native())
            .unwrap_or_else(|| LocalSolverConfig::default().approx_chol);
        let schur = self
            .schur
            .as_ref()
            .map(|s| s.bind(py).get().to_native())
            .unwrap_or_default();
        let scaling = self
            .scaling
            .as_ref()
            .map(|c| c.bind(py).get().to_native())
            .unwrap_or_default();
        LocalSolverConfig {
            approx_chol,
            schur,
            dense_threshold: self.dense_threshold,
            scaling,
        }
    }
}

#[pyclass(frozen, module = "within._within")]
#[pyo3(name = "AdditiveSchwarz")]
pub struct PyAdditiveSchwarz {
    #[pyo3(get)]
    pub local_solver: Option<Py<PyAny>>,
    #[pyo3(get)]
    pub reduction: PyReductionStrategy,
}

#[pymethods]
impl PyAdditiveSchwarz {
    #[new]
    #[pyo3(signature = (local_solver=None, reduction=PyReductionStrategy::Auto))]
    fn new(local_solver: Option<Py<PyAny>>, reduction: PyReductionStrategy) -> Self {
        Self {
            local_solver,
            reduction,
        }
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
}

/// Must run under the GIL; the result is all-native, so it can move into a released closure.
pub(crate) fn resolve_precond_input(
    py: Python<'_>,
    preconditioner: Option<&Bound<'_, PyAny>>,
) -> PyResult<PreconditionerInput> {
    if let Some(obj) = preconditioner {
        if let Ok(built) = obj.cast::<PyPreconditioner>() {
            return Ok(PreconditionerInput::Prebuilt(built.get().inner.clone()));
        }
    }
    Ok(extract_preconditioner_config(py, preconditioner)?
        .map_or(PreconditionerInput::Default, PreconditionerInput::Config))
}

fn extract_preconditioner_config(
    py: Python<'_>,
    preconditioner: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<PreconditionerConfig>> {
    let Some(obj) = preconditioner else {
        return Ok(None);
    };

    if let Ok(p) = obj.extract::<PyPreconditionerConfig>() {
        return Ok(Some(match p {
            PyPreconditionerConfig::Off => PreconditionerConfig::Off,
            PyPreconditionerConfig::Additive => PreconditionerConfig::default(),
            PyPreconditionerConfig::Diagonal => PreconditionerConfig::Diagonal,
        }));
    }

    if let Ok(schwarz) = obj.cast::<PyAdditiveSchwarz>() {
        let s = schwarz.get();
        let local = match &s.local_solver {
            None => LocalSolverConfig::default(),
            Some(obj) => {
                let obj = obj.bind(py);
                let Ok(sc) = obj.cast::<PyLocalSolverConfig>() else {
                    return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                        "local_solver must be LocalSolverConfig or None",
                    ));
                };
                sc.get().to_native(py)
            }
        };
        let reduction = s.reduction.to_native();
        return Ok(Some(PreconditionerConfig::Additive {
            local_solver: local,
            reduction,
        }));
    }

    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
        "preconditioner must be PreconditionerConfig.Additive, PreconditionerConfig.Off, \
         PreconditionerConfig.Diagonal, AdditiveSchwarz(...), a pre-built Preconditioner, or None",
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
