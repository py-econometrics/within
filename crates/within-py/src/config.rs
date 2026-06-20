//! PyO3 config wrapper classes exposed via `within._within`.
//!
//! These mirror the native [`within::config`] types and host the
//! Python→native config conversions (`to_native`,
//! `resolve_precond_input`, `resolve_lsmr_config`).

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

use within::config::{
    ApproxCholConfig, ApproxSchurConfig, LocalSolverConfig, LsmrOptions, PreconditionerConfig,
    ReductionStrategy,
};
use within::Preconditioner;

// ---------------------------------------------------------------------------
// Low-level config classes (available via `_within` for benchmarks)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// PreconditionerConfig enum (IntEnum shortcut)
// ---------------------------------------------------------------------------

/// Preconditioner selection shortcut for the LSMR solver.
///
/// - ``PreconditionerConfig.Additive`` — additive Schwarz (default)
/// - ``PreconditionerConfig.Off`` — no preconditioner
/// - ``PreconditionerConfig.Diagonal`` — diagonal/Jacobi preconditioner
#[pyclass(frozen, eq, eq_int, module = "within._within")]
#[pyo3(name = "PreconditionerConfig")]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyPreconditionerConfig {
    Additive = 0,
    Off = 1,
    Diagonal = 2,
}

#[pyclass(frozen, eq, eq_int, module = "within._within")]
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

// ---------------------------------------------------------------------------
// Local solver config (available via `_within` for benchmarks)
// ---------------------------------------------------------------------------

#[pyclass(frozen, subclass, module = "within._within")]
#[pyo3(name = "LocalSolverConfig")]
pub struct PyLocalSolverConfig {
    #[pyo3(get)]
    pub approx_chol: Option<Py<PyApproxCholConfig>>,
    #[pyo3(get)]
    pub approx_schur: Option<Py<PyApproxSchurConfig>>,
    #[pyo3(get)]
    pub dense_threshold: usize,
}

#[pymethods]
impl PyLocalSolverConfig {
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
            dense_threshold: dense_threshold
                .unwrap_or_else(|| LocalSolverConfig::default().dense_threshold),
        }
    }
}

// ---------------------------------------------------------------------------
// Schwarz preconditioner config (available via `_within` for benchmarks)
// ---------------------------------------------------------------------------

#[pyclass(frozen, module = "within._within")]
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

// ---------------------------------------------------------------------------
// LSMR config
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Python → native config conversion
// ---------------------------------------------------------------------------

/// Native interpretation of the Python `preconditioner` argument.
///
/// A pre-built [`Preconditioner`] takes the reuse path; everything else is a
/// [`PreconditionerConfig`] (or `None` for the library default) to build from.
/// Both variants hold only native data, so a resolved value is safe to move
/// into a GIL-released closure (`Preconditioner` clones are `Arc`-cheap).
pub(crate) enum PrecondInput {
    Prebuilt(Preconditioner),
    Config(Option<PreconditionerConfig>),
}

/// Resolve the Python `preconditioner` argument into a [`PrecondInput`].
///
/// Must run while the GIL is held (it inspects Python objects). A pre-built
/// `Preconditioner` is detected first and taken verbatim; anything else is
/// parsed as a `PreconditionerConfig` via [`extract_preconditioner_config`].
pub(crate) fn resolve_precond_input(
    py: Python<'_>,
    preconditioner: Option<&Bound<'_, PyAny>>,
) -> PyResult<PrecondInput> {
    if let Some(obj) = preconditioner {
        if let Ok(built) = obj.downcast::<PyPreconditioner>() {
            return Ok(PrecondInput::Prebuilt(built.get().inner.clone()));
        }
    }
    Ok(PrecondInput::Config(extract_preconditioner_config(
        py,
        preconditioner,
    )?))
}

fn extract_preconditioner_config(
    py: Python<'_>,
    preconditioner: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<PreconditionerConfig>> {
    let Some(obj) = preconditioner else {
        return Ok(None);
    };

    // Enum shorthand
    if let Ok(p) = obj.extract::<PyPreconditionerConfig>() {
        return Ok(Some(match p {
            PyPreconditionerConfig::Off => PreconditionerConfig::Off,
            PyPreconditionerConfig::Additive => PreconditionerConfig::default(),
            PyPreconditionerConfig::Diagonal => PreconditionerConfig::Diagonal,
        }));
    }

    // Advanced: AdditiveSchwarz object
    if let Ok(schwarz) = obj.downcast::<PyAdditiveSchwarz>() {
        let s = schwarz.get();
        let local = match &s.local_solver {
            None => LocalSolverConfig::default(),
            Some(obj) => {
                let obj = obj.bind(py);
                let Ok(sc) = obj.downcast::<PyLocalSolverConfig>() else {
                    return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                        "local_solver must be LocalSolverConfig or None",
                    ));
                };
                let sc = sc.get();
                let approx_chol = sc
                    .approx_chol
                    .as_ref()
                    .map(|c| c.bind(py).get().to_native())
                    .unwrap_or_else(|| LocalSolverConfig::default().approx_chol);
                let approx_schur = sc
                    .approx_schur
                    .as_ref()
                    .map(|c| c.bind(py).get().to_native());
                LocalSolverConfig {
                    approx_chol,
                    approx_schur,
                    dense_threshold: sc.dense_threshold,
                }
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
    if let Ok(lsmr) = c.downcast::<PyLsmrOptions>() {
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
