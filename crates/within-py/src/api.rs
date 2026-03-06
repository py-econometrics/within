//! Python API: typed config classes, solve entrypoint, and synthetic data generation.

use approx_chol::Config;
use numpy::ndarray::{Array1, Array2, ArrayView2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use within::config::{
    ApproxSchurConfig, CgPreconditioner, GmresPreconditioner, LocalSolverConfig, SchwarzConfig,
    SolverMethod, SolverParams,
};
use within::domain::WeightedDesign;
use within::observation::{
    CompressedStore, FactorMajorStore, ObservationStore, ObservationWeights, RowMajorStore,
};
use within::operator::design::DesignOperator;
use within::orchestrate::{solve_least_squares, SolveResult};
use within::WithinResult;

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
    fn to_native(&self) -> Config {
        let split_merge = if self.split > 1 {
            Some(self.split)
        } else {
            None
        };
        Config {
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

macro_rules! schwarz_pyclass {
    ($RustName:ident, $py_name:literal) => {
        #[pyclass(frozen)]
        #[pyo3(name = $py_name)]
        pub struct $RustName {
            #[pyo3(get)]
            pub smoother: Option<Py<PyApproxCholConfig>>,
            #[pyo3(get)]
            pub local_solver: Option<String>,
            #[pyo3(get)]
            pub approx_schur: Option<Py<PyApproxSchurConfig>>,
            #[pyo3(get)]
            pub dense_schur_threshold: Option<usize>,
        }

        #[pymethods]
        impl $RustName {
            #[new]
            #[pyo3(signature = (smoother=None, local_solver=None, approx_schur=None, dense_schur_threshold=None))]
            fn new(
                smoother: Option<Py<PyApproxCholConfig>>,
                local_solver: Option<String>,
                approx_schur: Option<Py<PyApproxSchurConfig>>,
                dense_schur_threshold: Option<usize>,
            ) -> Self {
                Self {
                    smoother,
                    local_solver,
                    approx_schur,
                    dense_schur_threshold,
                }
            }
        }
    };
}

schwarz_pyclass!(PyOneLevelSchwarz, "OneLevelSchwarz");
schwarz_pyclass!(
    PyMultiplicativeOneLevelSchwarz,
    "MultiplicativeOneLevelSchwarz"
);

#[pyclass(frozen)]
#[pyo3(name = "LSMR")]
pub struct PyLSMR {
    #[pyo3(get)]
    pub tol: f64,
    #[pyo3(get)]
    pub maxiter: usize,
    #[pyo3(get)]
    pub conlim: f64,
}

#[pymethods]
impl PyLSMR {
    #[new]
    #[pyo3(signature = (tol=1e-8, maxiter=1000, conlim=1e8))]
    fn new(tol: f64, maxiter: usize, conlim: f64) -> Self {
        Self {
            tol,
            maxiter,
            conlim,
        }
    }
}

#[pyclass(frozen)]
#[pyo3(name = "CG")]
pub struct PyCG {
    #[pyo3(get)]
    pub tol: f64,
    #[pyo3(get)]
    pub maxiter: usize,
    #[pyo3(get)]
    pub preconditioner: Option<PyObject>,
}

#[pymethods]
impl PyCG {
    #[new]
    #[pyo3(signature = (tol=1e-8, maxiter=1000, preconditioner=None))]
    fn new(tol: f64, maxiter: usize, preconditioner: Option<PyObject>) -> Self {
        Self {
            tol,
            maxiter,
            preconditioner,
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
    pub preconditioner: PyObject,
}

#[pymethods]
impl PyGMRES {
    #[new]
    #[pyo3(signature = (preconditioner, tol=1e-8, maxiter=10000, restart=30))]
    fn new(preconditioner: PyObject, tol: f64, maxiter: usize, restart: usize) -> Self {
        Self {
            tol,
            maxiter,
            restart,
            preconditioner,
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

/// Extract SchwarzConfig from the fields common to all Schwarz pyclasses.
fn extract_schwarz_config(
    py: Python<'_>,
    smoother: &Option<Py<PyApproxCholConfig>>,
    local_solver_str: &Option<String>,
    approx_schur: &Option<Py<PyApproxSchurConfig>>,
    dense_schur_threshold: &Option<usize>,
) -> PyResult<SchwarzConfig> {
    let approx_chol = match smoother {
        Some(config) => config.bind(py).get().to_native(),
        None => Config::default(),
    };
    let approx_schur_native = approx_schur
        .as_ref()
        .map(|cfg| cfg.bind(py).get().to_native());
    let local_solver = match local_solver_str.as_deref() {
        None | Some("schur_complement") => {
            let default = LocalSolverConfig::default();
            match default {
                LocalSolverConfig::SchurComplement {
                    approx_chol: ac,
                    dense_threshold: default_threshold,
                    ..
                } => LocalSolverConfig::SchurComplement {
                    approx_chol: ac,
                    approx_schur: approx_schur_native,
                    dense_threshold: dense_schur_threshold.unwrap_or(default_threshold),
                },
                other => other,
            }
        }
        Some("full_sddm") => LocalSolverConfig::FullSddm,
        Some(other) => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown local_solver '{}'. Expected 'schur_complement' or 'full_sddm'.",
                other
            )));
        }
    };
    Ok(SchwarzConfig {
        approx_chol,
        local_solver,
    })
}

/// Extract solver parameters from a Python config object (LSMR, CG, or GMRES).
fn extract_solver_params(py: Python<'_>, config: &Bound<'_, PyAny>) -> PyResult<SolverParams> {
    if let Ok(lsmr) = config.downcast::<PyLSMR>() {
        let lsmr = lsmr.get();
        return Ok(SolverParams {
            method: SolverMethod::Lsmr {
                conlim: lsmr.conlim,
            },
            tol: lsmr.tol,
            maxiter: lsmr.maxiter,
        });
    }

    if let Ok(cg) = config.downcast::<PyCG>() {
        let cg = cg.get();
        let preconditioner = match &cg.preconditioner {
            Some(preconditioner) => {
                let preconditioner = preconditioner.bind(py);
                if let Ok(s) = preconditioner.downcast::<PyOneLevelSchwarz>() {
                    let s = s.get();
                    CgPreconditioner::OneLevel(extract_schwarz_config(
                        py,
                        &s.smoother,
                        &s.local_solver,
                        &s.approx_schur,
                        &s.dense_schur_threshold,
                    )?)
                } else if let Ok(s) = preconditioner.downcast::<PyMultiplicativeOneLevelSchwarz>() {
                    let s = s.get();
                    CgPreconditioner::MultiplicativeOneLevel(extract_schwarz_config(
                        py,
                        &s.smoother,
                        &s.local_solver,
                        &s.approx_schur,
                        &s.dense_schur_threshold,
                    )?)
                } else {
                    return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                        "preconditioner must be OneLevelSchwarz or MultiplicativeOneLevelSchwarz",
                    ));
                }
            }
            None => CgPreconditioner::OneLevel(SchwarzConfig {
                approx_chol: Config::default(),
                local_solver: LocalSolverConfig::default(),
            }),
        };
        return Ok(SolverParams {
            method: SolverMethod::Cg { preconditioner },
            tol: cg.tol,
            maxiter: cg.maxiter,
        });
    }

    if let Ok(gmres) = config.downcast::<PyGMRES>() {
        let gmres = gmres.get();
        let preconditioner_obj = gmres.preconditioner.bind(py);
        let preconditioner =
            if let Ok(s) = preconditioner_obj.downcast::<PyMultiplicativeOneLevelSchwarz>() {
                let s = s.get();
                GmresPreconditioner::MultiplicativeOneLevel(extract_schwarz_config(
                    py,
                    &s.smoother,
                    &s.local_solver,
                    &s.approx_schur,
                    &s.dense_schur_threshold,
                )?)
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "GMRES preconditioner must be MultiplicativeOneLevelSchwarz",
                ));
            };

        return Ok(SolverParams {
            method: SolverMethod::Gmres {
                preconditioner,
                restart: gmres.restart,
            },
            tol: gmres.tol,
            maxiter: gmres.maxiter,
        });
    }

    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
        "config must be LSMR, CG, or GMRES",
    ))
}

#[pyfunction]
#[pyo3(signature = (categories, y, config, n_levels=None, weights=None, layout=None))]
pub fn py_solve<'py>(
    py: Python<'py>,
    categories: PyReadonlyArray2<'py, usize>,
    y: PyReadonlyArray1<'py, f64>,
    config: &Bound<'py, PyAny>,
    n_levels: Option<Vec<usize>>,
    weights: Option<PyReadonlyArray1<'py, f64>>,
    layout: Option<&str>,
) -> PyResult<PySolveResult> {
    let cats: Array2<usize> = categories.as_array().to_owned();
    let y_vec: Vec<f64> = y.as_array().to_vec();
    let w_vec: Option<Vec<f64>> = weights.map(|w| w.as_array().to_vec());

    let n_levels = match n_levels {
        Some(nl) => nl,
        None => {
            let n_factors = cats.ncols();
            (0..n_factors)
                .map(|col| cats.column(col).iter().copied().max().unwrap_or(0) + 1)
                .collect()
        }
    };

    let layout_str = layout.unwrap_or("factor_major");
    match layout_str {
        "factor_major" | "row_major" | "compressed" => {}
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Unknown layout: '{}'. Expected 'factor_major', 'row_major', or 'compressed'",
                layout_str
            )));
        }
    }

    let params = extract_solver_params(py, config)?;

    let result = py
        .allow_threads(|| {
            dispatch_solve(cats.view(), &n_levels, w_vec, layout_str, &y_vec, &params)
        })
        .map_err(|err| PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string()))?;

    Ok(into_py_result(py, result))
}

fn dispatch_solve(
    categories: ArrayView2<usize>,
    n_levels: &[usize],
    weights: Option<Vec<f64>>,
    layout: &str,
    y: &[f64],
    params: &SolverParams,
) -> WithinResult<SolveResult> {
    let obs_weights = match weights {
        Some(w) => ObservationWeights::Dense(w),
        None => ObservationWeights::Unit,
    };

    macro_rules! dispatch_on_layout {
        ($Store:ty) => {{
            let store = <$Store>::from_array(categories, obs_weights);
            let design = WeightedDesign::from_store(store, n_levels)?;
            solve_with_design(&design, y, params)
        }};
    }

    match layout {
        "row_major" => dispatch_on_layout!(RowMajorStore),
        "compressed" => dispatch_on_layout!(CompressedStore),
        _ => dispatch_on_layout!(FactorMajorStore),
    }
}

fn solve_with_design<S: ObservationStore>(
    design: &WeightedDesign<S>,
    y: &[f64],
    params: &SolverParams,
) -> WithinResult<SolveResult> {
    solve_least_squares(design, y, None, params)
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

fn randn(rng: &mut SmallRng) -> f64 {
    let u1: f64 = 1.0 - rng.random::<f64>();
    let u2: f64 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

#[pyfunction]
#[allow(clippy::type_complexity)]
#[pyo3(signature = (n_levels, n_rows, seed=None))]
pub fn py_generate_synthetic_data<'py>(
    py: Python<'py>,
    n_levels: Vec<usize>,
    n_rows: usize,
    seed: Option<u64>,
) -> PyResult<(
    Bound<'py, PyArray2<usize>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let mut rng = match seed {
        Some(s) => SmallRng::seed_from_u64(s),
        None => SmallRng::from_os_rng(),
    };

    let q_count = n_levels.len();
    let mut cats = Array2::<usize>::zeros((n_rows, q_count));
    for (q, &nl) in n_levels.iter().enumerate() {
        for i in 0..n_rows {
            cats[(i, q)] = rng.random_range(0..nl);
        }
    }

    let store = FactorMajorStore::from_array(cats.view(), ObservationWeights::Unit);
    let design = WeightedDesign::from_store(store, &n_levels)
        .map_err(|err| PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string()))?;
    let op = DesignOperator::new(&design);

    let n_dofs = design.n_dofs;
    let true_coeffs: Vec<f64> = (0..n_dofs).map(|_| randn(&mut rng)).collect();

    let mut y = vec![0.0f64; n_rows];
    op.matvec_d(&true_coeffs, &mut y);
    for yi in &mut y {
        *yi += 0.1 * randn(&mut rng);
    }

    let cats_py = cats.into_pyarray(py);
    let coeffs_py = Array1::from_vec(true_coeffs).into_pyarray(py);
    let y_py = Array1::from_vec(y).into_pyarray(py);

    Ok((cats_py, coeffs_py, y_py))
}
