//! R API for the `within` fixed-effects solver via extendr.

use extendr_api::prelude::*;
use ndarray::{ArrayView2, ShapeBuilder};

use within::config::{
    ApproxSchurConfig, KrylovMethod, LocalSolverConfig, OperatorRepr, Preconditioner, SolverParams,
    DEFAULT_DENSE_SCHUR_THRESHOLD,
};
use within::domain::WeightedDesign;
use within::observation::{FactorMajorStore, ObservationWeights};
use within::{
    solve as solve_native, solve_batch as solve_batch_native, FePreconditioner, SolveResult,
    Solver as NativeSolver,
};

// ---------------------------------------------------------------------------
// Persistent handles
// ---------------------------------------------------------------------------

struct SolverHandle {
    solver: NativeSolver<FactorMajorStore>,
}

struct PreconditionerHandle {
    preconditioner: FePreconditioner,
}

enum SolverPreconditionerArg {
    Config(Option<Preconditioner>),
    Built(FePreconditioner),
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn parse_operator(operator: Option<&str>) -> Result<OperatorRepr> {
    match operator.unwrap_or("implicit") {
        "implicit" => Ok(OperatorRepr::Implicit),
        "explicit" => Ok(OperatorRepr::Explicit),
        other => Err(Error::Other(format!(
            "Unknown operator '{}'. Use 'implicit' or 'explicit'.",
            other,
        ))),
    }
}

/// Parse method + scalar options into `SolverParams`.
fn build_params(
    method: Option<&str>,
    tol: Option<f64>,
    maxiter: Option<i32>,
    restart: Option<i32>,
    operator: Option<&str>,
) -> Result<SolverParams> {
    let tol = tol.unwrap_or(1e-8);
    if !tol.is_finite() || tol <= 0.0 {
        return Err(Error::Other("tol must be a positive finite number".to_string()));
    }

    let maxiter_i = maxiter.unwrap_or(1000);
    if maxiter_i <= 0 {
        return Err(Error::Other("maxiter must be >= 1".to_string()));
    }
    let maxiter = maxiter_i as usize;

    let krylov = match method.unwrap_or("cg") {
        "cg" => KrylovMethod::Cg,
        "gmres" => {
            let restart_i = restart.unwrap_or(30);
            if restart_i <= 0 {
                return Err(Error::Other("restart must be >= 1".to_string()));
            }
            KrylovMethod::Gmres {
                restart: restart_i as usize,
            }
        }
        other => {
            return Err(Error::Other(format!(
                "Unknown method '{}'. Use 'cg' or 'gmres'.",
                other,
            )))
        }
    };

    Ok(SolverParams {
        krylov,
        operator: parse_operator(operator)?,
        tol,
        maxiter,
    })
}

fn parse_nonnegative_integer(field: &Robj, name: &str) -> Result<i64> {
    let v = if let Some(v) = field.as_integer() {
        v as f64
    } else if let Some(v) = field.as_real() {
        v
    } else {
        return Err(Error::Other(format!("{} must be numeric", name)));
    };
    if !v.is_finite() || v < 0.0 || v.fract() != 0.0 {
        return Err(Error::Other(format!(
            "{} must be a non-negative integer",
            name
        )));
    }
    Ok(v as i64)
}

fn parse_positive_u32(field: &Robj, name: &str) -> Result<u32> {
    let v = parse_nonnegative_integer(field, name)?;
    if v < 1 || v > u32::MAX as i64 {
        return Err(Error::Other(format!("{} must be in 1..={} ", name, u32::MAX)));
    }
    Ok(v as u32)
}

fn get_field_or_null(obj: &Robj, field: &str) -> Robj {
    obj.dollar(field).unwrap_or_else(|_| nil_value())
}

fn parse_approx_chol_config(obj: &Robj) -> Result<approx_chol::Config> {
    if obj.is_null() {
        return Ok(approx_chol::Config::default());
    }
    if !obj.inherits("within_approx_chol_config") {
        return Err(Error::Other(
            "approx_chol must be an object created by approx_chol_config(...) or NULL".to_string(),
        ));
    }

    let seed_obj = get_field_or_null(obj, "seed");
    let split_obj = get_field_or_null(obj, "split");

    let seed_i = parse_nonnegative_integer(&seed_obj, "approx_chol$seed")?;
    if seed_i > u64::MAX as i64 {
        return Err(Error::Other("approx_chol$seed is too large".to_string()));
    }
    let split = parse_positive_u32(&split_obj, "approx_chol$split")?;

    Ok(approx_chol::Config {
        seed: seed_i as u64,
        split_merge: if split > 1 { Some(split) } else { None },
    })
}

fn parse_approx_schur_config(obj: &Robj) -> Result<Option<ApproxSchurConfig>> {
    if obj.is_null() {
        return Ok(None);
    }
    if !obj.inherits("within_approx_schur_config") {
        return Err(Error::Other(
            "approx_schur must be an object created by approx_schur_config(...) or NULL"
                .to_string(),
        ));
    }

    let seed_obj = get_field_or_null(obj, "seed");
    let split_obj = get_field_or_null(obj, "split");

    let seed_i = parse_nonnegative_integer(&seed_obj, "approx_schur$seed")?;
    if seed_i > u64::MAX as i64 {
        return Err(Error::Other("approx_schur$seed is too large".to_string()));
    }
    let split = parse_positive_u32(&split_obj, "approx_schur$split")?;

    Ok(Some(ApproxSchurConfig {
        seed: seed_i as u64,
        split,
    }))
}

fn parse_local_solver_config(obj: &Robj) -> Result<LocalSolverConfig> {
    if obj.is_null() {
        return Ok(LocalSolverConfig::solver_default());
    }

    if obj.inherits("within_schur_complement") {
        let approx_chol_obj = get_field_or_null(obj, "approx_chol");
        let approx_schur_obj = get_field_or_null(obj, "approx_schur");
        let dense_threshold_obj = get_field_or_null(obj, "dense_threshold");

        let dense_threshold_i = parse_nonnegative_integer(
            &dense_threshold_obj,
            "schur_complement$dense_threshold",
        )?;
        let dense_threshold = usize::try_from(dense_threshold_i).map_err(|_| {
            Error::Other("schur_complement$dense_threshold is too large".to_string())
        })?;

        return Ok(LocalSolverConfig::SchurComplement {
            approx_chol: parse_approx_chol_config(&approx_chol_obj)?,
            approx_schur: parse_approx_schur_config(&approx_schur_obj)?,
            dense_threshold,
        });
    }

    if obj.inherits("within_full_sddm") {
        let approx_chol_obj = get_field_or_null(obj, "approx_chol");
        return Ok(LocalSolverConfig::FullSddm {
            approx_chol: parse_approx_chol_config(&approx_chol_obj)?,
        });
    }

    Err(Error::Other(
        "local_solver must be schur_complement(...), full_sddm(...), or NULL".to_string(),
    ))
}

fn parse_preconditioner_string(name: &str) -> Result<Option<Preconditioner>> {
    match name {
        "additive" => Ok(Some(Preconditioner::Additive(
            LocalSolverConfig::solver_default(),
        ))),
        "multiplicative" => Ok(Some(Preconditioner::Multiplicative(
            LocalSolverConfig::solver_default(),
        ))),
        "off" => Ok(None),
        other => Err(Error::Other(format!(
            "Unknown preconditioner '{}'. Use 'additive', 'multiplicative', or 'off'.",
            other,
        ))),
    }
}

/// Parse preconditioner config from R.
///
/// Accepted values:
/// - `NULL` -> additive Schwarz default
/// - `"additive"`, `"multiplicative"`, `"off"`
/// - `additive_schwarz(local_solver=...)`
/// - `multiplicative_schwarz(local_solver=...)`
fn parse_preconditioner_config(preconditioner: Option<Robj>) -> Result<Option<Preconditioner>> {
    match preconditioner {
        None => Ok(Some(Preconditioner::Additive(
            LocalSolverConfig::solver_default(),
        ))),
        Some(obj) => {
            if obj.is_null() {
                return Ok(Some(Preconditioner::Additive(
                    LocalSolverConfig::solver_default(),
                )));
            }

            if let Some(name) = obj.as_str() {
                return parse_preconditioner_string(name);
            }

            if obj.inherits("within_additive_schwarz") {
                let local_solver = get_field_or_null(&obj, "local_solver");
                let cfg = parse_local_solver_config(&local_solver)?;
                return Ok(Some(Preconditioner::Additive(cfg)));
            }

            if obj.inherits("within_multiplicative_schwarz") {
                let local_solver = get_field_or_null(&obj, "local_solver");
                let cfg = parse_local_solver_config(&local_solver)?;
                return Ok(Some(Preconditioner::Multiplicative(cfg)));
            }

            Err(Error::Other(
                "preconditioner must be one of: NULL, 'additive', 'multiplicative', 'off', \
                 additive_schwarz(...), multiplicative_schwarz(...)"
                    .to_string(),
            ))
        }
    }
}

fn clone_preconditioner(p: &FePreconditioner) -> Result<FePreconditioner> {
    let bytes = postcard::to_stdvec(p).map_err(|e| Error::Other(e.to_string()))?;
    postcard::from_bytes(&bytes).map_err(|e| Error::Other(e.to_string()))
}

fn parse_solver_preconditioner(preconditioner: Option<Robj>) -> Result<SolverPreconditionerArg> {
    match preconditioner {
        None => Ok(SolverPreconditionerArg::Config(Some(
            Preconditioner::Additive(LocalSolverConfig::solver_default()),
        ))),
        Some(obj) => {
            if obj.is_null() {
                return Ok(SolverPreconditionerArg::Config(Some(
                    Preconditioner::Additive(LocalSolverConfig::solver_default()),
                )));
            }

            if let Ok(ptr) = ExternalPtr::<PreconditionerHandle>::try_from(obj.clone()) {
                let handle = ptr.try_addr()?;
                let precond = clone_preconditioner(&handle.preconditioner)?;
                return Ok(SolverPreconditionerArg::Built(precond));
            }

            let cfg = parse_preconditioner_config(Some(obj))?;
            Ok(SolverPreconditionerArg::Config(cfg))
        }
    }
}

/// Validate that CG is not paired with a multiplicative preconditioner.
fn validate_cg_preconditioner(
    params: &SolverParams,
    preconditioner: &Option<Preconditioner>,
) -> Result<()> {
    if matches!(params.krylov, KrylovMethod::Cg)
        && matches!(preconditioner, Some(Preconditioner::Multiplicative(_)))
    {
        return Err(Error::Other(
            "CG requires a symmetric preconditioner; use 'additive' or switch to method='gmres'"
                .to_string(),
        ));
    }
    Ok(())
}

/// Convert a `SolveResult` into a named R list.
fn result_to_list(r: SolveResult) -> List {
    list!(
        x = r.x,
        demeaned = r.demeaned,
        converged = r.converged,
        iterations = r.iterations as i32,
        residual = r.final_residual,
        time_total = r.time_total,
        time_setup = r.time_setup,
        time_solve = r.time_solve
    )
}

/// Convert a `BatchSolveResult` into a named R list.
fn batch_result_to_list(result: within::BatchSolveResult) -> List {
    let n_rhs = result.n_rhs();
    let n_dofs = if n_rhs > 0 {
        result.x_all().len() / n_rhs
    } else {
        0
    };
    let n_obs = if n_rhs > 0 {
        result.demeaned_all().len() / n_rhs
    } else {
        0
    };

    let mut x_mat = RMatrix::new(n_dofs, n_rhs);
    x_mat.data_mut().copy_from_slice(result.x_all());

    let mut dem_mat = RMatrix::new(n_obs, n_rhs);
    dem_mat.data_mut().copy_from_slice(result.demeaned_all());

    let converged: Vec<bool> = result.converged().to_vec();
    let iterations: Vec<i32> = result.iterations().iter().map(|&v| v as i32).collect();
    let residual: Vec<f64> = result.final_residual().to_vec();
    let time_solve: Vec<f64> = result.time_solve().to_vec();

    list!(
        x = x_mat,
        demeaned = dem_mat,
        converged = converged,
        iterations = iterations,
        residual = residual,
        time_solve = time_solve,
        time_total = result.time_total()
    )
}

/// Cast an i32 slice to a Vec<u32>, checking for negative values.
fn cast_categories(data: &[i32]) -> Result<Vec<u32>> {
    if let Some(pos) = data.iter().position(|&v| v < 0) {
        return Err(Error::Other(format!(
            "categories must be non-negative integers, found {} at position {}",
            data[pos], pos,
        )));
    }
    Ok(data.iter().map(|&v| v as u32).collect())
}

/// Extract weights from Nullable<Doubles> as Option<Vec<f64>>.
fn extract_weights(weights: Nullable<Doubles>) -> Option<Vec<f64>> {
    match weights {
        Nullable::NotNull(d) => Some(d.iter().map(|v| v.inner()).collect()),
        Nullable::Null => None,
    }
}

fn build_design(
    categories: &RMatrix<i32>,
    weights: Nullable<Doubles>,
) -> Result<WeightedDesign<FactorMajorStore>> {
    let n_obs = categories.nrows();
    let n_factors = categories.ncols();

    let cats_u32 = cast_categories(categories.data())?;
    let cats_view = ArrayView2::from_shape((n_obs, n_factors).f(), &cats_u32)
        .map_err(|e| Error::Other(e.to_string()))?;

    let factor_levels: Vec<Vec<u32>> = (0..n_factors)
        .map(|f| cats_view.column(f).iter().copied().collect())
        .collect();

    let weights = match extract_weights(weights) {
        Some(w) => ObservationWeights::Dense(w),
        None => ObservationWeights::Unit,
    };

    let store =
        FactorMajorStore::new(factor_levels, weights, n_obs).map_err(|e| Error::Other(e.to_string()))?;
    WeightedDesign::from_store(store).map_err(|e| Error::Other(e.to_string()))
}

// ---------------------------------------------------------------------------
// One-shot solve API
// ---------------------------------------------------------------------------

/// Solve fixed-effects demeaning for a single response vector.
///
/// @param categories Integer matrix (n_obs x n_factors) of factor levels (0-based).
/// @param y Numeric vector of length n_obs.
/// @param method Solver method: "cg" (default) or "gmres".
/// @param tol Convergence tolerance (default 1e-8).
/// @param maxiter Maximum iterations (default 1000).
/// @param restart GMRES restart parameter (default 30, ignored for CG).
/// @param preconditioner `NULL`, string ("additive"/"multiplicative"/"off"), or
///   advanced preconditioner config created by additive_schwarz(...) /
///   multiplicative_schwarz(...).
/// @param weights Optional observation weights (length n_obs).
/// @param operator Operator representation: "implicit" (default) or "explicit".
/// @return A named list with components x, demeaned, converged, iterations,
///         residual, time_total, time_setup, time_solve.
#[extendr]
fn solve(
    categories: RMatrix<i32>,
    y: &[f64],
    method: Nullable<&str>,
    tol: Nullable<f64>,
    maxiter: Nullable<i32>,
    restart: Nullable<i32>,
    preconditioner: Nullable<Robj>,
    weights: Nullable<Doubles>,
    operator: Nullable<&str>,
) -> Result<List> {
    let method: Option<&str> = method.into();
    let tol: Option<f64> = tol.into();
    let maxiter: Option<i32> = maxiter.into();
    let restart: Option<i32> = restart.into();
    let preconditioner: Option<Robj> = preconditioner.into();
    let operator: Option<&str> = operator.into();

    let nrow = categories.nrows();
    let ncol = categories.ncols();

    let cats_u32 = cast_categories(categories.data())?;
    let cats_view = ArrayView2::from_shape((nrow, ncol).f(), &cats_u32)
        .map_err(|e| Error::Other(e.to_string()))?;

    let params = build_params(method, tol, maxiter, restart, operator)?;
    let precond = parse_preconditioner_config(preconditioner)?;
    validate_cg_preconditioner(&params, &precond)?;

    let w_vec = extract_weights(weights);
    let w_ref = w_vec.as_deref();

    let result = solve_native(cats_view, y, w_ref, &params, precond.as_ref())
        .map_err(|e| Error::Other(e.to_string()))?;

    Ok(result_to_list(result))
}

/// Solve fixed-effects demeaning for multiple response vectors (batch).
///
/// @param categories Integer matrix (n_obs x n_factors) of factor levels (0-based).
/// @param Y Numeric matrix (n_obs x k) where each column is a response vector.
/// @param method Solver method: "cg" (default) or "gmres".
/// @param tol Convergence tolerance (default 1e-8).
/// @param maxiter Maximum iterations (default 1000).
/// @param restart GMRES restart parameter (default 30, ignored for CG).
/// @param preconditioner `NULL`, string ("additive"/"multiplicative"/"off"), or
///   advanced preconditioner config created by additive_schwarz(...) /
///   multiplicative_schwarz(...).
/// @param weights Optional observation weights (length n_obs).
/// @param operator Operator representation: "implicit" (default) or "explicit".
/// @return A named list with components x, demeaned (matrices), converged,
///         iterations, residual, time_solve (vectors), and time_total (scalar).
#[extendr]
fn solve_batch(
    categories: RMatrix<i32>,
    y: RMatrix<f64>,
    method: Nullable<&str>,
    tol: Nullable<f64>,
    maxiter: Nullable<i32>,
    restart: Nullable<i32>,
    preconditioner: Nullable<Robj>,
    weights: Nullable<Doubles>,
    operator: Nullable<&str>,
) -> Result<List> {
    let method: Option<&str> = method.into();
    let tol: Option<f64> = tol.into();
    let maxiter: Option<i32> = maxiter.into();
    let restart: Option<i32> = restart.into();
    let preconditioner: Option<Robj> = preconditioner.into();
    let operator: Option<&str> = operator.into();

    let nrow = categories.nrows();
    let ncol = categories.ncols();

    let cats_u32 = cast_categories(categories.data())?;
    let cats_view = ArrayView2::from_shape((nrow, ncol).f(), &cats_u32)
        .map_err(|e| Error::Other(e.to_string()))?;

    let params = build_params(method, tol, maxiter, restart, operator)?;
    let precond = parse_preconditioner_config(preconditioner)?;
    validate_cg_preconditioner(&params, &precond)?;

    let w_vec = extract_weights(weights);
    let w_ref = w_vec.as_deref();

    let y_data = y.data();
    let y_nrow = y.nrows();
    let y_ncol = y.ncols();

    let columns: Vec<Vec<f64>> = (0..y_ncol)
        .map(|j| y_data[j * y_nrow..(j + 1) * y_nrow].to_vec())
        .collect();
    let column_refs: Vec<&[f64]> = columns.iter().map(Vec::as_slice).collect();

    let result = solve_batch_native(cats_view, &column_refs, w_ref, &params, precond.as_ref())
        .map_err(|e| Error::Other(e.to_string()))?;

    Ok(batch_result_to_list(result))
}

// ---------------------------------------------------------------------------
// Persistent solver API
// ---------------------------------------------------------------------------

/// Build a persistent solver that can be reused across multiple solves.
#[extendr]
fn solver_new(
    categories: RMatrix<i32>,
    method: Nullable<&str>,
    tol: Nullable<f64>,
    maxiter: Nullable<i32>,
    restart: Nullable<i32>,
    preconditioner: Nullable<Robj>,
    weights: Nullable<Doubles>,
    operator: Nullable<&str>,
) -> Result<ExternalPtr<SolverHandle>> {
    let method: Option<&str> = method.into();
    let tol: Option<f64> = tol.into();
    let maxiter: Option<i32> = maxiter.into();
    let restart: Option<i32> = restart.into();
    let preconditioner: Option<Robj> = preconditioner.into();
    let operator: Option<&str> = operator.into();

    let params = build_params(method, tol, maxiter, restart, operator)?;
    let precond_arg = parse_solver_preconditioner(preconditioner)?;
    let design = build_design(&categories, weights)?;

    let solver = match precond_arg {
        SolverPreconditionerArg::Built(p) => {
            NativeSolver::from_design_with_preconditioner(design, &params, p)
        }
        SolverPreconditionerArg::Config(cfg) => {
            validate_cg_preconditioner(&params, &cfg)?;
            NativeSolver::from_design(design, &params, cfg.as_ref())
        }
    }
    .map_err(|e| Error::Other(e.to_string()))?;

    Ok(ExternalPtr::new(SolverHandle { solver }))
}

/// Solve one response vector with a persistent solver.
#[extendr]
fn solver_solve(solver: ExternalPtr<SolverHandle>, y: &[f64]) -> Result<List> {
    let handle = solver.try_addr()?;
    let result = handle
        .solver
        .solve(y)
        .map_err(|e| Error::Other(e.to_string()))?;
    Ok(result_to_list(result))
}

/// Solve multiple response vectors with a persistent solver.
#[extendr]
fn solver_solve_batch(solver: ExternalPtr<SolverHandle>, y: RMatrix<f64>) -> Result<List> {
    let handle = solver.try_addr()?;

    let y_data = y.data();
    let y_nrow = y.nrows();
    let y_ncol = y.ncols();

    let columns: Vec<Vec<f64>> = (0..y_ncol)
        .map(|j| y_data[j * y_nrow..(j + 1) * y_nrow].to_vec())
        .collect();
    let column_refs: Vec<&[f64]> = columns.iter().map(Vec::as_slice).collect();

    let result = handle
        .solver
        .solve_batch(&column_refs)
        .map_err(|e| Error::Other(e.to_string()))?;
    Ok(batch_result_to_list(result))
}

/// Return the built preconditioner from a persistent solver, or NULL.
#[extendr]
fn solver_preconditioner(
    solver: ExternalPtr<SolverHandle>,
) -> Result<Option<ExternalPtr<PreconditionerHandle>>> {
    let handle = solver.try_addr()?;
    match handle.solver.preconditioner() {
        None => Ok(None),
        Some(p) => Ok(Some(ExternalPtr::new(PreconditionerHandle {
            preconditioner: clone_preconditioner(p)?,
        }))),
    }
}

/// Number of DOFs (coefficients) in the persistent solver.
#[extendr]
fn solver_n_dofs(solver: ExternalPtr<SolverHandle>) -> Result<i32> {
    let handle = solver.try_addr()?;
    i32::try_from(handle.solver.n_dofs())
        .map_err(|_| Error::Other("n_dofs exceeds i32 range".to_string()))
}

/// Number of observations in the persistent solver.
#[extendr]
fn solver_n_obs(solver: ExternalPtr<SolverHandle>) -> Result<i32> {
    let handle = solver.try_addr()?;
    i32::try_from(handle.solver.n_obs())
        .map_err(|_| Error::Other("n_obs exceeds i32 range".to_string()))
}

// ---------------------------------------------------------------------------
// Preconditioner handle API
// ---------------------------------------------------------------------------

/// Apply a built preconditioner: y = M^{-1} x.
#[extendr]
fn fe_preconditioner_apply(
    preconditioner: ExternalPtr<PreconditionerHandle>,
    x: &[f64],
) -> Result<Vec<f64>> {
    let handle = preconditioner.try_addr()?;
    if x.len() != handle.preconditioner.ncols() {
        return Err(Error::Other(format!(
            "x has length {} but preconditioner expects {}",
            x.len(),
            handle.preconditioner.ncols()
        )));
    }
    let mut y = vec![0.0; handle.preconditioner.nrows()];
    handle.preconditioner.apply(x, &mut y);
    Ok(y)
}

/// Number of rows in a built preconditioner.
#[extendr]
fn fe_preconditioner_nrows(preconditioner: ExternalPtr<PreconditionerHandle>) -> Result<i32> {
    let handle = preconditioner.try_addr()?;
    i32::try_from(handle.preconditioner.nrows())
        .map_err(|_| Error::Other("nrows exceeds i32 range".to_string()))
}

/// Number of columns in a built preconditioner.
#[extendr]
fn fe_preconditioner_ncols(preconditioner: ExternalPtr<PreconditionerHandle>) -> Result<i32> {
    let handle = preconditioner.try_addr()?;
    i32::try_from(handle.preconditioner.ncols())
        .map_err(|_| Error::Other("ncols exceeds i32 range".to_string()))
}

/// Serialize a built preconditioner into raw bytes.
#[extendr]
fn fe_preconditioner_serialize(preconditioner: ExternalPtr<PreconditionerHandle>) -> Result<Raw> {
    let handle = preconditioner.try_addr()?;
    let bytes =
        postcard::to_stdvec(&handle.preconditioner).map_err(|e| Error::Other(e.to_string()))?;
    Ok(Raw::from_bytes(&bytes))
}

/// Deserialize a built preconditioner from raw bytes.
#[extendr]
fn fe_preconditioner_deserialize(data: Raw) -> Result<ExternalPtr<PreconditionerHandle>> {
    let preconditioner: FePreconditioner =
        postcard::from_bytes(data.as_slice()).map_err(|e| Error::Other(e.to_string()))?;
    Ok(ExternalPtr::new(PreconditionerHandle { preconditioner }))
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

extendr_module! {
    mod within;
    fn solve;
    fn solve_batch;
    fn solver_new;
    fn solver_solve;
    fn solver_solve_batch;
    fn solver_preconditioner;
    fn solver_n_dofs;
    fn solver_n_obs;
    fn fe_preconditioner_apply;
    fn fe_preconditioner_nrows;
    fn fe_preconditioner_ncols;
    fn fe_preconditioner_serialize;
    fn fe_preconditioner_deserialize;
}
