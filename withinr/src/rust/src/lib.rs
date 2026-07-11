// Extendr bridge exposing the `within` Rust crate to R as the `withinr` package.
//
// The bridge mirrors the Python binding shape: small R-facing configuration
// objects are converted to native `within` config values, then all numerical
// work is delegated to the Rust crate.
//
// R users pass **1-based** integer category matrices. This bridge validates
// that all entries are >= 1 and contain no `NA`, then subtracts 1 to produce
// the **0-based** `u32` indices expected by the Rust solver.

use extendr_api::prelude::*;
use ndarray::{ArrayView2, ShapeBuilder};
use std::time::Instant;

use within::config::{
    ApproxCholConfig, ApproxSchurConfig, LocalSolverConfig, LsmrOptions, PreconditionerConfig,
    ReductionStrategy,
};
use within::observation::FactorMajorStore;
use within::{
    solve as solve_native, solve_batch as solve_batch_native, Design, Preconditioner, SolveResult,
    Solver as NativeSolver,
};

// ---------------------------------------------------------------------------
// Persistent handles
// ---------------------------------------------------------------------------

struct SolverHandle {
    solver: NativeSolver<FactorMajorStore>,
    preconditioner_build_time_seconds: Option<f64>,
}

struct PreconditionerHandle {
    preconditioner: Preconditioner,
    build_time_seconds: Option<f64>,
}

enum PreconditionerArg {
    Config(Option<PreconditionerConfig>),
    Built {
        preconditioner: Preconditioner,
        build_time_seconds: Option<f64>,
    },
}

// ---------------------------------------------------------------------------
// Small conversion helpers
// ---------------------------------------------------------------------------

fn err(message: impl Into<String>) -> Error {
    Error::Other(message.into())
}

fn or_throw<T>(result: Result<T>) -> T {
    match result {
        Ok(value) => value,
        Err(error) => throw_r_error(error.to_string()),
    }
}

fn usize_to_i32(value: usize, name: &str) -> Result<i32> {
    i32::try_from(value).map_err(|_| err(format!("{name} exceeds i32 range")))
}

fn parse_nonnegative_integer(field: &Robj, name: &str) -> Result<i64> {
    let value = if let Some(value) = field.as_integer() {
        value as f64
    } else if let Some(value) = field.as_real() {
        value
    } else {
        return Err(err(format!("{name} must be numeric")));
    };

    if !value.is_finite() || value < 0.0 || value.fract() != 0.0 {
        return Err(err(format!("{name} must be a non-negative integer")));
    }
    Ok(value as i64)
}

fn parse_positive_u32(field: &Robj, name: &str) -> Result<u32> {
    let value = parse_nonnegative_integer(field, name)?;
    if value < 1 || value > u32::MAX as i64 {
        return Err(err(format!("{name} must be in 1..={}", u32::MAX)));
    }
    Ok(value as u32)
}

fn parse_optional_u32(field: &Robj, name: &str) -> Result<Option<u32>> {
    if field.is_null() {
        return Ok(None);
    }
    Ok(Some(parse_positive_u32(field, name)?))
}

fn get_field_or_null(obj: &Robj, field: &str) -> Robj {
    obj.dollar(field).unwrap_or_else(|_| nil_value())
}

fn extract_weights(weights: Robj) -> Result<Option<Vec<f64>>> {
    if weights.is_null() {
        return Ok(None);
    }
    weights
        .as_real_vector()
        .map(Some)
        .ok_or_else(|| err("weights must be a numeric vector or NULL"))
}

/// Convert an R integer matrix (1-based, column-major) to a 0-based `u32`
/// buffer preserving R's column-major layout.
fn cast_categories(data: &[i32]) -> Result<Vec<u32>> {
    let mut out = Vec::with_capacity(data.len());
    for (pos, &value) in data.iter().enumerate() {
        if value == i32::MIN {
            return Err(err(format!(
                "categories must not contain NA values (found at position {})",
                pos + 1
            )));
        }
        if value < 1 {
            return Err(err(format!(
                "categories must be >= 1 (1-based); found {value} at position {}",
                pos + 1
            )));
        }
        out.push((value - 1) as u32);
    }
    Ok(out)
}

fn categories_view<'a>(
    categories: &RMatrix<i32>,
    cats_u32: &'a [u32],
) -> Result<ArrayView2<'a, u32>> {
    ArrayView2::from_shape((categories.nrows(), categories.ncols()).f(), cats_u32)
        .map_err(|e| err(e.to_string()))
}

fn factor_major_store(categories: &RMatrix<i32>) -> Result<FactorMajorStore> {
    let n_obs = categories.nrows();
    let n_factors = categories.ncols();
    let cats_u32 = cast_categories(categories.data())?;
    let cats = categories_view(categories, &cats_u32)?;
    let factor_levels = (0..n_factors)
        .map(|factor| cats.column(factor).iter().copied().collect())
        .collect();
    FactorMajorStore::new(factor_levels, n_obs).map_err(|e| err(e.to_string()))
}

fn build_lsmr(tol: f64, maxiter: i32, local_size: Nullable<i32>) -> Result<LsmrOptions> {
    if !tol.is_finite() || tol <= 0.0 {
        return Err(err("tol must be a positive finite number"));
    }
    if maxiter < 1 {
        return Err(err("maxiter must be >= 1"));
    }

    let local_size: Option<i32> = local_size.into();
    let local_size = match local_size {
        None => None,
        Some(value) if value >= 1 => Some(value as usize),
        Some(_) => return Err(err("local_size must be NULL or >= 1")),
    };

    Ok(LsmrOptions {
        tol,
        maxiter: maxiter as usize,
        local_size,
    })
}

// ---------------------------------------------------------------------------
// Preconditioner config conversion
// ---------------------------------------------------------------------------

fn parse_approx_chol_config(obj: &Robj) -> Result<ApproxCholConfig> {
    if obj.is_null() {
        return Ok(LocalSolverConfig::default().approx_chol);
    }
    if !obj.inherits("within_approx_chol_config") {
        return Err(err(
            "approx_chol must be an object created by ApproxCholConfig(...) or NULL",
        ));
    }

    let seed_obj = get_field_or_null(obj, "seed");
    let split_merge_obj = get_field_or_null(obj, "split_merge");

    let seed = parse_nonnegative_integer(&seed_obj, "approx_chol$seed")?;

    Ok(ApproxCholConfig {
        seed: seed as u64,
        split_merge: parse_optional_u32(&split_merge_obj, "approx_chol$split_merge")?,
    })
}

fn parse_approx_schur_config(obj: &Robj) -> Result<Option<ApproxSchurConfig>> {
    if obj.is_null() {
        return Ok(None);
    }
    if !obj.inherits("within_approx_schur_config") {
        return Err(err(
            "approx_schur must be an object created by ApproxSchurConfig(...) or NULL",
        ));
    }

    let seed_obj = get_field_or_null(obj, "seed");
    let split_obj = get_field_or_null(obj, "split");

    let seed = parse_nonnegative_integer(&seed_obj, "approx_schur$seed")?;

    Ok(Some(ApproxSchurConfig {
        seed: seed as u64,
        split: parse_positive_u32(&split_obj, "approx_schur$split")?,
    }))
}

fn parse_local_solver_config(obj: &Robj) -> Result<LocalSolverConfig> {
    if obj.is_null() {
        return Ok(LocalSolverConfig::default());
    }
    if !obj.inherits("within_local_solver_config") {
        return Err(err(
            "local_solver must be an object created by LocalSolverConfig(...) or NULL",
        ));
    }

    let approx_chol_obj = get_field_or_null(obj, "approx_chol");
    let approx_schur_obj = get_field_or_null(obj, "approx_schur");
    let dense_threshold_obj = get_field_or_null(obj, "dense_threshold");

    let dense_threshold = if dense_threshold_obj.is_null() {
        LocalSolverConfig::default().dense_threshold
    } else {
        let value =
            parse_nonnegative_integer(&dense_threshold_obj, "local_solver$dense_threshold")?;
        usize::try_from(value).map_err(|_| err("local_solver$dense_threshold is too large"))?
    };

    Ok(LocalSolverConfig {
        approx_chol: parse_approx_chol_config(&approx_chol_obj)?,
        approx_schur: parse_approx_schur_config(&approx_schur_obj)?,
        dense_threshold,
    })
}

fn parse_reduction_strategy(obj: &Robj) -> Result<ReductionStrategy> {
    if obj.is_null() {
        return Ok(ReductionStrategy::default());
    }
    let Some(name) = obj.as_str() else {
        return Err(err("reduction must be a character scalar"));
    };
    match name {
        "auto" | "Auto" => Ok(ReductionStrategy::Auto),
        "atomic_scatter" | "AtomicScatter" => Ok(ReductionStrategy::AtomicScatter),
        "parallel_reduction" | "ParallelReduction" => Ok(ReductionStrategy::ParallelReduction),
        other => Err(err(format!(
            "unknown reduction strategy '{other}'; use 'auto', 'atomic_scatter', or 'parallel_reduction'"
        ))),
    }
}

fn parse_preconditioner_string(name: &str) -> Result<PreconditionerArg> {
    match name {
        "additive" | "Additive" => Ok(PreconditionerArg::Config(Some(
            PreconditionerConfig::default(),
        ))),
        "off" | "Off" => Ok(PreconditionerArg::Config(Some(PreconditionerConfig::Off))),
        "diagonal" | "Diagonal" => Ok(PreconditionerArg::Config(Some(
            PreconditionerConfig::Diagonal,
        ))),
        other => Err(err(format!(
            "unknown preconditioner '{other}'; use 'additive', 'off', 'diagonal', AdditiveSchwarz(...), a Preconditioner, or NULL"
        ))),
    }
}

fn parse_preconditioner(preconditioner: Robj) -> Result<PreconditionerArg> {
    if preconditioner.is_null() {
        return Ok(PreconditionerArg::Config(None));
    }

    if let Ok(ptr) = ExternalPtr::<PreconditionerHandle>::try_from(preconditioner.clone()) {
        let handle = ptr.try_addr()?;
        return Ok(PreconditionerArg::Built {
            preconditioner: handle.preconditioner.clone(),
            build_time_seconds: handle.build_time_seconds,
        });
    }

    if let Some(name) = preconditioner.as_str() {
        return parse_preconditioner_string(name);
    }

    if preconditioner.inherits("within_additive_schwarz") {
        let local_solver =
            parse_local_solver_config(&get_field_or_null(&preconditioner, "local_solver"))?;
        let reduction = parse_reduction_strategy(&get_field_or_null(&preconditioner, "reduction"))?;
        return Ok(PreconditionerArg::Config(Some(
            PreconditionerConfig::Additive {
                local_solver,
                reduction,
            },
        )));
    }

    Err(err(
        "preconditioner must be NULL, 'additive', 'off', 'diagonal', AdditiveSchwarz(...), or a Preconditioner",
    ))
}

// ---------------------------------------------------------------------------
// Result conversion
// ---------------------------------------------------------------------------

fn result_to_list(result: SolveResult) -> Result<List> {
    Ok(list!(
        x = result.x,
        demeaned = result.demeaned,
        converged = result.converged,
        iterations = usize_to_i32(result.iterations, "iterations")?,
        residual = result.residual,
        time_total = result.time_total,
        time_setup = result.time_setup,
        time_solve = result.time_solve
    ))
}

fn batch_result_to_list(result: within::BatchSolveResult) -> Result<List> {
    let n_rhs = result.converged.len();

    let mut x = RMatrix::new(result.n_dofs, n_rhs);
    x.data_mut().copy_from_slice(&result.x);

    let mut demeaned = RMatrix::new(result.n_obs, n_rhs);
    demeaned.data_mut().copy_from_slice(&result.demeaned);

    let iterations = result
        .iterations
        .iter()
        .map(|&value| usize_to_i32(value, "iterations"))
        .collect::<Result<Vec<_>>>()?;

    Ok(list!(
        x = x,
        demeaned = demeaned,
        converged = result.converged,
        iterations = iterations,
        residual = result.residual,
        time_solve = result.time_solve,
        time_total = result.time_total
    ))
}

// ---------------------------------------------------------------------------
// One-shot solve API
// ---------------------------------------------------------------------------

/// Solve fixed-effects normal equations for a single response vector.
///
/// @export
#[extendr]
fn solve_impl(
    categories: RMatrix<i32>,
    y: &[f64],
    weights: Robj,
    tol: f64,
    maxiter: i32,
    local_size: Nullable<i32>,
    preconditioner: Robj,
) -> List {
    or_throw(solve_impl_inner(
        categories,
        y,
        weights,
        tol,
        maxiter,
        local_size,
        preconditioner,
    ))
}

fn solve_impl_inner(
    categories: RMatrix<i32>,
    y: &[f64],
    weights: Robj,
    tol: f64,
    maxiter: i32,
    local_size: Nullable<i32>,
    preconditioner: Robj,
) -> Result<List> {
    let cats_u32 = cast_categories(categories.data())?;
    let cats = categories_view(&categories, &cats_u32)?;
    let lsmr = build_lsmr(tol, maxiter, local_size)?;
    let weights = extract_weights(weights)?;

    match parse_preconditioner(preconditioner)? {
        PreconditionerArg::Config(config) => {
            solve_native(cats, y, weights.as_deref(), &lsmr, config.as_ref())
                .map_err(|e| err(e.to_string()))
                .and_then(result_to_list)
        }
        PreconditionerArg::Built { preconditioner, .. } => {
            solve_native(cats, y, weights.as_deref(), &lsmr, preconditioner)
                .map_err(|e| err(e.to_string()))
                .and_then(result_to_list)
        }
    }
}

/// Solve fixed-effects normal equations for multiple response vectors.
///
/// @export
#[extendr]
fn solve_batch_impl(
    categories: RMatrix<i32>,
    y_matrix: RMatrix<f64>,
    weights: Robj,
    tol: f64,
    maxiter: i32,
    local_size: Nullable<i32>,
    preconditioner: Robj,
) -> List {
    or_throw(solve_batch_impl_inner(
        categories,
        y_matrix,
        weights,
        tol,
        maxiter,
        local_size,
        preconditioner,
    ))
}

fn solve_batch_impl_inner(
    categories: RMatrix<i32>,
    y_matrix: RMatrix<f64>,
    weights: Robj,
    tol: f64,
    maxiter: i32,
    local_size: Nullable<i32>,
    preconditioner: Robj,
) -> Result<List> {
    if y_matrix.nrows() != categories.nrows() {
        return Err(err(format!(
            "Y has {} rows but categories has {} observations",
            y_matrix.nrows(),
            categories.nrows()
        )));
    }

    let cats_u32 = cast_categories(categories.data())?;
    let cats = categories_view(&categories, &cats_u32)?;
    let lsmr = build_lsmr(tol, maxiter, local_size)?;
    let weights = extract_weights(weights)?;

    let y_data = y_matrix.data();
    let y_nrow = y_matrix.nrows();
    let y_ncol = y_matrix.ncols();
    let columns: Vec<Vec<f64>> = (0..y_ncol)
        .map(|j| y_data[j * y_nrow..(j + 1) * y_nrow].to_vec())
        .collect();
    let column_refs: Vec<&[f64]> = columns.iter().map(Vec::as_slice).collect();

    match parse_preconditioner(preconditioner)? {
        PreconditionerArg::Config(config) => solve_batch_native(
            cats,
            &column_refs,
            weights.as_deref(),
            &lsmr,
            config.as_ref(),
        )
        .map_err(|e| err(e.to_string()))
        .and_then(batch_result_to_list),
        PreconditionerArg::Built { preconditioner, .. } => solve_batch_native(
            cats,
            &column_refs,
            weights.as_deref(),
            &lsmr,
            preconditioner,
        )
        .map_err(|e| err(e.to_string()))
        .and_then(batch_result_to_list),
    }
}

// ---------------------------------------------------------------------------
// Persistent solver API
// ---------------------------------------------------------------------------

/// Build a persistent solver that can be reused across multiple solves.
///
/// @export
#[extendr]
fn solver_new_impl(
    categories: RMatrix<i32>,
    weights: Robj,
    preconditioner: Robj,
) -> ExternalPtr<SolverHandle> {
    or_throw(solver_new_impl_inner(categories, weights, preconditioner))
}

fn solver_new_impl_inner(
    categories: RMatrix<i32>,
    weights: Robj,
    preconditioner: Robj,
) -> Result<ExternalPtr<SolverHandle>> {
    let weights = extract_weights(weights)?;
    let store = factor_major_store(&categories)?;
    let design = Design::from_store(store).map_err(|e| err(e.to_string()))?;

    let (solver, preconditioner_build_time_seconds) = match parse_preconditioner(preconditioner)? {
        PreconditionerArg::Config(config) => {
            let started = Instant::now();
            let solver = NativeSolver::new(design, weights, config.as_ref())
                .map_err(|e| err(e.to_string()))?;
            let build_time_seconds = solver
                .preconditioner()
                .map(|_| started.elapsed().as_secs_f64());
            (solver, build_time_seconds)
        }
        PreconditionerArg::Built {
            preconditioner,
            build_time_seconds,
        } => {
            let solver = NativeSolver::new(design, weights, preconditioner)
                .map_err(|e| err(e.to_string()))?;
            (solver, build_time_seconds)
        }
    };

    Ok(ExternalPtr::new(SolverHandle {
        solver,
        preconditioner_build_time_seconds,
    }))
}

/// Solve one response vector with a persistent solver.
///
/// @export
#[extendr]
fn solver_solve_impl(
    solver: ExternalPtr<SolverHandle>,
    y: &[f64],
    tol: f64,
    maxiter: i32,
    local_size: Nullable<i32>,
) -> List {
    or_throw(solver_solve_impl_inner(solver, y, tol, maxiter, local_size))
}

fn solver_solve_impl_inner(
    solver: ExternalPtr<SolverHandle>,
    y: &[f64],
    tol: f64,
    maxiter: i32,
    local_size: Nullable<i32>,
) -> Result<List> {
    let lsmr = build_lsmr(tol, maxiter, local_size)?;
    let handle = solver.try_addr()?;
    handle
        .solver
        .solve(y, &lsmr)
        .map_err(|e| err(e.to_string()))
        .and_then(result_to_list)
}

/// Solve multiple response vectors with a persistent solver.
///
/// @export
#[extendr]
fn solver_solve_batch_impl(
    solver: ExternalPtr<SolverHandle>,
    y_matrix: RMatrix<f64>,
    tol: f64,
    maxiter: i32,
    local_size: Nullable<i32>,
) -> List {
    or_throw(solver_solve_batch_impl_inner(
        solver, y_matrix, tol, maxiter, local_size,
    ))
}

fn solver_solve_batch_impl_inner(
    solver: ExternalPtr<SolverHandle>,
    y_matrix: RMatrix<f64>,
    tol: f64,
    maxiter: i32,
    local_size: Nullable<i32>,
) -> Result<List> {
    let lsmr = build_lsmr(tol, maxiter, local_size)?;
    let handle = solver.try_addr()?;

    if y_matrix.nrows() != handle.solver.n_obs() {
        return Err(err(format!(
            "Y has {} rows but solver has {} observations",
            y_matrix.nrows(),
            handle.solver.n_obs()
        )));
    }

    let y_data = y_matrix.data();
    let y_nrow = y_matrix.nrows();
    let y_ncol = y_matrix.ncols();
    let columns: Vec<Vec<f64>> = (0..y_ncol)
        .map(|j| y_data[j * y_nrow..(j + 1) * y_nrow].to_vec())
        .collect();
    let column_refs: Vec<&[f64]> = columns.iter().map(Vec::as_slice).collect();

    handle
        .solver
        .solve_batch(&column_refs, &lsmr)
        .map_err(|e| err(e.to_string()))
        .and_then(batch_result_to_list)
}

/// Return the built preconditioner from a persistent solver, or NULL.
///
/// @export
#[extendr]
fn solver_preconditioner_impl(
    solver: ExternalPtr<SolverHandle>,
) -> Option<ExternalPtr<PreconditionerHandle>> {
    or_throw(solver_preconditioner_impl_inner(solver))
}

fn solver_preconditioner_impl_inner(
    solver: ExternalPtr<SolverHandle>,
) -> Result<Option<ExternalPtr<PreconditionerHandle>>> {
    let handle = solver.try_addr()?;
    let build_time_seconds = handle.preconditioner_build_time_seconds;
    Ok(handle.solver.preconditioner().map(|preconditioner| {
        ExternalPtr::new(PreconditionerHandle {
            preconditioner: preconditioner.clone(),
            build_time_seconds,
        })
    }))
}

/// Number of DOFs (coefficients) in the persistent solver.
///
/// @export
#[extendr]
fn solver_n_dofs_impl(solver: ExternalPtr<SolverHandle>) -> i32 {
    or_throw(solver_n_dofs_impl_inner(solver))
}

fn solver_n_dofs_impl_inner(solver: ExternalPtr<SolverHandle>) -> Result<i32> {
    let handle = solver.try_addr()?;
    usize_to_i32(handle.solver.n_dofs(), "n_dofs")
}

/// Number of observations in the persistent solver.
///
/// @export
#[extendr]
fn solver_n_obs_impl(solver: ExternalPtr<SolverHandle>) -> i32 {
    or_throw(solver_n_obs_impl_inner(solver))
}

fn solver_n_obs_impl_inner(solver: ExternalPtr<SolverHandle>) -> Result<i32> {
    let handle = solver.try_addr()?;
    usize_to_i32(handle.solver.n_obs(), "n_obs")
}

// ---------------------------------------------------------------------------
// Preconditioner handle API
// ---------------------------------------------------------------------------

/// Apply a built preconditioner: y = M^{-1} x.
///
/// @export
#[extendr]
fn preconditioner_apply_impl(
    preconditioner: ExternalPtr<PreconditionerHandle>,
    x: &[f64],
) -> Vec<f64> {
    or_throw(preconditioner_apply_impl_inner(preconditioner, x))
}

fn preconditioner_apply_impl_inner(
    preconditioner: ExternalPtr<PreconditionerHandle>,
    x: &[f64],
) -> Result<Vec<f64>> {
    let handle = preconditioner.try_addr()?;
    if x.len() != handle.preconditioner.ncols() {
        return Err(err(format!(
            "x has length {} but preconditioner expects {}",
            x.len(),
            handle.preconditioner.ncols()
        )));
    }
    let mut y = vec![0.0; handle.preconditioner.nrows()];
    handle
        .preconditioner
        .apply(x, &mut y)
        .map_err(|e| err(e.to_string()))?;
    Ok(y)
}

/// Number of rows in a built preconditioner.
///
/// @export
#[extendr]
fn preconditioner_nrows_impl(preconditioner: ExternalPtr<PreconditionerHandle>) -> i32 {
    or_throw(preconditioner_nrows_impl_inner(preconditioner))
}

fn preconditioner_nrows_impl_inner(
    preconditioner: ExternalPtr<PreconditionerHandle>,
) -> Result<i32> {
    let handle = preconditioner.try_addr()?;
    usize_to_i32(handle.preconditioner.nrows(), "nrows")
}

/// Number of columns in a built preconditioner.
///
/// @export
#[extendr]
fn preconditioner_ncols_impl(preconditioner: ExternalPtr<PreconditionerHandle>) -> i32 {
    or_throw(preconditioner_ncols_impl_inner(preconditioner))
}

fn preconditioner_ncols_impl_inner(
    preconditioner: ExternalPtr<PreconditionerHandle>,
) -> Result<i32> {
    let handle = preconditioner.try_addr()?;
    usize_to_i32(handle.preconditioner.ncols(), "ncols")
}

/// Concrete preconditioner variant name.
///
/// @export
#[extendr]
fn preconditioner_variant_impl(preconditioner: ExternalPtr<PreconditionerHandle>) -> String {
    or_throw(preconditioner_variant_impl_inner(preconditioner))
}

fn preconditioner_variant_impl_inner(
    preconditioner: ExternalPtr<PreconditionerHandle>,
) -> Result<String> {
    let handle = preconditioner.try_addr()?;
    Ok(handle.preconditioner.variant_name().to_string())
}

/// Build time for a preconditioner returned by Solver, or NULL if unknown.
///
/// @export
#[extendr]
fn preconditioner_build_time_seconds_impl(
    preconditioner: ExternalPtr<PreconditionerHandle>,
) -> Robj {
    or_throw(preconditioner_build_time_seconds_impl_inner(preconditioner))
}

fn preconditioner_build_time_seconds_impl_inner(
    preconditioner: ExternalPtr<PreconditionerHandle>,
) -> Result<Robj> {
    let handle = preconditioner.try_addr()?;
    Ok(handle
        .build_time_seconds
        .map_or_else(nil_value, |seconds| r!(seconds)))
}

/// Serialize a built preconditioner into raw bytes.
///
/// @export
#[extendr]
fn preconditioner_serialize_impl(preconditioner: ExternalPtr<PreconditionerHandle>) -> Raw {
    or_throw(preconditioner_serialize_impl_inner(preconditioner))
}

fn preconditioner_serialize_impl_inner(
    preconditioner: ExternalPtr<PreconditionerHandle>,
) -> Result<Raw> {
    let handle = preconditioner.try_addr()?;
    let bytes = postcard::to_stdvec(&handle.preconditioner).map_err(|e| err(e.to_string()))?;
    Ok(Raw::from_bytes(&bytes))
}

/// Deserialize a built preconditioner from raw bytes.
///
/// @export
#[extendr]
fn preconditioner_deserialize_impl(data: Raw) -> ExternalPtr<PreconditionerHandle> {
    or_throw(preconditioner_deserialize_impl_inner(data))
}

fn preconditioner_deserialize_impl_inner(data: Raw) -> Result<ExternalPtr<PreconditionerHandle>> {
    let preconditioner: Preconditioner =
        postcard::from_bytes(data.as_slice()).map_err(|e| err(e.to_string()))?;
    Ok(ExternalPtr::new(PreconditionerHandle {
        preconditioner,
        build_time_seconds: None,
    }))
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

extendr_module! {
    mod withinr;
    fn solve_impl;
    fn solve_batch_impl;
    fn solver_new_impl;
    fn solver_solve_impl;
    fn solver_solve_batch_impl;
    fn solver_preconditioner_impl;
    fn solver_n_dofs_impl;
    fn solver_n_obs_impl;
    fn preconditioner_apply_impl;
    fn preconditioner_nrows_impl;
    fn preconditioner_ncols_impl;
    fn preconditioner_variant_impl;
    fn preconditioner_build_time_seconds_impl;
    fn preconditioner_serialize_impl;
    fn preconditioner_deserialize_impl;
}
