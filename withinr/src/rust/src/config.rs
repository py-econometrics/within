//! R config objects → native [`within::config`] conversions, plus the
//! built-preconditioner handle exposed to R.
//!
//! Mirrors `crates/within-py/src/config.rs`: this is where R-facing
//! configuration values (classed lists, shortcut strings) are parsed into
//! native `within` config types.

use extendr_api::prelude::*;

use within::config::{
    ApproxCholConfig, ApproxSchurConfig, LocalSolverConfig, LsmrOptions, PreconditionerConfig,
    ReductionStrategy,
};
use within::Preconditioner;

use crate::convert::{
    err, get_field_or_null, or_throw, parse_nonnegative_integer, parse_optional_u32,
    parse_positive_f64, parse_positive_u32, usize_to_i32,
};

// ---------------------------------------------------------------------------
// Persistent preconditioner handle
// ---------------------------------------------------------------------------

pub(crate) struct PreconditionerHandle {
    pub(crate) preconditioner: Preconditioner,
    pub(crate) build_time_seconds: Option<f64>,
}

/// Native interpretation of the R `preconditioner` argument.
///
/// A pre-built preconditioner takes the reuse path; everything else is a
/// [`PreconditionerConfig`] (or `None` for the library default) to build from.
pub(crate) enum PreconditionerArg {
    Config(Option<PreconditionerConfig>),
    Built {
        preconditioner: Preconditioner,
        build_time_seconds: Option<f64>,
    },
}

// ---------------------------------------------------------------------------
// LSMR options
// ---------------------------------------------------------------------------

/// Resolve the R `options` argument into native [`LsmrOptions`].
///
/// `NULL` means library defaults; anything else must be created by
/// `LsmrOptions(...)`. Mirrors `resolve_lsmr_config` on the Python side;
/// value validation lives here, not in the R constructor.
pub(crate) fn parse_lsmr_options(options: &Robj) -> Result<LsmrOptions> {
    if options.is_null() {
        return Ok(LsmrOptions::default());
    }
    if !options.inherits("within_lsmr_options") {
        return Err(err("options must be created by LsmrOptions(...) or NULL"));
    }

    let tol = parse_positive_f64(&get_field_or_null(options, "tol"), "tol")?;

    let maxiter = parse_nonnegative_integer(&get_field_or_null(options, "maxiter"), "maxiter")?;
    if maxiter < 1 {
        return Err(err("maxiter must be >= 1"));
    }

    let local_size_obj = get_field_or_null(options, "local_size");
    let local_size = if local_size_obj.is_null() {
        None
    } else {
        let value = parse_nonnegative_integer(&local_size_obj, "local_size")?;
        if value < 1 {
            return Err(err("local_size must be NULL or >= 1"));
        }
        Some(value as usize)
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

pub(crate) fn parse_preconditioner(preconditioner: Robj) -> Result<PreconditionerArg> {
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
// Preconditioner handle API
// ---------------------------------------------------------------------------

// Apply a built preconditioner: y = M^{-1} x.
#[extendr]
fn preconditioner_apply_impl(
    preconditioner: ExternalPtr<PreconditionerHandle>,
    x: &[f64],
) -> Vec<f64> {
    or_throw((|| -> Result<Vec<f64>> {
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
    })())
}

// Number of rows in a built preconditioner.
#[extendr]
fn preconditioner_nrows_impl(preconditioner: ExternalPtr<PreconditionerHandle>) -> i32 {
    or_throw((|| -> Result<i32> {
        let handle = preconditioner.try_addr()?;
        usize_to_i32(handle.preconditioner.nrows(), "nrows")
    })())
}

// Number of columns in a built preconditioner.
#[extendr]
fn preconditioner_ncols_impl(preconditioner: ExternalPtr<PreconditionerHandle>) -> i32 {
    or_throw((|| -> Result<i32> {
        let handle = preconditioner.try_addr()?;
        usize_to_i32(handle.preconditioner.ncols(), "ncols")
    })())
}

// Concrete preconditioner variant name.
#[extendr]
fn preconditioner_variant_impl(preconditioner: ExternalPtr<PreconditionerHandle>) -> String {
    or_throw((|| -> Result<String> {
        let handle = preconditioner.try_addr()?;
        Ok(handle.preconditioner.variant_name().to_string())
    })())
}

// Build time for a preconditioner returned by Solver, or NULL if unknown.
#[extendr]
fn preconditioner_build_time_seconds_impl(
    preconditioner: ExternalPtr<PreconditionerHandle>,
) -> Robj {
    or_throw((|| -> Result<Robj> {
        let handle = preconditioner.try_addr()?;
        Ok(handle
            .build_time_seconds
            .map_or_else(nil_value, |seconds| r!(seconds)))
    })())
}

// Serialize a built preconditioner into raw bytes.
#[extendr]
fn preconditioner_serialize_impl(preconditioner: ExternalPtr<PreconditionerHandle>) -> Raw {
    or_throw((|| -> Result<Raw> {
        let handle = preconditioner.try_addr()?;
        let bytes = postcard::to_stdvec(&handle.preconditioner).map_err(|e| err(e.to_string()))?;
        Ok(Raw::from_bytes(&bytes))
    })())
}

// Deserialize a built preconditioner from raw bytes.
#[extendr]
fn preconditioner_deserialize_impl(data: Raw) -> ExternalPtr<PreconditionerHandle> {
    or_throw((|| -> Result<ExternalPtr<PreconditionerHandle>> {
        let preconditioner: Preconditioner =
            postcard::from_bytes(data.as_slice()).map_err(|e| err(e.to_string()))?;
        Ok(ExternalPtr::new(PreconditionerHandle {
            preconditioner,
            build_time_seconds: None,
        }))
    })())
}

extendr_module! {
    mod config;
    fn preconditioner_apply_impl;
    fn preconditioner_nrows_impl;
    fn preconditioner_ncols_impl;
    fn preconditioner_variant_impl;
    fn preconditioner_build_time_seconds_impl;
    fn preconditioner_serialize_impl;
    fn preconditioner_deserialize_impl;
}
