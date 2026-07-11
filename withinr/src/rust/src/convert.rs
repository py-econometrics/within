//! Shared R ↔ Rust coercion helpers and error plumbing for the bridge.
//!
//! R users pass **1-based** integer category matrices. This module validates
//! that all entries are >= 1 and contain no `NA`, then subtracts 1 to produce
//! the **0-based** `u32` indices expected by the Rust solver.

use extendr_api::prelude::*;
use ndarray::{ArrayView2, ShapeBuilder};

use within::observation::FactorMajorStore;

pub(crate) fn err(message: impl Into<String>) -> Error {
    Error::Other(message.into())
}

/// Raise `Err` as an R error, preserving its message.
///
/// extendr's default handling of `Result` return values unwraps them (a
/// panic), which surfaces in R as a generic "User function panicked" message.
/// Every `#[extendr]` entry point therefore wraps its fallible body in a
/// `?`-friendly closure and funnels the outcome through this helper instead.
pub(crate) fn or_throw<T>(result: Result<T>) -> T {
    match result {
        Ok(value) => value,
        Err(error) => throw_r_error(error.to_string()),
    }
}

pub(crate) fn usize_to_i32(value: usize, name: &str) -> Result<i32> {
    i32::try_from(value).map_err(|_| err(format!("{name} exceeds i32 range")))
}

pub(crate) fn parse_positive_f64(field: &Robj, name: &str) -> Result<f64> {
    let value = if let Some(value) = field.as_real() {
        value
    } else if let Some(value) = field.as_integer() {
        value as f64
    } else {
        return Err(err(format!("{name} must be a positive finite number")));
    };

    if !value.is_finite() || value <= 0.0 {
        return Err(err(format!("{name} must be a positive finite number")));
    }
    Ok(value)
}

pub(crate) fn parse_nonnegative_integer(field: &Robj, name: &str) -> Result<i64> {
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

pub(crate) fn parse_positive_u32(field: &Robj, name: &str) -> Result<u32> {
    let value = parse_nonnegative_integer(field, name)?;
    if value < 1 || value > u32::MAX as i64 {
        return Err(err(format!("{name} must be in 1..={}", u32::MAX)));
    }
    Ok(value as u32)
}

pub(crate) fn parse_optional_u32(field: &Robj, name: &str) -> Result<Option<u32>> {
    if field.is_null() {
        return Ok(None);
    }
    Ok(Some(parse_positive_u32(field, name)?))
}

pub(crate) fn get_field_or_null(obj: &Robj, field: &str) -> Robj {
    obj.dollar(field).unwrap_or_else(|_| nil_value())
}

/// Owned weights for the persistent solver (which stores them across solves).
pub(crate) fn extract_weights(weights: Robj) -> Result<Option<Vec<f64>>> {
    if weights.is_null() {
        return Ok(None);
    }
    weights
        .as_real_vector()
        .map(Some)
        .ok_or_else(|| err("weights must be a numeric vector or NULL"))
}

/// Borrowed weights for the one-shot paths (`NULL` → `None`), avoiding a copy.
pub(crate) fn weights_slice(weights: &Robj) -> Result<Option<&[f64]>> {
    if weights.is_null() {
        return Ok(None);
    }
    weights
        .as_real_slice()
        .map(Some)
        .ok_or_else(|| err("weights must be a numeric vector or NULL"))
}

/// Convert an R integer matrix (1-based, column-major) to a 0-based `u32`
/// buffer preserving R's column-major layout.
pub(crate) fn cast_categories(data: &[i32]) -> Result<Vec<u32>> {
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

pub(crate) fn categories_view<'a>(
    categories: &RMatrix<i32>,
    cats_u32: &'a [u32],
) -> Result<ArrayView2<'a, u32>> {
    ArrayView2::from_shape((categories.nrows(), categories.ncols()).f(), cats_u32)
        .map_err(|e| err(e.to_string()))
}

pub(crate) fn factor_major_store(categories: &RMatrix<i32>) -> Result<FactorMajorStore> {
    let n_obs = categories.nrows();
    let n_factors = categories.ncols();
    let cats_u32 = cast_categories(categories.data())?;
    let cats = categories_view(categories, &cats_u32)?;
    let factor_levels = (0..n_factors)
        .map(|factor| cats.column(factor).iter().copied().collect())
        .collect();
    FactorMajorStore::new(factor_levels, n_obs).map_err(|e| err(e.to_string()))
}
