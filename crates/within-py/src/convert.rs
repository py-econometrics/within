//! Shared numpy/Python coercion helpers bridging array and error types to the
//! native [`within`] API.

use std::borrow::Cow;

use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// Shared conversion helpers
// ---------------------------------------------------------------------------

/// Convert a numpy array view to a contiguous slice, copying only if non-contiguous.
pub(crate) fn coerce_to_slice<'a>(arr: &'a numpy::ndarray::ArrayView1<'_, f64>) -> Cow<'a, [f64]> {
    match arr.as_slice() {
        Some(s) => Cow::Borrowed(s),
        None => Cow::Owned(arr.to_vec()),
    }
}

/// Wrap a display-able error as a `PyValueError`.
pub(crate) fn value_err(e: impl std::fmt::Display) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())
}

/// A `TypeError` naming the dtype an input array must have (and the dtype it
/// actually had, when the object exposes one) — the opaque PyO3 extraction
/// failure names neither.
fn dtype_err(name: &str, expected: &str, obj: &Bound<'_, PyAny>) -> PyErr {
    let got = obj
        .getattr("dtype")
        .and_then(|d| d.str())
        .and_then(|s| s.extract::<String>())
        .map(|s| format!(", got dtype {s}"))
        .unwrap_or_default();
    PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
        "{name} must be a {expected} array{got}"
    ))
}

/// Extract a `float64` 1-D array, or raise a `TypeError` naming the dtype.
pub(crate) fn readonly_f64_1d<'py>(
    name: &str,
    obj: &Bound<'py, PyAny>,
) -> PyResult<PyReadonlyArray1<'py, f64>> {
    obj.extract()
        .map_err(|_| dtype_err(name, "1-D float64", obj))
}

/// Extract a `float64` 2-D array, or raise a `TypeError` naming the dtype.
pub(crate) fn readonly_f64_2d<'py>(
    name: &str,
    obj: &Bound<'py, PyAny>,
) -> PyResult<PyReadonlyArray2<'py, f64>> {
    obj.extract()
        .map_err(|_| dtype_err(name, "2-D float64", obj))
}

/// Extract a `uint32` 2-D array, or raise a `TypeError` naming the dtype.
pub(crate) fn readonly_u32_2d<'py>(
    name: &str,
    obj: &Bound<'py, PyAny>,
) -> PyResult<PyReadonlyArray2<'py, u32>> {
    obj.extract()
        .map_err(|_| dtype_err(name, "2-D uint32", obj))
}

/// Build a slice-of-slices reference view from borrowed-or-owned columns.
pub(crate) fn column_refs<'a>(columns: &'a [Cow<'_, [f64]>]) -> Vec<&'a [f64]> {
    columns.iter().map(|c| &**c).collect()
}

// ---------------------------------------------------------------------------
// Misc helpers
// ---------------------------------------------------------------------------

/// Extract the columns of a 2-D array as borrowed-or-owned slices.
///
/// A column of an F-contiguous (column-major) array is contiguous in memory and
/// is borrowed directly (`Cow::Borrowed`); a strided column (e.g. from C-order
/// input) is copied (`Cow::Owned`). Borrows are tied to the view's data lifetime.
pub(crate) fn extract_columns<'a>(
    arr: &numpy::ndarray::ArrayView2<'a, f64>,
) -> Vec<Cow<'a, [f64]>> {
    (0..arr.ncols())
        .map(|j| {
            let col = arr.index_axis_move(numpy::ndarray::Axis(1), j);
            match col.to_slice() {
                Some(s) => Cow::Borrowed(s),
                None => Cow::Owned(col.iter().copied().collect()),
            }
        })
        .collect()
}

pub(crate) fn warn_c_contiguous(
    py: Python<'_>,
    cats: &numpy::ndarray::ArrayView2<'_, u32>,
) -> PyResult<()> {
    // Warn only when the per-factor columns are NOT readable as contiguous
    // slices -- i.e. exactly when ingest must copy them: row stride != 1.
    // The column stride is irrelevant: even reversed (negative), each column
    // stays a contiguous slice and is borrowed zero-copy in logical order. A
    // single row or empty input is trivially contiguous regardless of strides.
    // Sortedness is unknown here (the locality sort happens later, inside
    // Design construction), so the advice is hedged: when the dominant factor
    // is unsorted, the sort copies the columns into contiguous owned storage
    // anyway and asfortranarray would only add a redundant copy.
    let strides = cats.strides();
    if cats.nrows() > 1 && strides[0] != 1 {
        PyErr::warn(
            py,
            &py.get_type::<pyo3::exceptions::PyUserWarning>(),
            c"categories array is not F-contiguous (column-major). If the data \
             is already sorted by the largest factor, np.asfortranarray(categories) \
             gives faster solves; unsorted input is copied internally either way.",
            1,
        )?;
    }
    Ok(())
}
