//! Shared numpy/Python coercion helpers bridging array and error types to the
//! native [`within`] API.

use std::borrow::Cow;

use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use within::{SolveError, WithinError};

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

/// Input-validation failures become `ValueError`, runtime failures `RuntimeError`.
pub(crate) trait IntoPyErr {
    fn into_py_err(self) -> PyErr;
}

impl IntoPyErr for SolveError {
    fn into_py_err(self) -> PyErr {
        match self {
            SolveError::InvalidInput { .. } => value_err(self),
            _ => PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(self.to_string()),
        }
    }
}

impl IntoPyErr for WithinError {
    fn into_py_err(self) -> PyErr {
        match self {
            WithinError::Solve(e) => e.into_py_err(),
            _ => value_err(self),
        }
    }
}

/// The opaque PyO3 extraction failure names neither the expected dtype nor the actual one.
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

/// An F-contiguous column is contiguous so it is borrowed; a strided one is copied.
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
    // Warn only when row stride != 1, i.e. exactly when ingest must copy.
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
