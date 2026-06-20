// __reduce__ methods return noisy PyO3 tuple types; allow the lint crate-wide.
#![allow(clippy::type_complexity)]

//! Thin PyO3 bridge exposing the [`within`] crate to Python as `within._within`.
//! Converts Python/numpy types to the native API and delegates all computation
//! to [`within`]; every heavy call releases the GIL via [`Python::allow_threads`].
//! Usage docs live in `python/within/` and the `within._within.pyi` stub.

use pyo3::prelude::*;

mod api;
mod config;
mod convert;
mod results;

use api::{solve, solve_batch, PySolver};
use config::{
    PyAdditiveSchwarz, PyApproxCholConfig, PyApproxSchurConfig, PyLocalSolverConfig, PyLsmrOptions,
    PyPreconditioner, PyPreconditionerConfig, PyReductionStrategy,
};
use results::{PyBatchSolveResult, PySolveResult};

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

#[pymodule]
fn _within(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySolveResult>()?;
    m.add_class::<PyBatchSolveResult>()?;
    m.add_class::<PyLsmrOptions>()?;
    m.add_class::<PyAdditiveSchwarz>()?;
    m.add_class::<PyReductionStrategy>()?;
    m.add_class::<PyPreconditionerConfig>()?;
    m.add_class::<PyApproxCholConfig>()?;
    m.add_class::<PyApproxSchurConfig>()?;
    m.add_class::<PyLocalSolverConfig>()?;
    m.add_class::<PyPreconditioner>()?;
    m.add_class::<PySolver>()?;
    m.add_function(wrap_pyfunction!(solve, m)?)?;
    m.add_function(wrap_pyfunction!(solve_batch, m)?)?;
    Ok(())
}
