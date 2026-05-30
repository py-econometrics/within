// The various __reduce__ methods return tuples whose Rust type signatures are
// inherently noisy (Bound<'py, PyAny>, (PyO3 fields...)). Suppressing the
// clippy lint keeps the PyO3 boilerplate readable per-method.
#![allow(clippy::type_complexity)]

//! Thin PyO3 bridge exposing the [`within`] Rust crate to Python as `within._within`.
//!
//! This crate is intentionally minimal: it converts between Python/numpy types
//! and the native Rust API, then delegates all computation to [`within`].
//!
//! # GIL release strategy
//!
//! Every call that performs substantial computation ([`api::solve`],
//! [`api::solve_batch`], `PySolver::solve_py`, `PySolver::solve_batch_py`, and
//! `PySolver::new`) releases the GIL via [`Python::allow_threads`] before
//! entering the Rust solver. This means Python threads are **not** blocked
//! during solve operations and the solver's internal rayon parallelism can run
//! freely.
//!
//! # Type mapping
//!
//! | Python / numpy              | Rust                              |
//! |-----------------------------|-----------------------------------|
//! | `NDArray[np.uint32]` (2-D)  | `ndarray::ArrayView2<u32>`        |
//! | `NDArray[np.float64]` (1-D) | `&[f64]`                          |
//! | `NDArray[np.float64]` (2-D) | `Vec<Vec<f64>>` (columns)         |
//! | `LsmrOptions`                | [`within::LsmrOptions`]            |
//! | `PreconditionerConfig` enum | [`within::PreconditionerConfig`]  |
//! | `Preconditioner` (built)    | [`within::Preconditioner`]        |
//! | `SolveResult`               | [`within::SolveResult`]           |
//!
//! Category arrays are read directly via numpy's ndarray bridge (zero-copy
//! when F-contiguous). Response vectors and weights are borrowed as slices
//! or copied when non-contiguous. Results are converted to numpy arrays on
//! return.
//!
//! # User-facing documentation
//!
//! For usage examples and the public API surface, see the Python package at
//! `python/within/`. This crate's types are re-exported through
//! `within.__init__` and documented in `within._within.pyi`.

use pyo3::prelude::*;

mod api;
mod config;
mod convert;
mod objects;
mod results;

use api::{solve, solve_approx_parallel, solve_approx_parallel_batch, solve_batch};
use config::{
    PyAdditiveSchwarz, PyApproxCholConfig, PyApproxSchurConfig, PyLocalSolverConfig, PyLsmrOptions,
    PyPreconditionerConfig, PyReductionStrategy,
};
use objects::{PyPreconditioner, PySolver};
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
    m.add_function(wrap_pyfunction!(solve_approx_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(solve_batch, m)?)?;
    m.add_function(wrap_pyfunction!(solve_approx_parallel_batch, m)?)?;
    Ok(())
}
