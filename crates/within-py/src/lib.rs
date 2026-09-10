// `__reduce__` methods return noisy PyO3 tuple types; allow the lint crate-wide.
#![allow(clippy::type_complexity)]

//! Thin PyO3 bridge exposing the [`within`] crate to Python as `within._within`.
//! Converts Python/numpy types to the native API and delegates all computation
//! to [`within`]; every heavy call detaches from the interpreter via [`Python::detach`].
//! Usage docs live in `python/within/` and the `within._within.pyi` stub.
//!
//! `gil_used = false`: no `&mut self`, no global state; numpy inputs are read in place.
//! Callers must not mutate an array from another thread while a call reads it.

use std::sync::{Mutex, OnceLock};

use pyo3::prelude::*;

mod api;
mod config;
mod convert;
mod results;

use api::{solve, solve_batch, PyDesign, PyEffect, PySolver};
use config::{
    PyApproxCholConfig, PyApproxSchurConfig, PyLocalSolverConfig, PyLsmrOptions, PyPreconditioner,
    PyPreconditionerConfig, PyReductionStrategy, PyScalingConfig, PySchur,
};
use convert::IntoPyErr;
use results::{PyBatchSolveResult, PyCoefficientLayout, PySolveResult, PyUnidentifiedDirection};

static LOG_CACHE: OnceLock<pyo3_log::ResetHandle> = OnceLock::new();

/// First exception a Python log handler raised since the current call began, from any thread.
static HANDLER_ERROR: Mutex<Option<PyErr>> = Mutex::new(None);

/// pyo3-log leaves a raising handler's exception pending on the emitting thread, where a rayon
/// worker's would be lost; this moves it where the calling thread can return it.
struct CapturingLogger(pyo3_log::Logger);

impl log::Log for CapturingLogger {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        self.0.enabled(metadata)
    }

    fn log(&self, record: &log::Record<'_>) {
        // On a worker, pyo3-log's own attach scope owns a temporary thread state whose pending
        // exception dies with it; holding an outer scope keeps that state alive until taken.
        let raised = Python::attach(|py| {
            self.0.log(record);
            PyErr::take(py)
        });
        if let Some(raised) = raised {
            let mut slot = HANDLER_ERROR.lock().unwrap_or_else(|e| e.into_inner());
            slot.get_or_insert(raised);
        }
    }

    fn flush(&self) {
        self.0.flush();
    }
}

/// Runs heavy work detached. Python may have reconfigured logging since the last call, and a
/// log handler that raised during the work becomes the call's exception.
pub(crate) fn detach<T, E, F>(py: Python<'_>, work: F) -> PyResult<T>
where
    T: Send,
    E: IntoPyErr + Send,
    F: Send + FnOnce() -> Result<T, E>,
{
    if let Some(cache) = LOG_CACHE.get() {
        cache.reset();
    }
    let result = py.detach(work).map_err(IntoPyErr::into_py_err);
    let raised = HANDLER_ERROR
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .take();
    match raised {
        Some(raised) => Err(raised),
        None => result,
    }
}

#[pymodule(gil_used = false)]
fn _within(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Another logger may already own the process; the channel is then theirs, not an error.
    let logger = pyo3_log::Logger::new(m.py(), pyo3_log::Caching::LoggersAndLevels)?
        .filter(log::LevelFilter::Trace)
        .filter_target("tracing::span".to_owned(), log::LevelFilter::Off);
    let cache = logger.reset_handle();
    if log::set_boxed_logger(Box::new(CapturingLogger(logger))).is_ok() {
        log::set_max_level(log::LevelFilter::Trace);
        let _ = LOG_CACHE.set(cache);
    }
    m.add_class::<PySolveResult>()?;
    m.add_class::<PyBatchSolveResult>()?;
    m.add_class::<PyUnidentifiedDirection>()?;
    m.add_class::<PyCoefficientLayout>()?;
    m.add_class::<PyLsmrOptions>()?;
    m.add_class::<PyReductionStrategy>()?;
    m.add_class::<PyPreconditionerConfig>()?;
    m.add_class::<PyApproxCholConfig>()?;
    m.add_class::<PyApproxSchurConfig>()?;
    m.add_class::<PyLocalSolverConfig>()?;
    m.add_class::<PyScalingConfig>()?;
    m.add_class::<PySchur>()?;
    m.add_class::<PyPreconditioner>()?;
    m.add_class::<PySolver>()?;
    m.add_class::<PyDesign>()?;
    m.add_class::<PyEffect>()?;
    m.add_function(wrap_pyfunction!(solve, m)?)?;
    m.add_function(wrap_pyfunction!(solve_batch, m)?)?;
    Ok(())
}
