//! End-to-end solve orchestration for normal equations.
//!
//! This module keeps the public orchestration API stable while implementation
//! details live in focused submodules.

mod api;
mod batch;
mod common;
mod normal_equations;

/// Common solve output for all orchestration entry points.
#[derive(Debug, Clone)]
#[must_use]
pub struct SolveResult {
    pub x: Vec<f64>,
    pub converged: bool,
    pub iterations: usize,
    pub final_residual: f64,
    pub time_total: f64,
    pub time_setup: f64,
    pub time_solve: f64,
}

pub use api::solve;
pub use batch::{demean_batch, BatchDemeanResult};
pub use normal_equations::solve_normal_equations;
