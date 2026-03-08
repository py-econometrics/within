//! End-to-end solve orchestration.
//!
//! This module provides the public convenience API ([`solve`], [`solve_batch`])
//! and the result types shared by all entry points.  The heavy lifting is done
//! by [`crate::solver::Solver`].

use std::time::Instant;

use ndarray::ArrayView2;

use crate::config::{Preconditioner, SolverParams};
use crate::WithinResult;

/// Common solve output for all orchestration entry points.
#[derive(Debug, Clone)]
#[must_use]
pub struct SolveResult {
    pub x: Vec<f64>,
    pub demeaned: Vec<f64>,
    pub converged: bool,
    pub iterations: usize,
    pub final_residual: f64,
    pub time_total: f64,
    pub time_setup: f64,
    pub time_solve: f64,
}

/// Result of a batch solve across multiple RHS vectors.
#[derive(Debug, Clone)]
pub struct BatchSolveResult {
    x: Vec<f64>,
    demeaned: Vec<f64>,
    converged: Vec<bool>,
    iterations: Vec<usize>,
    final_residual: Vec<f64>,
    time_solve: Vec<f64>,
    time_total: f64,
    n_dofs: usize,
    n_obs: usize,
    n_rhs: usize,
}

impl BatchSolveResult {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        x: Vec<f64>,
        demeaned: Vec<f64>,
        converged: Vec<bool>,
        iterations: Vec<usize>,
        final_residual: Vec<f64>,
        time_solve: Vec<f64>,
        time_total: f64,
        n_dofs: usize,
        n_obs: usize,
        n_rhs: usize,
    ) -> Self {
        Self {
            x,
            demeaned,
            converged,
            iterations,
            final_residual,
            time_solve,
            time_total,
            n_dofs,
            n_obs,
            n_rhs,
        }
    }

    pub fn n_rhs(&self) -> usize {
        self.n_rhs
    }
    pub fn x(&self, i: usize) -> &[f64] {
        &self.x[i * self.n_dofs..(i + 1) * self.n_dofs]
    }
    pub fn demeaned(&self, i: usize) -> &[f64] {
        &self.demeaned[i * self.n_obs..(i + 1) * self.n_obs]
    }
    pub fn x_all(&self) -> &[f64] {
        &self.x
    }
    pub fn demeaned_all(&self) -> &[f64] {
        &self.demeaned
    }
    pub fn converged(&self) -> &[bool] {
        &self.converged
    }
    pub fn iterations(&self) -> &[usize] {
        &self.iterations
    }
    pub fn final_residual(&self) -> &[f64] {
        &self.final_residual
    }
    pub fn time_solve(&self) -> &[f64] {
        &self.time_solve
    }
    pub fn time_total(&self) -> f64 {
        self.time_total
    }

    pub(crate) fn set_time_total(&mut self, t: f64) {
        self.time_total = t;
    }
}

// ===========================================================================
// High-level API
// ===========================================================================

/// Solve fixed-effects least squares from raw category data.
///
/// `categories` is an observation-major `(n_obs, n_factors)` array where
/// `categories[[i, q]]` is the level of observation `i` in factor `q`.
/// Levels must be `0..max_level` per factor; the number of levels is inferred.
/// `y` is the response vector (length = n_obs).
///
/// Zero-copy: the category array is borrowed, not copied.
///
/// This is a convenience wrapper around [`crate::Solver::new`] + [`crate::Solver::solve`].
pub fn solve(
    categories: ArrayView2<u32>,
    y: &[f64],
    weights: Option<&[f64]>,
    params: &SolverParams,
    preconditioner: Option<&Preconditioner>,
) -> WithinResult<SolveResult> {
    let t_start = Instant::now();
    let solver = crate::solver::Solver::new(categories, weights, params, preconditioner)?;
    let time_setup = t_start.elapsed().as_secs_f64();
    let mut result = solver.solve(y)?;
    // Include solver construction (preconditioner build) in setup time
    result.time_setup += time_setup;
    result.time_total = t_start.elapsed().as_secs_f64();
    Ok(result)
}

/// Solve fixed-effects least squares for multiple response vectors.
///
/// Same as [`solve`] but solves all RHS vectors in parallel (via rayon),
/// reusing the preconditioner across all solves.
pub fn solve_batch(
    categories: ArrayView2<u32>,
    ys: &[&[f64]],
    weights: Option<&[f64]>,
    params: &SolverParams,
    preconditioner: Option<&Preconditioner>,
) -> WithinResult<BatchSolveResult> {
    let t_start = Instant::now();
    let solver = crate::solver::Solver::new(categories, weights, params, preconditioner)?;
    let mut result = solver.solve_batch(ys)?;
    result.set_time_total(t_start.elapsed().as_secs_f64());
    Ok(result)
}
