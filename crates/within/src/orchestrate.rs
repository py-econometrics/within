//! End-to-end solve orchestration — the public entry points for `within`.
//!
//! This module provides the convenience functions [`solve`] and [`solve_batch`]
//! that tie together all four architectural layers:
//!
//! ```text
//! solve(categories, y, weights, params, preconditioner)
//!   1. Validate  → observation layer builds an ArrayStore, checks dimensions
//!   2. Design    → domain layer wraps the store in a Design
//!   3. Precond   → operator layer builds subdomains + local solvers (Schwarz)
//!   4. Solve     → Krylov solver (CG/GMRES) with iterative refinement
//!   5. Extract   → return coefficients x and demeaned residuals y - Dx
//! ```
//!
//! For one-shot solves, [`solve`] and [`solve_batch`] are the simplest API.
//! When the same design matrix is reused with multiple response vectors,
//! prefer [`crate::Solver`] directly — it caches the preconditioner across
//! calls (see [`crate::solver`] for details).
//!
//! # Result types
//!
//! - [`SolveResult`] — output of a single solve: coefficients `x`, demeaned
//!   values `y - Dx`, convergence flag, iteration count, and timing breakdown.
//! - [`BatchSolveResult`] — output of a batch solve: concatenated coefficients
//!   and demeaned values with per-RHS convergence and timing metadata.

use std::time::Instant;

use ndarray::ArrayView2;

use crate::config::{Preconditioner, SolverParams};
use crate::WithinResult;

/// Common solve output for all orchestration entry points.
#[derive(Debug, Clone)]
#[must_use]
pub struct SolveResult {
    /// Fixed-effect coefficients (length = total DOFs across all factors).
    pub x: Vec<f64>,
    /// Demeaned response: `y - D x` (length = n_obs).
    pub demeaned: Vec<f64>,
    /// Whether the iterative solver converged within `maxiter` iterations.
    pub converged: bool,
    /// Number of Krylov iterations used.
    pub iterations: usize,
    /// Final relative residual norm `‖r‖ / ‖b‖`.
    pub final_residual: f64,
    /// Wall-clock time for the entire solve (setup + Krylov), in seconds.
    pub time_total: f64,
    /// Wall-clock time for preconditioner construction, in seconds.
    pub time_setup: f64,
    /// Wall-clock time for the Krylov solve phase, in seconds.
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
}

impl BatchSolveResult {
    pub(crate) fn new(
        x: Vec<f64>,
        demeaned: Vec<f64>,
        converged: Vec<bool>,
        iterations: Vec<usize>,
        final_residual: Vec<f64>,
        time_solve: Vec<f64>,
        time_total: f64,
    ) -> Self {
        Self {
            x,
            demeaned,
            converged,
            iterations,
            final_residual,
            time_solve,
            time_total,
        }
    }

    /// Number of right-hand sides in the batch.
    pub fn n_rhs(&self) -> usize {
        self.converged.len()
    }
    /// Coefficient vector for the `i`-th RHS.
    pub fn x(&self, i: usize) -> &[f64] {
        let n_dofs = self.x.len() / self.n_rhs();
        &self.x[i * n_dofs..(i + 1) * n_dofs]
    }
    /// Demeaned response for the `i`-th RHS.
    pub fn demeaned(&self, i: usize) -> &[f64] {
        let n_obs = self.demeaned.len() / self.n_rhs();
        &self.demeaned[i * n_obs..(i + 1) * n_obs]
    }
    /// All coefficient vectors concatenated (length = n_dofs * n_rhs).
    pub fn x_all(&self) -> &[f64] {
        &self.x
    }
    /// All demeaned responses concatenated (length = n_obs * n_rhs).
    pub fn demeaned_all(&self) -> &[f64] {
        &self.demeaned
    }
    /// Per-RHS convergence flags.
    pub fn converged(&self) -> &[bool] {
        &self.converged
    }
    /// Per-RHS iteration counts.
    pub fn iterations(&self) -> &[usize] {
        &self.iterations
    }
    /// Per-RHS final relative residual norms.
    pub fn final_residual(&self) -> &[f64] {
        &self.final_residual
    }
    /// Per-RHS solve times in seconds.
    pub fn time_solve(&self) -> &[f64] {
        &self.time_solve
    }
    /// Total wall-clock time for the entire batch (setup + all solves), in seconds.
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
