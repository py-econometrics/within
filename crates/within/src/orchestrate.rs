//! End-to-end solve orchestration for normal equations.
//!
//! This module provides the public API for solving fixed-effects problems,
//! with implementation details in focused submodules.

mod normal_equations;

use rayon::prelude::*;

use crate::config::SolverParams;
use crate::domain::WeightedDesign;
use crate::observation::{FactorMajorStore, ObservationStore, ObservationWeights};
use crate::WithinResult;

use normal_equations::NormalEquationSolver;

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

pub use normal_equations::solve_normal_equations;

// ===========================================================================
// Batch demeaning (formerly batch.rs)
// ===========================================================================

/// Result of a batch demean operation (multiple RHS columns).
#[derive(Debug, Clone)]
pub struct BatchDemeanResult {
    /// Demeaned columns, each of length `n_obs`. Column-major order.
    pub columns: Vec<Vec<f64>>,
    /// True if every column converged within the iteration limit.
    pub all_converged: bool,
}

/// Demean multiple RHS columns in parallel, building Schwarz once.
///
/// The normal-equation solver is built once, then shared across rayon threads
/// for parallel per-column solves. Same algorithm as `within::solve`.
pub fn demean_batch<S: ObservationStore + Sync>(
    design: &WeightedDesign<S>,
    columns: &[Vec<f64>],
    params: &SolverParams,
) -> WithinResult<BatchDemeanResult> {
    let solver = NormalEquationSolver::build(design, params)?;
    let solves: WithinResult<Vec<(Vec<f64>, bool)>> = columns
        .par_iter()
        .map(|y_col| {
            let mut rhs = vec![0.0; design.n_dofs];
            design.rmatvec_wdt(y_col, &mut rhs);
            let solve = solver.solve(&rhs)?;
            let mut fitted = vec![0.0; y_col.len()];
            design.matvec_d(&solve.x, &mut fitted);
            let demeaned = y_col
                .iter()
                .zip(fitted.iter())
                .map(|(y, f)| y - f)
                .collect();
            Ok((demeaned, solve.converged))
        })
        .collect::<Vec<_>>()
        .into_iter()
        .collect();
    let solves = solves?;
    let all_converged = solves.iter().all(|(_, converged)| *converged);
    let columns = solves.into_iter().map(|(column, _)| column).collect();

    Ok(BatchDemeanResult {
        columns,
        all_converged,
    })
}

// ===========================================================================
// High-level API (formerly api.rs)
// ===========================================================================

fn design_from_categories(
    categories: &[Vec<u32>],
    n_levels: &[usize],
    y_len: usize,
    weights: Option<&[f64]>,
) -> WithinResult<WeightedDesign<FactorMajorStore>> {
    let weights = match weights {
        Some(weights) => ObservationWeights::Dense(weights.to_vec()),
        None => ObservationWeights::Unit,
    };
    let store = FactorMajorStore::new(categories.to_vec(), weights, y_len)?;
    WeightedDesign::from_store(store, n_levels)
}

/// Solve fixed-effects least squares from raw category data.
///
/// Each element of `categories` is a factor's level assignments (length = n_obs).
/// `n_levels[q]` is the number of distinct levels in factor `q`.
/// `y` is the response vector (length = n_obs).
///
/// Constructs a `FactorMajorStore` and `WeightedDesign` internally, forms
/// `D^T W y`, and solves the normal equations.
pub fn solve(
    categories: &[Vec<u32>],
    n_levels: &[usize],
    y: &[f64],
    weights: Option<&[f64]>,
    params: &SolverParams,
) -> WithinResult<SolveResult> {
    let design = design_from_categories(categories, n_levels, y.len(), weights)?;
    solve_response(&design, y, params)
}

fn solve_response<S: ObservationStore>(
    design: &WeightedDesign<S>,
    y: &[f64],
    params: &SolverParams,
) -> WithinResult<SolveResult> {
    let mut rhs = vec![0.0; design.n_dofs];
    design.rmatvec_wdt(y, &mut rhs);
    solve_normal_equations(design, &rhs, params)
}
