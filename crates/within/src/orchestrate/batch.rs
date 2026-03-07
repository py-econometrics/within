use std::time::Instant;

use rayon::prelude::*;

use crate::config::{CgPreconditioner, SolverMethod, SolverParams};
use crate::domain::WeightedDesign;
use crate::observation::ObservationStore;
use crate::operator::schwarz::{build_schwarz_default, FeSchwarz};
use crate::{WithinError, WithinResult};

use super::least_squares::solve_least_squares;

/// Per-column intermediate collected inside the parallel loop.
struct ColumnSolve {
    x: Vec<f64>,
    converged: bool,
    iterations: usize,
    final_residual: f64,
    time_solve: f64,
}

/// Result of a batch solve operation (multiple RHS columns).
#[derive(Debug, Clone)]
pub struct BatchSolveResult {
    /// Per-column coefficient vectors, in the same order as input columns.
    pub x: Vec<Vec<f64>>,
    /// Per-column convergence flags.
    pub converged: Vec<bool>,
    /// Per-column iteration counts.
    pub iterations: Vec<usize>,
    /// Per-column final residuals.
    pub final_residual: Vec<f64>,
    /// End-to-end wall time for the entire batch (setup + all solves).
    pub time_total: f64,
    /// Setup time (Schwarz build), shared across columns.
    pub time_setup: f64,
    /// Per-column solve times.
    pub time_solve: Vec<f64>,
}

/// Result of a batch demean operation (multiple RHS columns).
#[derive(Debug, Clone)]
pub struct BatchDemeanResult {
    /// Per-column demeaned responses, in the same order as `columns` input.
    pub y_demean: Vec<Vec<f64>>,
    /// Per-column convergence flags.
    pub converged: Vec<bool>,
    /// Per-column iteration counts.
    pub iterations: Vec<usize>,
    /// Per-column final residuals.
    pub final_residual: Vec<f64>,
    /// End-to-end wall time for the entire batch (setup + all solves).
    pub time_total: f64,
    /// Setup time (Schwarz build), shared across columns.
    pub time_setup: f64,
    /// Per-column solve times.
    pub time_solve: Vec<f64>,
}

/// Solve multiple RHS columns in parallel, building Schwarz once.
///
/// The Schwarz preconditioner is built once, then shared across rayon threads
/// for parallel per-column CG solves. Same algorithm as `within::solve`.
pub fn solve_batch<S: ObservationStore + Sync>(
    design: &WeightedDesign<S>,
    columns: &[Vec<f64>],
    params: &SolverParams,
) -> WithinResult<BatchSolveResult> {
    let t_start = Instant::now();

    let schwarz: Option<FeSchwarz> = match &params.method {
        SolverMethod::Cg {
            preconditioner: CgPreconditioner::OneLevel(cfg),
        } => Some(build_schwarz_default(
            design,
            &cfg.approx_chol,
            &cfg.local_solver,
        )?),
        _ => None,
    };
    let time_setup = t_start.elapsed().as_secs_f64();

    let expected_len = design.n_rows;
    let per_column: Vec<WithinResult<ColumnSolve>> = columns
        .par_iter()
        .map(|y_col| {
            if y_col.len() != expected_len {
                return Err(WithinError::RhsCountMismatch {
                    expected: expected_len,
                    got: y_col.len(),
                });
            }
            let result = solve_least_squares(design, y_col, schwarz.as_ref(), params)?;
            Ok(ColumnSolve {
                x: result.x,
                converged: result.converged,
                iterations: result.iterations,
                final_residual: result.final_residual,
                time_solve: result.time_solve,
            })
        })
        .collect();

    let n = per_column.len();
    let mut x = Vec::with_capacity(n);
    let mut converged = Vec::with_capacity(n);
    let mut iterations = Vec::with_capacity(n);
    let mut final_residual = Vec::with_capacity(n);
    let mut time_solve = Vec::with_capacity(n);

    for col in per_column {
        let col = col?;
        x.push(col.x);
        converged.push(col.converged);
        iterations.push(col.iterations);
        final_residual.push(col.final_residual);
        time_solve.push(col.time_solve);
    }

    Ok(BatchSolveResult {
        x,
        converged,
        iterations,
        final_residual,
        time_total: t_start.elapsed().as_secs_f64(),
        time_setup,
        time_solve,
    })
}

/// Demean multiple RHS columns in parallel, building Schwarz once.
pub fn demean_batch<S: ObservationStore + Sync>(
    design: &WeightedDesign<S>,
    columns: &[Vec<f64>],
    params: &SolverParams,
) -> WithinResult<BatchDemeanResult> {
    let solve_result = solve_batch(design, columns, params)?;
    let mut y_demean = Vec::with_capacity(columns.len());

    for (y_col, x_col) in columns.iter().zip(solve_result.x.iter()) {
        let mut fitted = vec![0.0; y_col.len()];
        design.matvec_d(x_col, &mut fitted);
        y_demean.push(
            y_col
                .iter()
                .zip(fitted.iter())
                .map(|(y, f)| y - f)
                .collect(),
        );
    }

    Ok(BatchDemeanResult {
        y_demean,
        converged: solve_result.converged,
        iterations: solve_result.iterations,
        final_residual: solve_result.final_residual,
        time_total: solve_result.time_total,
        time_setup: solve_result.time_setup,
        time_solve: solve_result.time_solve,
    })
}
