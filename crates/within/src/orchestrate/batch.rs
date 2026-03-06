use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::config::{CgPreconditioner, SchwarzConfig, SolverMethod, SolverParams};
use crate::domain::WeightedDesign;
use crate::observation::ObservationStore;
use crate::operator::schwarz::{build_schwarz_default, FeSchwarz};
use crate::WithinResult;

use super::least_squares::solve_least_squares;

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
/// The Schwarz preconditioner is built once, then shared across rayon threads
/// for parallel per-column CG solves. Same algorithm as `within::solve`.
pub fn demean_batch<S: ObservationStore + Sync>(
    design: &WeightedDesign<S>,
    columns: &[Vec<f64>],
    params: &SolverParams,
) -> WithinResult<BatchDemeanResult> {
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

    let all_converged = AtomicBool::new(true);

    let results: Vec<Vec<f64>> = columns
        .par_iter()
        .map(|y_col| {
            let result = solve_least_squares(design, y_col, schwarz.as_ref(), params)
                .expect("solve_least_squares failed");
            if !result.converged {
                all_converged.store(false, Ordering::Relaxed);
            }
            // demeaned = y - D·x
            let mut fitted = vec![0.0; y_col.len()];
            design.matvec_d(&result.x, &mut fitted);
            y_col.iter().zip(fitted.iter()).map(|(y, f)| y - f).collect()
        })
        .collect();

    Ok(BatchDemeanResult {
        columns: results,
        all_converged: all_converged.load(Ordering::Relaxed),
    })
}

/// Convenience: build a weighted design and demean a batch of columns.
pub fn demean_batch_default(
    categories: Vec<Vec<i64>>,
    n_levels: Vec<usize>,
    n_obs: usize,
    weights: Vec<f64>,
    columns: &[Vec<f64>],
    tol: f64,
    maxiter: usize,
) -> WithinResult<BatchDemeanResult> {
    use crate::FixedEffectsDesign;

    let design = FixedEffectsDesign::new_weighted(categories, n_levels, n_obs, weights)?;

    let params = SolverParams {
        method: SolverMethod::Cg {
            preconditioner: CgPreconditioner::OneLevel(SchwarzConfig::default()),
        },
        tol,
        maxiter,
    };

    demean_batch(&design, columns, &params)
}
