use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::config::{LocalSolverConfig, OperatorRepr, SolverMethod, SolverParams};
use crate::domain::WeightedDesign;
use crate::observation::ObservationStore;
use crate::WithinResult;

use super::normal_equations::{build_normal_equation_system, solve_with_normal_equation_system};

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
    let all_converged = AtomicBool::new(true);
    let system = build_normal_equation_system(design, params)?;
    let results: Vec<Vec<f64>> = columns
        .par_iter()
        .map(|y_col| {
            let mut rhs = vec![0.0; design.n_dofs];
            design.rmatvec_wdt(y_col, &mut rhs);
            let solve = solve_with_normal_equation_system(&system, &rhs, params)
                .expect("normal-equation solve failed");
            if !solve.converged {
                all_converged.store(false, Ordering::Relaxed);
            }
            let mut fitted = vec![0.0; y_col.len()];
            design.matvec_d(&solve.x, &mut fitted);
            y_col
                .iter()
                .zip(fitted.iter())
                .map(|(y, f)| y - f)
                .collect()
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
            preconditioner: Some(LocalSolverConfig::cg_default()),
            operator: OperatorRepr::Implicit,
        },
        tol,
        maxiter,
    };

    demean_batch(&design, columns, &params)
}
