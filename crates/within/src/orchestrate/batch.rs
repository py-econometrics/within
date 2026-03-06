use rayon::prelude::*;

use crate::config::SolverParams;
use crate::domain::WeightedDesign;
use crate::observation::ObservationStore;
use crate::WithinResult;

use super::normal_equations::NormalEquationSolver;

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
        tol,
        maxiter,
        ..Default::default()
    };

    demean_batch(&design, columns, &params)
}
