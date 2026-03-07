use crate::domain::WeightedDesign;
use crate::observation::{FactorMajorStore, ObservationWeights};
use crate::WithinResult;

use super::normal_equations::solve_normal_equations;
use super::SolveResult;
use crate::config::SolverParams;

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

fn solve_response<S: crate::observation::ObservationStore>(
    design: &WeightedDesign<S>,
    y: &[f64],
    params: &SolverParams,
) -> WithinResult<SolveResult> {
    let mut rhs = vec![0.0; design.n_dofs];
    design.rmatvec_wdt(y, &mut rhs);
    solve_normal_equations(design, &rhs, params)
}
