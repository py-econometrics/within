use crate::domain::WeightedDesign;
use crate::observation::{FactorMajorStore, ObservationWeights};
use crate::operator::schwarz::FeSchwarz;
use crate::WithinResult;

use super::least_squares::solve_least_squares;
use super::SolveResult;
use crate::config::SolverParams;

fn design_from_categories(
    categories: &[Vec<u32>],
    n_levels: &[usize],
    y_len: usize,
    weights: ObservationWeights,
) -> WithinResult<WeightedDesign<FactorMajorStore>> {
    let store = FactorMajorStore::new(categories.to_vec(), weights, y_len)?;
    WeightedDesign::from_store(store, n_levels)
}

/// Solve fixed-effects least squares from raw category data.
///
/// Each element of `categories` is a factor's level assignments (length = n_obs).
/// `n_levels[q]` is the number of distinct levels in factor `q`.
/// `y` is the response vector (length = n_obs).
///
/// Constructs a `FactorMajorStore` and `WeightedDesign` internally, then delegates
/// to [`solve_least_squares`].
pub fn solve(
    categories: &[Vec<u32>],
    n_levels: &[usize],
    y: &[f64],
    params: &SolverParams,
    prebuilt_schwarz: Option<&FeSchwarz>,
) -> WithinResult<SolveResult> {
    let design = design_from_categories(categories, n_levels, y.len(), ObservationWeights::Unit)?;
    solve_least_squares(&design, y, prebuilt_schwarz, params)
}

/// Solve weighted fixed-effects least squares from raw category data.
///
/// Same as [`solve`] but with per-observation weights.
pub fn solve_weighted(
    categories: &[Vec<u32>],
    n_levels: &[usize],
    y: &[f64],
    weights: &[f64],
    params: &SolverParams,
    prebuilt_schwarz: Option<&FeSchwarz>,
) -> WithinResult<SolveResult> {
    let design = design_from_categories(
        categories,
        n_levels,
        y.len(),
        ObservationWeights::Dense(weights.to_vec()),
    )?;
    solve_least_squares(&design, y, prebuilt_schwarz, params)
}
