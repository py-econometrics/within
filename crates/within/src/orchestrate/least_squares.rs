use crate::config::SolverParams;
use crate::domain::WeightedDesign;
use crate::observation::ObservationStore;
use crate::WithinResult;

use super::normal_equations::solve_normal_equations;
use super::SolveResult;

/// Solve weighted least-squares `min ||W^{1/2}(D x - y)||`.
pub fn solve_least_squares<S: ObservationStore>(
    design: &WeightedDesign<S>,
    y: &[f64],
    params: &SolverParams,
) -> WithinResult<SolveResult> {
    let rhs = design.normal_equation_rhs(y);
    solve_normal_equations(design, &rhs, params)
}
