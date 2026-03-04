mod execute;

use crate::config::{SolverMethod, SolverParams};
use crate::domain::WeightedDesign;
use crate::observation::ObservationStore;
use crate::operator::schwarz::FeSchwarz;
use crate::WithinResult;

use super::normal_equations::solve_normal_equations;
use super::SolveResult;
use execute::solve_least_squares_lsmr;

fn dispatch_method<S: ObservationStore>(
    design: &WeightedDesign<S>,
    y: &[f64],
    prebuilt_schwarz: Option<&FeSchwarz>,
    params: &SolverParams,
) -> WithinResult<SolveResult> {
    match &params.method {
        SolverMethod::Cg { .. } | SolverMethod::Gmres { .. } => {
            let mut rhs = vec![0.0; design.n_dofs];
            design.rmatvec_wdt(y, &mut rhs);
            solve_normal_equations(design, &rhs, prebuilt_schwarz, params)
        }
        SolverMethod::Lsmr { conlim } => solve_least_squares_lsmr(design, y, params, *conlim),
    }
}

/// Solve weighted least-squares `min ||W^{1/2}(D x - y)||`.
pub fn solve_least_squares<S: ObservationStore>(
    design: &WeightedDesign<S>,
    y: &[f64],
    prebuilt_schwarz: Option<&FeSchwarz>,
    params: &SolverParams,
) -> WithinResult<SolveResult> {
    dispatch_method(design, y, prebuilt_schwarz, params)
}
