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
