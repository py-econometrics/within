use std::time::Instant;

use schwarz_precond::solve::lsmr::{lsmr_solve, vec_norm};

use crate::config::SolverParams;
use crate::domain::WeightedDesign;
use crate::observation::ObservationStore;
use crate::operator::design::PreconditionedDesign;
use crate::operator::gramian::GramianOperator;
use crate::WithinResult;

use super::super::common::{interpret_lsmr_istop, solve_and_assemble, TimingContext};
use super::super::SolveResult;

fn inverse_gramian_diagonal<S: ObservationStore>(design: &WeightedDesign<S>) -> Vec<f64> {
    let mut inv_diag = design.gramian_diagonal();
    for d in &mut inv_diag {
        *d = if *d == 0.0 { 1.0 } else { 1.0 / *d };
    }
    inv_diag
}

fn lsmr_rhs<S: ObservationStore>(design: &WeightedDesign<S>, y: &[f64]) -> Vec<f64> {
    if design.store.is_unweighted() {
        y.to_vec()
    } else {
        (0..design.n_rows)
            .map(|i| design.weight(i).sqrt() * y[i])
            .collect()
    }
}

fn normal_equation_rhs<S: ObservationStore>(design: &WeightedDesign<S>, y: &[f64]) -> Vec<f64> {
    let mut rhs = vec![0.0; design.n_dofs];
    design.rmatvec_wdt(y, &mut rhs);
    rhs
}

pub(super) fn solve_least_squares_lsmr<S: ObservationStore>(
    design: &WeightedDesign<S>,
    y: &[f64],
    params: &SolverParams,
    conlim: f64,
) -> WithinResult<SolveResult> {
    let t_start = Instant::now();

    let t_setup_start = Instant::now();
    let inv_diag = inverse_gramian_diagonal(design);
    let time_setup = t_setup_start.elapsed().as_secs_f64();

    let t_solve = Instant::now();

    // Preconditioned: B = W^{1/2} D diag^{-1}, solve min||Bz - W^{1/2}y||, recover x = diag^{-1}·z
    let precond_design = PreconditionedDesign::new(design, inv_diag);
    let lsmr_res = lsmr_solve(
        &precond_design,
        &lsmr_rhs(design, y),
        params.tol,
        params.tol,
        conlim,
        params.maxiter,
    )?;

    // Recover: x = inv_diag · z
    let x: Vec<f64> = lsmr_res
        .x
        .iter()
        .zip(precond_design.inv_diag().iter())
        .map(|(z, d)| z * d)
        .collect();

    // Normal-equations residual for consistency with CG path: ||G x - D^T W y|| / ||D^T W y||
    let rhs = normal_equation_rhs(design, y);
    let rhs_norm = vec_norm(&rhs).max(1e-15);
    let gram_op = GramianOperator::new(design);

    Ok(solve_and_assemble(
        &gram_op,
        x,
        interpret_lsmr_istop(lsmr_res.istop),
        lsmr_res.itn,
        &rhs,
        rhs_norm,
        TimingContext {
            t_start,
            time_setup,
            t_solve_start: t_solve,
        },
    ))
}
