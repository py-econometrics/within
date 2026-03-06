use std::time::Instant;

use schwarz_precond::solve::cg::{cg_solve, cg_solve_preconditioned};
use schwarz_precond::solve::gmres::gmres_solve;
use schwarz_precond::solve::lsmr::{lsmr_solve, vec_norm};
use schwarz_precond::Operator;

use crate::config::{
    CgPreconditioner, GmresPreconditioner, SchwarzConfig, SolverMethod, SolverParams,
};
use crate::domain::{build_domains_and_gramian_blocks, WeightedDesign};
use crate::observation::ObservationStore;
use crate::operator::gramian::{Gramian, GramianOperator};
use crate::operator::schwarz::{
    build_multiplicative_schwarz, build_multiplicative_schwarz_from_parts, build_schwarz_default,
    FeSchwarz,
};
use crate::WithinResult;

use super::common::{interpret_lsmr_istop, solve_and_assemble, TimingContext};
use super::SolveResult;

/// Solver output normalized across CG/GMRES/LSMR branches.
struct MethodSolve {
    x: Vec<f64>,
    converged: bool,
    iterations: usize,
}

/// Solver output plus exact solve start timestamp for timing accounting.
struct TimedMethodSolve {
    solve: MethodSolve,
    solve_start: Instant,
}

fn run_timed<F>(run: F) -> WithinResult<TimedMethodSolve>
where
    F: FnOnce() -> WithinResult<MethodSolve>,
{
    let solve_start = Instant::now();
    let solve = run()?;
    Ok(TimedMethodSolve { solve, solve_start })
}

fn run_cg_unpreconditioned<A: Operator>(
    op: &A,
    rhs: &[f64],
    params: &SolverParams,
) -> WithinResult<TimedMethodSolve> {
    run_timed(|| {
        let result = cg_solve(op, rhs, params.tol, params.maxiter)?;
        Ok(MethodSolve {
            x: result.x,
            converged: result.converged,
            iterations: result.iterations,
        })
    })
}

fn run_cg_preconditioned<A: Operator, M: Operator>(
    op: &A,
    preconditioner: &M,
    rhs: &[f64],
    params: &SolverParams,
) -> WithinResult<TimedMethodSolve> {
    run_timed(|| {
        let result = cg_solve_preconditioned(op, preconditioner, rhs, params.tol, params.maxiter)?;
        Ok(MethodSolve {
            x: result.x,
            converged: result.converged,
            iterations: result.iterations,
        })
    })
}

fn run_gmres_preconditioned<A: Operator, M: Operator>(
    op: &A,
    preconditioner: &M,
    rhs: &[f64],
    params: &SolverParams,
    restart: usize,
) -> WithinResult<TimedMethodSolve> {
    run_timed(|| {
        let result = gmres_solve(op, preconditioner, rhs, params.tol, params.maxiter, restart)?;
        Ok(MethodSolve {
            x: result.x,
            converged: result.converged,
            iterations: result.iterations,
        })
    })
}

fn run_lsmr<A: Operator>(
    op: &A,
    rhs: &[f64],
    params: &SolverParams,
    conlim: f64,
) -> WithinResult<TimedMethodSolve> {
    run_timed(|| {
        // Right-preconditioned LSMR.
        //
        // We use right-preconditioning rather than left-preconditioning because
        // LSMR minimizes ||A z - b||_2 and requires both A*z and A^T*r. With
        // right-preconditioning we substitute x = M^{-1} z, giving the system
        // (G * M^{-1}) z = rhs, which preserves the symmetric structure of the
        // residual and lets LSMR converge to the minimum-norm solution.
        // After solving we recover x = M^{-1} z.
        //
        // Left-preconditioning (M^{-1} G x = M^{-1} rhs) would change the norm
        // being minimized and break LSMR's optimality properties.
        let result = lsmr_solve(op, rhs, params.tol, params.tol, conlim, params.maxiter)?;
        Ok(MethodSolve {
            x: result.x,
            converged: interpret_lsmr_istop(result.istop),
            iterations: result.itn,
        })
    })
}

fn with_one_level_schwarz<S: ObservationStore, R>(
    design: &WeightedDesign<S>,
    prebuilt_schwarz: Option<&FeSchwarz>,
    cfg: &SchwarzConfig,
    f: impl FnOnce(&FeSchwarz) -> WithinResult<R>,
) -> WithinResult<R> {
    if let Some(schwarz) = prebuilt_schwarz {
        return f(schwarz);
    }
    let schwarz = build_schwarz_default(design, &cfg.approx_chol, &cfg.local_solver)?;
    f(&schwarz)
}

fn dispatch_method<S: ObservationStore>(
    design: &WeightedDesign<S>,
    gramian_op: &GramianOperator<'_, S>,
    rhs: &[f64],
    prebuilt_schwarz: Option<&FeSchwarz>,
    params: &SolverParams,
) -> WithinResult<TimedMethodSolve> {
    match &params.method {
        SolverMethod::Cg { preconditioner } => dispatch_cg(
            design,
            gramian_op,
            rhs,
            prebuilt_schwarz,
            params,
            preconditioner,
        ),
        SolverMethod::Gmres {
            preconditioner,
            restart,
        } => dispatch_gmres(design, gramian_op, rhs, params, preconditioner, *restart),
        SolverMethod::Lsmr { conlim } => run_lsmr(gramian_op, rhs, params, *conlim),
    }
}

fn dispatch_cg<S: ObservationStore>(
    design: &WeightedDesign<S>,
    gramian_op: &GramianOperator<'_, S>,
    rhs: &[f64],
    prebuilt_schwarz: Option<&FeSchwarz>,
    params: &SolverParams,
    preconditioner: &CgPreconditioner,
) -> WithinResult<TimedMethodSolve> {
    match preconditioner {
        CgPreconditioner::OneLevel(cfg) => {
            with_one_level_schwarz(design, prebuilt_schwarz, cfg, |schwarz| {
                run_cg_preconditioned(gramian_op, schwarz, rhs, params)
            })
        }
        CgPreconditioner::MultiplicativeOneLevel(cfg) => {
            let schwarz =
                build_multiplicative_schwarz(design, &cfg.approx_chol, &cfg.local_solver, true)?;
            run_cg_preconditioned(gramian_op, &schwarz, rhs, params)
        }
        CgPreconditioner::None => run_cg_unpreconditioned(gramian_op, rhs, params),
    }
}

fn dispatch_gmres<S: ObservationStore>(
    design: &WeightedDesign<S>,
    _gramian_op: &GramianOperator<'_, S>,
    rhs: &[f64],
    params: &SolverParams,
    preconditioner: &GmresPreconditioner,
    restart: usize,
) -> WithinResult<TimedMethodSolve> {
    match preconditioner {
        GmresPreconditioner::MultiplicativeOneLevel(cfg) => {
            // Build domains (parallel observation scan) and compose Gramian from blocks.
            // This replaces the sequential Gramian::build() + extract-from-gramian path.
            let (domain_pairs, blocks) = build_domains_and_gramian_blocks(design, None);
            let gramian = Gramian::from_pair_blocks(&blocks, &design.factors, design.n_dofs);
            let schwarz = build_multiplicative_schwarz_from_parts(
                domain_pairs,
                &gramian,
                design.n_dofs,
                &cfg.approx_chol,
                &cfg.local_solver,
                false,
            )?;
            run_gmres_preconditioned(&gramian, &schwarz, rhs, params, restart)
        }
    }
}

/// Solve normal equations `G x = rhs` where `G = D^T W D`.
pub fn solve_normal_equations<S: ObservationStore>(
    design: &WeightedDesign<S>,
    rhs: &[f64],
    prebuilt_schwarz: Option<&FeSchwarz>,
    params: &SolverParams,
) -> WithinResult<SolveResult> {
    let t_start = Instant::now();
    let rhs_norm = vec_norm(rhs).max(1e-15);
    let gramian_op = GramianOperator::new(design);

    let t_setup_start = Instant::now();
    let timed = dispatch_method(design, &gramian_op, rhs, prebuilt_schwarz, params)?;
    let time_setup = timed
        .solve_start
        .duration_since(t_setup_start)
        .as_secs_f64();

    Ok(solve_and_assemble(
        &gramian_op,
        timed.solve.x,
        timed.solve.converged,
        timed.solve.iterations,
        rhs,
        rhs_norm,
        TimingContext {
            t_start,
            time_setup,
            t_solve_start: timed.solve_start,
        },
    ))
}
