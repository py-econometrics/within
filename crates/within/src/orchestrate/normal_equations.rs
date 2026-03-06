use std::time::Instant;

use schwarz_precond::solve::cg::{cg_solve, cg_solve_preconditioned};
use schwarz_precond::solve::gmres::gmres_solve;
use schwarz_precond::solve::lsmr::{lsmr_solve, vec_norm};
use schwarz_precond::{IdentityOperator, Operator};

use crate::config::{OperatorRepr, Preconditioner, SolverMethod, SolverParams};
use crate::domain::{build_domains_and_gramian_blocks, build_local_domains, WeightedDesign};
use crate::observation::ObservationStore;
use crate::operator::gramian::{Gramian, GramianOperator};
use crate::operator::schwarz::{
    build_additive, build_multiplicative_obs, build_multiplicative_sparse, DomainSource, FeSchwarz,
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
        let result = lsmr_solve(op, rhs, params.tol, params.tol, conlim, params.maxiter)?;
        Ok(MethodSolve {
            x: result.x,
            converged: interpret_lsmr_istop(result.istop),
            iterations: result.itn,
        })
    })
}

/// Extract (operator, preconditioner) from solver method.
fn method_axes(method: &SolverMethod) -> (OperatorRepr, &Preconditioner) {
    match method {
        SolverMethod::Cg {
            preconditioner,
            operator,
        } => (*operator, preconditioner),
        SolverMethod::Gmres {
            preconditioner,
            operator,
            ..
        } => (*operator, preconditioner),
        SolverMethod::Lsmr { .. } => unreachable!("LSMR handled in least_squares.rs"),
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

    // Handle LSMR separately (implicit operator, no preconditioner)
    if let SolverMethod::Lsmr { conlim } = &params.method {
        let gramian_op = GramianOperator::new(design);
        let t_setup_start = Instant::now();
        let timed = run_lsmr(&gramian_op, rhs, params, *conlim)?;
        let time_setup = timed
            .solve_start
            .duration_since(t_setup_start)
            .as_secs_f64();
        return Ok(solve_and_assemble(
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
        ));
    }

    let (operator, preconditioner) = method_axes(&params.method);

    let needs_gramian = operator == OperatorRepr::Explicit;
    let needs_domains =
        !matches!(preconditioner, Preconditioner::None) && prebuilt_schwarz.is_none();

    let t_setup_start = Instant::now();

    // Build phase: one of 4 branches
    let (gramian_opt, domain_pairs_opt) = match (needs_gramian, needs_domains) {
        (true, true) => {
            let (domain_pairs, blocks) = build_domains_and_gramian_blocks(design, None);
            let gramian = Gramian::from_pair_blocks(&blocks, &design.factors, design.n_dofs);
            (Some(gramian), Some(domain_pairs))
        }
        (true, false) => {
            let gramian = Gramian::build(design);
            (Some(gramian), None)
        }
        (false, true) => {
            let domain_pairs = build_local_domains(design, None);
            (None, Some(domain_pairs))
        }
        (false, false) => (None, None),
    };

    // Dispatch: build preconditioner and run solver, using Gramian or GramianOperator
    match (gramian_opt, &params.method) {
        // Explicit Gramian path
        (Some(gramian), SolverMethod::Cg { preconditioner, .. }) => {
            let timed = dispatch_cg(
                &gramian,
                rhs,
                preconditioner,
                prebuilt_schwarz,
                domain_pairs_opt,
                design,
                Some(&gramian),
                params,
            )?;
            finish(gramian, timed, rhs, rhs_norm, t_start, t_setup_start)
        }
        (
            Some(gramian),
            SolverMethod::Gmres {
                preconditioner,
                restart,
                ..
            },
        ) => {
            let timed = dispatch_gmres(
                &gramian,
                rhs,
                preconditioner,
                prebuilt_schwarz,
                domain_pairs_opt,
                design,
                Some(&gramian),
                params,
                *restart,
            )?;
            finish(gramian, timed, rhs, rhs_norm, t_start, t_setup_start)
        }
        // Implicit operator path
        (None, SolverMethod::Cg { preconditioner, .. }) => {
            let gramian_op = GramianOperator::new(design);
            let timed = dispatch_cg(
                &gramian_op,
                rhs,
                preconditioner,
                prebuilt_schwarz,
                domain_pairs_opt,
                design,
                None,
                params,
            )?;
            finish(gramian_op, timed, rhs, rhs_norm, t_start, t_setup_start)
        }
        (
            None,
            SolverMethod::Gmres {
                preconditioner,
                restart,
                ..
            },
        ) => {
            let gramian_op = GramianOperator::new(design);
            let timed = dispatch_gmres(
                &gramian_op,
                rhs,
                preconditioner,
                prebuilt_schwarz,
                domain_pairs_opt,
                design,
                None,
                params,
                *restart,
            )?;
            finish(gramian_op, timed, rhs, rhs_norm, t_start, t_setup_start)
        }
        _ => unreachable!("LSMR handled above"),
    }
}

fn finish<A: Operator>(
    op: A,
    timed: TimedMethodSolve,
    rhs: &[f64],
    rhs_norm: f64,
    t_start: Instant,
    t_setup_start: Instant,
) -> WithinResult<SolveResult> {
    let time_setup = timed
        .solve_start
        .duration_since(t_setup_start)
        .as_secs_f64();
    Ok(solve_and_assemble(
        &op,
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

type DomainPairs = Vec<(crate::domain::Subdomain, crate::operator::gramian::CrossTab)>;

fn make_source<S: ObservationStore>(
    domain_pairs_opt: Option<DomainPairs>,
    design: &WeightedDesign<S>,
) -> DomainSource<'_, S> {
    match domain_pairs_opt {
        Some(pairs) => DomainSource::FromParts(pairs),
        None => DomainSource::FromDesign(design),
    }
}

/// Dispatch CG solver with the right preconditioner variant.
#[allow(clippy::too_many_arguments)]
fn dispatch_cg<A: Operator, S: ObservationStore>(
    op: &A,
    rhs: &[f64],
    preconditioner: &Preconditioner,
    prebuilt_schwarz: Option<&FeSchwarz>,
    domain_pairs_opt: Option<DomainPairs>,
    design: &WeightedDesign<S>,
    gramian: Option<&Gramian>,
    params: &SolverParams,
) -> WithinResult<TimedMethodSolve> {
    match preconditioner {
        Preconditioner::None => run_cg_unpreconditioned(op, rhs, params),
        Preconditioner::Additive(cfg) => {
            if let Some(schwarz) = prebuilt_schwarz {
                return run_cg_preconditioned(op, schwarz, rhs, params);
            }
            let schwarz =
                build_additive(make_source(domain_pairs_opt, design), design.n_dofs, cfg)?;
            run_cg_preconditioned(op, &schwarz, rhs, params)
        }
        Preconditioner::Multiplicative(cfg) => {
            let source = make_source(domain_pairs_opt, design);
            if let Some(gramian) = gramian {
                let schwarz =
                    build_multiplicative_sparse(source, gramian, design.n_dofs, cfg, true)?;
                run_cg_preconditioned(op, &schwarz, rhs, params)
            } else {
                let schwarz = build_multiplicative_obs(source, design, cfg, true)?;
                run_cg_preconditioned(op, &schwarz, rhs, params)
            }
        }
    }
}

/// Dispatch GMRES solver with the right preconditioner variant.
#[allow(clippy::too_many_arguments)]
fn dispatch_gmres<A: Operator, S: ObservationStore>(
    op: &A,
    rhs: &[f64],
    preconditioner: &Preconditioner,
    prebuilt_schwarz: Option<&FeSchwarz>,
    domain_pairs_opt: Option<DomainPairs>,
    design: &WeightedDesign<S>,
    gramian: Option<&Gramian>,
    params: &SolverParams,
    restart: usize,
) -> WithinResult<TimedMethodSolve> {
    match preconditioner {
        Preconditioner::None => {
            let identity = IdentityOperator::new(rhs.len());
            run_gmres_preconditioned(op, &identity, rhs, params, restart)
        }
        Preconditioner::Additive(cfg) => {
            if let Some(schwarz) = prebuilt_schwarz {
                return run_gmres_preconditioned(op, schwarz, rhs, params, restart);
            }
            let schwarz =
                build_additive(make_source(domain_pairs_opt, design), design.n_dofs, cfg)?;
            run_gmres_preconditioned(op, &schwarz, rhs, params, restart)
        }
        Preconditioner::Multiplicative(cfg) => {
            let source = make_source(domain_pairs_opt, design);
            if let Some(gramian) = gramian {
                let schwarz =
                    build_multiplicative_sparse(source, gramian, design.n_dofs, cfg, false)?;
                run_gmres_preconditioned(op, &schwarz, rhs, params, restart)
            } else {
                let schwarz = build_multiplicative_obs(source, design, cfg, false)?;
                run_gmres_preconditioned(op, &schwarz, rhs, params, restart)
            }
        }
    }
}
