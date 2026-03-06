use std::time::Instant;

use schwarz_precond::solve::cg::{cg_solve, cg_solve_preconditioned};
use schwarz_precond::solve::gmres::gmres_solve;
use schwarz_precond::solve::vec_norm;
use schwarz_precond::{IdentityOperator, Operator};

use crate::config::{GmresPrecond, LocalSolverConfig, OperatorRepr, SolverMethod, SolverParams};
use crate::domain::{build_domains_and_gramian_blocks, build_local_domains, WeightedDesign};
use crate::observation::ObservationStore;
use crate::operator::gramian::{Gramian, GramianOperator};
use crate::operator::schwarz::{
    build_additive, build_multiplicative_obs, build_multiplicative_sparse, DomainSource,
};
use crate::WithinResult;

use super::common::{solve_and_assemble, TimingContext};
use super::SolveResult;

pub(super) struct MethodSolve {
    pub(super) x: Vec<f64>,
    pub(super) converged: bool,
    pub(super) iterations: usize,
}

fn run_cg_unpreconditioned<A: Operator + ?Sized>(
    op: &A,
    rhs: &[f64],
    params: &SolverParams,
) -> WithinResult<MethodSolve> {
    let result = cg_solve(op, rhs, params.tol, params.maxiter)?;
    Ok(MethodSolve {
        x: result.x,
        converged: result.converged,
        iterations: result.iterations,
    })
}

fn run_cg_preconditioned<A: Operator + ?Sized, M: Operator + ?Sized>(
    op: &A,
    preconditioner: &M,
    rhs: &[f64],
    params: &SolverParams,
) -> WithinResult<MethodSolve> {
    let result = cg_solve_preconditioned(op, preconditioner, rhs, params.tol, params.maxiter)?;
    Ok(MethodSolve {
        x: result.x,
        converged: result.converged,
        iterations: result.iterations,
    })
}

fn run_gmres<A: Operator + ?Sized, M: Operator + ?Sized>(
    op: &A,
    preconditioner: &M,
    rhs: &[f64],
    params: &SolverParams,
    restart: usize,
) -> WithinResult<MethodSolve> {
    let result = gmres_solve(op, preconditioner, rhs, params.tol, params.maxiter, restart)?;
    Ok(MethodSolve {
        x: result.x,
        converged: result.converged,
        iterations: result.iterations,
    })
}

enum RequestedPreconditioner<'a> {
    None,
    Additive(&'a LocalSolverConfig),
    Multiplicative(&'a LocalSolverConfig),
}

fn requested_preconditioner(method: &SolverMethod) -> RequestedPreconditioner<'_> {
    match method {
        SolverMethod::Cg {
            preconditioner: None,
            ..
        } => RequestedPreconditioner::None,
        SolverMethod::Cg {
            preconditioner: Some(cfg),
            ..
        } => RequestedPreconditioner::Additive(cfg),
        SolverMethod::Gmres {
            preconditioner: None,
            ..
        } => RequestedPreconditioner::None,
        SolverMethod::Gmres {
            preconditioner: Some(GmresPrecond::Additive(cfg)),
            ..
        } => RequestedPreconditioner::Additive(cfg),
        SolverMethod::Gmres {
            preconditioner: Some(GmresPrecond::Multiplicative(cfg)),
            ..
        } => RequestedPreconditioner::Multiplicative(cfg),
    }
}

fn requested_operator(method: &SolverMethod) -> OperatorRepr {
    match method {
        SolverMethod::Cg { operator, .. } | SolverMethod::Gmres { operator, .. } => *operator,
    }
}

pub(super) struct NormalEquationSystem<'a> {
    operator: Box<dyn Operator + 'a>,
    preconditioner: Option<Box<dyn Operator + 'a>>,
}

impl NormalEquationSystem<'_> {
    fn operator(&self) -> &dyn Operator {
        self.operator.as_ref()
    }

    fn preconditioner(&self) -> Option<&dyn Operator> {
        self.preconditioner.as_deref()
    }
}

pub(super) fn build_normal_equation_system<'a, S: ObservationStore>(
    design: &'a WeightedDesign<S>,
    params: &SolverParams,
) -> WithinResult<NormalEquationSystem<'a>> {
    match (
        requested_operator(&params.method),
        requested_preconditioner(&params.method),
    ) {
        (OperatorRepr::Explicit, RequestedPreconditioner::None) => Ok(NormalEquationSystem {
            operator: Box::new(Gramian::build(design)),
            preconditioner: None,
        }),
        (OperatorRepr::Explicit, RequestedPreconditioner::Additive(cfg)) => {
            let (domains, blocks) = build_domains_and_gramian_blocks(design, None);
            let gramian = Gramian::from_pair_blocks(&blocks, &design.factors, design.n_dofs);
            let schwarz =
                build_additive(DomainSource::<S>::FromParts(domains), design.n_dofs, cfg)?;
            Ok(NormalEquationSystem {
                operator: Box::new(gramian),
                preconditioner: Some(Box::new(schwarz)),
            })
        }
        (OperatorRepr::Explicit, RequestedPreconditioner::Multiplicative(cfg)) => {
            let (domains, blocks) = build_domains_and_gramian_blocks(design, None);
            let gramian = Gramian::from_pair_blocks(&blocks, &design.factors, design.n_dofs);
            let schwarz = build_multiplicative_sparse(
                DomainSource::<S>::FromParts(domains),
                &gramian,
                design.n_dofs,
                cfg,
            )?;
            Ok(NormalEquationSystem {
                operator: Box::new(gramian),
                preconditioner: Some(Box::new(schwarz)),
            })
        }
        (OperatorRepr::Implicit, RequestedPreconditioner::None) => Ok(NormalEquationSystem {
            operator: Box::new(GramianOperator::new(design)),
            preconditioner: None,
        }),
        (OperatorRepr::Implicit, RequestedPreconditioner::Additive(cfg)) => {
            let domains = build_local_domains(design, None);
            let schwarz =
                build_additive(DomainSource::<S>::FromParts(domains), design.n_dofs, cfg)?;
            Ok(NormalEquationSystem {
                operator: Box::new(GramianOperator::new(design)),
                preconditioner: Some(Box::new(schwarz)),
            })
        }
        (OperatorRepr::Implicit, RequestedPreconditioner::Multiplicative(cfg)) => {
            let domains = build_local_domains(design, None);
            let schwarz =
                build_multiplicative_obs(DomainSource::<S>::FromParts(domains), design, cfg)?;
            Ok(NormalEquationSystem {
                operator: Box::new(GramianOperator::new(design)),
                preconditioner: Some(Box::new(schwarz)),
            })
        }
    }
}

pub(super) fn solve_with_normal_equation_system(
    system: &NormalEquationSystem<'_>,
    rhs: &[f64],
    params: &SolverParams,
) -> WithinResult<MethodSolve> {
    match (&params.method, system.preconditioner()) {
        (SolverMethod::Cg { .. }, None) => run_cg_unpreconditioned(system.operator(), rhs, params),
        (SolverMethod::Cg { .. }, Some(p)) => {
            run_cg_preconditioned(system.operator(), p, rhs, params)
        }
        (SolverMethod::Gmres { restart, .. }, None) => {
            let id = IdentityOperator::new(rhs.len());
            run_gmres(system.operator(), &id, rhs, params, *restart)
        }
        (SolverMethod::Gmres { restart, .. }, Some(p)) => {
            run_gmres(system.operator(), p, rhs, params, *restart)
        }
    }
}

/// Solve normal equations `G x = rhs` where `G = D^T W D`.
pub fn solve_normal_equations<S: ObservationStore>(
    design: &WeightedDesign<S>,
    rhs: &[f64],
    params: &SolverParams,
) -> WithinResult<SolveResult> {
    let t_start = Instant::now();
    let rhs_norm = vec_norm(rhs).max(1e-15);
    let t_setup_start = Instant::now();
    let system = build_normal_equation_system(design, params)?;
    let t_solve_start = Instant::now();
    let solve = solve_with_normal_equation_system(&system, rhs, params)?;
    let time_setup = t_solve_start.duration_since(t_setup_start).as_secs_f64();
    Ok(solve_and_assemble(
        system.operator(),
        solve.x,
        solve.converged,
        solve.iterations,
        rhs,
        rhs_norm,
        TimingContext {
            t_start,
            time_setup,
            t_solve_start,
        },
    ))
}
