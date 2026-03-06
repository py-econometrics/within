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

#[derive(Clone, Copy)]
enum KrylovMethod {
    Cg,
    Gmres { restart: usize },
}

#[derive(Clone, Copy)]
enum PreconditionerBuild<'a> {
    None,
    Additive(&'a LocalSolverConfig),
    Multiplicative(&'a LocalSolverConfig),
}

#[derive(Clone, Copy)]
struct SolverBuildSpec<'a> {
    operator: OperatorRepr,
    preconditioner: PreconditionerBuild<'a>,
    krylov: KrylovMethod,
    tol: f64,
    maxiter: usize,
}

impl<'a> SolverBuildSpec<'a> {
    fn from_params(params: &'a SolverParams) -> Self {
        let (operator, preconditioner, krylov) = match &params.method {
            SolverMethod::Cg {
                preconditioner,
                operator,
            } => (
                *operator,
                match preconditioner {
                    Some(cfg) => PreconditionerBuild::Additive(cfg),
                    None => PreconditionerBuild::None,
                },
                KrylovMethod::Cg,
            ),
            SolverMethod::Gmres {
                preconditioner,
                operator,
                restart,
            } => (
                *operator,
                match preconditioner {
                    Some(GmresPrecond::Additive(cfg)) => PreconditionerBuild::Additive(cfg),
                    Some(GmresPrecond::Multiplicative(cfg)) => {
                        PreconditionerBuild::Multiplicative(cfg)
                    }
                    None => PreconditionerBuild::None,
                },
                KrylovMethod::Gmres { restart: *restart },
            ),
        };
        Self {
            operator,
            preconditioner,
            krylov,
            tol: params.tol,
            maxiter: params.maxiter,
        }
    }
}

fn run_cg_unpreconditioned<A: Operator + ?Sized>(
    op: &A,
    rhs: &[f64],
    tol: f64,
    maxiter: usize,
) -> WithinResult<MethodSolve> {
    let result = cg_solve(op, rhs, tol, maxiter)?;
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
    tol: f64,
    maxiter: usize,
) -> WithinResult<MethodSolve> {
    let result = cg_solve_preconditioned(op, preconditioner, rhs, tol, maxiter)?;
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
    tol: f64,
    maxiter: usize,
    restart: usize,
) -> WithinResult<MethodSolve> {
    let result = gmres_solve(op, preconditioner, rhs, tol, maxiter, restart)?;
    Ok(MethodSolve {
        x: result.x,
        converged: result.converged,
        iterations: result.iterations,
    })
}

pub(super) struct NormalEquationSolver<'a> {
    operator: Box<dyn Operator + 'a>,
    preconditioner: Option<Box<dyn Operator + 'a>>,
    krylov: KrylovMethod,
    tol: f64,
    maxiter: usize,
}

type BoxedOperator<'a> = Box<dyn Operator + 'a>;
type AssembledSystem<'a> = (BoxedOperator<'a>, Option<BoxedOperator<'a>>);

impl<'a> NormalEquationSolver<'a> {
    pub(super) fn operator(&self) -> &dyn Operator {
        self.operator.as_ref()
    }

    fn preconditioner(&self) -> Option<&dyn Operator> {
        self.preconditioner.as_deref()
    }
    pub(super) fn build<S: ObservationStore>(
        design: &'a WeightedDesign<S>,
        params: &SolverParams,
    ) -> WithinResult<Self> {
        let spec = SolverBuildSpec::from_params(params);
        let (operator, preconditioner) =
            build_operator_and_preconditioner(design, spec.operator, spec.preconditioner)?;
        Ok(Self {
            operator,
            preconditioner,
            krylov: spec.krylov,
            tol: spec.tol,
            maxiter: spec.maxiter,
        })
    }

    pub(super) fn solve(&self, rhs: &[f64]) -> WithinResult<MethodSolve> {
        match (self.krylov, self.preconditioner()) {
            (KrylovMethod::Cg, None) => {
                run_cg_unpreconditioned(self.operator(), rhs, self.tol, self.maxiter)
            }
            (KrylovMethod::Cg, Some(p)) => {
                run_cg_preconditioned(self.operator(), p, rhs, self.tol, self.maxiter)
            }
            (KrylovMethod::Gmres { restart }, None) => {
                let id = IdentityOperator::new(rhs.len());
                run_gmres(self.operator(), &id, rhs, self.tol, self.maxiter, restart)
            }
            (KrylovMethod::Gmres { restart }, Some(p)) => {
                run_gmres(self.operator(), p, rhs, self.tol, self.maxiter, restart)
            }
        }
    }
}

fn build_operator_and_preconditioner<'a, S: ObservationStore>(
    design: &'a WeightedDesign<S>,
    operator: OperatorRepr,
    preconditioner: PreconditionerBuild<'_>,
) -> WithinResult<AssembledSystem<'a>> {
    match operator {
        OperatorRepr::Explicit => build_explicit_system(design, preconditioner),
        OperatorRepr::Implicit => build_implicit_system(design, preconditioner),
    }
}

fn build_explicit_system<'a, S: ObservationStore>(
    design: &'a WeightedDesign<S>,
    preconditioner: PreconditionerBuild<'_>,
) -> WithinResult<AssembledSystem<'a>> {
    if matches!(preconditioner, PreconditionerBuild::None) {
        return Ok((Box::new(Gramian::build(design)), None));
    }

    let (domains, blocks) = build_domains_and_gramian_blocks(design, None);
    let gramian = Gramian::from_pair_blocks(&blocks, &design.factors, design.n_dofs);
    let preconditioner = match preconditioner {
        PreconditionerBuild::None => None,
        PreconditionerBuild::Additive(cfg) => Some(Box::new(build_additive(
            DomainSource::<S>::FromParts(domains),
            design.n_dofs,
            cfg,
        )?) as Box<dyn Operator>),
        PreconditionerBuild::Multiplicative(cfg) => Some(Box::new(build_multiplicative_sparse(
            DomainSource::<S>::FromParts(domains),
            &gramian,
            design.n_dofs,
            cfg,
        )?) as Box<dyn Operator>),
    };
    Ok((Box::new(gramian), preconditioner))
}

fn build_implicit_system<'a, S: ObservationStore>(
    design: &'a WeightedDesign<S>,
    preconditioner: PreconditionerBuild<'_>,
) -> WithinResult<AssembledSystem<'a>> {
    if matches!(preconditioner, PreconditionerBuild::None) {
        return Ok((Box::new(GramianOperator::new(design)), None));
    }

    let domains = build_local_domains(design, None);
    let preconditioner = match preconditioner {
        PreconditionerBuild::None => None,
        PreconditionerBuild::Additive(cfg) => Some(Box::new(build_additive(
            DomainSource::<S>::FromParts(domains),
            design.n_dofs,
            cfg,
        )?) as Box<dyn Operator>),
        PreconditionerBuild::Multiplicative(cfg) => Some(Box::new(build_multiplicative_obs(
            DomainSource::<S>::FromParts(domains),
            design,
            cfg,
        )?) as Box<dyn Operator>),
    };
    Ok((Box::new(GramianOperator::new(design)), preconditioner))
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
    let solver = NormalEquationSolver::build(design, params)?;
    let t_solve_start = Instant::now();
    let solve = solver.solve(rhs)?;
    let time_setup = t_solve_start.duration_since(t_setup_start).as_secs_f64();
    Ok(solve_and_assemble(
        solver.operator(),
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
