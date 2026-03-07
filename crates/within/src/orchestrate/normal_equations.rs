use std::time::Instant;

use schwarz_precond::solve::cg::{cg_solve, cg_solve_preconditioned};
use schwarz_precond::solve::gmres::gmres_solve;
use schwarz_precond::solve::vec_norm;
use schwarz_precond::{IdentityOperator, Operator};

use crate::config::{KrylovMethod, LocalSolverConfig, OperatorRepr, Preconditioner, SolverParams};
use crate::domain::{
    build_domains_and_gramian_blocks, build_local_domains, Subdomain, WeightedDesign,
};
use crate::observation::ObservationStore;
use crate::operator::gramian::{CrossTab, Gramian, GramianOperator};
use crate::operator::schwarz::{
    build_additive, build_multiplicative_obs, build_multiplicative_sparse, DomainSource,
};
use crate::WithinResult;

use super::SolveResult;

// ===========================================================================
// TimingContext + solve finalization (formerly common.rs)
// ===========================================================================

/// Timing state captured during solve orchestration.
pub(super) struct TimingContext {
    pub(super) t_start: Instant,
    pub(super) time_setup: f64,
    pub(super) t_solve_start: Instant,
}

/// Finalize a solve: compute residual, assemble timings, and return `SolveResult`.
pub(super) fn solve_and_assemble<A: Operator + ?Sized>(
    op: &A,
    x: Vec<f64>,
    converged: bool,
    iterations: usize,
    rhs: &[f64],
    rhs_norm: f64,
    timing: TimingContext,
) -> SolveResult {
    let time_solve = timing.t_solve_start.elapsed().as_secs_f64();
    let mut scratch = vec![0.0; x.len()];
    let final_residual = compute_relative_residual(op, &x, rhs, rhs_norm, &mut scratch);
    SolveResult {
        x,
        converged,
        iterations,
        final_residual,
        time_total: timing.t_start.elapsed().as_secs_f64(),
        time_setup: timing.time_setup,
        time_solve,
    }
}

/// Compute `||A x - b|| / ||b||` using caller-provided scratch.
fn compute_relative_residual<A: Operator + ?Sized>(
    op: &A,
    x: &[f64],
    b: &[f64],
    b_norm: f64,
    scratch: &mut [f64],
) -> f64 {
    op.apply(x, scratch);
    for i in 0..b.len() {
        scratch[i] -= b[i];
    }
    vec_norm(scratch) / b_norm
}

pub(super) struct MethodSolve {
    pub(super) x: Vec<f64>,
    pub(super) converged: bool,
    pub(super) iterations: usize,
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
type DomainPairs = Vec<(Subdomain, CrossTab)>;

trait AssemblyMode<'a>: Sized {
    type Prepared;

    fn unpreconditioned(self) -> AssembledSystem<'a>;
    fn prepare(self) -> Self::Prepared;
    fn additive(
        prepared: Self::Prepared,
        config: &LocalSolverConfig,
    ) -> WithinResult<AssembledSystem<'a>>;
    fn multiplicative(
        prepared: Self::Prepared,
        config: &LocalSolverConfig,
    ) -> WithinResult<AssembledSystem<'a>>;
}

struct ExplicitAssembly<'a, S: ObservationStore> {
    design: &'a WeightedDesign<S>,
}

struct ExplicitPrepared {
    domains: DomainPairs,
    gramian: Gramian,
    n_dofs: usize,
}

impl<'a, S: ObservationStore> ExplicitAssembly<'a, S> {
    fn new(design: &'a WeightedDesign<S>) -> Self {
        Self { design }
    }
}

impl<'a, S: ObservationStore> AssemblyMode<'a> for ExplicitAssembly<'a, S> {
    type Prepared = ExplicitPrepared;

    fn unpreconditioned(self) -> AssembledSystem<'a> {
        (Box::new(Gramian::build(self.design)), None)
    }

    fn prepare(self) -> Self::Prepared {
        let (domains, blocks) = build_domains_and_gramian_blocks(self.design, None);
        let gramian = Gramian::from_pair_blocks(&blocks, &self.design.factors, self.design.n_dofs);
        ExplicitPrepared {
            domains,
            gramian,
            n_dofs: self.design.n_dofs,
        }
    }

    fn additive(
        prepared: Self::Prepared,
        config: &LocalSolverConfig,
    ) -> WithinResult<AssembledSystem<'a>> {
        let preconditioner = build_additive(
            DomainSource::<S>::FromParts(prepared.domains),
            prepared.n_dofs,
            config,
        )?;
        Ok((Box::new(prepared.gramian), Some(Box::new(preconditioner))))
    }

    fn multiplicative(
        prepared: Self::Prepared,
        config: &LocalSolverConfig,
    ) -> WithinResult<AssembledSystem<'a>> {
        let preconditioner = build_multiplicative_sparse(
            DomainSource::<S>::FromParts(prepared.domains),
            &prepared.gramian,
            prepared.n_dofs,
            config,
        )?;
        Ok((Box::new(prepared.gramian), Some(Box::new(preconditioner))))
    }
}

struct ImplicitAssembly<'a, S: ObservationStore> {
    design: &'a WeightedDesign<S>,
}

struct ImplicitPrepared<'a, S: ObservationStore> {
    design: &'a WeightedDesign<S>,
    domains: DomainPairs,
}

impl<'a, S: ObservationStore> ImplicitAssembly<'a, S> {
    fn new(design: &'a WeightedDesign<S>) -> Self {
        Self { design }
    }
}

impl<'a, S: ObservationStore> AssemblyMode<'a> for ImplicitAssembly<'a, S> {
    type Prepared = ImplicitPrepared<'a, S>;

    fn unpreconditioned(self) -> AssembledSystem<'a> {
        (Box::new(GramianOperator::new(self.design)), None)
    }

    fn prepare(self) -> Self::Prepared {
        ImplicitPrepared {
            design: self.design,
            domains: build_local_domains(self.design, None),
        }
    }

    fn additive(
        prepared: Self::Prepared,
        config: &LocalSolverConfig,
    ) -> WithinResult<AssembledSystem<'a>> {
        let preconditioner = build_additive(
            DomainSource::<S>::FromParts(prepared.domains),
            prepared.design.n_dofs,
            config,
        )?;
        Ok((
            Box::new(GramianOperator::new(prepared.design)),
            Some(Box::new(preconditioner)),
        ))
    }

    fn multiplicative(
        prepared: Self::Prepared,
        config: &LocalSolverConfig,
    ) -> WithinResult<AssembledSystem<'a>> {
        let preconditioner = build_multiplicative_obs(
            DomainSource::<S>::FromParts(prepared.domains),
            prepared.design,
            config,
        )?;
        Ok((
            Box::new(GramianOperator::new(prepared.design)),
            Some(Box::new(preconditioner)),
        ))
    }
}

fn assemble_system<'a, A: AssemblyMode<'a>>(
    assembly: A,
    preconditioner: Option<&Preconditioner>,
) -> WithinResult<AssembledSystem<'a>> {
    match preconditioner {
        None => Ok(assembly.unpreconditioned()),
        Some(Preconditioner::Additive(config)) => A::additive(assembly.prepare(), config),
        Some(Preconditioner::Multiplicative(config)) => {
            A::multiplicative(assembly.prepare(), config)
        }
    }
}

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
        let (operator, preconditioner) = match params.operator {
            OperatorRepr::Explicit => assemble_system(
                ExplicitAssembly::new(design),
                params.preconditioner.as_ref(),
            )?,
            OperatorRepr::Implicit => assemble_system(
                ImplicitAssembly::new(design),
                params.preconditioner.as_ref(),
            )?,
        };
        Ok(Self {
            operator,
            preconditioner,
            krylov: params.krylov,
            tol: params.tol,
            maxiter: params.maxiter,
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
