//! End-to-end solve orchestration for normal equations.
//!
//! This module provides the public API for solving fixed-effects problems.

use std::time::Instant;

use schwarz_precond::solve::cg::{cg_solve, cg_solve_preconditioned};
use schwarz_precond::solve::gmres::gmres_solve;
use schwarz_precond::solve::vec_norm;
use schwarz_precond::{IdentityOperator, Operator};

use crate::config::{KrylovMethod, OperatorRepr, Preconditioner, SolverParams};
use crate::domain::{build_domains_and_gramian_blocks, build_local_domains, WeightedDesign};
use crate::observation::{FactorMajorStore, ObservationStore, ObservationWeights};
use crate::operator::gramian::{Gramian, GramianOperator};
use crate::operator::schwarz::{
    build_additive, build_multiplicative_obs, build_multiplicative_sparse, DomainSource,
};
use crate::WithinResult;

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

// ===========================================================================
// TimingContext + solve finalization
// ===========================================================================

struct TimingContext {
    t_start: Instant,
    time_setup: f64,
    t_solve_start: Instant,
}

fn solve_and_assemble<A: Operator>(
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

fn compute_relative_residual<A: Operator>(
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

// ===========================================================================
// Krylov method wrappers
// ===========================================================================

struct MethodSolve {
    x: Vec<f64>,
    converged: bool,
    iterations: usize,
}

fn krylov_dispatch<A: Operator, M: Operator>(
    op: &A,
    preconditioner: &M,
    rhs: &[f64],
    params: &SolverParams,
) -> WithinResult<MethodSolve> {
    match params.krylov {
        KrylovMethod::Cg => {
            let r = cg_solve_preconditioned(op, preconditioner, rhs, params.tol, params.maxiter)?;
            Ok(MethodSolve {
                x: r.x,
                converged: r.converged,
                iterations: r.iterations,
            })
        }
        KrylovMethod::Gmres { restart } => {
            let r = gmres_solve(op, preconditioner, rhs, params.tol, params.maxiter, restart)?;
            Ok(MethodSolve {
                x: r.x,
                converged: r.converged,
                iterations: r.iterations,
            })
        }
    }
}

fn krylov_dispatch_unpreconditioned<A: Operator>(
    op: &A,
    rhs: &[f64],
    params: &SolverParams,
) -> WithinResult<MethodSolve> {
    match params.krylov {
        KrylovMethod::Cg => {
            let r = cg_solve(op, rhs, params.tol, params.maxiter)?;
            Ok(MethodSolve {
                x: r.x,
                converged: r.converged,
                iterations: r.iterations,
            })
        }
        KrylovMethod::Gmres { restart } => {
            let id = IdentityOperator::new(rhs.len());
            let r = gmres_solve(op, &id, rhs, params.tol, params.maxiter, restart)?;
            Ok(MethodSolve {
                x: r.x,
                converged: r.converged,
                iterations: r.iterations,
            })
        }
    }
}

// ===========================================================================
// Concrete-type solve paths (no dynamic dispatch)
// ===========================================================================

fn assemble_result<A: Operator>(
    op: &A,
    solve: MethodSolve,
    rhs: &[f64],
    rhs_norm: f64,
    timing: TimingContext,
) -> SolveResult {
    solve_and_assemble(
        op,
        solve.x,
        solve.converged,
        solve.iterations,
        rhs,
        rhs_norm,
        timing,
    )
}

fn solve_implicit<S: ObservationStore>(
    design: &WeightedDesign<S>,
    rhs: &[f64],
    rhs_norm: f64,
    params: &SolverParams,
    t_start: Instant,
    t_setup_start: Instant,
) -> WithinResult<SolveResult> {
    match &params.preconditioner {
        None => {
            let op = GramianOperator::new(design);
            let t_solve_start = Instant::now();
            let time_setup = t_solve_start.duration_since(t_setup_start).as_secs_f64();
            let solve = krylov_dispatch_unpreconditioned(&op, rhs, params)?;
            Ok(assemble_result(
                &op,
                solve,
                rhs,
                rhs_norm,
                TimingContext {
                    t_start,
                    time_setup,
                    t_solve_start,
                },
            ))
        }
        Some(Preconditioner::Additive(config)) => {
            let domains = build_local_domains(design);
            let precond =
                build_additive(DomainSource::<S>::FromParts(domains), design.n_dofs, config)?;
            let op = GramianOperator::new(design);
            let t_solve_start = Instant::now();
            let time_setup = t_solve_start.duration_since(t_setup_start).as_secs_f64();
            let solve = krylov_dispatch(&op, &precond, rhs, params)?;
            Ok(assemble_result(
                &op,
                solve,
                rhs,
                rhs_norm,
                TimingContext {
                    t_start,
                    time_setup,
                    t_solve_start,
                },
            ))
        }
        Some(Preconditioner::Multiplicative(config)) => {
            let domains = build_local_domains(design);
            let precond =
                build_multiplicative_obs(DomainSource::<S>::FromParts(domains), design, config)?;
            let op = GramianOperator::new(design);
            let t_solve_start = Instant::now();
            let time_setup = t_solve_start.duration_since(t_setup_start).as_secs_f64();
            let solve = krylov_dispatch(&op, &precond, rhs, params)?;
            Ok(assemble_result(
                &op,
                solve,
                rhs,
                rhs_norm,
                TimingContext {
                    t_start,
                    time_setup,
                    t_solve_start,
                },
            ))
        }
    }
}

fn solve_explicit<S: ObservationStore>(
    design: &WeightedDesign<S>,
    rhs: &[f64],
    rhs_norm: f64,
    params: &SolverParams,
    t_start: Instant,
    t_setup_start: Instant,
) -> WithinResult<SolveResult> {
    match &params.preconditioner {
        None => {
            let op = Gramian::build(design);
            let t_solve_start = Instant::now();
            let time_setup = t_solve_start.duration_since(t_setup_start).as_secs_f64();
            let solve = krylov_dispatch_unpreconditioned(&op, rhs, params)?;
            Ok(assemble_result(
                &op,
                solve,
                rhs,
                rhs_norm,
                TimingContext {
                    t_start,
                    time_setup,
                    t_solve_start,
                },
            ))
        }
        Some(Preconditioner::Additive(config)) => {
            let (domains, blocks) = build_domains_and_gramian_blocks(design);
            let op = Gramian::from_pair_blocks(&blocks, &design.factors, design.n_dofs);
            let precond =
                build_additive(DomainSource::<S>::FromParts(domains), design.n_dofs, config)?;
            let t_solve_start = Instant::now();
            let time_setup = t_solve_start.duration_since(t_setup_start).as_secs_f64();
            let solve = krylov_dispatch(&op, &precond, rhs, params)?;
            Ok(assemble_result(
                &op,
                solve,
                rhs,
                rhs_norm,
                TimingContext {
                    t_start,
                    time_setup,
                    t_solve_start,
                },
            ))
        }
        Some(Preconditioner::Multiplicative(config)) => {
            let (domains, blocks) = build_domains_and_gramian_blocks(design);
            let op = Gramian::from_pair_blocks(&blocks, &design.factors, design.n_dofs);
            let precond = build_multiplicative_sparse(
                DomainSource::<S>::FromParts(domains),
                &op,
                design.n_dofs,
                config,
            )?;
            let t_solve_start = Instant::now();
            let time_setup = t_solve_start.duration_since(t_setup_start).as_secs_f64();
            let solve = krylov_dispatch(&op, &precond, rhs, params)?;
            Ok(assemble_result(
                &op,
                solve,
                rhs,
                rhs_norm,
                TimingContext {
                    t_start,
                    time_setup,
                    t_solve_start,
                },
            ))
        }
    }
}

// ===========================================================================
// Public API
// ===========================================================================

/// Solve normal equations `G x = rhs` where `G = D^T W D`.
pub fn solve_normal_equations<S: ObservationStore>(
    design: &WeightedDesign<S>,
    rhs: &[f64],
    params: &SolverParams,
) -> WithinResult<SolveResult> {
    let t_start = Instant::now();
    let rhs_norm = vec_norm(rhs).max(1e-15);
    let t_setup_start = Instant::now();
    match params.operator {
        OperatorRepr::Implicit => {
            solve_implicit(design, rhs, rhs_norm, params, t_start, t_setup_start)
        }
        OperatorRepr::Explicit => {
            solve_explicit(design, rhs, rhs_norm, params, t_start, t_setup_start)
        }
    }
}

// ===========================================================================
// High-level API
// ===========================================================================

fn design_from_categories(
    categories: &[Vec<u32>],
    n_levels: &[usize],
    y_len: usize,
    weights: Option<&[f64]>,
) -> WithinResult<WeightedDesign<FactorMajorStore>> {
    let weights = match weights {
        Some(weights) => ObservationWeights::Dense(weights.to_vec()),
        None => ObservationWeights::Unit,
    };
    let store = FactorMajorStore::new(categories.to_vec(), weights, y_len)?;
    WeightedDesign::from_store(store, n_levels)
}

/// Solve fixed-effects least squares from raw category data.
///
/// Each element of `categories` is a factor's level assignments (length = n_obs).
/// `n_levels[q]` is the number of distinct levels in factor `q`.
/// `y` is the response vector (length = n_obs).
///
/// Constructs a `FactorMajorStore` and `WeightedDesign` internally, forms
/// `D^T W y`, and solves the normal equations.
pub fn solve(
    categories: &[Vec<u32>],
    n_levels: &[usize],
    y: &[f64],
    weights: Option<&[f64]>,
    params: &SolverParams,
) -> WithinResult<SolveResult> {
    let design = design_from_categories(categories, n_levels, y.len(), weights)?;
    solve_response(&design, y, params)
}

fn solve_response<S: ObservationStore>(
    design: &WeightedDesign<S>,
    y: &[f64],
    params: &SolverParams,
) -> WithinResult<SolveResult> {
    let mut rhs = vec![0.0; design.n_dofs];
    design.rmatvec_wdt(y, &mut rhs);
    solve_normal_equations(design, &rhs, params)
}
