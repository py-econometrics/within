//! Persistent solver that reuses preconditioners across multiple RHS solves.

use std::time::Instant;

use ndarray::ArrayView2;
use rayon::prelude::*;
use schwarz_precond::solve::cg::pcg;
use schwarz_precond::solve::gmres::pgmres;
use schwarz_precond::solve::vec_norm;
use schwarz_precond::Operator;

use crate::config::{KrylovMethod, OperatorRepr, Preconditioner, SolverParams};
use crate::domain::WeightedDesign;
use crate::observation::{ArrayStore, ObservationStore, ObservationWeights};
use crate::operator::gramian::{Gramian, GramianOperator};
use crate::orchestrate::{BatchSolveResult, SolveResult};
use crate::preconditioner::{build_preconditioner, build_preconditioner_fused, FePreconditioner};
use crate::WithinResult;

/// Persistent solver that owns its preconditioner for reuse across multiple solves.
///
/// Build once with [`Solver::new`] or [`Solver::from_design`], then call
/// [`Solver::solve`] or [`Solver::solve_batch`] repeatedly with different RHS
/// vectors. The expensive preconditioner factorization happens only at
/// construction time.
pub struct Solver<S: ObservationStore> {
    design: WeightedDesign<S>,
    gramian: Option<Gramian>,
    preconditioner: Option<FePreconditioner>,
    krylov: KrylovMethod,
    tol: f64,
    maxiter: usize,
}

impl<S: ObservationStore> Solver<S> {
    /// Build from an existing [`WeightedDesign`].
    ///
    /// When both an explicit Gramian and a preconditioner are needed, uses
    /// a fused single-scan path that builds domains and Gramian blocks
    /// simultaneously, avoiding a redundant observation scan.
    pub fn from_design(
        design: WeightedDesign<S>,
        params: &SolverParams,
        preconditioner: Option<&Preconditioner>,
    ) -> WithinResult<Self> {
        let needs_gramian = params.operator == OperatorRepr::Explicit
            || matches!(preconditioner, Some(Preconditioner::Multiplicative(_)));

        let (gramian, built_precond) = match (needs_gramian, preconditioner) {
            // Fused path: single observation scan → domains + Gramian blocks.
            (true, Some(config)) => {
                let (gramian, precond) = build_preconditioner_fused(&design, config)?;
                (Some(gramian), Some(precond))
            }
            // Gramian only, no preconditioner.
            (true, None) => (Some(Gramian::build(&design)), None),
            // Preconditioner only (implicit operator).
            (false, Some(config)) => {
                let precond = build_preconditioner(&design, None, config)?;
                (None, Some(precond))
            }
            // Neither.
            (false, None) => (None, None),
        };

        Ok(Self {
            design,
            gramian,
            preconditioner: built_precond,
            krylov: params.krylov,
            tol: params.tol,
            maxiter: params.maxiter,
        })
    }

    /// Build from a design with a pre-built preconditioner (e.g. deserialized).
    pub fn from_design_with_preconditioner(
        design: WeightedDesign<S>,
        params: &SolverParams,
        preconditioner: FePreconditioner,
    ) -> WithinResult<Self> {
        let gramian = if params.operator == OperatorRepr::Explicit {
            Some(Gramian::build(&design))
        } else {
            None
        };

        Ok(Self {
            design,
            gramian,
            preconditioner: Some(preconditioner),
            krylov: params.krylov,
            tol: params.tol,
            maxiter: params.maxiter,
        })
    }

    /// Solve for a single RHS vector.
    pub fn solve(&self, y: &[f64]) -> WithinResult<SolveResult> {
        let t_start = Instant::now();
        let t_setup_start = Instant::now();

        // Project to normal equations: rhs = D^T W y
        let mut rhs = vec![0.0; self.design.n_dofs];
        self.design.rmatvec_wdt(y, &mut rhs);
        let rhs_norm = vec_norm(&rhs).max(1e-15);

        let t_solve_start = Instant::now();
        let time_setup = t_solve_start.duration_since(t_setup_start).as_secs_f64();

        // Dispatch Krylov solve
        let solve = self.krylov_solve(&rhs)?;

        let time_solve = t_solve_start.elapsed().as_secs_f64();

        // Compute relative residual in normal-equation space
        let final_residual = self.compute_residual(&solve.x, &rhs, rhs_norm);

        // Compute demeaned: y - D*x
        let mut demeaned = vec![0.0; self.design.n_rows];
        self.design.matvec_d(&solve.x, &mut demeaned);
        for (d, &yi) in demeaned.iter_mut().zip(y.iter()) {
            *d = yi - *d;
        }

        Ok(SolveResult {
            x: solve.x,
            demeaned,
            converged: solve.converged,
            iterations: solve.iterations,
            final_residual,
            time_total: t_start.elapsed().as_secs_f64(),
            time_setup,
            time_solve,
        })
    }

    /// Solve for multiple RHS vectors in parallel.
    pub fn solve_batch(&self, ys: &[&[f64]]) -> WithinResult<BatchSolveResult> {
        let t_start = Instant::now();
        let n_rhs = ys.len();
        let n_dofs = self.design.n_dofs;
        let n_obs = self.design.n_rows;

        // Solve each RHS in parallel
        let results: Vec<WithinResult<SolveResult>> =
            ys.par_iter().map(|y| self.solve(y)).collect();

        // Collect into BatchSolveResult
        let mut x = Vec::with_capacity(n_dofs * n_rhs);
        let mut demeaned = Vec::with_capacity(n_obs * n_rhs);
        let mut converged = Vec::with_capacity(n_rhs);
        let mut iterations = Vec::with_capacity(n_rhs);
        let mut final_residual = Vec::with_capacity(n_rhs);
        let mut time_solve = Vec::with_capacity(n_rhs);

        for r in results {
            let r = r?;
            x.extend_from_slice(&r.x);
            demeaned.extend_from_slice(&r.demeaned);
            converged.push(r.converged);
            iterations.push(r.iterations);
            final_residual.push(r.final_residual);
            time_solve.push(r.time_solve);
        }

        Ok(BatchSolveResult::new(
            x,
            demeaned,
            converged,
            iterations,
            final_residual,
            time_solve,
            t_start.elapsed().as_secs_f64(),
            n_dofs,
            n_obs,
            n_rhs,
        ))
    }

    /// Access the preconditioner (for serialization).
    pub fn preconditioner(&self) -> Option<&FePreconditioner> {
        self.preconditioner.as_ref()
    }

    /// Number of DOFs (coefficients).
    pub fn n_dofs(&self) -> usize {
        self.design.n_dofs
    }

    /// Number of observations.
    pub fn n_obs(&self) -> usize {
        self.design.n_rows
    }

    // --- Internal ---

    fn krylov_solve(&self, rhs: &[f64]) -> WithinResult<KrylovSolve> {
        match (&self.gramian, &self.preconditioner) {
            (Some(gramian), Some(precond)) => self.dispatch_krylov(gramian, Some(precond), rhs),
            (Some(gramian), None) => {
                self.dispatch_krylov::<_, FePreconditioner>(gramian, None, rhs)
            }
            (None, Some(precond)) => {
                let op = GramianOperator::new(&self.design);
                self.dispatch_krylov(&op, Some(precond), rhs)
            }
            (None, None) => {
                let op = GramianOperator::new(&self.design);
                self.dispatch_krylov::<_, FePreconditioner>(&op, None, rhs)
            }
        }
    }

    fn dispatch_krylov<A: Operator, M: Operator>(
        &self,
        op: &A,
        preconditioner: Option<&M>,
        rhs: &[f64],
    ) -> WithinResult<KrylovSolve> {
        match self.krylov {
            KrylovMethod::Cg => {
                let r = pcg(op, rhs, preconditioner, self.tol, self.maxiter)?;
                Ok(KrylovSolve {
                    x: r.x,
                    converged: r.converged,
                    iterations: r.iterations,
                })
            }
            KrylovMethod::Gmres { restart } => {
                let r = pgmres(op, rhs, preconditioner, self.tol, self.maxiter, restart)?;
                Ok(KrylovSolve {
                    x: r.x,
                    converged: r.converged,
                    iterations: r.iterations,
                })
            }
        }
    }

    fn compute_residual(&self, x: &[f64], rhs: &[f64], rhs_norm: f64) -> f64 {
        let mut scratch = vec![0.0; self.design.n_dofs];
        match &self.gramian {
            Some(g) => g.apply(x, &mut scratch),
            None => {
                let op = GramianOperator::new(&self.design);
                op.apply(x, &mut scratch);
            }
        }
        for i in 0..rhs.len() {
            scratch[i] -= rhs[i];
        }
        vec_norm(&scratch) / rhs_norm
    }
}

struct KrylovSolve {
    x: Vec<f64>,
    converged: bool,
    iterations: usize,
}

// Convenience constructors for ArrayStore
impl<'a> Solver<ArrayStore<'a>> {
    /// Build a solver from raw category data (zero-copy).
    pub fn new(
        categories: ArrayView2<'a, u32>,
        weights: Option<&[f64]>,
        params: &SolverParams,
        preconditioner: Option<&Preconditioner>,
    ) -> WithinResult<Self> {
        let weights = match weights {
            Some(w) => ObservationWeights::Dense(w.to_vec()),
            None => ObservationWeights::Unit,
        };
        let store = ArrayStore::new(categories, weights)?;
        let design = WeightedDesign::from_store(store)?;
        Self::from_design(design, params, preconditioner)
    }

    /// Build a solver with a pre-built preconditioner (e.g. deserialized).
    pub fn with_preconditioner(
        categories: ArrayView2<'a, u32>,
        weights: Option<&[f64]>,
        params: &SolverParams,
        preconditioner: FePreconditioner,
    ) -> WithinResult<Self> {
        let weights = match weights {
            Some(w) => ObservationWeights::Dense(w.to_vec()),
            None => ObservationWeights::Unit,
        };
        let store = ArrayStore::new(categories, weights)?;
        let design = WeightedDesign::from_store(store)?;
        Self::from_design_with_preconditioner(design, params, preconditioner)
    }
}
