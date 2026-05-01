//! Persistent solver that caches the preconditioner for reuse across multiple
//! right-hand sides.
//!
//! # Motivation
//!
//! Building the Schwarz preconditioner is the most expensive step in a
//! fixed-effects solve: it scans observations to build subdomains, assembles
//! local operators, and computes approximate Cholesky factorizations. For a
//! single right-hand side (RHS) this cost is unavoidable, but econometric
//! workflows frequently solve the same design matrix with many different
//! response vectors. [`Solver`] lets callers pay the preconditioner cost once
//! and amortize it across all subsequent solves.
//!
//! # Iterative refinement
//!
//! After each Krylov solve, recompute the residual from observation space
//! (`D^T W (y - D x)`) and solve for a correction. Each refinement step costs
//! two inexpensive matrix-vector products plus one Krylov solve whose RHS is
//! already small.
//!
//! # Usage
//!
//! ```no_run
//! use within::{Solver, SolverParams, Preconditioner, LocalSolverConfig};
//! use ndarray::Array2;
//!
//! let categories = Array2::<u32>::zeros((1000, 2));
//! let params = SolverParams::default();
//! let precond = Preconditioner::Additive(
//!     LocalSolverConfig::solver_default(),
//!     Default::default(),
//! );
//!
//! let solver = Solver::new(categories.view(), None, &params, Some(&precond)).unwrap();
//!
//! let y1 = vec![1.0; 1000];
//! let r1 = solver.solve(&y1).unwrap();
//! ```

use std::time::Instant;

use ndarray::ArrayView2;
use rayon::prelude::*;
use schwarz_precond::solve::cg::pcg;
use schwarz_precond::solve::gmres::pgmres;
use schwarz_precond::solve::lsmr::mlsmr;
use schwarz_precond::solve::vec_norm;
use schwarz_precond::Operator;

use crate::config::{KrylovMethod, OperatorRepr, Preconditioner, SolverParams};
use crate::domain::Design;
use crate::error::WithinError;
use crate::observation::{ArrayStore, Store};
use crate::operator::gramian::{Gramian, GramianOperator};
use crate::operator::preconditioner::{
    build_preconditioner, build_preconditioner_fused, FePreconditioner,
};
use crate::operator::DesignOperator;
use crate::orchestrate::{BatchSolveResult, SolveResult};
use crate::WithinResult;

enum PreconditionerSource<'a> {
    Config(&'a Preconditioner),
    Built(FePreconditioner),
}

impl PreconditionerSource<'_> {
    fn requires_explicit_gramian(&self) -> bool {
        matches!(self, Self::Config(Preconditioner::Multiplicative(_)))
    }
}

/// Persistent solver that owns its preconditioner for reuse across multiple solves.
pub struct Solver<S: Store> {
    design: Design<S>,
    weights: Option<Vec<f64>>,
    gramian: Option<Gramian>,
    preconditioner: Option<FePreconditioner>,
    krylov: KrylovMethod,
    tol: f64,
    maxiter: usize,
    max_refinements: usize,
}

impl<S: Store> Solver<S> {
    /// Build from an existing [`Design`] and optional observation weights.
    pub fn from_design(
        design: Design<S>,
        weights: Option<Vec<f64>>,
        params: &SolverParams,
        preconditioner: Option<&Preconditioner>,
    ) -> WithinResult<Self> {
        Self::from_design_with_source(
            design,
            weights,
            params,
            preconditioner.map(PreconditionerSource::Config),
        )
    }

    /// Build from a design with a pre-built preconditioner (e.g. deserialized).
    pub fn from_design_with_preconditioner(
        design: Design<S>,
        weights: Option<Vec<f64>>,
        params: &SolverParams,
        preconditioner: FePreconditioner,
    ) -> WithinResult<Self> {
        Self::from_design_with_source(
            design,
            weights,
            params,
            Some(PreconditionerSource::Built(preconditioner)),
        )
    }

    fn from_design_with_source(
        design: Design<S>,
        weights: Option<Vec<f64>>,
        params: &SolverParams,
        preconditioner: Option<PreconditionerSource<'_>>,
    ) -> WithinResult<Self> {
        if let Some(w) = &weights {
            if w.len() != design.n_rows {
                return Err(WithinError::WeightCountMismatch {
                    expected: design.n_rows,
                    got: w.len(),
                });
            }
        }

        let weights_slice = weights.as_deref();
        let needs_gramian = params.operator == OperatorRepr::Explicit
            || preconditioner
                .as_ref()
                .is_some_and(PreconditionerSource::requires_explicit_gramian);

        let (gramian, preconditioner) = match (needs_gramian, preconditioner) {
            (true, Some(PreconditionerSource::Config(config))) => {
                let (gramian, preconditioner) =
                    build_preconditioner_fused(&design, weights_slice, config)?;
                (Some(gramian), Some(preconditioner))
            }
            (false, Some(PreconditionerSource::Config(config))) => {
                let preconditioner = build_preconditioner(&design, weights_slice, None, config)?;
                (None, Some(preconditioner))
            }
            (true, Some(PreconditionerSource::Built(preconditioner))) => (
                Some(Gramian::build(&design, weights_slice)),
                Some(preconditioner),
            ),
            (false, Some(PreconditionerSource::Built(preconditioner))) => {
                (None, Some(preconditioner))
            }
            (true, None) => (Some(Gramian::build(&design, weights_slice)), None),
            (false, None) => (None, None),
        };

        Ok(Self {
            design,
            weights,
            gramian,
            preconditioner,
            krylov: params.krylov,
            tol: params.tol,
            maxiter: params.maxiter,
            max_refinements: params.max_refinements,
        })
    }

    /// Solve for a single RHS vector.
    pub fn solve(&self, y: &[f64]) -> WithinResult<SolveResult> {
        let t_start = Instant::now();
        let t_setup_start = Instant::now();

        if let KrylovMethod::Lsmr { local_size } = self.krylov {
            return self.solve_lsmr(y, local_size, t_start, t_setup_start);
        }

        // CG/GMRES path: normal equations + iterative refinement
        let mut rhs = vec![0.0; self.design.n_dofs];
        self.adjoint_apply_weighted(y, &mut rhs); // rhs = D^T W y
        let rhs_norm = vec_norm(&rhs).max(1e-15);
        let abs_tol = self.tol * rhs_norm;

        let t_solve_start = Instant::now();
        let time_setup = t_solve_start.duration_since(t_setup_start).as_secs_f64();

        let (x0, conv0, iter0) = self.krylov_solve(&rhs)?;
        let (x, demeaned, converged, iterations) =
            self.iterative_refinement(y, abs_tol, x0, iter0, conv0)?;

        let time_solve = t_solve_start.elapsed().as_secs_f64();

        // Residual via observation space: ||D^T W (y - Dx)|| / ||rhs||.
        let mut residual_dof = vec![0.0; self.design.n_dofs];
        self.adjoint_apply_weighted(&demeaned, &mut residual_dof);
        let final_residual = vec_norm(&residual_dof) / rhs_norm;

        Ok(SolveResult {
            x,
            demeaned,
            converged,
            iterations,
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

        let results: Vec<WithinResult<SolveResult>> =
            ys.par_iter().map(|y| self.solve(y)).collect();

        let mut x = Vec::with_capacity(self.design.n_dofs * n_rhs);
        let mut demeaned = Vec::with_capacity(self.design.n_rows * n_rhs);
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

        Ok(BatchSolveResult {
            x,
            demeaned,
            converged,
            iterations,
            final_residual,
            time_solve,
            time_total: t_start.elapsed().as_secs_f64(),
        })
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

    /// `dst = D^T W r` (or `D^T r` when unweighted).
    fn adjoint_apply_weighted(&self, r: &[f64], dst: &mut [f64]) {
        match &self.weights {
            Some(w) => {
                use crate::operator::scatter_apply;
                dst.fill(0.0);
                scatter_apply(&self.design, dst, |i| w[i] * r[i]);
            }
            None => {
                DesignOperator::new(&self.design, None)
                    .apply_adjoint(r, dst)
                    .expect("DesignOperator::apply_adjoint is infallible");
            }
        }
    }

    fn solve_lsmr(
        &self,
        y: &[f64],
        local_size: Option<usize>,
        t_start: Instant,
        t_setup_start: Instant,
    ) -> WithinResult<SolveResult> {
        let rect_op = DesignOperator::new(&self.design, self.weights.as_deref());
        let b: Vec<f64> = match rect_op.sqrt_weights() {
            Some(sw) => y.iter().zip(sw).map(|(&yi, &swi)| swi * yi).collect(),
            None => y.to_vec(),
        };

        let t_solve_start = Instant::now();
        let time_setup = t_solve_start.duration_since(t_setup_start).as_secs_f64();

        let r = match self.preconditioner.as_ref() {
            Some(p) => mlsmr(&rect_op, &b, Some(p), self.tol, self.maxiter, local_size)?,
            None => mlsmr::<_, FePreconditioner>(
                &rect_op,
                &b,
                None,
                self.tol,
                self.maxiter,
                local_size,
            )?,
        };

        let time_solve = t_solve_start.elapsed().as_secs_f64();

        let mut demeaned = vec![0.0; self.design.n_rows];
        DesignOperator::new(&self.design, None)
            .apply(&r.x, &mut demeaned)
            .expect("DesignOperator::apply is infallible");
        for (d, &yi) in demeaned.iter_mut().zip(y.iter()) {
            *d = yi - *d;
        }

        let mut rhs = vec![0.0; self.design.n_dofs];
        self.adjoint_apply_weighted(y, &mut rhs);
        let rhs_norm = vec_norm(&rhs).max(1e-15);
        let mut residual_dof = vec![0.0; self.design.n_dofs];
        self.adjoint_apply_weighted(&demeaned, &mut residual_dof);
        let final_residual = vec_norm(&residual_dof) / rhs_norm;

        Ok(SolveResult {
            x: r.x,
            demeaned,
            converged: r.converged,
            iterations: r.iterations,
            final_residual,
            time_total: t_start.elapsed().as_secs_f64(),
            time_setup,
            time_solve,
        })
    }

    /// Iterative refinement starting from `(initial_x, initial_iters, initial_converged)`.
    ///
    /// Returns `(x, demeaned, converged, iterations)`.
    fn iterative_refinement(
        &self,
        y: &[f64],
        abs_tol: f64,
        initial_x: Vec<f64>,
        initial_iters: usize,
        initial_converged: bool,
    ) -> WithinResult<(Vec<f64>, Vec<f64>, bool, usize)> {
        let mut x = initial_x;
        let mut iterations = initial_iters;
        let mut converged = initial_converged;

        let mut demeaned = vec![0.0; self.design.n_rows];
        let mut rhs_corr = vec![0.0; self.design.n_dofs];

        for _ in 0..self.max_refinements {
            // Observation-space residual: demeaned = y - D·x
            DesignOperator::new(&self.design, None)
                .apply(&x, &mut demeaned)
                .expect("DesignOperator::apply is infallible");
            for (d, &yi) in demeaned.iter_mut().zip(y.iter()) {
                *d = yi - *d;
            }

            // Correction RHS in normal-equation space: D^T W (y - Dx)
            self.adjoint_apply_weighted(&demeaned, &mut rhs_corr);

            let corr_norm = vec_norm(&rhs_corr);
            if corr_norm <= abs_tol {
                break;
            }

            let corr_tol = (abs_tol / corr_norm).min(1.0);
            let (corr_x, corr_converged, corr_iterations) =
                self.krylov_solve_with_tol(&rhs_corr, corr_tol)?;
            for (xi, &di) in x.iter_mut().zip(corr_x.iter()) {
                *xi += di;
            }
            iterations += corr_iterations;
            converged = corr_converged;
        }

        // Final demeaned: y - D*x
        DesignOperator::new(&self.design, None)
            .apply(&x, &mut demeaned)
            .expect("DesignOperator::apply is infallible");
        for (d, &yi) in demeaned.iter_mut().zip(y.iter()) {
            *d = yi - *d;
        }

        Ok((x, demeaned, converged, iterations))
    }

    /// Run the configured Krylov solver. Returns `(x, converged, iterations)`.
    fn krylov_solve(&self, rhs: &[f64]) -> WithinResult<(Vec<f64>, bool, usize)> {
        self.krylov_solve_with_tol(rhs, self.tol)
    }

    fn krylov_solve_with_tol(
        &self,
        rhs: &[f64],
        tol: f64,
    ) -> WithinResult<(Vec<f64>, bool, usize)> {
        match (&self.gramian, &self.preconditioner) {
            (Some(gramian), Some(precond)) => {
                self.dispatch_krylov(gramian, Some(precond), rhs, tol)
            }
            (Some(gramian), None) => {
                self.dispatch_krylov::<_, FePreconditioner>(gramian, None, rhs, tol)
            }
            (None, precond) => {
                let op = GramianOperator::new(&self.design, self.weights.as_deref());
                self.dispatch_krylov(&op, precond.as_ref(), rhs, tol)
            }
        }
    }

    fn dispatch_krylov<A: Operator, M: Operator>(
        &self,
        op: &A,
        preconditioner: Option<&M>,
        rhs: &[f64],
        tol: f64,
    ) -> WithinResult<(Vec<f64>, bool, usize)> {
        match self.krylov {
            KrylovMethod::Cg => {
                let r = pcg(op, rhs, preconditioner, tol, self.maxiter)?;
                Ok((r.x, r.converged, r.iterations))
            }
            KrylovMethod::Gmres { restart } => {
                let r = pgmres(op, rhs, preconditioner, tol, self.maxiter, restart)?;
                Ok((r.x, r.converged, r.iterations))
            }
            KrylovMethod::Lsmr { .. } => unreachable!("LSMR is dispatched inline in solve()"),
        }
    }
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
        let store = ArrayStore::new(categories);
        let design = Design::from_store(store)?;
        let weights = weights.map(|w| w.to_vec());
        Self::from_design(design, weights, params, preconditioner)
    }
}
