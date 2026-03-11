//! Schwarz domain decomposition preconditioner library.
//!
//! Provides one-level additive and multiplicative Schwarz preconditioners,
//! generic over local solvers and residual update strategies. Includes
//! iterative solvers (CG, GMRES).
//!
//! # Quick start
//!
//! ```rust,no_run
//! use schwarz_precond::{
//!     LocalSolver, Operator, SchwarzPreconditioner, SubdomainCore, SubdomainEntry,
//! };
//! use schwarz_precond::solve::cg::cg_solve_preconditioned;
//!
//! // 1. Implement Operator for your system matrix A.
//! // 2. Implement LocalSolver for your local solve strategy.
//! // 3. Build subdomain entries with restriction indices + weights.
//! // 4. Construct the preconditioner and call CG.
//! ```
//!
//! See the [`examples/`](https://github.com/kristof-mattei/domain-decomp-chol/tree/main/crates/schwarz-precond/examples)
//! directory for complete runnable examples.
//!
//! # References
//!
//! - Toselli & Widlund (2005). *Domain Decomposition Methods — Algorithms and Theory*. Springer.
//! - Smith, Bjørstad & Gropp (1996). *Domain Decomposition: Parallel Multilevel Methods for Elliptic PDEs*. Cambridge University Press.

#![deny(missing_docs)]
#![warn(clippy::all)]

// ============================================================================
// Operator trait + IdentityOperator
// ============================================================================

/// A linear operator A: R^ncols -> R^nrows with its adjoint A^T.
///
/// Preconditioners are operators too (M^{-1} is a linear map).
/// All implementors must be Send + Sync to enable Rayon parallelism.
pub trait Operator: Send + Sync {
    /// Number of rows in the operator.
    fn nrows(&self) -> usize;
    /// Number of columns in the operator.
    fn ncols(&self) -> usize;
    /// y = A*x
    fn apply(&self, x: &[f64], y: &mut [f64]);
    /// Computes y = A^T * x. For symmetric operators, this should delegate to `apply`.
    fn apply_adjoint(&self, x: &[f64], y: &mut [f64]);

    /// Fallible version of [`Operator::apply`].
    ///
    /// Implementors with runtime failure modes should override this.
    fn try_apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), error::ApplyError> {
        self.apply(x, y);
        Ok(())
    }

    /// Fallible version of [`Operator::apply_adjoint`].
    ///
    /// Implementors with runtime failure modes should override this.
    fn try_apply_adjoint(&self, x: &[f64], y: &mut [f64]) -> Result<(), error::ApplyError> {
        self.apply_adjoint(x, y);
        Ok(())
    }
}

/// Identity operator: applies the identity map (y = x).
///
/// Used to deduplicate CG: unpreconditioned CG delegates to preconditioned CG
/// with this as the preconditioner. The monomorphizer fully inlines the copies.
pub struct IdentityOperator {
    n: usize,
}

impl IdentityOperator {
    /// Create an identity operator of dimension `n`.
    pub fn new(n: usize) -> Self {
        Self { n }
    }
}

impl Operator for IdentityOperator {
    fn nrows(&self) -> usize {
        self.n
    }
    fn ncols(&self) -> usize {
        self.n
    }
    #[inline(always)]
    fn apply(&self, x: &[f64], y: &mut [f64]) {
        y.copy_from_slice(x);
    }
    #[inline(always)]
    fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) {
        y.copy_from_slice(x);
    }
}

// ============================================================================
// Modules
// ============================================================================

/// Domain decomposition primitives: subdomain cores and partition weights.
pub mod domain;
mod error;
mod local_solve;
mod schwarz;
/// Iterative solvers.
pub mod solve;
mod sparse_matrix;

pub use domain::{PartitionWeights, SubdomainCore};
pub use error::{ApplyError, LocalSolveError, PreconditionerBuildError, SolveError};
pub use local_solve::{LocalSolver, SubdomainEntry};
pub use schwarz::{
    local_solver_inner_parallelism_enabled, with_local_solver_inner_parallelism,
    MultiplicativeSchwarzPreconditioner, OperatorResidualUpdater, ReductionStrategy,
    ResidualUpdater, SchwarzPreconditioner,
};
pub use sparse_matrix::SparseMatrix;
