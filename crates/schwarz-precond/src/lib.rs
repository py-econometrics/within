//! Schwarz domain decomposition preconditioner library.
//!
//! Provides one-level additive and multiplicative Schwarz preconditioners,
//! generic over local solvers and residual update strategies. Includes
//! iterative solvers (CG, GMRES).
//!
//! # Why domain decomposition?
//!
//! Domain decomposition splits s global problem
//! `A x = b` into overlapping *subdomains*, solving each
//! one cheaply, and stitching the local solutions back together.
//!
//! The result is a preconditioner `M^{-1}` that approximates `A^{-1}` well
//! enough for a Krylov solver (CG or GMRES) to converge in far fewer
//! iterations than it would unpreconditioned. Because the local solves are
//! independent in the additive variant, they parallelize naturally.
//!
//! ## The additive Schwarz formula
//!
//! For `N` overlapping subdomains the one-level additive Schwarz
//! preconditioner is:
//!
//! ```text
//! M⁻¹ = Σᵢ Rᵢᵀ D̃ᵢ Aᵢ⁻¹ D̃ᵢ Rᵢ
//! ```
//!
//! where each operator plays a specific role:
//!
//! | Symbol | Meaning |
//! |--------|---------|
//! | `Rᵢ` | **Restriction** — gathers global DOFs belonging to subdomain *i* into a local vector |
//! | `D̃ᵢ` | **Partition-of-unity weight** — scales each DOF so that overlapping contributions sum to the identity (`Σ D̃ᵢ² = I`) |
//! | `Aᵢ⁻¹` | **Local solve** — inverts the local system (exact or approximate) |
//! | `Rᵢᵀ` | **Prolongation** — scatters the local correction back to the global vector |
//!
//! The two-sided weighting `Rᵢᵀ D̃ᵢ · · · D̃ᵢ Rᵢ` ensures that where
//! subdomains overlap, their contributions blend smoothly rather than
//! double-counting. Concretely, if a DOF belongs to *c* subdomains its
//! weight in each is `1/√c`, so the squared weights sum to one.
//!
//! The **multiplicative** variant applies subdomains sequentially, updating
//! the residual after each local solve. This is analogous to block
//! Gauss-Seidel vs. block Jacobi: it converges faster per iteration but
//! cannot parallelize the subdomain loop.
//!
//! # Module structure
//!
//! ```text
//! schwarz-precond
//! ├── domain          SubdomainCore, PartitionWeights
//! ├── local_solve     LocalSolver trait, LocalSolveInvoker, SubdomainEntry
//! ├── schwarz         Schwarz preconditioners
//! │   ├── additive       SchwarzPreconditioner (parallel local solves)
//! │   └── multiplicative MultiplicativeSchwarzPreconditioner, ResidualUpdater
//! ├── solve           Iterative solvers
//! │   ├── cg             Preconditioned conjugate gradient
//! │   └── gmres          Right-preconditioned GMRES(m) with restarts
//! ├── sparse_matrix   SparseMatrix (internal CSR representation)
//! └── error           Typed errors for build and runtime failures
//! ```
//!
//! # Trait relationships
//!
//! The crate is built around three core traits that compose to form the
//! preconditioner:
//!
//! - **[`Operator`]** — A linear map `R^n -> R^n` with `apply` and
//!   `apply_adjoint`. Both the system matrix `A` and the preconditioner
//!   `M^{-1}` implement this trait, so solvers are generic over both.
//!   All operators must be `Send + Sync` to enable Rayon parallelism.
//!
//! - **[`LocalSolver`]** — The `Aᵢ⁻¹` abstraction. Given a local
//!   right-hand side, produce the local solution. Implementations range
//!   from exact Cholesky to approximate incomplete factorizations.
//!   The solver declares its DOF count and scratch buffer requirements,
//!   which are validated at construction time.
//!
//! - **[`LocalSolveInvoker`]** — An execution-policy adapter that wraps
//!   a `LocalSolver` call. The default ([`DefaultLocalSolveInvoker`])
//!   delegates directly; specialized invokers can add instrumentation or
//!   nested-parallelism control without polluting the solver trait itself.
//!
//! These compose inside [`SubdomainEntry`], which bundles a
//! [`SubdomainCore`] (restriction indices + partition-of-unity weights)
//! with a `LocalSolver`. The preconditioner owns a collection of entries
//! and orchestrates the restrict-solve-prolongate loop.
//!
//! See [`examples/`](https://github.com/kristof-mattei/domain-decomp-chol/tree/main/crates/schwarz-precond/examples)
//! directory for complete runnable examples.
//!
//! # References
//!
//! - Toselli & Widlund (2005). *Domain Decomposition Methods — Algorithms and Theory*. Springer.

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
pub use error::{
    ApplyError, LocalSolveError, PreconditionerBuildError, SolveError, SubdomainCoreBuildError,
    SubdomainEntryBuildError,
};
pub use local_solve::{DefaultLocalSolveInvoker, LocalSolveInvoker, LocalSolver, SubdomainEntry};
pub use schwarz::{
    AdditiveSchwarzDiagnostics, MultiplicativeSchwarzPreconditioner, OperatorResidualUpdater,
    ReductionStrategy, ResidualUpdater, SchwarzPreconditioner,
};
pub use sparse_matrix::SparseMatrix;
