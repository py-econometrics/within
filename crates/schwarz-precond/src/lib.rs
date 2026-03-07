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

/// Domain decomposition primitives: subdomain cores and partition weights.
pub mod domain;
mod error;
mod local_solve;
mod multiplicative_schwarz;
mod operator;
mod residual_update;
mod schwarz;
/// Iterative solvers.
pub mod solve;
mod sparse_matrix;

pub use domain::{PartitionWeights, SubdomainCore};
pub use error::{ApplyError, LocalSolveError, PreconditionerBuildError, SolveError};
pub use local_solve::{LocalSolver, SubdomainEntry};
pub use multiplicative_schwarz::MultiplicativeSchwarzPreconditioner;
pub use operator::{IdentityOperator, Operator};
pub use residual_update::{OperatorResidualUpdater, ResidualUpdater};
pub use schwarz::SchwarzPreconditioner;
pub use sparse_matrix::SparseMatrix;
