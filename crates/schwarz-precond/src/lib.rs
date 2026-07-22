//! Schwarz domain decomposition preconditioner library.
//!
//! Provides a one-level additive Schwarz preconditioner generic over local
//! solvers, plus a Modified LSMR iterative solver for rectangular
//! least-squares problems. The preconditioner approximates `(AᵀA)^{-1}` by
//! solving overlapping subproblems and stitching them together with a
//! partition-of-unity weighting; each local solve runs in parallel.
//!
//! See `examples/` for runnable usage. Reference: Toselli & Widlund (2005).
//! *Domain Decomposition Methods — Algorithms and Theory*. Springer.
//!
//! [`lsmr`] solves `min ‖b − A x‖₂` for any [`Operator`], with no
//! preconditioner required:
//!
//! ```
//! use schwarz_precond::{lsmr, Operator, SolveError};
//!
//! struct Diag(Vec<f64>);
//! impl Operator for Diag {
//!     fn nrows(&self) -> usize {
//!         self.0.len()
//!     }
//!     fn ncols(&self) -> usize {
//!         self.0.len()
//!     }
//!     fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
//!         for i in 0..self.0.len() {
//!             y[i] = self.0[i] * x[i];
//!         }
//!         Ok(())
//!     }
//!     fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
//!         self.apply(x, y)
//!     }
//! }
//!
//! let a = Diag(vec![2.0, 3.0]);
//! let r = lsmr(&a, &[4.0, 9.0], 1e-10, 50, None).unwrap();
//! assert!(r.converged);
//! assert!((r.x[0] - 2.0).abs() < 1e-6);
//! assert!((r.x[1] - 3.0).abs() < 1e-6);
//! ```

#![deny(missing_docs)]
#![warn(clippy::all)]

/// A linear operator A: R^ncols -> R^nrows with its adjoint A^T.
///
/// Preconditioners are operators too (M^{-1} is a linear map).
/// All implementors must be Send + Sync to enable Rayon parallelism.
///
/// Both apply methods are fallible: implementors that cannot fail in practice
/// (matrices, identity operators) still return `Result<(), SolveError>` so
/// callers can use a uniform `?` propagation path. Symmetric operators
/// should delegate `apply_adjoint` to `apply`.
pub trait Operator: Send + Sync {
    /// Number of rows in the operator.
    fn nrows(&self) -> usize;
    /// Number of columns in the operator.
    fn ncols(&self) -> usize;
    /// Computes y = A * x. Returns an error if the apply fails at runtime
    /// (e.g. a local subdomain solver diverges).
    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), error::SolveError>;
    /// Computes y = A^T * x. For symmetric operators, this should delegate to `apply`.
    /// Returns an error under the same conditions as `apply`.
    fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) -> Result<(), error::SolveError>;
}

/// Domain decomposition primitives: subdomain cores and partition weights.
pub mod domain;
/// Typed errors for build and runtime failures.
pub mod error;
mod local_solve;
mod lsmr;
mod schwarz;

pub use domain::{PartitionWeights, SubdomainCore};
pub use error::{BuildError, LocalSolveError, SolveError};
pub use local_solve::{LocalSolver, SubdomainEntry};
pub use lsmr::{lsmr, mlsmr, LsmrResult, LsmrStopReason};
pub use schwarz::{ReductionStrategy, SchwarzPreconditioner};
