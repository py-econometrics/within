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
//! [`mlsmr`] solves `min ‖b − A x‖₂` for any [`Operator`]; [`IdentityOp`] runs it
//! unpreconditioned:
//!
//! ```
//! use schwarz_precond::{mlsmr, IdentityOp, MemStack, MlsmrOptions, Operator, SolveError};
//!
//! struct Diag(Vec<f64>);
//! impl Operator for Diag {
//!     fn nrows(&self) -> usize {
//!         self.0.len()
//!     }
//!     fn ncols(&self) -> usize {
//!         self.0.len()
//!     }
//!     fn apply_with(&self, x: &[f64], y: &mut [f64], _: &mut MemStack) -> Result<(), SolveError> {
//!         for i in 0..self.0.len() {
//!             y[i] = self.0[i] * x[i];
//!         }
//!         Ok(())
//!     }
//!     fn apply_adjoint_with(&self, x: &[f64], y: &mut [f64], stack: &mut MemStack) -> Result<(), SolveError> {
//!         self.apply_with(x, y, stack)
//!     }
//! }
//!
//! let a = Diag(vec![2.0, 3.0]);
//! let none = IdentityOp { n: 2 };
//! let r = mlsmr(&a, &[4.0, 9.0], &none, 1e-10, 50, None, MlsmrOptions::default()).unwrap();
//! assert!(r.converged());
//! assert!((r.x[0] - 2.0).abs() < 1e-6);
//! assert!((r.x[1] - 3.0).abs() < 1e-6);
//! ```

#![deny(missing_docs)]
#![warn(clippy::all)]

/// A linear operator with its adjoint; both applies are fallible for one uniform `?` path.
///
/// An apply's working memory comes from the `stack` its caller lends; the thread limit is the
/// rayon pool it runs in.
pub trait Operator: Send + Sync {
    /// Number of rows in the operator.
    fn nrows(&self) -> usize;
    /// Number of columns in the operator.
    fn ncols(&self) -> usize;
    /// What one apply, of either side, takes from its stack; nothing unless stated. An operator
    /// that applies another on its stack must include the other's: the default leaves it out, and
    /// the inner apply then panics on a stack too small.
    fn scratch(&self) -> StackReq {
        StackReq::EMPTY
    }
    /// Computes `y = A x`; `stack` holds at least [`scratch`](Self::scratch), aligned as it
    /// states, which [`MemBuffer::new`] gives.
    fn apply_with(
        &self,
        x: &[f64],
        y: &mut [f64],
        stack: &mut MemStack,
    ) -> Result<(), error::SolveError>;
    /// Computes `y = Aᵀ x`; symmetric operators should delegate to `apply_with`.
    fn apply_adjoint_with(
        &self,
        x: &[f64],
        y: &mut [f64],
        stack: &mut MemStack,
    ) -> Result<(), error::SolveError>;
    /// [`apply_with`](Self::apply_with) on a stack allocated for this call.
    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), error::SolveError> {
        self.apply_with(x, y, MemStack::new(&mut MemBuffer::new(self.scratch())))
    }
    /// [`apply_adjoint_with`](Self::apply_adjoint_with) on a stack allocated for this call.
    fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) -> Result<(), error::SolveError> {
        self.apply_adjoint_with(x, y, MemStack::new(&mut MemBuffer::new(self.scratch())))
    }
}

// Compiles the README usage example as a doctest so it cannot drift from the public API.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDoctests;

/// Domain decomposition primitives: subdomain cores and partition weights.
pub mod domain;
/// Typed errors for build and runtime failures.
pub mod error;
mod local_solve;
mod lsmr;
pub mod parallel;
mod prolongation;
mod schwarz;
mod threads;

pub use domain::{PartitionWeights, SubdomainCore};
pub use dyn_stack::{MemBuffer, MemStack, StackReq};
pub use error::{BuildError, LocalSolveError, SolveError};
pub use local_solve::{LocalSolver, SubdomainEntry};
pub use lsmr::{
    mlsmr, EscalationHandler, EscalationPolicy, IdentityOp, LsmrResult, LsmrStopReason,
    MlsmrOptions, Progress, Staleness, StalenessError,
};
pub use schwarz::SchwarzPreconditioner;
pub use threads::Threads;
