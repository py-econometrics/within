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
//! use schwarz_precond::{
//!     mlsmr, mlsmr_scratch, IdentityOp, MemBuffer, MemStack, MlsmrOptions, Operator, SolveError,
//!     StackReq, Threads,
//! };
//!
//! struct Diag(Vec<f64>);
//! impl Operator for Diag {
//!     fn nrows(&self) -> usize {
//!         self.0.len()
//!     }
//!     fn ncols(&self) -> usize {
//!         self.0.len()
//!     }
//!     fn scratch(&self, _columns: usize, _threads: Threads) -> StackReq {
//!         StackReq::EMPTY
//!     }
//!     fn apply(&self, x: &[f64], y: &mut [f64], _: Threads, _: &mut MemStack) -> Result<(), SolveError> {
//!         for i in 0..self.0.len() {
//!             y[i] = self.0[i] * x[i];
//!         }
//!         Ok(())
//!     }
//!     fn apply_adjoint(
//!         &self,
//!         x: &[f64],
//!         y: &mut [f64],
//!         threads: Threads,
//!         stack: &mut MemStack,
//!     ) -> Result<(), SolveError> {
//!         self.apply(x, y, threads, stack)
//!     }
//! }
//!
//! let a = Diag(vec![2.0, 3.0]);
//! let none = IdentityOp { n: 2 };
//! let mut scratch = MemBuffer::new(mlsmr_scratch(&a, &none, Threads::All));
//! let stack = MemStack::new(&mut scratch);
//! let r = mlsmr(&a, &[4.0, 9.0], &none, 1e-10, 50, MlsmrOptions::default(), Threads::All, stack)
//!     .unwrap();
//! assert!(r.converged);
//! assert!((r.x[0] - 2.0).abs() < 1e-6);
//! assert!((r.x[1] - 3.0).abs() < 1e-6);
//! ```

#![deny(missing_docs)]
#![warn(clippy::all)]

/// A linear operator with its adjoint; both applies are fallible for one uniform `?` path.
///
/// Each apply runs on at most `threads` and takes its scratch from the caller's `stack`.
pub trait Operator: Send + Sync {
    /// Number of rows in the operator.
    fn nrows(&self) -> usize;
    /// Number of columns in the operator.
    fn ncols(&self) -> usize;
    /// What either apply on `columns` columns can use at `threads`; at `Threads::ONE`, all it needs.
    fn scratch(&self, columns: usize, threads: Threads) -> StackReq;
    /// Computes `y = A x`; `stack` holds at least `scratch(1, Threads::ONE)`, with stale contents.
    fn apply(
        &self,
        x: &[f64],
        y: &mut [f64],
        threads: Threads,
        stack: &mut MemStack,
    ) -> Result<(), error::SolveError>;
    /// Computes `y = Aᵀ x`; symmetric operators should delegate to `apply`.
    fn apply_adjoint(
        &self,
        x: &[f64],
        y: &mut [f64],
        threads: Threads,
        stack: &mut MemStack,
    ) -> Result<(), error::SolveError>;
}

// Compiles the README usage example as a doctest so it cannot drift from the public API.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDoctests;

/// Domain decomposition primitives: subdomain cores and partition weights.
pub mod domain;
/// Typed errors for build and runtime failures.
pub mod error;
mod fold;
mod local_solve;
mod lsmr;
mod prolongation;
mod schwarz;
mod threads;

pub use domain::{PartitionWeights, SubdomainCore};
pub use dyn_stack::{MemBuffer, MemStack, StackReq};
pub use error::{BuildError, LocalSolveError, SolveError};
pub use local_solve::{LocalSolver, SubdomainEntry};
pub use lsmr::{
    mlsmr, mlsmr_scratch, EscalationHandler, EscalationPolicy, IdentityOp, LsmrResult,
    LsmrStopReason, MlsmrOptions, Progress, Staleness, StalenessError,
};
pub use schwarz::SchwarzPreconditioner;
pub use threads::Threads;
