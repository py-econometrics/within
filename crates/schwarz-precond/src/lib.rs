//! Schwarz domain decomposition preconditioner library.
//!
//! Provides a one-level additive Schwarz preconditioner generic over local
//! solvers, plus a Modified LSMR iterative solver for rectangular
//! least-squares problems. The preconditioner approximates `(AᵀA)^{-1}` by
//! solving overlapping subproblems and stitching them together with a
//! partition-of-unity weighting.
//!
//! Operators are faer [`LinOp`]s: each apply gets a budget ([`Par`]) and scratch ([`MemStack`]).
//!
//! See `examples/` for runnable usage. Reference: Toselli & Widlund (2005).
//! *Domain Decomposition Methods — Algorithms and Theory*. Springer.
//!
//! [`mlsmr`] solves `min ‖b − A x‖₂` for any [`BiLinOp`], such as a faer matrix;
//! [`IdentityPrecond`] runs it unpreconditioned:
//!
//! ```
//! use schwarz_precond::{mlsmr, IdentityPrecond, InitialGuess, LsmrOptions, MemBuffer, Par};
//!
//! let a = faer::mat![[2.0, 0.0], [0.0, 3.0]];
//! let (b, mut x) = ([4.0, 9.0], [0.0f64; 2]);
//! let options = LsmrOptions { tol: 1e-10, ..Default::default() };
//! let none = IdentityPrecond { dim: 2 };
//! let mut scratch = MemBuffer::new(Default::default());
//! let info = mlsmr(&a, &b, &none, &mut x, InitialGuess::Zero, &options, None, Par::Seq, &mut scratch)
//!     .unwrap();
//! assert!(info.converged);
//! assert!((x[0] - 2.0).abs() < 1e-6 && (x[1] - 3.0).abs() < 1e-6);
//! ```

#![deny(missing_docs)]
#![warn(clippy::all)]

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
#[doc(hidden)]
pub mod par;
mod prolongation;
mod schwarz;

pub use domain::{PartitionWeights, SubdomainCore};
pub use error::{BuildError, SolveError};
pub use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
pub use faer::matrix_free::{BiLinOp, IdentityPrecond, LinOp};
pub use faer::Par;
pub use faer::{ColMut, ColRef, MatMut, MatRef};
pub use local_solve::{LocalSolver, SubdomainEntry};
pub use lsmr::{
    mlsmr, InitialGuess, LsmrInfo, LsmrOptions, LsmrStopReason, Staleness, StalenessError,
};
pub use schwarz::SchwarzPreconditioner;

#[doc(hidden)]
pub use lsmr::{with_unit_stride, NonExhaustive};
