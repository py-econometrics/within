//! Error types for the `schwarz-precond` crate.
//!
//! Errors are partitioned by lifecycle phase:
//!
//! - **Build** ([`BuildError`]) — caught during construction, before any
//!   solve begins. Covers partition-weight validation, subdomain DOF/scratch
//!   contracts, and preconditioner-wide index checks.
//! - **Solve** ([`SolveError`]) — runtime failures during a solve, including
//!   operator/preconditioner application (e.g. a local solver diverges) and
//!   iterative-solver input validation.
//!
//! [`crate::error::LocalSolveError`] is the narrow trait contract returned by
//! [`crate::LocalSolver::solve_local`]. The Schwarz executor lifts it into
//! [`SolveError::LocalSolveFailed`] at the one apply site that knows the
//! subdomain index, so there is no `From` chain between the two.

use thiserror::Error;

/// Construction-time validation errors for the Schwarz building blocks.
///
/// Consolidates failures from subdomain core/entry construction and
/// preconditioner-wide index checks into a single flat enum.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum BuildError {
    /// Partition-of-unity weight vector length does not match index count.
    #[error("partition weight count ({weight_count}) does not match index count ({index_count})")]
    PartitionWeightLengthMismatch {
        /// Number of global indices in the subdomain core.
        index_count: usize,
        /// Number of partition weights in the subdomain core.
        weight_count: usize,
    },
    /// Local solver `n_local` does not match the subdomain index count.
    #[error("index count ({index_count}) does not match solver n_local ({solver_n_local})")]
    LocalDofCountMismatch {
        /// Number of global indices in the subdomain core.
        index_count: usize,
        /// Local DOF count reported by the solver implementation.
        solver_n_local: usize,
    },
    /// Local solver scratch size is too small for the subdomain gather/scatter buffers.
    #[error("scratch size ({scratch_size}) is smaller than required minimum ({required_min})")]
    ScratchSizeTooSmall {
        /// Scratch size reported by the local solver.
        scratch_size: usize,
        /// Minimum scratch size required by the subdomain core.
        required_min: usize,
    },
}

/// Runtime error emitted by a local subdomain solver during a solve call.
///
/// Returned by [`crate::LocalSolver::solve_local`].
/// Backend-agnostic by design: backends report a `context` site and a
/// free-form `message` rather than enumerating their internal error modes
/// through this generic crate.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum LocalSolveError {
    /// The backend implementation reported a failure during a local solve.
    #[error("{context}: {message}")]
    BackendFailed {
        /// Context string identifying where the failure occurred.
        context: &'static str,
        /// Backend error text.
        message: String,
    },
}

/// Runtime failure while executing a solve.
///
/// Covers both operator/preconditioner application failures (e.g. a local
/// solver diverges) and iterative-solver input validation.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum SolveError {
    /// A local subdomain solve failed during a preconditioner apply.
    #[error("subdomain {subdomain} local solve failed: {source}")]
    LocalSolveFailed {
        /// Index of the failing subdomain entry in the preconditioner.
        subdomain: usize,
        /// Local solver error.
        #[source]
        source: LocalSolveError,
    },
    /// Internal synchronization failed (e.g. poisoned mutex) during an apply.
    #[error("synchronization failure at {context}")]
    Synchronization {
        /// Context string identifying the lock/synchronization site.
        context: &'static str,
    },
    /// Solver input was invalid before any iteration was attempted.
    #[error("invalid solver input at {context}: {message}")]
    InvalidInput {
        /// Context string identifying the validation site.
        context: &'static str,
        /// Validation failure details.
        message: String,
    },
}
