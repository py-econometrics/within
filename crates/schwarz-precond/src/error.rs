//! Error types for the `schwarz-precond` crate.
//!
//! Errors are partitioned by lifecycle phase:
//!
//! - **Build** ([`BuildError`]) — caught during construction, before any
//!   solve begins. Covers partition-weight validation, subdomain DOF
//!   contracts, and preconditioner-wide index checks.
//! - **Solve** ([`SolveError`]) — solver input and breakdowns; operator applies are infallible.

use thiserror::Error;

/// Construction-time validation errors for the Schwarz building blocks.
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
}

/// Runtime failure while executing a solve.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum SolveError {
    /// Solver input was invalid before any iteration was attempted.
    #[error("invalid solver input at {context}: {message}")]
    InvalidInput {
        /// Context string identifying the validation site.
        context: &'static str,
        /// Validation failure details.
        message: String,
    },
}
