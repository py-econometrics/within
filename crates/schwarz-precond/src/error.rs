//! Error types for the `schwarz-precond` crate.
//!
//! Two enums:
//!
//! - [`BuildError`] — caught during construction, before any solve begins.
//! - [`SolveError`] — runtime failures during operator application or
//!   iterative solver execution.

use std::error::Error;
use std::fmt::{Display, Formatter};

use crate::local_solve::{LocalSolver, SubdomainEntry};

/// Construction-time validation errors for subdomains and preconditioners.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BuildError {
    /// Partition-of-unity weight vector length does not match index count.
    PartitionWeightLengthMismatch {
        /// Number of global indices in the subdomain core.
        index_count: usize,
        /// Number of partition weights in the subdomain core.
        weight_count: usize,
    },
    /// Local solver `n_local` does not match the subdomain index count.
    LocalDofCountMismatch {
        /// Number of global indices in the subdomain core.
        index_count: usize,
        /// Local DOF count reported by the solver implementation.
        solver_n_local: usize,
    },
    /// Local solver scratch size is too small for the subdomain gather/scatter buffers.
    ScratchSizeTooSmall {
        /// Scratch size reported by the local solver.
        scratch_size: usize,
        /// Minimum scratch size required by the subdomain core.
        required_min: usize,
    },
    /// A subdomain references a global DOF outside `[0, n_dofs)`.
    GlobalIndexOutOfBounds {
        /// Index of the failing subdomain entry in the provided list.
        subdomain: usize,
        /// Position inside `global_indices` where the invalid DOF was found.
        local_index: usize,
        /// Global DOF index that exceeded the valid range.
        global_index: u32,
        /// Total number of global DOFs configured for the preconditioner.
        n_dofs: usize,
    },
}

impl Display for BuildError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PartitionWeightLengthMismatch {
                index_count,
                weight_count,
            } => write!(
                f,
                "partition weight count ({weight_count}) does not match index count ({index_count})",
            ),
            Self::LocalDofCountMismatch {
                index_count,
                solver_n_local,
            } => write!(
                f,
                "index count ({index_count}) does not match solver n_local ({solver_n_local})",
            ),
            Self::ScratchSizeTooSmall {
                scratch_size,
                required_min,
            } => write!(
                f,
                "scratch size ({scratch_size}) is smaller than required minimum ({required_min})",
            ),
            Self::GlobalIndexOutOfBounds {
                subdomain,
                local_index,
                global_index,
                n_dofs,
            } => write!(
                f,
                "subdomain {subdomain}: global index at local position {local_index} is out of bounds ({global_index} >= {n_dofs})",
            ),
        }
    }
}

impl Error for BuildError {}

/// Runtime failure while applying an operator/preconditioner or running an iterative solver.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum SolveError {
    /// A local subdomain solve failed (typically inside a Schwarz preconditioner).
    LocalSolveFailed {
        /// Index of the failing subdomain entry in the preconditioner.
        subdomain: usize,
        /// Context string identifying where the failure occurred.
        context: &'static str,
        /// Backend error text.
        message: String,
    },
    /// Internal synchronization failed (e.g. poisoned mutex).
    Synchronization {
        /// Context string identifying the lock/synchronization site.
        context: &'static str,
    },
    /// Solver input was invalid before any iteration was attempted.
    InvalidInput {
        /// Context string identifying the validation site.
        context: &'static str,
        /// Validation failure details.
        message: String,
    },
}

impl Display for SolveError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LocalSolveFailed {
                subdomain,
                context,
                message,
            } => write!(
                f,
                "subdomain {subdomain} local solve failed at {context}: {message}"
            ),
            Self::Synchronization { context } => {
                write!(f, "synchronization failure at {context}")
            }
            Self::InvalidInput { context, message } => {
                write!(f, "invalid solver input at {context}: {message}")
            }
        }
    }
}

impl Error for SolveError {}

/// Re-tag a `SolveError::LocalSolveFailed` with the correct subdomain index.
///
/// Local solvers don't know their own subdomain index, so they construct
/// `LocalSolveFailed` with a placeholder. The Schwarz exec loop catches the
/// error and uses this helper to attach the correct index. Other variants
/// pass through unchanged.
pub(crate) fn tag_subdomain(error: SolveError, subdomain: usize) -> SolveError {
    match error {
        SolveError::LocalSolveFailed {
            context, message, ..
        } => SolveError::LocalSolveFailed {
            subdomain,
            context,
            message,
        },
        other => other,
    }
}

pub(crate) fn validate_entries<S: LocalSolver>(
    entries: &[SubdomainEntry<S>],
    n_dofs: usize,
) -> Result<(), BuildError> {
    for (subdomain, entry) in entries.iter().enumerate() {
        for (local_index, &global_index) in entry.global_indices().iter().enumerate() {
            if (global_index as usize) >= n_dofs {
                return Err(BuildError::GlobalIndexOutOfBounds {
                    subdomain,
                    local_index,
                    global_index,
                    n_dofs,
                });
            }
        }
    }
    Ok(())
}
