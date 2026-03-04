use std::error::Error;
use std::fmt::{Display, Formatter};

use crate::local_solve::{LocalSolver, SubdomainEntry};

/// Construction-time validation errors for Schwarz preconditioners.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PreconditionerBuildError {
    /// Local solver `n_local` does not match the subdomain index count.
    LocalDofCountMismatch {
        /// Index of the failing subdomain entry in the provided list.
        subdomain: usize,
        /// Number of global indices in the subdomain core.
        index_count: usize,
        /// Local DOF count reported by the solver implementation.
        solver_n_local: usize,
    },
    /// Local solver scratch size is too small for the subdomain gather/scatter buffers.
    ScratchSizeTooSmall {
        /// Index of the failing subdomain entry in the provided list.
        subdomain: usize,
        /// Scratch size reported by the local solver.
        scratch_size: usize,
        /// Minimum scratch size required by the subdomain core.
        required_min: usize,
    },
    /// Partition-of-unity weight vector length does not match index count.
    PartitionWeightLengthMismatch {
        /// Index of the failing subdomain entry in the provided list.
        subdomain: usize,
        /// Number of global indices in the subdomain core.
        index_count: usize,
        /// Number of partition weights in the subdomain core.
        weight_count: usize,
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

impl Display for PreconditionerBuildError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LocalDofCountMismatch {
                subdomain,
                index_count,
                solver_n_local,
            } => write!(
                f,
                "subdomain {subdomain}: index count ({index_count}) does not match solver n_local ({solver_n_local})",
            ),
            Self::ScratchSizeTooSmall {
                subdomain,
                scratch_size,
                required_min,
            } => write!(
                f,
                "subdomain {subdomain}: scratch size ({scratch_size}) is smaller than required minimum ({required_min})",
            ),
            Self::PartitionWeightLengthMismatch {
                subdomain,
                index_count,
                weight_count,
            } => write!(
                f,
                "subdomain {subdomain}: partition weight count ({weight_count}) does not match index count ({index_count})",
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

impl Error for PreconditionerBuildError {}

/// Runtime error emitted by a local subdomain solver during a solve call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LocalSolveError {
    /// Approximate Cholesky back-substitution failed.
    ApproxCholSolveFailed {
        /// Context string identifying where the failure occurred.
        context: &'static str,
        /// Backend error text.
        message: String,
    },
}

impl Display for LocalSolveError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ApproxCholSolveFailed { context, message } => {
                write!(f, "{context}: {message}")
            }
        }
    }
}

impl Error for LocalSolveError {}

/// Runtime failure while applying a Schwarz preconditioner/operator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApplyError {
    /// A local subdomain solve failed.
    LocalSolveFailed {
        /// Index of the failing subdomain entry in the preconditioner.
        subdomain: usize,
        /// Local solver error.
        source: LocalSolveError,
    },
    /// Internal synchronization failed (e.g. poisoned mutex).
    Synchronization {
        /// Context string identifying the lock/synchronization site.
        context: &'static str,
    },
}

impl Display for ApplyError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LocalSolveFailed { subdomain, source } => {
                write!(f, "subdomain {subdomain} local solve failed: {source}")
            }
            Self::Synchronization { context } => {
                write!(f, "synchronization failure at {context}")
            }
        }
    }
}

impl Error for ApplyError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::LocalSolveFailed { source, .. } => Some(source),
            Self::Synchronization { .. } => None,
        }
    }
}

/// Runtime failure while executing an iterative solver.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SolveError {
    /// Operator/preconditioner apply failed.
    Apply(ApplyError),
}

impl Display for SolveError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Apply(err) => write!(f, "operator apply failed: {err}"),
        }
    }
}

impl Error for SolveError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Apply(err) => Some(err),
        }
    }
}

impl From<ApplyError> for SolveError {
    fn from(value: ApplyError) -> Self {
        Self::Apply(value)
    }
}

pub(crate) fn validate_entries<S: LocalSolver>(
    entries: &[SubdomainEntry<S>],
    n_dofs: usize,
) -> Result<(), PreconditionerBuildError> {
    for (subdomain, entry) in entries.iter().enumerate() {
        let index_count = entry.core.global_indices.len();
        let solver_n_local = entry.solver.n_local();
        if solver_n_local != index_count {
            return Err(PreconditionerBuildError::LocalDofCountMismatch {
                subdomain,
                index_count,
                solver_n_local,
            });
        }

        let scratch_size = entry.scratch_size();
        if scratch_size < index_count {
            return Err(PreconditionerBuildError::ScratchSizeTooSmall {
                subdomain,
                scratch_size,
                required_min: index_count,
            });
        }

        let weight_count = entry.core.partition_weights.len();
        if weight_count != index_count {
            return Err(PreconditionerBuildError::PartitionWeightLengthMismatch {
                subdomain,
                index_count,
                weight_count,
            });
        }

        for (local_index, &global_index) in entry.core.global_indices.iter().enumerate() {
            if (global_index as usize) >= n_dofs {
                return Err(PreconditionerBuildError::GlobalIndexOutOfBounds {
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
