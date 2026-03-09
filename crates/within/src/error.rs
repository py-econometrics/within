use std::error::Error;
use std::fmt::{Display, Formatter};

use schwarz_precond::{PreconditionerBuildError, SolveError};

/// Result alias used by fallible public APIs in this crate.
pub type WithinResult<T> = Result<T, WithinError>;

/// Errors produced while validating inputs or building solver components.
#[derive(Debug)]
pub enum WithinError {
    /// No observations provided.
    EmptyObservations,
    /// One factor column does not match the expected observation count.
    ObservationCountMismatch {
        factor: usize,
        expected: usize,
        got: usize,
    },
    /// Weight vector does not match the number of observations.
    WeightCountMismatch { expected: usize, got: usize },
    /// Numeric overflow during assembly.
    Overflow(String),
    /// A zero diagonal was encountered during block elimination.
    SingularDiagonal { block: &'static str, index: usize },
    /// Local solver construction failed.
    LocalSolverBuild(String),
    /// Preconditioner structural validation failed.
    PreconditionerBuild(PreconditionerBuildError),
    /// Iterative solver runtime error.
    IterativeSolve(SolveError),
}

impl Display for WithinError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyObservations => write!(f, "no observations provided"),
            Self::ObservationCountMismatch {
                factor,
                expected,
                got,
            } => write!(
                f,
                "factor {factor} has {got} observations, expected {expected}",
            ),
            Self::WeightCountMismatch { expected, got } => {
                write!(f, "weights has length {got}, expected {expected}")
            }
            Self::Overflow(msg) => write!(f, "numeric overflow: {msg}"),
            Self::SingularDiagonal { block, index } => {
                write!(f, "zero diagonal in {block} block at index {index}")
            }
            Self::LocalSolverBuild(msg) => write!(f, "local solver build failed: {msg}"),
            Self::PreconditionerBuild(err) => write!(f, "{err}"),
            Self::IterativeSolve(err) => write!(f, "{err}"),
        }
    }
}

impl Error for WithinError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::PreconditionerBuild(err) => Some(err),
            Self::IterativeSolve(err) => Some(err),
            _ => None,
        }
    }
}

impl From<PreconditionerBuildError> for WithinError {
    fn from(value: PreconditionerBuildError) -> Self {
        Self::PreconditionerBuild(value)
    }
}

impl From<SolveError> for WithinError {
    fn from(value: SolveError) -> Self {
        Self::IterativeSolve(value)
    }
}
