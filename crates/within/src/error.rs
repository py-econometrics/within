use std::error::Error;
use std::fmt::{Display, Formatter};

use schwarz_precond::{PreconditionerBuildError, SolveError};

/// Result alias used by fallible public APIs in this crate.
pub type WithinResult<T> = Result<T, WithinError>;

/// Errors produced while validating inputs or building solver components.
#[derive(Debug)]
pub enum WithinError {
    /// Number of factor columns differs from the number of factor level counts.
    FactorCountMismatch {
        category_factors: usize,
        level_factors: usize,
    },
    /// One factor column does not match the expected observation count.
    ObservationCountMismatch {
        factor: usize,
        expected: usize,
        got: usize,
    },
    /// Weight vector does not match the number of observations.
    WeightCountMismatch { expected: usize, got: usize },
    /// RHS vector does not match the number of observations.
    RhsCountMismatch { expected: usize, got: usize },
    /// A factor declares zero levels.
    EmptyLevelSet { factor: usize },
    /// Category value is negative in i64-based constructors.
    NegativeCategory {
        factor: usize,
        observation: usize,
        level: i64,
    },
    /// Category level is out of range for a factor.
    LevelOutOfRange {
        factor: usize,
        observation: usize,
        level: u32,
        n_levels: usize,
    },
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
            Self::FactorCountMismatch {
                category_factors,
                level_factors,
            } => write!(
                f,
                "factor count mismatch: categories has {category_factors} factors, n_levels has {level_factors}",
            ),
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
            Self::RhsCountMismatch { expected, got } => {
                write!(f, "rhs has length {got}, expected {expected}")
            }
            Self::EmptyLevelSet { factor } => {
                write!(f, "factor {factor} declares zero levels")
            }
            Self::NegativeCategory {
                factor,
                observation,
                level,
            } => write!(
                f,
                "factor {factor}, observation {observation}: negative category level {level}",
            ),
            Self::LevelOutOfRange {
                factor,
                observation,
                level,
                n_levels,
            } => write!(
                f,
                "factor {factor}, observation {observation}: level {level} out of range for n_levels={n_levels}",
            ),
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
