//! Error types for the `within` crate.

use thiserror::Error;

pub use schwarz_precond::SolveError;

/// Errors produced while validating inputs or building solver components.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum BuildError {
    /// No observations provided.
    #[error("no observations provided")]
    EmptyObservations,
    /// One column does not match the expected observation count.
    #[error("column {column} has {got} observations, expected {expected}")]
    ObservationCountMismatch {
        /// Index of the mismatched column (categorical first, then continuous).
        column: usize,
        /// Expected number of observations.
        expected: usize,
        /// Actual number of observations in this column.
        got: usize,
    },
    /// An effect with neither an intercept nor a slope.
    #[error("an effect must have an intercept or at least one slope")]
    EmptyEffect,
    /// A slope covariate's length does not match the effect's level count.
    #[error("slope {slope} has {got} values, expected {expected}")]
    SlopeLengthMismatch {
        /// Index of the slope covariate within its effect.
        slope: usize,
        /// Expected length (the effect's level count).
        expected: usize,
        /// Actual length.
        got: usize,
    },
    /// An effect carries varying slopes in a form the solver does not yet support.
    #[error("effect {effect} has varying slopes, not yet supported alongside other effects or with more than one slope")]
    SlopesNotYetSupported {
        /// Index of the offending effect.
        effect: usize,
    },
    /// Weight vector does not match the number of observations.
    #[error("weights has length {got}, expected {expected}")]
    WeightCountMismatch {
        /// Expected number of weights.
        expected: usize,
        /// Actual weight vector length.
        got: usize,
    },
    /// A weight is not a usable variance. The operator applies `W^{1/2}`, so a
    /// negative or non-finite (NaN/∞) weight would take `sqrt` of a bad value
    /// and silently corrupt the solution; weights must be finite and `>= 0`.
    #[error("weight at index {index} must be finite and non-negative, got {value}")]
    InvalidWeight {
        /// Index of the offending weight.
        index: usize,
        /// The offending value.
        value: f64,
    },
    /// A zero diagonal was encountered during block elimination.
    #[error("zero diagonal in {block} block at index {index}")]
    SingularDiagonal {
        /// Which block contained the zero diagonal ("keep" or "elim").
        block: &'static str,
        /// Row/column index of the zero diagonal entry.
        index: usize,
    },
    /// Local solver construction failed.
    #[error("local solver build failed: {0}")]
    LocalSolverBuild(String),
    /// Schwarz preconditioner structural validation failed.
    #[error("preconditioner build failed: {0}")]
    Preconditioner(#[source] schwarz_precond::BuildError),
    /// A pre-built preconditioner's shape does not match the design's DOF count.
    #[error(
        "prebuilt preconditioner shape ({actual_rows}x{actual_cols}) does not match \
         design DOF count {expected}"
    )]
    PreconditionerDimensionMismatch {
        /// Expected number of rows and columns (design `n_dofs`).
        expected: usize,
        /// Actual row count of the supplied preconditioner.
        actual_rows: usize,
        /// Actual column count of the supplied preconditioner.
        actual_cols: usize,
    },
}

/// Top-level error type returned by [`crate::solve`] and [`crate::solve_batch`].
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum WithinError {
    /// Build-time failure.
    #[error(transparent)]
    Build(#[from] BuildError),
    /// Solve-time failure.
    #[error(transparent)]
    Solve(#[from] SolveError),
}
