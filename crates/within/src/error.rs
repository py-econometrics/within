//! Error types for the `within` crate.

use thiserror::Error;

use crate::channel::ChannelPair;

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
    /// A slope covariate's length does not match the observation count.
    #[error("slope {slope} has {got} values, expected {expected}")]
    SlopeLengthMismatch {
        /// Index of the slope covariate within its effect.
        slope: usize,
        /// Expected length (the observation count — one value per row).
        expected: usize,
        /// Actual length.
        got: usize,
    },
    /// Non-finite makes the level-Gram diagonal non-finite, so the column is silently dropped.
    #[error("slope {slope} loading at index {index} must be finite, got {value}")]
    InvalidLoading {
        /// Index of the offending slope covariate within its effect.
        slope: usize,
        /// Index of the offending loading value.
        index: usize,
        /// The offending value.
        value: f64,
    },
    /// Weight vector does not match the number of observations.
    #[error("weights has length {got}, expected {expected}")]
    WeightCountMismatch {
        /// Expected number of weights.
        expected: usize,
        /// Actual weight vector length.
        got: usize,
    },
    /// The operator applies `W^{1/2}`, so a negative or non-finite weight corrupts the solution.
    #[error("weight at index {index} must be finite and non-negative, got {value}")]
    InvalidWeight {
        /// Index of the offending weight.
        index: usize,
        /// The offending value.
        value: f64,
    },
    /// A signed component could not be certified as diagonally scalable to an SDDM operator.
    #[error("signed component between {pair} is not diagonally scalable to SDDM form")]
    UnscalableComponent {
        /// The offending channel pair.
        pair: ChannelPair,
    },
    /// A zero (or non-finite-reciprocal) diagonal was encountered while building a preconditioner.
    #[error("zero diagonal in preconditioner at index {index}")]
    SingularDiagonal {
        /// Row/column index of the degenerate entry.
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
    /// Usually raw entity IDs passed as factor codes, inflating `n_levels = max code + 1`.
    #[error("design has {n_dofs} degrees of freedom, exceeding the u32 column-index limit")]
    DofSpaceExceedsU32 {
        /// Total degrees of freedom implied by the design.
        n_dofs: usize,
    },
}

/// A non-fatal preconditioner-build event.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum BuildWarning {
    /// Residual deficits were clamped, degrading only preconditioner quality.
    UnscalableComponent {
        /// The offending channel pair.
        pair: ChannelPair,
        /// Relaxation sweeps spent before handing the scaling over.
        sweeps: usize,
        /// Largest relative dominance violation at hand-over.
        violation: f64,
    },
}

impl std::fmt::Display for BuildWarning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnscalableComponent {
                pair,
                sweeps,
                violation,
            } => write!(
                f,
                "signed component between {pair}: dominance scaling uncertified after \
                 {sweeps} sweeps (max relative violation {violation:.2e}); deficits \
                 clamped, preconditioner quality may degrade"
            ),
        }
    }
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
