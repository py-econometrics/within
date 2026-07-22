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
    /// A slope loading is not finite. A NaN/∞ loading makes its level-Gram
    /// diagonal non-finite, so reparameterization silently drops the column and
    /// misreports it as an unidentified direction; loadings must be finite.
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
    /// A signed component could not be certified as diagonally scalable to an
    /// SDDM operator.
    #[error("signed component between {pair} is not diagonally scalable to SDDM form")]
    UnscalableComponent {
        /// The offending channel pair.
        pair: SignedPair,
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

/// The channel pair whose signed cross-factor component an error or warning
/// refers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SignedPair {
    /// Term index of the pair's first channel.
    pub term_q: usize,
    /// Coefficient column of the first channel within its term.
    pub column_q: usize,
    /// Term index of the pair's second channel.
    pub term_r: usize,
    /// Coefficient column of the second channel within its term.
    pub column_r: usize,
}

impl std::fmt::Display for SignedPair {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "term {} column {} and term {} column {}",
            self.term_q, self.column_q, self.term_r, self.column_r
        )
    }
}

/// A non-fatal preconditioner-build event, surfaced via
/// [`Solver::warnings`](crate::Solver::warnings).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum BuildWarning {
    /// A signed component's dominance scaling was not certified within the
    /// configured tolerance; residual deficits were clamped, which degrades
    /// only preconditioner quality.
    UnscalableComponent {
        /// The offending channel pair.
        pair: SignedPair,
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
