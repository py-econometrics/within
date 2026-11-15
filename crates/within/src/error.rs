//! Error types for the `within` crate.

use thiserror::Error;

use crate::channel::ChannelPair;

pub use schwarz_precond::SolveError;

/// Errors produced while validating inputs or building solver components.
#[derive(Debug, Clone, Error)]
#[non_exhaustive]
pub enum BuildError {
    /// No observations provided.
    #[error("no observations provided")]
    EmptyObservations,
    /// One effect does not match the first effect's observation count.
    #[error("column {column} has {got} observations, expected {expected}")]
    ObservationCountMismatch {
        /// Index of the mismatched effect.
        column: usize,
        /// Expected number of observations.
        expected: usize,
        /// Actual number of observations in this effect.
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
    /// The thread pool a deferred preconditioner build runs on could not be created.
    #[error("thread pool build failed: {0}")]
    ThreadPool(String),
    /// Schwarz preconditioner structural validation failed.
    #[error("preconditioner build failed: {0}")]
    Preconditioner(#[source] schwarz_precond::BuildError),
    /// A reused preconditioner was built for other coefficients than this solver's.
    #[error("the prebuilt preconditioner was built for other coefficients: {0}")]
    PreconditionerMismatch(LayoutMismatch),
    /// A scaling tolerance outside `[0, 1)`.
    #[error("scaling tolerance must lie in [0, 1), got {value}")]
    InvalidScalingTolerance {
        /// The offending value.
        value: f64,
    },
    /// The reduced system needs the rows sorted by the eliminated term, as `Design::new` sorts them.
    #[error("the reduced system needs a design sorted by its dominant term")]
    UnsortedReducedSystem,
    /// The compact coefficient space exceeds the internal `u32` index width.
    #[error("design has {n_dofs} degrees of freedom, exceeding the u32 column-index limit")]
    DofSpaceExceedsU32 {
        /// Total degrees of freedom implied by the design.
        n_dofs: usize,
    },
}

/// A non-fatal solver-build event.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum BuildWarning {
    /// Residual deficits were clamped, degrading only preconditioner quality.
    UnscalableComponent {
        /// The offending channel pair.
        pair: ChannelPair,
        /// Largest relative dominance violation, clamped away.
        violation: f64,
    },
    /// The two-level rung built no map; the solve stays on one-level Schwarz.
    CoarseDeclined(CoarseDecline),
}

/// Why the two-level rung built no map.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum CoarseDecline {
    /// No well-conditioned spell glues levels into a component that needs a coarse column.
    NoCoarseSpace,
    /// The coarse factor would hold more values than the design has observations.
    OverBudget,
}

impl std::fmt::Display for BuildWarning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnscalableComponent { pair, violation } => write!(
                f,
                "signed component between {pair}: dominance scaling uncertified (max relative \
                 violation {violation:.2e}); deficits clamped, preconditioner quality may degrade"
            ),
            Self::CoarseDeclined(CoarseDecline::NoCoarseSpace) => write!(
                f,
                "two-level Schwarz declined: no coarse space; the solve stays one-level"
            ),
            Self::CoarseDeclined(CoarseDecline::OverBudget) => write!(
                f,
                "two-level Schwarz declined: its coarse factor exceeds the budget; the solve \
                 stays one-level"
            ),
        }
    }
}

/// How a reused preconditioner's coefficients differ from the solver's.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum LayoutMismatch {
    /// Built on the full system and reused on the reduced one, or the reverse.
    #[error("built on the other system (full vs reduced)")]
    System,
    /// Built on a design with another number of terms.
    #[error("built on {found} terms, the design has {expected}")]
    TermCount {
        /// The design's term count.
        expected: usize,
        /// The term count the preconditioner was built on.
        found: usize,
    },
    /// Term `term` has another number of levels.
    #[error("term {term} has {expected} levels, the preconditioner was built on {found}")]
    Levels {
        /// Index of the first differing term.
        term: usize,
        /// The design's level count for it.
        expected: usize,
        /// The level count the preconditioner was built on.
        found: usize,
    },
    /// Term `term` has another number of coefficient columns.
    #[error("term {term} has {expected} columns, the preconditioner was built on {found}")]
    Columns {
        /// Index of the first differing term.
        term: usize,
        /// The design's column count for it.
        expected: usize,
        /// The column count the preconditioner was built on.
        found: usize,
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
