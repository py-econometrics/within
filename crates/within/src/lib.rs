#![deny(missing_docs)]
//! Fixed-effects normal-equation solver. Solves `G x = D^T W y` (with
//! `G = D^T W D`) for a sparse categorical design `D` via modified LSMR,
//! preconditioned by default with a diagonal that escalates to additive
//! Schwarz over factor-pair subdomains on a stalled contraction.
//!
//! ```
//! use ndarray::Array2;
//! use within::{solve, LsmrOptions};
//!
//! let categories = Array2::<u32>::zeros((10_000, 2));
//! let y = vec![0.0; 10_000];
//! let r = solve(categories.view(), &y, None, &LsmrOptions::default(), None).unwrap();
//! assert!(r.converged);
//! ```
//!
//! # Reproducibility
//!
//! A solve returns the same bits at any Rayon width, run after run: every parallel
//! sum is split where the data says, every parallel build is put back in a fixed order,
//! and the width only decides how many pieces run at once.
//!
//! [`PreconditionerConfig::Adaptive`] reproduces raw coefficients per solve, not
//! across solves; fitted values agree.

pub mod config;
pub mod error;

pub(crate) mod design;
pub(crate) mod linalg;
pub(crate) mod precond;
pub(crate) mod solver;
pub(crate) mod system;

pub use config::{
    ApproxCholConfig, ApproxSchurConfig, LocalSolverConfig, LsmrOptions, PreconditionerConfig,
    ScalingConfig, ScalingFailure, SchurMode, Schwarz, Staleness, StalenessError,
};
pub use design::{Channel, ChannelPair, CoefficientAddress, CoefficientLayout, Design, Effect};
pub use error::{BuildError, BuildWarning, SolveError, WithinError};
pub use precond::Preconditioner;
pub use solver::{
    solve, solve_batch, BatchSolveResult, IntoDesign, PreconditionerInput, SolveResult, Solver,
};

/// Rows from which a sweep over observations runs in parallel.
pub(crate) const PAR_THRESHOLD: usize = 10_000;
