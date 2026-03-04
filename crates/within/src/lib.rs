//! Fixed-effects normal equation solver with Schwarz preconditioning.
//!
//! `within` solves the linear fixed-effects problem **y = D x + e** where D is a
//! sparse categorical design matrix (each row has exactly Q ones, one per factor).
//! The normal equations **G x = D^T W y** (G = D^T W D) are solved via LSMR,
//! preconditioned CG, or right-preconditioned GMRES with additive or multiplicative
//! Schwarz preconditioners backed by approximate Cholesky local solvers.
//!
//! # Quick start
//!
//! ```no_run
//! use within::{solve, SolverParams};
//!
//! // Two factors: 100 levels each, 10 000 observations
//! let factor_0: Vec<u32> = (0..10_000).map(|i| (i % 100) as u32).collect();
//! let factor_1: Vec<u32> = (0..10_000).map(|i| (i / 100) as u32).collect();
//! let y: Vec<f64> = (0..10_000).map(|i| i as f64 * 0.01).collect();
//!
//! let result = solve(
//!     &[factor_0, factor_1],
//!     &[100, 100],
//!     &y,
//!     &SolverParams::default(),
//!     None,
//! )
//! .expect("solve should succeed");
//! assert!(result.converged);
//! ```
//!
//! # Architecture
//!
//! The crate is organized in four layers:
//!
//! - **`observation`** — Storage backends for per-observation data: [`FactorMajorStore`],
//!   [`RowMajorStore`], [`CompressedStore`]. All implement the [`ObservationStore`] trait.
//! - **`domain`** — Domain decomposition: [`WeightedDesign`] wraps a store with factor
//!   metadata; factor-pair subdomains are built with partition-of-unity weights.
//! - **`operator`** — Linear algebra primitives: [`Gramian`] (explicit CSR), [`GramianOperator`]
//!   (implicit D^T W D), [`DesignOperator`] (D and D^T), Schwarz preconditioner builders.
//! - **`orchestrate`** — End-to-end solve: [`solve`], [`solve_weighted`],
//!   [`solve_least_squares`], [`solve_normal_equations`] with typed configuration.
//!
//! # References
//!
//! - Correia, S. (2017). *Linear Models with High-Dimensional Fixed Effects: An Efficient and Feasible Estimator.*
//! - Gaure, S. (2013). *OLS with Multiple High Dimensional Category Variables.* Computational Statistics & Data Analysis.
//! - Spielman, D. & Teng, S.-H. (2014). *Nearly Linear Time Algorithms for Preconditioning and Solving Symmetric, Diagonally Dominant Linear Systems.* SIAM J. Matrix Anal. Appl.

pub mod config;
pub mod domain;
pub mod error;
pub mod observation;
pub mod operator;
pub mod orchestrate;
// ---------------------------------------------------------------------------
// High-level API
// ---------------------------------------------------------------------------

pub use orchestrate::{solve, solve_least_squares, solve_normal_equations, solve_weighted};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

pub use config::{
    ApproxSchurConfig, CgPreconditioner, GmresPreconditioner, LocalSolverConfig, SchwarzConfig,
    SolverMethod, SolverParams, DEFAULT_DENSE_SCHUR_THRESHOLD,
};
pub use error::{WithinError, WithinResult};
pub use orchestrate::SolveResult;

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

pub use domain::{FixedEffectsDesign, Subdomain, WeightedDesign};
pub use observation::{
    CompressedStore, FactorMajorStore, FactorMeta, ObservationStore, ObservationWeights,
    RowMajorStore,
};

// ---------------------------------------------------------------------------
// Operators & builders
// ---------------------------------------------------------------------------

pub use operator::design::{DesignOperator, PreconditionedDesign};
pub use operator::gramian::{Gramian, GramianOperator};
pub use operator::schwarz::{build_schwarz_default, FeSchwarz};
