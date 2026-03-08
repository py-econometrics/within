//! Fixed-effects normal equation solver with Schwarz preconditioning.
//!
//! `within` solves the linear fixed-effects problem **y = D x + e** where D is a
//! sparse categorical design matrix (each row has exactly Q ones, one per factor).
//! The normal equations **G x = D^T W y** (G = D^T W D) are solved via
//! preconditioned CG or right-preconditioned GMRES with additive or
//! multiplicative Schwarz preconditioners backed by approximate Cholesky
//! local solvers.
//!
//! # Quick start
//!
//! ```no_run
//! use ndarray::Array2;
//! use within::{solve, SolverParams};
//!
//! // Two factors: 100 levels each, 10 000 observations
//! let n_obs = 10_000usize;
//! let mut categories = Array2::<u32>::zeros((n_obs, 2));
//! for i in 0..n_obs {
//!     categories[[i, 0]] = (i % 100) as u32;
//!     categories[[i, 1]] = (i / 100) as u32;
//! }
//! let y: Vec<f64> = (0..n_obs).map(|i| i as f64 * 0.01).collect();
//!
//! let result = solve(categories.view(), &y, None, &SolverParams::default(), None)
//!     .expect("solve should succeed");
//! assert!(result.converged);
//! ```
//!
//! # Architecture
//!
//! The crate is organized in four layers:
//!
//! - **`observation`** — Per-observation data storage via [`FactorMajorStore`]
//!   and the [`ObservationStore`] trait.
//! - **`domain`** — Domain decomposition: [`WeightedDesign`] wraps a store with factor
//!   metadata; factor-pair subdomains are built with partition-of-unity weights.
//! - **`operator`** — Linear algebra primitives: [`Gramian`] (explicit CSR), [`GramianOperator`]
//!   (implicit D^T W D), [`DesignOperator`] (D and D^T), Schwarz preconditioner builders.
//! - **`orchestrate`** — End-to-end solve: [`solve`] with typed configuration.
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
pub mod solver;
// ---------------------------------------------------------------------------
// High-level API
// ---------------------------------------------------------------------------

pub use operator::preconditioner::FePreconditioner;
pub use orchestrate::solve;
pub use orchestrate::solve_batch;
pub use orchestrate::BatchSolveResult;
pub use solver::Solver;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

pub use config::{
    ApproxSchurConfig, KrylovMethod, LocalSolverConfig, OperatorRepr, Preconditioner, SolverParams,
    DEFAULT_DENSE_SCHUR_THRESHOLD,
};
pub use error::{WithinError, WithinResult};
pub use orchestrate::SolveResult;

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

pub use domain::{FixedEffectsDesign, Subdomain, WeightedDesign};
pub use observation::{
    ArrayStore, FactorMajorStore, FactorMeta, ObservationStore, ObservationWeights,
};

// ---------------------------------------------------------------------------
// Operators & builders
// ---------------------------------------------------------------------------

pub use operator::gramian::{Gramian, GramianOperator};
pub use operator::schwarz::{build_schwarz, FeSchwarz};
pub use operator::DesignOperator;
pub use schwarz_precond::Operator;
