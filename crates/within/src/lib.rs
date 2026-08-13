#![deny(missing_docs)]
//! Fixed-effects normal-equation solver. Solves `G x = D^T W y` (with
//! `G = D^T W D`) for a sparse categorical design `D` via modified LSMR with a
//! Schwarz preconditioner over factor-pair subdomains.
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
//! A single-threaded run (a one-thread Rayon pool, or `RAYON_NUM_THREADS=1`)
//! is bitwise-reproducible. Parallel reductions sum in an order that depends on
//! the Rayon width, so coefficients from different thread counts differ at the
//! ULP scale — reproducible within solver tolerance, not bitwise. Pinning the
//! Rayon width across runs holds estimates stable within solver tolerance (and
//! is bitwise in practice at a fixed width, though only the single-threaded
//! case is a guarantee); when the width may vary, also pin an explicit
//! [`ReductionStrategy`] rather than [`ReductionStrategy::Auto`], which selects
//! its backend from the width.

pub mod config;
pub mod error;
pub mod observation;

pub(crate) mod block_elim;
pub(crate) mod channel;
pub(crate) mod csr_block;
pub(crate) mod domain;
pub(crate) mod operator;
pub(crate) mod solver;

pub use channel::{Channel, ChannelPair};
pub use config::{
    ApproxCholConfig, ApproxSchurConfig, LocalSolverConfig, LsmrOptions, PreconditionerConfig,
    ReductionStrategy, ScalingConfig, ScalingFailure, SchurMode,
};
pub use domain::{Design, Effect};
pub use error::{BuildError, BuildWarning, SolveError, WithinError};
pub use operator::schwarz::Preconditioner;
pub use solver::{
    solve, solve_batch, BatchSolveResult, CoefficientAddress, CoefficientLayout, IntoDesign,
    PreconditionerInput, SolveResult, Solver,
};
