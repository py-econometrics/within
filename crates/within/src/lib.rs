#![deny(missing_docs)]
//! Fixed-effects normal-equation solver. Solves `G x = D^T W y` (with
//! `G = D^T W D`) for a sparse categorical design `D` via modified LSMR with a
//! Schwarz preconditioner over factor-pair subdomains.
//!
//! ```no_run
//! use ndarray::Array2;
//! use within::{solve, LsmrOptions};
//!
//! let categories = Array2::<u32>::zeros((10_000, 2));
//! let y = vec![0.0; 10_000];
//! let r = solve(categories.view(), &y, None, &LsmrOptions::default(), None).unwrap();
//! assert!(r.converged);
//! ```

pub mod config;
pub mod error;
pub mod observation;

pub(crate) mod block_elim;
pub(crate) mod csr_block;
pub(crate) mod domain;
pub(crate) mod operator;
pub(crate) mod solver;

pub use config::{LsmrOptions, PreconditionerConfig};
pub use domain::Design;
pub use error::{BuildError, SolveError, WithinError};
pub use operator::schwarz::Preconditioner;
pub use solver::{solve, solve_batch, BatchSolveResult, SolveResult, Solver};
