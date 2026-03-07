//! End-to-end solve orchestration for normal equations and least squares.
//!
//! This module keeps the public orchestration API stable while implementation
//! details live in focused submodules.

mod api;
mod batch;
mod common;
mod least_squares;
mod normal_equations;
mod result;

pub use api::{solve, solve_weighted};
pub use batch::{demean_batch, solve_batch, BatchDemeanResult, BatchSolveResult};
pub use least_squares::solve_least_squares;
pub use normal_equations::solve_normal_equations;
pub use result::SolveResult;
