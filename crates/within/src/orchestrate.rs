//! End-to-end solve orchestration for normal equations.
//!
//! This module keeps the public orchestration API stable while implementation
//! details live in focused submodules.

mod api;
mod batch;
mod common;
mod normal_equations;
mod result;

pub use api::{solve, solve_weighted};
pub use batch::{demean_batch, demean_batch_default, BatchDemeanResult};
pub use normal_equations::solve_normal_equations;
pub use result::SolveResult;
