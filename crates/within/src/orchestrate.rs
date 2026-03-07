//! End-to-end solve orchestration for normal equations.
//!
//! This module keeps the public orchestration API stable while implementation
//! details live in focused submodules.

mod api;
mod batch;
mod common;
mod normal_equations;
mod result;

pub use api::solve;
pub use batch::{demean_batch, BatchDemeanResult};
pub use normal_equations::solve_normal_equations;
pub use result::SolveResult;
