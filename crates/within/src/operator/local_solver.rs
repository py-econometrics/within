//! Approximate Cholesky-backed local solver for Schwarz subdomains.

mod approx_chol_solver;
mod block_elim_solver;
mod fe_local_solver;
mod strategy;
mod transforms;

pub(crate) use approx_chol_solver::ApproxCholSolver;
pub(crate) use block_elim_solver::{BlockElimSolver, ReducedFactor};
pub(crate) use fe_local_solver::FeLocalSolver;
pub(crate) use strategy::LocalSolveStrategy;
