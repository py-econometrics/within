//! Iterative solvers.
//!
//! Contains CG and GMRES solvers.

/// Conjugate gradient solver (unpreconditioned and left-preconditioned).
pub mod cg;
/// Right-preconditioned GMRES(m) with restarts.
pub mod gmres;
/// Shared BLAS-like primitives for iterative solvers.
pub(crate) mod util;

pub use util::vec_norm;
