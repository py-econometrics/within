//! Iterative solvers and spectral tools.
//!
//! Contains CG and GMRES solvers alongside spectral extraction utilities
//! (deflated CG and tridiagonal eigensolvers).

/// Conjugate gradient solver (unpreconditioned and left-preconditioned).
pub mod cg;
/// Two-phase deflated CG with Ritz pair extraction.
pub mod deflated_cg;
/// Right-preconditioned GMRES(m) with restarts.
pub mod gmres;
/// Small dense symmetric eigenvalue solver (Jacobi rotations).
pub(crate) mod tridiagonal_eigen;
/// Shared BLAS-like primitives for iterative solvers.
pub(crate) mod util;

pub use util::vec_norm;
