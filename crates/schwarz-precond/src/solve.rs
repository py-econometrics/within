//! Iterative solvers.
//!
//! Contains CG and GMRES solvers.

/// Conjugate gradient solver (unpreconditioned and left-preconditioned).
pub mod cg;
/// Right-preconditioned GMRES(m) with restarts.
pub mod gmres;

/// Inner product of two vectors.
#[inline]
pub(crate) fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

/// Euclidean norm of a vector.
#[inline]
pub fn vec_norm(v: &[f64]) -> f64 {
    let mut s = 0.0f64;
    for &x in v {
        s += x * x;
    }
    s.sqrt()
}
