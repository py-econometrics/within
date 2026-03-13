//! Iterative Krylov solvers for `A x = b`.
//!
//! Both solvers accept an optional preconditioner `M` (any [`Operator`](crate::Operator)):
//!
//! - **`cg`** — Preconditioned conjugate gradient. Requires `A` symmetric
//!   positive (semi-)definite and `M` symmetric. Optimal for the Schwarz
//!   additive preconditioner.
//! - **`gmres`** — Right-preconditioned GMRES(m) with restarts. Works with
//!   any non-singular `A` and any `M`, including non-symmetric
//!   (multiplicative Schwarz). Uses Modified Gram-Schmidt and Givens
//!   rotations.

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
