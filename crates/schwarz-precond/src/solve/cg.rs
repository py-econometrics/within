//! Conjugate gradient solver, generic over Operator.
//!
//! Two entry points avoid `Option` type inference issues:
//! - `cg_solve`: unpreconditioned CG (delegates to preconditioned with IdentityOperator)
//! - `cg_solve_preconditioned`: left-preconditioned CG

use super::vec_norm;
use crate::{IdentityOperator, Operator, SolveError};

/// Inner product of two vectors.
#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Result of a conjugate gradient solve.
#[must_use]
pub struct CgResult {
    /// Solution vector.
    pub x: Vec<f64>,
    /// Whether CG converged within the tolerance.
    pub converged: bool,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Final residual norm `||r||`.
    pub residual_norm: f64,
}

/// Unpreconditioned conjugate gradient: solve A x = b.
///
/// `tol`: relative tolerance (||r|| / ||b|| < tol -> converged)
pub fn cg_solve<A: Operator + ?Sized>(
    operator: &A,
    b: &[f64],
    tol: f64,
    maxiter: usize,
) -> Result<CgResult, SolveError> {
    cg_solve_preconditioned(
        operator,
        &IdentityOperator::new(operator.ncols()),
        b,
        tol,
        maxiter,
    )
}

/// Left-preconditioned conjugate gradient: solve A x = b with preconditioner M.
///
/// Applies M^{-1} (via `preconditioner.apply()`) to the residual each iteration.
pub fn cg_solve_preconditioned<A: Operator + ?Sized, M: Operator + ?Sized>(
    operator: &A,
    preconditioner: &M,
    b: &[f64],
    tol: f64,
    maxiter: usize,
) -> Result<CgResult, SolveError> {
    let n = operator.ncols();
    debug_assert_eq!(b.len(), n);
    let b_norm = vec_norm(b);
    if b_norm == 0.0 {
        return Ok(CgResult {
            x: vec![0.0; n],
            converged: true,
            iterations: 0,
            residual_norm: 0.0,
        });
    }

    let mut x = vec![0.0; n];
    let mut r = b.to_vec();
    let mut z = vec![0.0; n];
    let mut ap = vec![0.0; n];

    preconditioner.try_apply(&r, &mut z)?;
    let mut p = z.clone();
    let mut rz = dot(&r, &z);
    let mut r_norm = b_norm;

    for itn in 1..=maxiter {
        operator.try_apply(&p, &mut ap)?;
        let pap = dot(&p, &ap);
        if pap <= 0.0 {
            return Ok(CgResult {
                x,
                converged: false,
                iterations: itn,
                residual_norm: r_norm,
            });
        }
        let alpha = rz / pap;

        for ((xi, &pi), (ri, &api)) in x.iter_mut().zip(&p).zip(r.iter_mut().zip(&ap)) {
            *xi += alpha * pi;
            *ri -= alpha * api;
        }

        r_norm = vec_norm(&r);
        if r_norm / b_norm <= tol {
            return Ok(CgResult {
                x,
                converged: true,
                iterations: itn,
                residual_norm: r_norm,
            });
        }

        preconditioner.try_apply(&r, &mut z)?;
        let rz_new = dot(&r, &z);
        if rz_new.abs() < 1e-300 {
            return Ok(CgResult {
                x,
                converged: false,
                iterations: itn,
                residual_norm: r_norm,
            });
        }
        let beta = rz_new / rz;
        for (pi, &zi) in p.iter_mut().zip(&z) {
            *pi = zi + beta * *pi;
        }
        rz = rz_new;
    }

    Ok(CgResult {
        x,
        converged: false,
        iterations: maxiter,
        residual_norm: r_norm,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    struct SpdMatrix3;

    impl Operator for SpdMatrix3 {
        fn nrows(&self) -> usize {
            3
        }
        fn ncols(&self) -> usize {
            3
        }
        fn apply(&self, x: &[f64], y: &mut [f64]) {
            y[0] = 4.0 * x[0] + 1.0 * x[1];
            y[1] = 1.0 * x[0] + 3.0 * x[1] + 1.0 * x[2];
            y[2] = 1.0 * x[1] + 2.0 * x[2];
        }
        fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) {
            self.apply(x, y);
        }
    }

    struct JacobiPrecond3;

    impl Operator for JacobiPrecond3 {
        fn nrows(&self) -> usize {
            3
        }
        fn ncols(&self) -> usize {
            3
        }
        fn apply(&self, r: &[f64], z: &mut [f64]) {
            z[0] = r[0] / 4.0;
            z[1] = r[1] / 3.0;
            z[2] = r[2] / 2.0;
        }
        fn apply_adjoint(&self, r: &[f64], z: &mut [f64]) {
            self.apply(r, z);
        }
    }

    #[test]
    fn test_cg_unpreconditioned() {
        let b = vec![1.0, 2.0, 3.0];
        let result = cg_solve(&SpdMatrix3, &b, 1e-10, 100).expect("cg solve");
        assert!(result.converged, "CG did not converge");
        assert!(result.iterations <= 3, "CG took too many iterations");

        let mut ax = vec![0.0; 3];
        SpdMatrix3.apply(&result.x, &mut ax);
        let err: f64 = ax
            .iter()
            .zip(b.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(err < 1e-8, "CG residual too large: {err}");
    }

    #[test]
    fn test_cg_preconditioned() {
        let b = vec![1.0, 2.0, 3.0];
        let result = cg_solve_preconditioned(&SpdMatrix3, &JacobiPrecond3, &b, 1e-10, 100)
            .expect("preconditioned cg solve");
        assert!(result.converged, "Preconditioned CG did not converge");

        let mut ax = vec![0.0; 3];
        SpdMatrix3.apply(&result.x, &mut ax);
        let err: f64 = ax
            .iter()
            .zip(b.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(err < 1e-8, "Preconditioned CG residual too large: {err}");
    }

    #[test]
    fn test_cg_zero_rhs() {
        let b = vec![0.0; 3];
        let result = cg_solve(&SpdMatrix3, &b, 1e-10, 100).expect("cg solve");
        assert!(result.converged);
        assert_eq!(result.iterations, 0);
        assert!(result.x.iter().all(|&v| v == 0.0));
        assert_eq!(result.residual_norm, 0.0);
    }
}
