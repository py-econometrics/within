//! Conjugate gradient solver, generic over Operator.
//!
//! Public entry point: `pcg` (preconditioned or unpreconditioned via Option dispatch).

use super::{dot, vec_norm};
use crate::{Operator, SolveError};

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

#[inline]
fn apply_preconditioner<M: Operator + ?Sized>(
    preconditioner: Option<&M>,
    r: &[f64],
    z: &mut [f64],
) -> Result<(), SolveError> {
    match preconditioner {
        Some(m) => m.apply(r, z)?,
        None => z.copy_from_slice(r),
    }
    Ok(())
}

/// Unpreconditioned conjugate gradient.
///
/// Solves `A x = b` with no preconditioner. `tol` is the relative tolerance
/// `||r|| / ||b|| < tol`.
pub fn cg<A: Operator + ?Sized>(
    operator: &A,
    b: &[f64],
    tol: f64,
    maxiter: usize,
) -> Result<CgResult, SolveError> {
    pcg_impl::<A, A>(operator, b, None, tol, maxiter)
}

/// Left-preconditioned conjugate gradient.
///
/// Solves `A x = b` with preconditioner `M` applied to the residual each
/// iteration. `tol` is the relative tolerance `||r|| / ||b|| < tol`.
pub fn pcg<A: Operator + ?Sized, M: Operator + ?Sized>(
    operator: &A,
    b: &[f64],
    preconditioner: &M,
    tol: f64,
    maxiter: usize,
) -> Result<CgResult, SolveError> {
    pcg_impl(operator, b, Some(preconditioner), tol, maxiter)
}

fn pcg_impl<A: Operator + ?Sized, M: Operator + ?Sized>(
    operator: &A,
    b: &[f64],
    preconditioner: Option<&M>,
    tol: f64,
    maxiter: usize,
) -> Result<CgResult, SolveError> {
    let n = operator.ncols();
    debug_assert_eq!(b.len(), n);
    let b_norm = vec_norm(b);
    if b_norm < f64::EPSILON {
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

    apply_preconditioner(preconditioner, &r, &mut z)?;
    let mut p = z.clone();
    let mut rz = dot(&r, &z);
    let rz_init = rz;
    let mut r_norm = b_norm;

    for itn in 1..=maxiter {
        operator.apply(&p, &mut ap)?;
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

        let mut r_norm_sq = 0.0;
        for ((xi, &pi), (ri, &api)) in x.iter_mut().zip(&p).zip(r.iter_mut().zip(&ap)) {
            *xi += alpha * pi;
            *ri -= alpha * api;
            r_norm_sq += *ri * *ri;
        }
        r_norm = r_norm_sq.sqrt();
        if r_norm / b_norm <= tol {
            return Ok(CgResult {
                x,
                converged: true,
                iterations: itn,
                residual_norm: r_norm,
            });
        }

        apply_preconditioner(preconditioner, &r, &mut z)?;
        let rz_new = dot(&r, &z);
        // Guard threshold is EPS^2 (not EPS) so that rz must reach true
        // numerical noise before we bail. At EPS*rz_init the recursive
        // residual is roughly sqrt(EPS)*||b|| ~ 1.5e-8·||b||, which collides
        // with user tolerances near 1e-8 and causes spurious non-convergence.
        let stagnation_threshold =
            f64::EPSILON * f64::EPSILON * rz_init.abs().max(f64::MIN_POSITIVE);
        if rz_new.abs() < stagnation_threshold {
            return Ok(CgResult {
                x,
                converged: r_norm / b_norm <= tol,
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
        fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), crate::SolveError> {
            y[0] = 4.0 * x[0] + 1.0 * x[1];
            y[1] = 1.0 * x[0] + 3.0 * x[1] + 1.0 * x[2];
            y[2] = 1.0 * x[1] + 2.0 * x[2];
            Ok(())
        }
        fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) -> Result<(), crate::SolveError> {
            self.apply(x, y)
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
        fn apply(&self, r: &[f64], z: &mut [f64]) -> Result<(), crate::SolveError> {
            z[0] = r[0] / 4.0;
            z[1] = r[1] / 3.0;
            z[2] = r[2] / 2.0;
            Ok(())
        }
        fn apply_adjoint(&self, r: &[f64], z: &mut [f64]) -> Result<(), crate::SolveError> {
            self.apply(r, z)
        }
    }

    #[test]
    fn test_cg_unpreconditioned() {
        let b = vec![1.0, 2.0, 3.0];
        let result = cg(&SpdMatrix3, &b, 1e-10, 100).expect("cg solve");
        assert!(result.converged, "CG did not converge");
        assert!(result.iterations <= 3, "CG took too many iterations");

        let mut ax = vec![0.0; 3];
        SpdMatrix3
            .apply(&result.x, &mut ax)
            .expect("operator apply");
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
        let result =
            pcg(&SpdMatrix3, &b, &JacobiPrecond3, 1e-10, 100).expect("preconditioned cg solve");
        assert!(result.converged, "Preconditioned CG did not converge");

        let mut ax = vec![0.0; 3];
        SpdMatrix3
            .apply(&result.x, &mut ax)
            .expect("operator apply");
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
        let result = cg(&SpdMatrix3, &b, 1e-10, 100).expect("cg solve");
        assert!(result.converged);
        assert_eq!(result.iterations, 0);
        assert!(result.x.iter().all(|&v| v == 0.0));
        assert_eq!(result.residual_norm, 0.0);
    }

    struct NegDiagOperator;

    impl Operator for NegDiagOperator {
        fn nrows(&self) -> usize {
            3
        }
        fn ncols(&self) -> usize {
            3
        }
        fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), crate::SolveError> {
            y[0] = -x[0];
            y[1] = -2.0 * x[1];
            y[2] = -3.0 * x[2];
            Ok(())
        }
        fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) -> Result<(), crate::SolveError> {
            self.apply(x, y)
        }
    }

    #[test]
    fn test_cg_indefinite_operator_returns_not_converged() {
        // NegDiagOperator has all negative eigenvalues, so pap <= 0 on the
        // first iteration, triggering the early-exit branch.
        let b = vec![1.0, 1.0, 1.0];
        let result = cg(&NegDiagOperator, &b, 1e-10, 100).expect("cg solve");
        assert!(
            !result.converged,
            "expected non-convergence for indefinite operator"
        );
        assert_eq!(
            result.iterations, 1,
            "expected early exit on first iteration"
        );
    }

    #[test]
    fn test_cg_maxiter_exhaustion_returns_not_converged() {
        // A single iteration cannot achieve 1e-15 relative tolerance on a
        // non-trivial system, so maxiter=1 forces early termination.
        let b = vec![1.0, 2.0, 3.0];
        let result = cg(&SpdMatrix3, &b, 1e-15, 1).expect("cg solve");
        assert!(
            !result.converged,
            "expected non-convergence when maxiter exhausted"
        );
        assert_eq!(result.iterations, 1);
        assert!(
            result.residual_norm > 0.0,
            "residual should be nonzero after 1 iteration"
        );
    }
}
