//! LSMR solver (Fong & Saunders, 2011), generic over Operator.
//!
//! Single implementation that handles both square and rectangular operators.
//! The `Operator` trait carries the matvec/adjoint logic, so specializations
//! like `PreconditionedDesign` get their fused performance automatically.

use crate::{Operator, SolveError};

// Re-export for backward compatibility (other modules import from here).
pub use super::util::vec_norm;

#[inline]
fn vec_scale(v: &mut [f64], s: f64) {
    for x in v.iter_mut() {
        *x *= s;
    }
}

/// Stable symmetric Givens rotation (Fong & Saunders 2011, S2.3).
///
/// Returns (c, s, r) such that:
///   [ c  s ] [ a ] = [ r ]
///   [ s -c ] [ b ]   [ 0 ]
#[inline]
fn sym_ortho(a: f64, b: f64) -> (f64, f64, f64) {
    if b == 0.0 {
        return (a.signum(), 0.0, a.abs());
    }
    if a == 0.0 {
        return (0.0, b.signum(), b.abs());
    }
    let (abs_a, abs_b) = (a.abs(), b.abs());
    if abs_b >= abs_a {
        let tau = a / b;
        let s = (1.0 + tau * tau).sqrt().recip() * b.signum();
        let c = s * tau;
        let r = b / s;
        (c, s, r)
    } else {
        let tau = b / a;
        let c = (1.0 + tau * tau).sqrt().recip() * a.signum();
        let s = c * tau;
        let r = a / c;
        (c, s, r)
    }
}

// ---------------------------------------------------------------------------
// LSMR result
// ---------------------------------------------------------------------------

/// Result of an LSMR solve.
#[must_use]
pub struct LsmrResult {
    /// Solution vector.
    pub x: Vec<f64>,
    /// Stopping condition code (0 = zero RHS, 1-2 = converged, 3-6 = limits, 7 = maxiter).
    pub istop: i32,
    /// Number of iterations performed.
    pub itn: usize,
}

// ---------------------------------------------------------------------------
// LSMR core (Fong & Saunders 2011, damp=0)
// ---------------------------------------------------------------------------

/// LSMR iterative solver: minimize ||A x - b||_2 (damp = 0).
///
/// Implements Algorithm 3.1 from Fong & Saunders, "LSMR: An Iterative Algorithm
/// for Sparse Least-Squares Problems", SIAM J. Sci. Comput. 33(5), 2011.
///
/// Works with any `Operator` (rectangular or square, symmetric or not).
/// `apply` computes A*x and `apply_adjoint` computes A^T*x.
pub fn lsmr_solve<A: Operator>(
    operator: &A,
    b: &[f64],
    atol: f64,
    btol: f64,
    conlim: f64,
    maxiter: usize,
) -> Result<LsmrResult, SolveError> {
    let m = operator.nrows();
    let n = operator.ncols();
    let ctol = if conlim > 0.0 { 1.0 / conlim } else { 0.0 };

    let mut u = vec![0.0f64; m];
    let mut v = vec![0.0f64; n];
    let mut h = vec![0.0f64; n];
    let mut hbar = vec![0.0f64; n];
    let mut scratch_m = vec![0.0f64; m];
    let mut scratch_n = vec![0.0f64; n];

    u.copy_from_slice(b);
    let mut beta = vec_norm(&u);

    if beta == 0.0 {
        return Ok(LsmrResult {
            x: vec![0.0; n],
            istop: 0,
            itn: 0,
        });
    }

    vec_scale(&mut u, 1.0 / beta);

    operator.try_apply_adjoint(&u, &mut v)?;
    let mut alpha = vec_norm(&v);

    if alpha == 0.0 {
        return Ok(LsmrResult {
            x: vec![0.0; n],
            istop: 0,
            itn: 0,
        });
    }

    vec_scale(&mut v, 1.0 / alpha);

    let normb = beta;
    h.copy_from_slice(&v);
    hbar.fill(0.0);

    let mut x = vec![0.0f64; n];
    let mut alphabar = alpha;
    let mut zetabar = alpha * beta;
    let mut rho = 1.0;
    let mut rhobar = 1.0;
    let mut cbar = 1.0;
    let mut sbar = 0.0;

    let mut betadd = beta;
    let mut betad = 0.0;
    let mut rhodold = 1.0;
    let mut tautildeold = 0.0;
    let mut thetatilde = 0.0;
    let mut zeta = 0.0;
    let d = 0.0;

    let mut norm_a2 = alpha * alpha;
    let mut maxrbar = 0.0f64;
    let mut minrbar = 1e100f64;

    let mut normr;
    let mut normar;
    let mut normx;
    let mut istop = 0i32;
    let mut itn = 0usize;

    for _iter in 0..maxiter {
        itn += 1;

        operator.try_apply(&v, &mut scratch_m)?;
        for (ui, &si) in u.iter_mut().zip(&scratch_m) {
            *ui = si - alpha * *ui;
        }
        beta = vec_norm(&u);

        if beta > 0.0 {
            vec_scale(&mut u, 1.0 / beta);
            operator.try_apply_adjoint(&u, &mut scratch_n)?;
            for (vi, &si) in v.iter_mut().zip(&scratch_n) {
                *vi = si - beta * *vi;
            }
            alpha = vec_norm(&v);
            if alpha > 0.0 {
                vec_scale(&mut v, 1.0 / alpha);
            }
        }

        let alphahat = alphabar;

        let rhoold = rho;
        let c;
        let s;
        (c, s, rho) = sym_ortho(alphahat, beta);
        let thetanew = s * alpha;
        alphabar = c * alpha;

        let rhobarold = rhobar;
        let zetaold = zeta;
        let thetabar = sbar * rho;
        let rhotemp = cbar * rho;
        (cbar, sbar, rhobar) = sym_ortho(rhotemp, thetanew);
        zeta = cbar * zetabar;
        zetabar *= -sbar;

        let hbar_scale = -(thetabar * rho) / (rhoold * rhobarold);
        let x_scale = zeta / (rho * rhobar);
        let theta_rho = thetanew / rho;
        for (((hbi, &hi), xi), &_vi) in hbar.iter_mut().zip(&h).zip(x.iter_mut()).zip(&v) {
            *hbi = hi + hbar_scale * *hbi;
            *xi += x_scale * *hbi;
            // h[i] updated in separate pass (borrows conflict)
        }
        for (hi, &vi) in h.iter_mut().zip(&v) {
            *hi = vi - theta_rho * *hi;
        }

        let betaacute = betadd;
        let betahat = c * betaacute;
        betadd = -s * betaacute;

        let thetatildeold = thetatilde;
        let ctildeold;
        let stildeold;
        let rhotildeold;
        (ctildeold, stildeold, rhotildeold) = sym_ortho(rhodold, thetabar);
        thetatilde = stildeold * rhobar;
        rhodold = ctildeold * rhobar;
        betad = -stildeold * betad + ctildeold * betahat;

        tautildeold = (zetaold - thetatildeold * tautildeold) / rhotildeold;
        let taud = (zeta - thetatilde * tautildeold) / rhodold;
        normr = (d + (betad - taud).powi(2) + betadd.powi(2)).sqrt();

        normar = zetabar.abs();

        norm_a2 += beta * beta;
        let norma = norm_a2.sqrt();
        norm_a2 += alpha * alpha;

        maxrbar = maxrbar.max(rhobarold);
        if itn > 1 {
            minrbar = minrbar.min(rhobarold);
        }
        let conda = maxrbar.max(rhotemp) / minrbar.min(rhotemp);

        normx = vec_norm(&x);

        let test1 = normr / normb;
        let test2 = if norma * normr != 0.0 {
            normar / (norma * normr)
        } else {
            f64::INFINITY
        };
        let test3 = 1.0 / conda;
        let t1 = test1 / (1.0 + norma * normx / normb);
        let rtol = btol + atol * norma * normx / normb;

        if itn >= maxiter {
            istop = 7;
        }
        if 1.0 + test3 <= 1.0 {
            istop = 6;
        }
        if 1.0 + test2 <= 1.0 {
            istop = 4;
        }
        if 1.0 + t1 <= 1.0 {
            istop = 4;
        }
        if test3 <= ctol {
            istop = 3;
        }
        if test2 <= atol {
            istop = 2;
        }
        if test1 <= rtol {
            istop = 1;
        }

        if istop != 0 {
            break;
        }
    }

    if istop == 0 && itn >= maxiter {
        istop = 7;
    }

    Ok(LsmrResult { x, istop, itn })
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

    struct RectMatrix32;

    impl Operator for RectMatrix32 {
        fn nrows(&self) -> usize {
            3
        }
        fn ncols(&self) -> usize {
            2
        }
        fn apply(&self, x: &[f64], y: &mut [f64]) {
            y[0] = x[0];
            y[1] = x[1];
            y[2] = x[0] + x[1];
        }
        fn apply_adjoint(&self, y: &[f64], x: &mut [f64]) {
            x[0] = y[0] + y[2];
            x[1] = y[1] + y[2];
        }
    }

    #[test]
    fn test_lsmr_square_system() {
        let b = vec![1.0, 2.0, 3.0];
        let result = lsmr_solve(&SpdMatrix3, &b, 1e-10, 1e-10, 1e8, 100).expect("lsmr solve");
        assert!(
            result.istop == 1 || result.istop == 2 || result.istop == 4,
            "LSMR did not converge: istop={}",
            result.istop
        );

        let mut ax = vec![0.0; 3];
        SpdMatrix3.apply(&result.x, &mut ax);
        let err: f64 = ax
            .iter()
            .zip(b.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(err < 1e-6, "LSMR residual too large: {err}");
    }

    #[test]
    fn test_lsmr_rectangular() {
        let b = vec![1.0, 2.0, 3.0];
        let result = lsmr_solve(&RectMatrix32, &b, 1e-10, 1e-10, 1e8, 100).expect("lsmr solve");
        assert!(
            result.istop == 1 || result.istop == 2 || result.istop == 4,
            "LSMR did not converge: istop={}",
            result.istop
        );

        assert!((result.x[0] - 1.0).abs() < 1e-6, "x[0] = {}", result.x[0]);
        assert!((result.x[1] - 2.0).abs() < 1e-6, "x[1] = {}", result.x[1]);
    }

    #[test]
    fn test_lsmr_zero_rhs() {
        let b = vec![0.0; 3];
        let result = lsmr_solve(&SpdMatrix3, &b, 1e-10, 1e-10, 1e8, 100).expect("lsmr solve");
        assert_eq!(result.istop, 0);
        assert_eq!(result.itn, 0);
    }
}
