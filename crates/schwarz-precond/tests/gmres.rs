mod common;

use schwarz_precond::solve::gmres::pgmres;
use schwarz_precond::{Operator, SolveError};

use common::{check_residual, IdentityOp, JacobiPrecond3, NonsymMatrix3, SpdMatrix3};

/// Diagonal matrix diag(2, 5, 8)
struct DiagMatrix3;

impl Operator for DiagMatrix3 {
    fn nrows(&self) -> usize {
        3
    }
    fn ncols(&self) -> usize {
        3
    }
    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        y[0] = 2.0 * x[0];
        y[1] = 5.0 * x[1];
        y[2] = 8.0 * x[2];
        Ok(())
    }
    fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        self.apply(x, y)
    }
}

#[test]
fn test_gmres_spd_system() {
    let b = vec![1.0, 2.0, 3.0];
    let id = IdentityOp { n: 3 };
    let result = pgmres(&SpdMatrix3, &b, &id, 1e-10, 100, 30).expect("gmres solve");
    assert!(result.converged, "GMRES did not converge");
    assert!(
        result.iterations <= 3,
        "took too many iterations: {}",
        result.iterations
    );
    check_residual(&SpdMatrix3, &result.x, &b, 1e-8);
}

#[test]
fn test_gmres_diagonal_system() {
    let b = vec![4.0, 10.0, 24.0];
    let id = IdentityOp { n: 3 };
    let result = pgmres(&DiagMatrix3, &b, &id, 1e-12, 100, 30).expect("gmres solve");
    assert!(
        result.converged,
        "GMRES did not converge on diagonal system"
    );
    assert!(result.iterations <= 3);
    assert!((result.x[0] - 2.0).abs() < 1e-10);
    assert!((result.x[1] - 2.0).abs() < 1e-10);
    assert!((result.x[2] - 3.0).abs() < 1e-10);
}

#[test]
fn test_gmres_nonsymmetric_system() {
    let b = vec![1.0, 2.0, 3.0];
    let id = IdentityOp { n: 3 };
    let result = pgmres(&NonsymMatrix3, &b, &id, 1e-10, 100, 30).expect("gmres solve");
    assert!(
        result.converged,
        "GMRES did not converge on nonsymmetric system"
    );
    assert!(result.iterations <= 3);
    check_residual(&NonsymMatrix3, &result.x, &b, 1e-8);
}

#[test]
fn test_gmres_preconditioned() {
    let b = vec![1.0, 2.0, 3.0];
    let result = pgmres(&SpdMatrix3, &b, &JacobiPrecond3, 1e-10, 100, 30).expect("gmres solve");
    assert!(result.converged, "Preconditioned GMRES did not converge");
    check_residual(&SpdMatrix3, &result.x, &b, 1e-8);
}

#[test]
fn test_gmres_zero_rhs() {
    let b = vec![0.0; 3];
    let id = IdentityOp { n: 3 };
    let result = pgmres(&SpdMatrix3, &b, &id, 1e-10, 100, 30).expect("gmres solve");
    assert!(result.converged);
    assert_eq!(result.iterations, 0);
    assert!(result.x.iter().all(|&v| v == 0.0));
    assert_eq!(result.residual_norm, 0.0);
}

#[test]
fn test_gmres_restart_small() {
    let b = vec![1.0, 2.0, 3.0];
    let id = IdentityOp { n: 3 };
    let result = pgmres(&SpdMatrix3, &b, &id, 1e-10, 200, 1).expect("gmres solve");
    assert!(result.converged, "GMRES(1) did not converge");
    check_residual(&SpdMatrix3, &result.x, &b, 1e-8);
}

#[test]
fn test_gmres_restart_medium_vs_full() {
    let b = vec![1.0, 2.0, 3.0];
    let id = IdentityOp { n: 3 };

    let r2 = pgmres(&SpdMatrix3, &b, &id, 1e-10, 200, 2).expect("gmres solve");
    let r50 = pgmres(&SpdMatrix3, &b, &id, 1e-10, 200, 50).expect("gmres solve");

    assert!(r2.converged, "GMRES(2) did not converge");
    assert!(r50.converged, "GMRES(50) did not converge");

    check_residual(&SpdMatrix3, &r2.x, &b, 1e-8);
    check_residual(&SpdMatrix3, &r50.x, &b, 1e-8);

    assert!(r50.iterations <= 3);
}

#[test]
fn test_gmres_tolerance() {
    let b = vec![1.0, 2.0, 3.0];
    let id = IdentityOp { n: 3 };

    let loose = pgmres(&SpdMatrix3, &b, &id, 1e-2, 100, 30).expect("gmres solve");
    assert!(loose.converged);

    let tight = pgmres(&SpdMatrix3, &b, &id, 1e-12, 100, 30).expect("gmres solve");
    assert!(tight.converged);

    assert!(tight.residual_norm <= loose.residual_norm + 1e-15);
}

#[test]
fn test_gmres_maxiter_reached() {
    let b = vec![1.0, 2.0, 3.0];
    let id = IdentityOp { n: 3 };
    let result = pgmres(&SpdMatrix3, &b, &id, 1e-12, 1, 30).expect("gmres solve");
    assert!(!result.converged || result.iterations <= 1);
    assert!(result.iterations <= 1);
}

#[test]
fn test_gmres_preconditioned_nonsymmetric() {
    struct NonsymJacobi;
    impl Operator for NonsymJacobi {
        fn nrows(&self) -> usize {
            3
        }
        fn ncols(&self) -> usize {
            3
        }
        fn apply(&self, r: &[f64], z: &mut [f64]) -> Result<(), SolveError> {
            z[0] = r[0] / 3.0;
            z[1] = r[1] / 4.0;
            z[2] = r[2] / 5.0;
            Ok(())
        }
        fn apply_adjoint(&self, r: &[f64], z: &mut [f64]) -> Result<(), SolveError> {
            self.apply(r, z)
        }
    }

    let b = vec![1.0, 2.0, 3.0];
    let result = pgmres(&NonsymMatrix3, &b, &NonsymJacobi, 1e-10, 100, 30).expect("gmres solve");
    assert!(
        result.converged,
        "Preconditioned GMRES on nonsymmetric did not converge"
    );
    check_residual(&NonsymMatrix3, &result.x, &b, 1e-8);
}
