mod common;

use schwarz_precond::solve::cg::pcg;
use schwarz_precond::solve::gmres::pgmres;
use schwarz_precond::Operator;
use schwarz_precond::{IdentityOperator, SolveError};

use common::{check_residual, DiagOperator, IdentityOp, SpdMatrix3, TridiagOperator};

// ---------------------------------------------------------------------------
// Local test operators
// ---------------------------------------------------------------------------

/// Zero operator: A x = 0 for all x.
///
/// With a non-zero rhs, p^T A p = 0 on the first iteration, which exercises
/// the `pap <= 0` early-exit branch in CG.
struct ZeroOperator {
    n: usize,
}

impl Operator for ZeroOperator {
    fn nrows(&self) -> usize {
        self.n
    }
    fn ncols(&self) -> usize {
        self.n
    }
    fn apply(&self, _x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        y.fill(0.0);
        Ok(())
    }
    fn apply_adjoint(&self, _x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        y.fill(0.0);
        Ok(())
    }
}

/// Preconditioner that scales every component by a very small constant.
///
/// Used to drive rz_init toward the lower boundary of representable values,
/// exercising the `rz_init.abs().max(f64::MIN_POSITIVE)` guard in the
/// stagnation check.
struct TinyPreconditioner {
    n: usize,
}

impl Operator for TinyPreconditioner {
    fn nrows(&self) -> usize {
        self.n
    }
    fn ncols(&self) -> usize {
        self.n
    }
    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        for (yi, &xi) in y.iter_mut().zip(x) {
            *yi = xi * 1e-300;
        }
        Ok(())
    }
    fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        self.apply(x, y)
    }
}

/// Rank-1 projection operator: A x = (v^T x) v.
///
/// Rank-deficient by construction. Used to exercise GMRES lucky breakdown:
/// the Krylov subspace is exhausted after 1 step when b lies in range(A).
struct Rank1Projector {
    v: Vec<f64>,
}

impl Operator for Rank1Projector {
    fn nrows(&self) -> usize {
        self.v.len()
    }
    fn ncols(&self) -> usize {
        self.v.len()
    }
    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        let dot: f64 = self.v.iter().zip(x).map(|(a, b)| a * b).sum();
        for (yi, &vi) in y.iter_mut().zip(&self.v) {
            *yi = dot * vi;
        }
        Ok(())
    }
    fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        // Self-adjoint since A = v v^T
        self.apply(x, y)
    }
}

// ---------------------------------------------------------------------------
// CG edge-case tests
// ---------------------------------------------------------------------------

/// The zero operator causes p^T A p = 0 on the very first iteration.
///
/// CG must exit the `pap <= 0` branch cleanly: not-converged after exactly
/// one iteration, with a finite solution.
#[test]
fn test_cg_zero_pap_clean_exit() {
    let op = ZeroOperator { n: 3 };
    let b = vec![1.0, 1.0, 1.0];
    let result =
        pcg(&op, &b, None::<&IdentityOperator>, 1e-10, 100).expect("cg solve should not error");

    assert!(!result.converged, "zero operator cannot satisfy Ax=b");
    assert_eq!(result.iterations, 1, "should exit on first pap <= 0 check");
    assert!(
        result.x.iter().all(|v| v.is_finite()),
        "solution components must be finite"
    );
}

/// The zero operator with an explicit preconditioner also hits pap <= 0.
///
/// With right-hand side [1,0,0] the initial search direction p = M^{-1} b,
/// and A p = 0 regardless of the preconditioner, so pap is still zero.
#[test]
fn test_cg_preconditioned_zero_pap_clean_exit() {
    let op = ZeroOperator { n: 3 };
    let prec = IdentityOp { n: 3 };
    let b = vec![1.0, 0.0, 0.0];
    let result = pcg(&op, &b, Some(&prec), 1e-10, 100).expect("cg solve should not error");

    assert!(!result.converged);
    assert_eq!(result.iterations, 1);
    assert!(result.x.iter().all(|v| v.is_finite()));
}

/// A preconditioner that scales by 1e-300 pushes rz_init toward the floor
/// used by the stagnation guard: `rz_init.abs().max(f64::MIN_POSITIVE)`.
///
/// Because 1e-300 > f64::MIN_POSITIVE (≈2.2e-308), the scale factor is the
/// preconditioner magnitude rather than the fallback constant. The test
/// asserts that the solver terminates without panic and returns finite results
/// whether it converges or exits via the stagnation path.
#[test]
fn test_cg_stagnation_guard_near_zero_rz_init() {
    let op = SpdMatrix3;
    let prec = TinyPreconditioner { n: 3 };
    let b = vec![1.0, 2.0, 3.0];
    let result = pcg(&op, &b, Some(&prec), 1e-10, 100).expect("cg solve should not error");

    // Regardless of convergence, the result must be numerically safe.
    assert!(
        result.x.iter().all(|v| v.is_finite()),
        "solution must be finite even with tiny preconditioner"
    );
    assert!(
        result.residual_norm.is_finite(),
        "residual norm must be finite"
    );
}

/// An extremely ill-conditioned diagonal system stresses the stagnation guard.
///
/// The condition number of ~1e30 forces rz to decay across iterations at a
/// rate that can trigger the relative stagnation check
/// `rz_new.abs() < f64::EPSILON * rz_init.abs().max(f64::MIN_POSITIVE)`.
/// The test asserts safe termination — convergence or stagnation exit, never
/// a panic or infinite loop.
#[test]
fn test_cg_stagnation_guard_extreme_conditioning() {
    let op = DiagOperator {
        values: vec![1e15, 1.0, 1e-15],
    };
    let b = vec![1.0, 1.0, 1.0];
    let result =
        pcg(&op, &b, None::<&IdentityOperator>, 1e-12, 100).expect("cg solve should not error");

    assert!(
        result.x.iter().all(|v| v.is_finite()),
        "solution must be finite for ill-conditioned system"
    );
    assert!(
        result.residual_norm.is_finite(),
        "residual norm must be finite"
    );
    // Iterations must be bounded by maxiter
    assert!(result.iterations <= 100);
}

/// CG must return immediately with a converged zero solution for a zero rhs,
/// without entering the iteration loop at all (iterations == 0).
#[test]
fn test_cg_zero_rhs_exits_immediately() {
    let op = SpdMatrix3;
    let b = vec![0.0; 3];
    let result =
        pcg(&op, &b, None::<&IdentityOperator>, 1e-10, 100).expect("cg solve should not error");

    assert!(result.converged);
    assert_eq!(result.iterations, 0);
    assert_eq!(result.residual_norm, 0.0);
    assert!(result.x.iter().all(|&v| v == 0.0));
}

/// When maxiter=0, CG returns immediately without performing any iteration.
///
/// The loop runs zero times, so neither convergence nor stagnation is checked.
/// The result should be the initial guess (all zeros) and not-converged.
#[test]
fn test_cg_maxiter_zero_returns_initial_guess() {
    let op = SpdMatrix3;
    let b = vec![1.0, 2.0, 3.0];
    let result =
        pcg(&op, &b, None::<&IdentityOperator>, 1e-10, 0).expect("cg solve should not error");

    assert!(!result.converged);
    assert_eq!(result.iterations, 0);
    // x starts at zero (initial guess)
    assert!(result.x.iter().all(|&v| v == 0.0));
}

/// A single-element system converges in exactly one iteration.
///
/// Exercises CG on the smallest possible system (n=1) where the Krylov
/// subspace spans the full space after one step.
#[test]
fn test_cg_single_element_system() {
    let op = DiagOperator { values: vec![5.0] };
    let b = vec![10.0];
    let result =
        pcg(&op, &b, None::<&IdentityOperator>, 1e-12, 100).expect("cg solve should not error");

    assert!(result.converged);
    assert_eq!(result.iterations, 1);
    assert!((result.x[0] - 2.0).abs() < 1e-12, "expected x[0] = 2.0");
}

/// A single-element system with preconditioner converges in one iteration.
#[test]
fn test_cg_preconditioned_single_element_system() {
    let op = DiagOperator { values: vec![5.0] };
    let prec = DiagOperator { values: vec![0.2] }; // exact inverse
    let b = vec![10.0];
    let result = pcg(&op, &b, Some(&prec), 1e-12, 100).expect("cg solve should not error");

    assert!(result.converged);
    assert!((result.x[0] - 2.0).abs() < 1e-12, "expected x[0] = 2.0");
}

// ---------------------------------------------------------------------------
// GMRES edge-case tests
// ---------------------------------------------------------------------------

/// GMRES with restart=0 is degenerate: no inner iterations can be performed
/// per restart cycle. The maxiter guard fires immediately and returns
/// not-converged with 0 iterations.
#[test]
fn test_gmres_maxiter_zero_returns_initial_guess() {
    let op = SpdMatrix3;
    let id = IdentityOp { n: 3 };
    let b = vec![1.0, 2.0, 3.0];
    let result = pgmres(&op, &b, Some(&id), 1e-10, 0, 30).expect("gmres solve should not error");

    assert!(!result.converged);
    assert_eq!(result.iterations, 0);
}

/// GMRES on a zero rhs returns immediately with the zero solution (iterations=0).
#[test]
fn test_gmres_zero_rhs_exits_immediately() {
    let op = SpdMatrix3;
    let id = IdentityOp { n: 3 };
    let b = vec![0.0; 3];
    let result = pgmres(&op, &b, Some(&id), 1e-10, 100, 30).expect("gmres solve should not error");

    assert!(result.converged);
    assert_eq!(result.iterations, 0);
    assert_eq!(result.residual_norm, 0.0);
    assert!(result.x.iter().all(|&v| v == 0.0));
}

/// GMRES with restart=1 and restart=30 must both converge to the same solution
/// on a well-conditioned 10x10 tridiagonal system.
///
/// This exercises the restart accumulation path: GMRES(1) requires many
/// restart cycles whereas GMRES(30) solves in a single cycle for this size.
#[test]
fn test_gmres_restart_accumulates_progress() {
    let n = 10;
    let op = TridiagOperator::new(n, 3.0);
    let b: Vec<f64> = (0..n).map(|i| i as f64 + 1.0).collect();
    let id = IdentityOp { n };

    let r1 = pgmres(&op, &b, Some(&id), 1e-10, 500, 1).expect("GMRES(1) solve");
    let r30 = pgmres(&op, &b, Some(&id), 1e-10, 500, 30).expect("GMRES(30) solve");

    assert!(r1.converged, "GMRES(1) did not converge");
    assert!(r30.converged, "GMRES(30) did not converge");

    // A larger Krylov subspace should not need more iterations than restart=1.
    assert!(
        r30.iterations <= r1.iterations,
        "GMRES(30) used {} iters, GMRES(1) used {} — larger restart should not be worse",
        r30.iterations,
        r1.iterations
    );

    check_residual(&op, &r1.x, &b, 1e-8);
    check_residual(&op, &r30.x, &b, 1e-8);
}

/// A rank-1 operator triggers GMRES lucky breakdown after the first Arnoldi step.
///
/// The Krylov subspace is exhausted (h_{j+1,j} ≈ 0) when b lies exactly in
/// range(A), causing the `Breakdown` arm to fire and the true residual to be
/// recomputed for the convergence decision.
#[test]
fn test_gmres_lucky_breakdown_rank1_operator() {
    // v = [1, 0, 0], A x = (v^T x) v = [x[0], 0, 0]
    // b = [1, 0, 0] lies in range(A), so the system Ax=b has solution x = [1, 0, 0] + ker(A)
    // GMRES finds the minimum-norm solution.
    let v = vec![1.0, 0.0, 0.0];
    let op = Rank1Projector { v };
    let id = IdentityOp { n: 3 };
    let b = vec![1.0, 0.0, 0.0];

    let result = pgmres(&op, &b, Some(&id), 1e-10, 100, 30).expect("gmres solve should not error");

    // The solver must terminate cleanly (breakdown path) and the solution
    // must satisfy Ax = b.
    assert!(
        result.x.iter().all(|v| v.is_finite()),
        "solution must be finite"
    );
    check_residual(&op, &result.x, &b, 1e-10);
}

/// The identity operator causes GMRES to converge in exactly one iteration,
/// which triggers the lucky breakdown path (h_{j+1,j} collapses to zero).
#[test]
fn test_gmres_lucky_breakdown_identity_operator() {
    let id_op = IdentityOp { n: 3 };
    let id_prec = IdentityOp { n: 3 };
    let b = vec![1.0, 2.0, 3.0];

    let result =
        pgmres(&id_op, &b, Some(&id_prec), 1e-10, 100, 30).expect("gmres solve should not error");

    assert!(result.converged, "GMRES on identity system must converge");
    // With A = I, the residual is zero after 1 step; breakdown fires immediately.
    assert!(
        result.iterations <= 3,
        "identity system should converge in very few iterations"
    );
    check_residual(&id_op, &result.x, &b, 1e-10);
}

/// A nearly-singular diagonal operator stresses the `solve_upper_triangular`
/// fallback: when the Hessenberg diagonal satisfies
/// `diag.abs() < 1e-14 * (1 + sum.abs())`, y[i] is set to 0 rather than
/// dividing by a near-zero value.
///
/// The solver must complete without panic and return a finite result.
#[test]
fn test_gmres_near_singular_hessenberg_diagonal() {
    // Smallest diagonal entry is below double precision noise level.
    let op = DiagOperator {
        values: vec![1e-16, 1.0, 2.0],
    };
    let id = IdentityOp { n: 3 };
    let b = vec![1.0, 1.0, 1.0];

    let result = pgmres(&op, &b, Some(&id), 1e-10, 100, 30).expect("gmres solve should not error");

    assert!(
        result.x.iter().all(|v| v.is_finite()),
        "solution must be finite even with near-singular Hessenberg"
    );
    assert!(
        result.residual_norm.is_finite(),
        "residual norm must be finite"
    );
}

/// GMRES with restart=1 on a large tridiagonal system exercises many restart
/// cycles and the update_solution accumulation across them.
#[test]
fn test_gmres_many_restarts_large_system() {
    let n = 20;
    let op = TridiagOperator::new(n, 4.0);
    let b: Vec<f64> = (0..n).map(|i| (i as f64).sin() + 1.0).collect();
    let id = IdentityOp { n };

    let result = pgmres(&op, &b, Some(&id), 1e-8, 1000, 1).expect("gmres solve should not error");

    assert!(
        result.converged,
        "GMRES(1) on 20x20 tridiagonal must converge"
    );
    check_residual(&op, &result.x, &b, 1e-6);
}

/// GMRES with restart equal to the system size behaves as full GMRES (no restarts).
///
/// For an n×n system, restart=n guarantees convergence in at most n iterations
/// (exact arithmetic). This exercises the path where `iters_this_cycle` is
/// never capped by the restart bound within a single cycle.
#[test]
fn test_gmres_full_restart_equals_system_size() {
    let n = 5;
    let op = TridiagOperator::new(n, 3.0);
    let b = vec![1.0, 2.0, 3.0, 2.0, 1.0];
    let id = IdentityOp { n };

    let result = pgmres(&op, &b, Some(&id), 1e-12, 100, n).expect("gmres solve should not error");

    assert!(
        result.converged,
        "full GMRES on 5x5 tridiagonal must converge"
    );
    assert!(
        result.iterations <= n,
        "full GMRES must converge within n iterations for n×n system"
    );
    check_residual(&op, &result.x, &b, 1e-10);
}

/// GMRES tolerance comparison: tighter tolerance yields a smaller residual norm.
///
/// Both solves must converge; the tighter one's residual must be no larger than
/// the loose one's.
#[test]
fn test_gmres_tighter_tolerance_gives_smaller_residual() {
    let n = 8;
    let op = TridiagOperator::new(n, 3.0);
    let b: Vec<f64> = (0..n).map(|i| i as f64 + 1.0).collect();
    let id = IdentityOp { n };

    let loose = pgmres(&op, &b, Some(&id), 1e-4, 200, 20).expect("gmres loose solve");
    let tight = pgmres(&op, &b, Some(&id), 1e-10, 200, 20).expect("gmres tight solve");

    assert!(loose.converged, "loose tolerance must converge");
    assert!(tight.converged, "tight tolerance must converge");
    assert!(
        tight.residual_norm <= loose.residual_norm + 1e-15,
        "tight residual {} should be <= loose residual {}",
        tight.residual_norm,
        loose.residual_norm
    );
}
