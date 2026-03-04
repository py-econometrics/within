mod common;

use schwarz_precond::solve::cg::{cg_solve, cg_solve_preconditioned};
use schwarz_precond::{Operator, SchwarzPreconditioner};

use common::{make_schwarz_entries, TridiagOperator};

#[test]
fn test_additive_schwarz_reduces_iterations() {
    let n = 20;
    let a = TridiagOperator::new(n, 3.0);
    let rhs = vec![1.0; n];

    let unprecond = cg_solve(&a, &rhs, 1e-8, 200).expect("unpreconditioned cg");
    assert!(unprecond.converged, "Unpreconditioned CG did not converge");

    let schwarz = SchwarzPreconditioner::new(make_schwarz_entries(n), n)
        .expect("valid additive schwarz preconditioner");
    let precond =
        cg_solve_preconditioned(&a, &schwarz, &rhs, 1e-8, 200).expect("preconditioned cg");
    assert!(precond.converged, "Preconditioned CG did not converge");

    assert!(
        precond.iterations <= unprecond.iterations,
        "Preconditioned ({}) should be <= unpreconditioned ({})",
        precond.iterations,
        unprecond.iterations
    );
}

#[test]
fn test_clone_produces_independent_preconditioner() {
    let n = 20;
    let original = SchwarzPreconditioner::new(make_schwarz_entries(n), n)
        .expect("valid additive schwarz preconditioner");
    let cloned = original.clone();

    // Apply on different inputs and verify both produce correct results.
    let r1: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
    let r2: Vec<f64> = (0..n).map(|i| ((n - i) as f64) * 0.5).collect();

    let mut z_orig = vec![0.0; n];
    let mut z_clone = vec![0.0; n];

    original.apply(&r1, &mut z_orig);
    cloned.apply(&r2, &mut z_clone);

    // Verify independently: apply the original with r2 to check the clone's result.
    let mut z_check = vec![0.0; n];
    original.apply(&r2, &mut z_check);

    for i in 0..n {
        assert!(
            (z_clone[i] - z_check[i]).abs() < 1e-14,
            "clone result differs at index {}: {} vs {}",
            i,
            z_clone[i],
            z_check[i],
        );
    }

    // Verify the original was not corrupted by the clone's apply.
    let mut z_orig2 = vec![0.0; n];
    original.apply(&r1, &mut z_orig2);
    for i in 0..n {
        assert!(
            (z_orig[i] - z_orig2[i]).abs() < 1e-14,
            "original result changed after clone apply at index {i}",
        );
    }
}

#[test]
fn test_additive_schwarz_operator_dimensions() {
    let n = 10;
    let schwarz = SchwarzPreconditioner::new(make_schwarz_entries(n), n)
        .expect("valid additive schwarz preconditioner");

    assert_eq!(schwarz.nrows(), n);
    assert_eq!(schwarz.ncols(), n);

    // apply and apply_adjoint should produce the same result (symmetric)
    let r = vec![1.0; n];
    let mut z1 = vec![0.0; n];
    let mut z2 = vec![0.0; n];
    schwarz.apply(&r, &mut z1);
    schwarz.apply_adjoint(&r, &mut z2);

    for i in 0..n {
        assert!(
            (z1[i] - z2[i]).abs() < 1e-14,
            "apply != apply_adjoint at index {}: {} vs {}",
            i,
            z1[i],
            z2[i]
        );
    }
}
