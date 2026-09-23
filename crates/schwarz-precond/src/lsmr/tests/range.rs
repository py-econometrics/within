//! Extreme scales: no vector leaves the double range unless the answer itself does.

use rstest::rstest;

use super::super::*;
use crate::lsmr::fixtures::*;
use crate::Operator;

#[derive(Clone, Copy, Debug)]
enum Metric {
    None,
    Identity,
    Diagonal,
}

fn solve<A: Operator>(
    a: &A,
    b: &[f64],
    metric: Metric,
    local_size: Option<usize>,
    warm_start: Option<&[f64]>,
) -> Result<LsmrResult, SolveError> {
    let n = a.ncols();
    let diagonal = DiagOp(
        (0..n)
            .map(|j| if j % 2 == 0 { 4.0 } else { 0.25 })
            .collect(),
    );
    let options = MlsmrOptions {
        warm_start,
        local_size,
        ..Default::default()
    };
    match metric {
        Metric::None => lsmr(a, b, 1e-10, 50, local_size),
        Metric::Identity => mlsmr(a, b, &IdentityOp { n }, 1e-10, 50, options),
        Metric::Diagonal => mlsmr(a, b, &diagonal, 1e-10, 50, options),
    }
}

/// `α₁ = ‖Aᵀb‖ / ‖b‖ ≈ 1e-310` is subnormal although `‖A‖` and `‖b‖` are not, so `1/α₁` is ∞.
#[rstest]
fn a_subnormal_initial_gradient_normalizes_its_basis_vector(
    #[values(Metric::None, Metric::Identity, Metric::Diagonal)] metric: Metric,
    #[values(None, Some(2))] local_size: Option<usize>,
) {
    let r = solve(
        &DiagOp(vec![0.0, 1.0]),
        &[1e200, 1e-110],
        metric,
        local_size,
        None,
    )
    .expect("subnormal-gradient solve");

    assert!(r.converged, "{:?}", r.stop_reason);
    assert_eq!(r.x[0], 0.0);
    assert!((r.x[1] / 1e-110 - 1.0).abs() < 1e-9, "{:?}", r.x);
}

/// A subnormal `‖b‖` has no representable reciprocal either.
#[rstest]
fn a_subnormal_rhs_normalizes_its_first_vector(
    #[values(Metric::None, Metric::Identity, Metric::Diagonal)] metric: Metric,
) {
    let r = solve(&IdentityOp { n: 2 }, &[1e-310, 0.0], metric, None, None)
        .expect("subnormal-rhs solve");

    assert!(r.converged, "{:?}", r.stop_reason);
    assert!((r.x[0] / 1e-310 - 1.0).abs() < 1e-9, "{:?}", r.x);
    assert_eq!(r.x[1], 0.0);
}

/// `‖A‖ = f64::MAX` leaves `1/α₁` subnormal; a stop dropping column 2 must hold its backward error.
#[test]
fn an_operator_at_the_top_of_the_range_solves_to_its_backward_error() {
    let a = DiagOp(vec![f64::MAX, 1.0]);
    let b = [1.0, 1.0];
    let r = lsmr(&a, &b, 1e-10, 50, None).expect("top-of-range solve");

    assert!(r.converged, "{:?}", r.stop_reason);
    assert!((r.x[0] * f64::MAX - 1.0).abs() < 1e-9, "{:?}", r.x);
    let normr = vec_norm(&[b[0] - f64::MAX * r.x[0], b[1] - r.x[1]]);
    let backward_error = normal_equation_residual(&a, &r.x, &b) / f64::MAX / normr;
    assert!(backward_error <= 1e-10, "{backward_error:e}");
}
