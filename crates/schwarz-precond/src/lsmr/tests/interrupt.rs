use super::super::{fixtures::*, *};
use std::sync::atomic::{AtomicUsize, Ordering};

fn assert_same(a: &LsmrResult, b: &LsmrResult) {
    assert_eq!(
        a.x.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        b.x.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    assert_eq!(a.iterations, b.iterations);
    assert_eq!(a.stop_reason, b.stop_reason);
    assert_eq!(a.converged, b.converged);
    assert_eq!(a.residual_norm.to_bits(), b.residual_norm.to_bits());
    assert_eq!(
        a.normal_eq_residual.to_bits(),
        b.normal_eq_residual.to_bits()
    );
    assert_eq!(a.true_residual, b.true_residual);
}

#[test]
fn every_executed_poll_aborts_and_default_retries_are_identical() {
    let op = OverdeterminedOp;
    let pre = IdentityOp { n: 3 };
    let rhs = [1., 2., 3., 4.];
    for modified in [false, true] {
        let solve = |poll: Option<&(dyn Fn() -> bool + Sync)>| {
            if modified {
                mlsmr_with_poll(&op, &rhs, &pre, 1e-12, 100, MlsmrOptions::default(), poll)
            } else {
                lsmr_with_poll(&op, &rhs, 1e-12, 100, None, poll)
            }
        };
        let original = if modified {
            mlsmr(&op, &rhs, &pre, 1e-12, 100, MlsmrOptions::default()).unwrap()
        } else {
            lsmr(&op, &rhs, 1e-12, 100, None).unwrap()
        };
        let calls = AtomicUsize::new(0);
        let traced = solve(Some(&|| {
            calls.fetch_add(1, Ordering::SeqCst);
            true
        }))
        .unwrap();
        assert_same(&original, &traced);
        assert!(calls.load(Ordering::SeqCst) > original.iterations);
        for stop in 0..calls.load(Ordering::SeqCst) {
            let count = AtomicUsize::new(0);
            assert!(matches!(
                solve(Some(&|| count.fetch_add(1, Ordering::SeqCst) != stop)),
                Err(SolveError::Interrupted)
            ));
            assert_eq!(count.load(Ordering::SeqCst), stop + 1);
            assert_same(&original, &solve(None).unwrap());
        }
    }
}
