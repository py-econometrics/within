use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicUsize, Ordering};
use within::{
    BatchSolveResult, Effect, LocalSolverConfig, LsmrOptions, PreconditionerConfig,
    ReductionStrategy, SolveError, Solver, WithinError,
};

fn same(a: &BatchSolveResult, b: &BatchSolveResult) {
    for (left, right) in [
        (&a.x, &b.x),
        (&a.demeaned, &b.demeaned),
        (&a.residual, &b.residual),
    ] {
        assert!(left
            .iter()
            .zip(right)
            .all(|(a, b)| a.to_bits() == b.to_bits()));
        assert_eq!(left.len(), right.len());
    }
    assert_eq!(a.iterations, b.iterations);
    assert_eq!(a.converged, b.converged);
}

#[test]
fn batch_cancellation_and_panic_leave_the_solver_reusable() {
    let n = 700;
    let a: Vec<_> = (0..n).map(|i| (i % 29) as u32).collect();
    let b: Vec<_> = (0..n).map(|i| ((i / 3 + i % 5) % 17) as u32).collect();
    let y: Vec<_> = (0..n).map(|i| (i as f64 * 0.123).sin()).collect();
    let z: Vec<_> = (0..n).map(|i| (i as f64 * 0.37).cos()).collect();
    let options = LsmrOptions {
        tol: 1e-12,
        maxiter: 1000,
        local_size: None,
    };
    for config in [
        PreconditionerConfig::Off,
        PreconditionerConfig::Diagonal,
        PreconditionerConfig::Additive {
            local_solver: LocalSolverConfig::default(),
            reduction: ReductionStrategy::ParallelReduction,
        },
    ] {
        let mut reference = None;
        for workers in [1, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .unwrap();
            pool.install(|| {
                let solver = Solver::new_raw(
                    vec![
                        Effect::new(&a, true, []).unwrap(),
                        Effect::new(&b, true, []).unwrap(),
                    ],
                    None,
                    config.clone(),
                )
                .unwrap();
                let original = solver.solve_batch(&[&y, &z], &options).unwrap();
                if let Some(reference) = &reference {
                    same(reference, &original);
                }
                let calls = AtomicUsize::new(0);
                let traced = solver
                    .solve_batch_with_poll(
                        &[&y, &z],
                        &options,
                        Some(&|| {
                            calls.fetch_add(1, Ordering::SeqCst);
                            true
                        }),
                    )
                    .unwrap();
                same(&original, &traced);
                assert!(calls.load(Ordering::SeqCst) > original.iterations.iter().sum());
                for stop in [0, 3, calls.load(Ordering::SeqCst) / 2] {
                    let count = AtomicUsize::new(0);
                    assert!(matches!(
                        solver.solve_batch_with_poll(
                            &[&y, &z],
                            &options,
                            Some(&|| count.fetch_add(1, Ordering::SeqCst) < stop)
                        ),
                        Err(WithinError::Solve(SolveError::Interrupted))
                    ));
                    same(&original, &solver.solve_batch(&[&y, &z], &options).unwrap());
                }
                assert!(catch_unwind(AssertUnwindSafe(|| solver.solve_with_poll(
                    &y,
                    &options,
                    Some(&|| panic!("poll panic"))
                )))
                .is_err());
                same(&original, &solver.solve_batch(&[&y, &z], &options).unwrap());
                reference = Some(original);
            });
        }
    }
}
