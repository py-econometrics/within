use rstest::rstest;
use within::{
    Effect, LocalSolverConfig, LsmrOptions, PreconditionerConfig, ReductionStrategy, Solver,
};

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|x| x.to_bits()).collect()
}

#[rstest]
#[case::weighted_intercepts(true, false)]
#[case::unweighted_intercepts(false, false)]
#[case::weighted_slopes(true, true)]
#[case::unweighted_slopes(false, true)]
fn multiway_batch_is_identical_across_workers(#[case] weighted: bool, #[case] sloped: bool) {
    let n = 20_000;
    let a: Vec<u32> = (0..n).map(|i| (i % 311) as u32).collect();
    let b: Vec<u32> = (0..n).map(|i| ((i / 3 + i % 7) % 97) as u32).collect();
    let c: Vec<u32> = (0..n).map(|i| ((i / 19) % 11) as u32).collect();
    let d: Vec<u32> = (0..n).map(|i| ((i / 37) % 5) as u32).collect();
    let y: Vec<f64> = (0..n)
        .map(|i| (i as f64 * 0.123).sin() + (i % 17) as f64 / 19.)
        .collect();
    let other: Vec<f64> = (0..n).map(|i| (i as f64 * 0.37).cos()).collect();
    let weights: Vec<f64> = (0..n).map(|i| 0.7 + (i % 13) as f64 / 7.).collect();
    let slope: Vec<_> = (0..n).map(|i| (i as f64 * 0.173).sin()).collect();
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
        for workers in [1, 2, 4, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .unwrap();
            let result = pool.install(|| {
                let solver = Solver::new(
                    vec![
                        Effect::new(&a, true, sloped.then_some(slope.as_slice())).unwrap(),
                        Effect::new(&b, true, []).unwrap(),
                        Effect::new(&c, true, []).unwrap(),
                        Effect::new(&d, true, []).unwrap(),
                    ],
                    weighted.then_some(weights.as_slice()),
                    &config,
                )
                .unwrap();
                let result = solver.solve_batch(&[&y, &other], &options).unwrap();
                let again = solver.solve_batch(&[&y, &other], &options).unwrap();
                assert!(bits(&result.x) == bits(&again.x));
                assert!(bits(&result.demeaned) == bits(&again.demeaned));
                assert_eq!(bits(&result.residual), bits(&again.residual));
                assert_eq!(result.iterations, again.iterations);
                let scalar = solver.solve(&y, &options).unwrap();
                assert!(bits(&result.x[..result.n_dofs]) == bits(&scalar.x));
                assert!(bits(&result.demeaned[..n]) == bits(&scalar.demeaned));
                assert_eq!(result.residual[0].to_bits(), scalar.residual.to_bits());
                assert_eq!(result.iterations[0], scalar.iterations);
                assert_eq!(result.converged[0], scalar.converged);
                result
            });
            assert!(result.converged.iter().all(|&value| value));
            let actual = (
                bits(&result.x),
                bits(&result.demeaned),
                bits(&result.residual),
                result.iterations,
                result.converged,
                result.unidentified,
            );
            if let Some(expected) = &reference {
                // Keep a failing test's output bounded even for an observation-sized mismatch.
                assert!(expected == &actual, "{config:?}, workers={workers}");
            } else {
                reference = Some(actual);
            }
        }
    }
}
