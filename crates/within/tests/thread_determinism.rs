use within::{Effect, LocalSolverConfig, LsmrOptions, PreconditionerConfig, ReductionStrategy};

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn weighted_multiway_batch_is_identical_across_workers() {
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
                within::solve_batch(
                    vec![
                        Effect::new(&a, true, []).unwrap(),
                        Effect::new(&b, true, []).unwrap(),
                        Effect::new(&c, true, []).unwrap(),
                        Effect::new(&d, true, []).unwrap(),
                    ],
                    &[&y, &other],
                    Some(&weights),
                    &LsmrOptions {
                        tol: 1e-12,
                        maxiter: 1000,
                        local_size: None,
                    },
                    config.clone(),
                )
                .unwrap()
            });
            assert!(result.converged.iter().all(|&value| value));
            let actual = (bits(&result.x), bits(&result.demeaned), result.iterations);
            if let Some(expected) = &reference {
                // Keep a failing test's output bounded even for an observation-sized mismatch.
                assert!(expected == &actual, "{config:?}, workers={workers}");
            } else {
                reference = Some(actual);
            }
        }
    }
}
