use within::{Effect, LocalSolverConfig, LsmrOptions, PreconditionerConfig, ReductionStrategy};
#[test]
fn close_slopes_keep_the_declared_span() {
    let n = 80;
    let z: Vec<f64> = (0..n).map(|i| (i as f64 * 0.37).sin()).collect();
    let v: Vec<f64> = (0..n).map(|i| (i as f64 * 0.23).cos()).collect();
    let second: Vec<f64> = z.iter().zip(&v).map(|(a, b)| a + 1e-6 * b).collect();
    let codes = vec![0; n];
    let solver = within::Solver::new_raw(
        vec![Effect::new(&codes, false, [z.as_slice(), second.as_slice()]).unwrap()],
        None,
        PreconditionerConfig::Additive {
            local_solver: LocalSolverConfig::default(),
            reduction: ReductionStrategy::ParallelReduction,
        },
    )
    .unwrap();
    let result = solver
        .solve(
            &v,
            &LsmrOptions {
                tol: 1e-12,
                maxiter: 1000,
                local_size: None,
            },
        )
        .unwrap();
    let error = v
        .iter()
        .zip(z.iter().zip(&second))
        .map(|(y, (a, b))| (y - result.x[0] * a - result.x[1] * b).abs())
        .fold(0., f64::max);
    println!(
        "converged={}, iterations={}, coefficients={:?}, max_residual={error:e}",
        result.converged, result.iterations, result.x
    );
    assert!(result.converged && error < 2e-8);
}

#[test]
fn raw_weighted_slopes_use_the_same_batch_kernels_across_workers() {
    let n = 20_011;
    let a: Vec<u32> = (0..n).map(|i| (i % 97) as u32).collect();
    let b: Vec<u32> = (0..n).map(|i| ((i / 3 + i % 7) % 53) as u32).collect();
    let z: Vec<f64> = (0..n).map(|i| (i as f64 * 0.173).sin()).collect();
    let t: Vec<f64> = (0..n).map(|i| (i as f64 * 0.237).cos()).collect();
    let weights: Vec<f64> = (0..n).map(|i| 0.7 + (i % 13) as f64 / 7.).collect();
    let y: Vec<f64> = (0..n)
        .map(|i| f64::from(a[i]) / 97. * (1. + 0.3 * z[i]) - 0.2 * f64::from(b[i]) / 53. * t[i])
        .collect();
    let other: Vec<f64> = (0..n).map(|i| (i as f64 * 0.73).sin()).collect();
    let mut reference = None;
    for workers in [1, 2, 4, 8] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .unwrap();
        let result = pool.install(|| {
            let solver = within::Solver::new_raw(
                vec![
                    Effect::new(&a, true, [z.as_slice()]).unwrap(),
                    Effect::new(&b, false, [t.as_slice()]).unwrap(),
                ],
                Some(&weights),
                PreconditionerConfig::Additive {
                    local_solver: LocalSolverConfig::default(),
                    reduction: ReductionStrategy::ParallelReduction,
                },
            )
            .unwrap();
            solver
                .solve_batch(
                    &[&y, &other],
                    &LsmrOptions {
                        tol: 1e-12,
                        maxiter: 1000,
                        local_size: None,
                    },
                )
                .unwrap()
        });
        assert!(result.converged.iter().all(|&v| v));
        assert!(result.unidentified.is_empty());
        // Reconstruct in caller coordinates, independently of upstream demeaned values.
        let error = (0..n)
            .map(|i| {
                (y[i]
                    - result.x[a[i] as usize]
                    - result.x[97 + a[i] as usize] * z[i]
                    - result.x[194 + b[i] as usize] * t[i])
                    .abs()
            })
            .fold(0., f64::max);
        assert!(error < 1e-8, "raw-coordinate error {error:e}");
        let bits = |v: Vec<f64>| v.into_iter().map(f64::to_bits).collect::<Vec<_>>();
        let actual = (bits(result.x), bits(result.demeaned), result.iterations);
        if let Some(expected) = &reference {
            assert!(expected == &actual, "workers={workers}");
        } else {
            reference = Some(actual);
        }
    }
}
