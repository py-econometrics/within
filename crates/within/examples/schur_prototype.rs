#[cfg(feature = "schur-prototype")]
#[path = "../benches/shared/fixest_dgp.rs"]
mod fixest_dgp;

#[cfg(feature = "schur-prototype")]
fn main() {
    use std::time::Instant;
    use within::{LocalSolverConfig, LsmrOptions, PreconditionerConfig, Solver};
    let n: usize = std::env::args()
        .nth(1)
        .map(|v| v.parse().unwrap())
        .unwrap_or(100_000);
    let options = LsmrOptions {
        tol: 1e-12,
        maxiter: 10000,
        local_size: Some(10),
    };
    for difficult in [false, true] {
        let (design, y) = fixest_dgp::generate_fixest_like_case(n, 3, difficult, 0xC0FFEE);
        let reference_solver = Solver::new(
            &design,
            None,
            PreconditionerConfig::Additive {
                local_solver: LocalSolverConfig::default(),
                reduction: Default::default(),
            },
        )
        .unwrap();
        let reference = reference_solver
            .solve(
                &y,
                &LsmrOptions {
                    tol: 1e-13,
                    ..options.clone()
                },
            )
            .unwrap();
        assert!(reference.converged);
        for config in [
            PreconditionerConfig::Diagonal,
            PreconditionerConfig::Additive {
                local_solver: LocalSolverConfig::default(),
                reduction: Default::default(),
            },
        ] {
            let build = Instant::now();
            let solver = Solver::new(&design, None, &config).unwrap();
            println!(
                "n={n} difficult={difficult} config={config:?} dofs={} build={:.4}s",
                solver.n_dofs(),
                build.elapsed().as_secs_f64()
            );
            let scale = y.iter().map(|v| v * v).sum::<f64>().sqrt();
            let mut times = [Vec::new(), Vec::new()];
            for repeat in 0..4 {
                let mut results = Vec::new();
                for schur in if repeat % 2 == 0 {
                    [false, true]
                } else {
                    [true, false]
                } {
                    let start = Instant::now();
                    let result = if schur {
                        solver.solve_schur_prototype(&y, &options)
                    } else {
                        solver.solve(&y, &options)
                    }
                    .unwrap();
                    if repeat > 0 {
                        times[schur as usize].push(start.elapsed().as_secs_f64());
                    }
                    println!(
                        "  schur={schur} iterations={} converged={} residual={:.2e} total={:.4}s",
                        result.iterations, result.converged, result.residual, result.time_total
                    );
                    assert!(result.converged);
                    let reference_error = result
                        .demeaned
                        .iter()
                        .zip(&reference.demeaned)
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f64>()
                        .sqrt()
                        / scale;
                    println!("  fit error against tighter full solve={reference_error:.2e}");
                    assert!(reference_error < 1e-7);
                    results.push(result);
                }
                let error = results[0]
                    .demeaned
                    .iter()
                    .zip(&results[1].demeaned)
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let scale = y.iter().map(|v| v * v).sum::<f64>().sqrt();
                println!("  relative fit difference={:.2e}", error / scale);
                assert!(error / scale < 1e-6);
            }
            for time in &mut times {
                time.sort_by(f64::total_cmp);
            }
            println!(
                "median full={:.4}s schur={:.4}s speedup={:.2}x",
                times[0][1],
                times[1][1],
                times[0][1] / times[1][1]
            );
        }
    }
}

#[cfg(not(feature = "schur-prototype"))]
fn main() {
    eprintln!("Enable --features schur-prototype to run this experiment.");
}
