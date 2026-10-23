#[cfg(feature = "asca-prototype")]
#[path = "../benches/shared/fixest_dgp.rs"]
mod fixest_dgp;

#[cfg(feature = "asca-prototype")]
fn main() {
    use std::{num::NonZeroUsize, time::Instant};
    use within::{AscaConfig, LocalSolverConfig, LsmrOptions, PreconditionerConfig, Solver};
    let n: usize = std::env::args()
        .nth(1)
        .map(|v| v.parse().unwrap())
        .unwrap_or(100_000);
    let options = LsmrOptions {
        tol: 1e-9,
        maxiter: 2000,
        local_size: Some(10),
    };
    for difficult in [false, true] {
        let (design, y) = fixest_dgp::generate_fixest_like_case(n, 3, difficult, 0xC0FFEE);
        let start = Instant::now();
        let full = Solver::new(
            &design,
            None,
            PreconditionerConfig::Additive {
                local_solver: LocalSolverConfig::default(),
                reduction: Default::default(),
            },
        )
        .unwrap();
        let full_build = start.elapsed().as_secs_f64();
        let start = Instant::now();
        let reduced = Solver::new(&design, None, PreconditionerConfig::Off).unwrap();
        let reduced_build = start.elapsed().as_secs_f64();
        let reference = full
            .solve(
                &y,
                &LsmrOptions {
                    tol: 1e-13,
                    ..options.clone()
                },
            )
            .unwrap();
        assert!(reference.converged);
        let scale = y.iter().map(|v| v * v).sum::<f64>().sqrt();
        let check = |r: &within::SolveResult| {
            let error = r
                .demeaned
                .iter()
                .zip(&reference.demeaned)
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt()
                / scale;
            assert!(
                r.converged,
                "residual={} iterations={}",
                r.residual, r.iterations
            );
            assert!(error < 1e-7, "fit error={error}");
            error
        };
        println!("n={n} difficult={difficult} full_build={full_build:.6}s");
        for projected in [false, true] {
            let mut times = Vec::new();
            for repeat in 0..4 {
                let r = if projected {
                    full.solve_schur_prototype(&y, &options)
                } else {
                    full.solve(&y, &options)
                }
                .unwrap();
                let error = check(&r);
                if repeat > 0 {
                    times.push(r.time_total);
                }
                if repeat == 0 {
                    println!(
                        "  projected={projected} iterations={} fit_error={error:.2e}",
                        r.iterations
                    );
                }
            }
            times.sort_by(f64::total_cmp);
            println!(
                "  projected={projected} median={:.6}s first_rhs={:.6}s",
                times[1],
                full_build + times[1]
            );
        }
        for pairs_per_level in [None, NonZeroUsize::new(8), NonZeroUsize::new(32)] {
            let start = Instant::now();
            let asca = reduced
                .asca_prototype(AscaConfig {
                    pairs_per_level,
                    ..Default::default()
                })
                .unwrap();
            let build = reduced_build + start.elapsed().as_secs_f64();
            if pairs_per_level.is_none() {
                let mut times = Vec::new();
                for repeat in 0..4 {
                    let start = Instant::now();
                    let r = asca
                        .solve_with_full_preconditioner(
                            &y,
                            &options,
                            full.preconditioner().unwrap(),
                        )
                        .unwrap();
                    let elapsed = start.elapsed().as_secs_f64();
                    let error = check(&r);
                    if repeat > 0 {
                        times.push(elapsed);
                    }
                    if repeat == 0 {
                        println!("  restricted-PCG iterations={} full_gradient={:.2e} fit_error={error:.2e}", r.iterations, r.residual);
                    }
                }
                times.sort_by(f64::total_cmp);
                println!("  restricted-PCG median={:.6}s", times[1]);
            }
            println!(
                "  pairs={pairs_per_level:?} reduced={} nnz={} assembly={:.6}s factor={:.6}s",
                asca.n_reduced(),
                asca.preconditioner_nnz(),
                asca.assembly_seconds(),
                asca.factor_seconds()
            );
            let mut times = Vec::new();
            for repeat in 0..4 {
                let r = asca.solve(&y, &options).unwrap();
                let error = check(&r);
                if repeat > 0 {
                    times.push(r.time_total);
                }
                if repeat == 0 {
                    println!(
                        "    iterations={} full_gradient={:.2e} fit_error={error:.2e}",
                        r.iterations, r.residual
                    );
                }
            }
            times.sort_by(f64::total_cmp);
            println!(
                "    median={:.6}s first_rhs={:.6}s",
                times[1],
                build + times[1]
            );
        }
    }
}

#[cfg(not(feature = "asca-prototype"))]
fn main() {
    eprintln!("Enable --features asca-prototype to run this experiment.");
}
