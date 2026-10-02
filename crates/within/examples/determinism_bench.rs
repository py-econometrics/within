//! Small reproducibility qualification timings, including design/preconditioner setup.
//! Run baseline and candidate executables consecutively, with the same worker argument.
use std::time::Instant;
use within::{
    Effect, LocalSolverConfig, LsmrOptions, PreconditionerConfig, ReductionStrategy, Solver,
};

fn main() {
    let threads: usize = std::env::args()
        .nth(1)
        .expect("worker count")
        .parse()
        .unwrap();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();
    println!("case,threads,repeat,seconds,iterations,status");
    for (case, n, widths) in [
        ("small", 20_000, [311, 97, 11]),
        ("few_levels", 1_000_000, [317, 101, 7]),
        ("panel", 1_000_000, [100_000, 4348, 10]),
        ("many_levels", 300_000, [100_003, 100_019, 11]),
    ] {
        let levels: Vec<Vec<u32>> = widths
            .into_iter()
            .enumerate()
            .map(|(q, width)| {
                (0..n)
                    .map(|i| match (case, q) {
                        ("panel", 0) => (i / 10) as u32,
                        (_, 0) => (i % width) as u32,
                        (_, 1) => ((i / 3 + i % 7) % width) as u32,
                        _ => ((i / 19) % width) as u32,
                    })
                    .collect()
            })
            .collect();
        let y: Vec<_> = (0..n)
            .map(|i| (i as f64 * 0.123).sin() + (i % 17) as f64 / 19.)
            .collect();
        let weights: Vec<_> = (0..n).map(|i| 0.7 + (i % 13) as f64 / 7.).collect();
        for repeat in 0..6 {
            let attempt: Result<_, String> = pool.install(|| {
                let start = Instant::now();
                let solver = Solver::new(
                    levels
                        .iter()
                        .map(|column| Effect::new(column, true, []).unwrap())
                        .collect::<Vec<_>>(),
                    Some(&weights),
                    &PreconditionerConfig::Additive {
                        local_solver: LocalSolverConfig::default(),
                        reduction: ReductionStrategy::ParallelReduction,
                    },
                )
                .map_err(|error| error.to_string())?;
                let result = solver
                    .solve(
                        &y,
                        &LsmrOptions {
                            tol: 1e-8,
                            maxiter: 1000,
                            local_size: None,
                        },
                    )
                    .map_err(|error| error.to_string())?;
                if !result.converged {
                    return Err(format!(
                        "not converged after {} iterations",
                        result.iterations
                    ));
                }
                std::hint::black_box(&result);
                Ok((start.elapsed().as_secs_f64(), result.iterations))
            });
            match attempt {
                Ok((elapsed, iterations)) => {
                    println!("{case},{threads},{repeat},{elapsed:.9},{iterations},ok")
                }
                Err(error) => {
                    println!("{case},{threads},{repeat},,,error");
                    eprintln!("{case},threads={threads}: {error}");
                    break;
                }
            }
        }
    }
}
