#[cfg(feature = "pair-schur-prototype")]
#[path = "../benches/shared/fixest_dgp.rs"]
mod fixest_dgp;

#[cfg(feature = "pair-schur-prototype")]
fn main() {
    use ndarray::Array2;
    use within::Design;
    let arg = std::env::args().nth(1);
    match arg.as_deref().map(str::parse::<usize>) {
        None | Some(Ok(_)) => {
            let n = arg.map_or(100_000, |s| s.parse().unwrap());
            for difficult in [false, true] {
                let (design, y) = fixest_dgp::generate_fixest_like_case(n, 3, difficult, 0xC0FFEE);
                compare(&format!("fixest n={n} difficult={difficult}"), &design, &y);
            }
        }
        // Dumps of `u64 n_obs, u64 n_fe, u32 categories row-major, f64 y`.
        Some(Err(_)) => {
            for path in std::env::args().skip(1) {
                let bytes = std::fs::read(&path).unwrap();
                let word = |i: usize| u64::from_le_bytes(bytes[8 * i..][..8].try_into().unwrap());
                let (n_obs, n_fe) = (word(0) as usize, word(1) as usize);
                let (cats, y) = bytes[16..].split_at(4 * n_obs * n_fe);
                let cats = cats
                    .chunks_exact(4)
                    .map(|b| u32::from_le_bytes(b.try_into().unwrap()));
                let cats = Array2::from_shape_vec((n_obs, n_fe), cats.collect()).unwrap();
                let y: Vec<f64> = y
                    .chunks_exact(8)
                    .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
                    .collect();
                let design = Design::from_categories(cats.view()).unwrap();
                compare(&path, &design, &y);
            }
        }
    }
}

#[cfg(feature = "pair-schur-prototype")]
fn compare(label: &str, design: &within::Design<'_>, y: &[f64]) {
    use std::{hint::black_box, time::Instant};
    use within::{LocalSolverConfig, LsmrOptions, PreconditionerConfig, SchurMode, Solver};
    let config = LocalSolverConfig {
        dense_threshold: 0,
        schur: match std::env::var("SCHUR").as_deref() {
            Ok("approximate") => SchurMode::default(),
            _ => SchurMode::Exact,
        },
        ..Default::default()
    };
    let options = LsmrOptions {
        tol: 1e-9,
        maxiter: 2000,
        local_size: Some(10),
    };
    let start = Instant::now();
    let full = Solver::new(
        design,
        None,
        PreconditionerConfig::Additive {
            local_solver: config.clone(),
            reduction: Default::default(),
        },
    )
    .unwrap();
    let full_build = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let base = Solver::new(design, None, PreconditionerConfig::Off).unwrap();
    let reduced = base
        .pair_schur_prototype(&config, Default::default())
        .unwrap();
    let reduced_build = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let jacobi = base
        .pair_schur_block_jacobi(&config, Default::default())
        .unwrap();
    let jacobi_build = start.elapsed().as_secs_f64();
    let (operator, jacobi_operator) = (reduced.assemble().unwrap(), jacobi.assemble().unwrap());
    let reference = full
        .solve(
            y,
            &LsmrOptions {
                tol: 1e-13,
                maxiter: 10_000,
                ..options.clone()
            },
        )
        .unwrap();
    let scale = y.iter().map(|v| v * v).sum::<f64>().sqrt();
    println!("{label} schur={:?} reference_converged={} full_dofs={} reduced_dofs={} schur_domains={} retained_domains={}",config.schur,reference.converged,full.n_dofs(),reduced.n_reduced(),reduced.schur_domains(),reduced.retained_domains());
    println!("full_build={full_build:.6}s reduced_build={reduced_build:.6}s operator={operator:.6}s reduced_preconditioner={:.6}s",reduced.preconditioner_seconds());
    // Reduced mlsmr never reads the assembled `S`, so its build excludes that assembly.
    let methods = [
        ("full_mlsmr_full_schwarz", full_build),
        ("reduced_mlsmr_restricted_full", full_build),
        ("reduced_mlsmr_pair_schur", reduced_build),
        (
            "reduced_pcg_assembled_s_pair_schur",
            reduced_build + operator,
        ),
        ("reduced_mlsmr_block_jacobi", jacobi_build),
        (
            "reduced_pcg_assembled_s_block_jacobi",
            jacobi_build + jacobi_operator,
        ),
    ];
    let mut times = vec![Vec::new(); methods.len()];
    for repeat in 0..7 {
        let order: Vec<usize> = match repeat % 2 {
            0 => (0..methods.len()).collect(),
            _ => (0..methods.len()).rev().collect(),
        };
        for method in order {
            let start = Instant::now();
            let result = match method {
                0 => full.solve(y, &options),
                1 => full.solve_schur_prototype(y, &options),
                2 => reduced.solve_lsmr(y, &options),
                3 => reduced.solve(y, &options),
                4 => jacobi.solve_lsmr(y, &options),
                _ => jacobi.solve(y, &options),
            }
            .unwrap();
            let elapsed = start.elapsed().as_secs_f64();
            let error = result
                .demeaned
                .iter()
                .zip(&reference.demeaned)
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt()
                / scale;
            if repeat > 0 {
                times[method].push(elapsed);
            }
            if repeat == 0 {
                println!(
                    "{} converged={} iterations={} residual={:.2e} fit_error={error:.2e}",
                    methods[method].0, result.converged, result.iterations, result.residual
                );
            }
        }
    }
    for ((name, build), time) in methods.iter().zip(&mut times) {
        time.sort_by(f64::total_cmp);
        let median = (time[2] + time[3]) / 2.0;
        println!(
            "{name} median={median:.6}s build={build:.6}s first_rhs={:.6}s",
            build + median
        );
    }
    let r: Vec<f64> = (0..reduced.n_reduced()).map(|i| (i as f64).sin()).collect();
    let mut z = vec![0.0; r.len()];
    let mut full_r = vec![0.0; full.n_dofs()];
    let eliminated = full.n_dofs() - reduced.n_reduced();
    full_r[eliminated..].copy_from_slice(&r);
    let mut full_z = vec![0.0; full.n_dofs()];
    let start = Instant::now();
    for _ in 0..100 {
        full.preconditioner()
            .unwrap()
            .apply(black_box(&full_r), black_box(&mut full_z))
            .unwrap();
    }
    let old = start.elapsed().as_secs_f64() / 100.0;
    let start = Instant::now();
    for _ in 0..100 {
        reduced
            .apply_preconditioner(black_box(&r), black_box(&mut z))
            .unwrap();
    }
    let new = start.elapsed().as_secs_f64() / 100.0;
    println!(
        "preconditioner_apply full={old:.6}s local_schur={new:.6}s speedup={:.2}x",
        old / new
    );
}

#[cfg(not(feature = "pair-schur-prototype"))]
fn main() {
    eprintln!("Enable --features pair-schur-prototype to run this experiment.");
}
