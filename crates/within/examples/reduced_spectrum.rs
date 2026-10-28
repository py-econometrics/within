//! Iterations and Ritz extremes of the preconditioned reduced system on dumped worker/firm/year cases.
//! Dump: `u64 n, n_worker_slopes, n_firm_slopes; u32 worker, firm, year; f64 slopes…, y`.
//! Env: `LOCAL=exact`, `STEPS`, `WITHIN_REDUCED_RIGID` for the coarse rung, and `VECTORS=k` to
//! write the `k` lowest eigenvectors to `<dump>.ritz`: `u64 k, n_terms, (levels, columns)…; f64 θ…, x…`.

#[cfg(feature = "pair-schur-prototype")]
fn main() {
    use within::{
        Design, Effect, LocalSolverConfig, LsmrOptions, PreconditionerConfig, SchurMode, Solver,
    };
    let steps: usize = std::env::var("STEPS").map_or(1000, |s| s.parse().unwrap());
    let config = match std::env::var("LOCAL").as_deref() {
        Ok("exact") => LocalSolverConfig {
            schur: SchurMode::Exact,
            dense_threshold: 1 << 30,
            ..Default::default()
        },
        _ => LocalSolverConfig::default(),
    };
    for path in std::env::args().skip(1) {
        let bytes = std::fs::read(&path).unwrap();
        let word = |i: usize| u64::from_le_bytes(bytes[8 * i..][..8].try_into().unwrap()) as usize;
        let (n, n_ws, n_fs) = (word(0), word(1), word(2));
        let codes: Vec<Vec<u32>> = (0..3)
            .map(|k| {
                bytes[24 + 4 * n * k..][..4 * n]
                    .chunks_exact(4)
                    .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                    .collect()
            })
            .collect();
        let floats: Vec<Vec<f64>> = (0..n_ws + n_fs + 1)
            .map(|k| {
                bytes[24 + 12 * n + 8 * n * k..][..8 * n]
                    .chunks_exact(8)
                    .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
                    .collect()
            })
            .collect();
        let slopes = |r: std::ops::Range<usize>| -> Vec<&[f64]> {
            floats[r].iter().map(|v| &v[..]).collect()
        };
        let effects = vec![
            Effect::new(&codes[0], true, slopes(0..n_ws)).unwrap(),
            Effect::new(&codes[1], true, slopes(n_ws..n_ws + n_fs)).unwrap(),
            Effect::new(&codes[2], true, []).unwrap(),
        ];
        let design = Design::new(effects).unwrap();
        let base = Solver::new(&design, None, PreconditionerConfig::Off).unwrap();
        let reduced = base
            .pair_schur_prototype(&config, Default::default())
            .unwrap();
        let options = LsmrOptions {
            tol: 1e-12,
            maxiter: 20_000,
            ..Default::default()
        };
        let solved = reduced.solve_lsmr(&floats[n_ws + n_fs], &options).unwrap();
        let ritz = reduced.ritz_values(steps).unwrap();
        let hi = ritz[ritz.len() - 1].0;
        // Rounding outside `range(Ã)` is an exact zero eigenvector of `Ã M Ãᵀ`; values in transit
        // toward it have not converged.
        let zeros = ritz.iter().filter(|r| r.0 <= 1e-8 * hi).count();
        let converged: Vec<f64> = (ritz.iter())
            .filter(|r| r.0 > 1e-8 * hi && r.1 <= 1e-6 * hi)
            .map(|r| r.0)
            .collect();
        let lo = converged.first().copied().unwrap_or(f64::NAN);
        let vectors: usize = std::env::var("VECTORS").map_or(0, |s| s.parse().unwrap());
        if vectors > 0 {
            let which: Vec<usize> = (0..ritz.len())
                .filter(|&j| ritz[j].0 > 1e-8 * hi && ritz[j].1 <= 1e-6 * hi)
                .filter(|&j| j == 0 || ritz[j].0 - ritz[j - 1].0 > 1e-6 * ritz[j].0)
                .take(vectors)
                .collect();
            let pairs = reduced.ritz_vectors(steps, &which).unwrap();
            let terms = reduced.kept_terms();
            let mut out: Vec<u8> = Vec::new();
            for w in [which.len(), terms.len()]
                .into_iter()
                .chain(terms.iter().flat_map(|t| [t.0, t.1]))
            {
                out.extend((w as u64).to_le_bytes());
            }
            for &j in &which {
                out.extend(ritz[j].0.to_le_bytes());
            }
            for (x, residual) in &pairs {
                eprintln!("ritz vector residual {residual:.2e}");
                out.extend(x.iter().flat_map(|v| v.to_le_bytes()));
            }
            std::fs::write(format!("{path}.ritz"), out).unwrap();
        }
        let mut low: Vec<f64> = converged.clone();
        low.dedup_by(|a, b| (*a - *b).abs() <= 1e-6 * *b);
        let low: Vec<String> = low.iter().take(5).map(|v| format!("{v:.3e}")).collect();
        println!(
            "{path} n={n} kept={} it={}{} lanczos={} zeros={zeros} unconverged={} lambda=[{lo:.3e}, {hi:.3e}] kappa={:.3e} lowest={}",
            reduced.n_reduced(),
            solved.iterations,
            if solved.converged { "" } else { "!" },
            ritz.len(),
            ritz.len() - zeros - converged.len(),
            hi / lo,
            low.join(","),
        );
    }
}

#[cfg(not(feature = "pair-schur-prototype"))]
fn main() {
    eprintln!("Enable --features pair-schur-prototype to run this experiment.");
}
