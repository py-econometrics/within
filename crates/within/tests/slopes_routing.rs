//! Cross-factor signed routing (#61): slope terms solving alongside other
//! factors through balanced/scaled signed subdomains, plus the frustration
//! error path (lifted by #62).

use within::{BuildError, Effect, LsmrOptions, PreconditionerConfig, SignedPair, Solver};

fn lcg(seed: &mut u64) -> u64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *seed
}

#[test]
fn two_factor_slope_solves_with_bounded_iterations() {
    // f (~200 levels, intercept + slope) alongside a binary g: the signed
    // (f-slope, g-int) pair is structurally balanced — centering makes each
    // whitened slope row's two cells opposite-signed.
    let n = 50_000;
    let n_f = 200u64;
    let mut seed = 42u64;
    let f: Vec<u32> = (0..n).map(|_| (lcg(&mut seed) % n_f) as u32).collect();
    let g: Vec<u32> = (0..n).map(|_| (lcg(&mut seed) % 2) as u32).collect();
    let z: Vec<f64> = (0..n)
        .map(|_| (lcg(&mut seed) % 2001) as f64 / 1000.0 - 1.0)
        .collect();
    let y: Vec<f64> = (0..n)
        .map(|i| {
            let fl = f[i] as f64;
            (fl * 0.013).sin()
                + (0.5 + (fl * 0.037).cos()) * z[i]
                + 0.8 * g[i] as f64
                + (i as f64 * 0.61).sin() * 0.3
        })
        .collect();

    let effects = vec![
        Effect::new(&f, true, [&z[..]]).expect("slope effect"),
        Effect::new(&g, true, []).expect("plain effect"),
    ];
    let r = Solver::new(effects, None, PreconditionerConfig::default())
        .expect("signed routing builds")
        .solve(&y, &LsmrOptions::default())
        .expect("solve");

    assert!(r.converged);
    assert!(r.iterations <= 50, "iterations = {}", r.iterations);
    assert!(r.unidentified.is_empty());
}

#[test]
fn unit_trends_plus_time_effects_boundary() {
    // Balanced panel, unit trends + time effects. Odd T puts the whitened
    // trend at exactly 0 for the middle period, so the signed pair keeps a
    // live positive-diagonal singleton (trivial 1×1 route).
    let (n_units, n_times) = (30usize, 9usize);
    let n = n_units * n_times;
    let unit: Vec<u32> = (0..n).map(|i| (i / n_times) as u32).collect();
    let time: Vec<u32> = (0..n).map(|i| (i % n_times) as u32).collect();
    let t: Vec<f64> = time.iter().map(|&k| k as f64).collect();
    let y: Vec<f64> = (0..n)
        .map(|i| {
            let u = unit[i] as f64;
            let k = time[i] as f64;
            (u * 0.31).sin() * 2.0
                + (0.1 + (u * 0.7).cos() * 0.05) * k
                + (k * 0.9).sin()
                + (i as f64 * 1.3).sin() * 0.2
        })
        .collect();

    let effects = vec![
        Effect::new(&unit, true, [&t[..]]).expect("trend effect"),
        Effect::new(&time, true, []).expect("time effect"),
    ];
    let r = Solver::new(effects, None, PreconditionerConfig::default())
        .expect("PSD-boundary routing builds")
        .solve(&y, &LsmrOptions::default())
        .expect("solve");

    assert!(r.converged);
    assert!(r.iterations <= 60, "iterations = {}", r.iterations);
}

#[test]
fn frustrated_component_errors_cleanly() {
    // Per-level means are exactly 0, so whitening keeps the cell signs:
    // rows (−,+,+) / (−,−,+) contain a negative 4-cycle. Exactly one signed
    // component exists, so the reported pair is deterministic.
    let f = [0u32, 0, 0, 1, 1, 1];
    let g = [0u32, 1, 2, 0, 1, 2];
    let z = [-2.0, 1.0, 1.0, -1.0, -1.0, 2.0];
    let effects = vec![
        Effect::new(&f, true, [&z[..]]).expect("slope effect"),
        Effect::new(&g, true, []).expect("plain effect"),
    ];

    let err = Solver::new(effects, None, PreconditionerConfig::default()).unwrap_err();
    assert!(err.to_string().contains("frustrated"));
    match err {
        BuildError::FrustratedComponent {
            pair:
                SignedPair {
                    term_q: 0,
                    column_q: 1,
                    term_r: 1,
                    column_r: 0,
                },
        } => {}
        other => panic!("expected FrustratedComponent for (f-slope, g-int), got: {other:?}"),
    }
}

#[test]
fn near_collinear_cross_term_direction_survives_routing() {
    // Relative surplus 5e-10 sat below the former 1e-9 SURPLUS_TOL; routing
    // this PD component as singular projected out the identified [1, -1]
    // direction, returning x ≈ 0 with converged = true.
    let c = 1.0 - 5e-10;
    let f = [0u32, 0];
    let z1 = [1.0, 0.0];
    let z2 = [c, (1.0f64 - c * c).sqrt()];
    let y: Vec<f64> = (0..2).map(|i| z1[i] - z2[i]).collect();
    let effects = vec![
        Effect::new(&f, false, [&z1[..]]).unwrap(),
        Effect::new(&f, false, [&z2[..]]).unwrap(),
    ];
    let r = Solver::new(effects, None, PreconditionerConfig::default())
        .expect("near-collinear pair builds")
        .solve(&y, &LsmrOptions::default())
        .expect("solve");
    assert!(r.converged);
    assert!(
        (r.x[0] - 1.0).abs() < 1e-6 && (r.x[1] + 1.0).abs() < 1e-6,
        "x = {:?}",
        r.x
    );
    let rnorm: f64 = r.demeaned.iter().map(|v| v * v).sum::<f64>().sqrt();
    let ynorm: f64 = y.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(
        rnorm <= 1e-6 * ynorm,
        "relative residual {:e}",
        rnorm / ynorm
    );
}

#[test]
fn surplus_component_sampled_matches_exact_reduction() {
    // DGP in lockstep with `positive_slope_only_pair_grounds_beyond_dense_threshold`
    // (src/solver/tests.rs), which pins that this design grounds a surplus-carrying
    // component whose kept side exceeds the dense threshold: the default arm below
    // exercises the sparse SAMPLED reduction on it; `approx_schur: None` is the
    // exact-reduction reference (#83).
    let n = 8000usize;
    let f: Vec<u32> = (0..n).map(|i| (i % 80) as u32).collect();
    let g: Vec<u32> = (0..n).map(|i| ((i / 80) % 40) as u32).collect();
    let z: Vec<f64> = (0..n)
        .map(|i| 0.5 + ((i * 13) % 100) as f64 / 100.0)
        .collect();
    let y: Vec<f64> = (0..n)
        .map(|i| {
            let fl = f[i] as f64;
            (0.5 + (fl * 0.037).cos()) * z[i]
                + 0.8 * (g[i] as f64 * 0.11).sin()
                + (i as f64 * 0.61).sin() * 0.3
        })
        .collect();

    let solve = |approx_schur| {
        let effects = vec![
            Effect::new(&f, false, [&z[..]]).expect("slope effect"),
            Effect::new(&g, true, []).expect("plain effect"),
        ];
        let config = PreconditionerConfig::Additive {
            local_solver: within::config::LocalSolverConfig {
                approx_schur,
                ..Default::default()
            },
            reduction: Default::default(),
        };
        Solver::new(effects, None, config)
            .expect("build")
            .solve(&y, &LsmrOptions::default())
            .expect("solve")
    };

    let sampled = solve(Some(within::config::ApproxSchurConfig::default()));
    let exact = solve(None);
    assert!(sampled.converged, "sampled arm did not converge");
    assert!(exact.converged, "exact arm did not converge");
    assert!(sampled.unidentified.is_empty());
    assert!(
        sampled.iterations <= 30,
        "iterations = {}",
        sampled.iterations
    );
    // The demeaned response is the kernel-invariant deliverable; both arms
    // must land on the same one.
    for (i, ((&s, &e), &yi)) in sampled
        .demeaned
        .iter()
        .zip(&exact.demeaned)
        .zip(&y)
        .enumerate()
    {
        assert!(
            (s - e).abs() <= 1e-6 * (1.0 + yi.abs()),
            "demeaned[{i}]: sampled {s} vs exact {e}"
        );
    }
}
