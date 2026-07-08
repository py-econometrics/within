//! Cross-factor signed routing (#61): slope terms solving alongside other
//! factors through balanced/scaled signed subdomains, plus the frustration
//! error path (lifted by #62).

use within::{BuildError, Effect, LsmrOptions, PreconditionerConfig, SolveResult, Solver};

fn lcg(seed: &mut u64) -> u64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *seed
}

/// One channel of a term as seen by the projection oracle: its level column,
/// loading (`None` ≡ 1, an intercept), and level count.
struct ChannelColumn<'a> {
    levels: &'a [u32],
    loading: Option<&'a [f64]>,
    n_levels: usize,
}

/// Gauge-invariant projection oracle: the residual is orthogonal to every
/// design column — per level of each channel, `Σ demeaned_i · col_i ≈ 0`.
fn assert_residual_orthogonality(r: &SolveResult, channels: &[ChannelColumn<'_>]) {
    let scale: f64 = r.demeaned.iter().map(|v| v * v).sum::<f64>().sqrt();
    let tol = 1e-5 * (1.0 + scale);
    for (c, ch) in channels.iter().enumerate() {
        for level in 0..ch.n_levels {
            let dot: f64 = ch
                .levels
                .iter()
                .enumerate()
                .filter(|(_, &l)| l as usize == level)
                .map(|(i, _)| r.demeaned[i] * ch.loading.map_or(1.0, |z| z[i]))
                .sum();
            assert!(
                dot.abs() <= tol,
                "channel {c} level {level}: residual·column = {dot} (tol {tol})"
            );
        }
    }
}

#[test]
fn two_factor_slope_matches_projection_with_bounded_iterations() {
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
    assert_residual_orthogonality(
        &r,
        &[
            ChannelColumn {
                levels: &f,
                loading: None,
                n_levels: n_f as usize,
            },
            ChannelColumn {
                levels: &f,
                loading: Some(&z),
                n_levels: n_f as usize,
            },
            ChannelColumn {
                levels: &g,
                loading: None,
                n_levels: 2,
            },
        ],
    );
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
    assert_residual_orthogonality(
        &r,
        &[
            ChannelColumn {
                levels: &unit,
                loading: None,
                n_levels: n_units,
            },
            ChannelColumn {
                levels: &unit,
                loading: Some(&t),
                n_levels: n_units,
            },
            ChannelColumn {
                levels: &time,
                loading: None,
                n_levels: n_times,
            },
        ],
    );
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
            term_q: 0,
            column_q: 1,
            term_r: 1,
            column_r: 0,
        } => {}
        other => panic!("expected FrustratedComponent for (f-slope, g-int), got: {other:?}"),
    }
}
