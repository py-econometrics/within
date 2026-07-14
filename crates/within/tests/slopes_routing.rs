//! Cross-factor signed routing (#61): slope terms solving alongside other
//! factors through balanced/scaled signed subdomains, with frustrated
//! components solving through their Gremban double cover (#62).

use within::{Effect, LsmrOptions, Preconditioner, PreconditionerConfig, SchurMode, Solver};

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
fn frustrated_component_solves_via_cover() {
    // Per-level means are exactly 0, so whitening keeps the cell signs:
    // rows (−,+,+) / (−,−,+) contain a negative 4-cycle — the minimal
    // frustrated component. The cover path must reproduce the exact
    // least-squares projection: residuals orthogonal to every design column,
    // and the reported coefficients reconstruct the fit.
    let f = [0u32, 0, 0, 1, 1, 1];
    let g = [0u32, 1, 2, 0, 1, 2];
    let z = [-2.0, 1.0, 1.0, -1.0, -1.0, 2.0];
    let y = [1.0, -2.0, 0.5, 3.0, -1.5, 2.5];
    let effects = vec![
        Effect::new(&f, true, [&z[..]]).expect("slope effect"),
        Effect::new(&g, true, []).expect("plain effect"),
    ];

    let r = Solver::new(effects, None, PreconditionerConfig::default())
        .expect("frustrated component builds via its Gremban cover")
        .solve(&y, &LsmrOptions::default())
        .expect("solve");
    assert!(r.converged);
    assert!(r.unidentified.is_empty());

    for level in 0..2u32 {
        for slope in [None, Some(&z)] {
            let dot: f64 = (0..y.len())
                .filter(|&i| f[i] == level)
                .map(|i| r.demeaned[i] * slope.map_or(1.0, |z| z[i]))
                .sum();
            assert!(dot.abs() < 1e-6, "f level {level}: residual·column = {dot}");
        }
    }
    for level in 0..3u32 {
        let dot: f64 = (0..y.len())
            .filter(|&i| g[i] == level)
            .map(|i| r.demeaned[i])
            .sum();
        assert!(dot.abs() < 1e-6, "g level {level}: residual·column = {dot}");
    }
    for i in 0..y.len() {
        let fitted = r.x[f[i] as usize] + r.x[2 + f[i] as usize] * z[i] + r.x[4 + g[i] as usize];
        assert!(
            (y[i] - r.demeaned[i] - fitted).abs() < 1e-6,
            "fitted value {i}"
        );
    }
}

#[test]
fn frustrated_two_factor_slope_solves_with_bounded_iterations() {
    // Same shape as the balanced two-factor case, but g has five levels:
    // whitened slope rows are zero-sum, so a ≥3-level partner generically
    // closes negative cycles — the realistic frustrated regime.
    let n = 50_000;
    let n_f = 200u64;
    let mut seed = 47u64;
    let f: Vec<u32> = (0..n).map(|_| (lcg(&mut seed) % n_f) as u32).collect();
    let g: Vec<u32> = (0..n).map(|_| (lcg(&mut seed) % 5) as u32).collect();
    let z: Vec<f64> = (0..n)
        .map(|_| (lcg(&mut seed) % 2001) as f64 / 1000.0 - 1.0)
        .collect();
    let y: Vec<f64> = (0..n)
        .map(|i| {
            let fl = f[i] as f64;
            (fl * 0.013).sin()
                + (0.5 + (fl * 0.037).cos()) * z[i]
                + 0.4 * g[i] as f64
                + (i as f64 * 0.61).sin() * 0.3
        })
        .collect();

    let effects = vec![
        Effect::new(&f, true, [&z[..]]).expect("slope effect"),
        Effect::new(&g, true, []).expect("plain effect"),
    ];
    let r = Solver::new(effects, None, PreconditionerConfig::default())
        .expect("frustrated routing builds")
        .solve(&y, &LsmrOptions::default())
        .expect("solve");

    assert!(r.converged);
    assert!(r.iterations <= 80, "iterations = {}", r.iterations);
    assert!(r.unidentified.is_empty());
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
    // exercises the sparse SAMPLED reduction on it; `SchurMode::Exact` is the
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

    let solve = |schur: SchurMode| {
        let effects = vec![
            Effect::new(&f, false, [&z[..]]).expect("slope effect"),
            Effect::new(&g, true, []).expect("plain effect"),
        ];
        let config = PreconditionerConfig::Additive {
            local_solver: within::config::LocalSolverConfig {
                schur,
                ..Default::default()
            },
            reduction: Default::default(),
        };
        Solver::new(effects, None, config)
            .expect("build")
            .solve(&y, &LsmrOptions::default())
            .expect("solve")
    };

    let sampled = solve(SchurMode::Approximate(
        within::config::ApproxSchurConfig::default(),
    ));
    let exact = solve(SchurMode::Exact);
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

/// Regression for #98: with the default (additive Schwarz) preconditioner, a
/// level observed exactly once in a slope-carrying term that is not the first
/// term used to crash LSMR with a preconditioner one column short of the
/// operator. That singleton-slope direction is unidentified — a structural-zero
/// operator column no subdomain covers — yet it must still count toward the
/// preconditioner's shape. The solve must succeed and land the same identified
/// fit the `Off`/`Diagonal` preconditioners already produce.
#[test]
fn singleton_level_in_non_first_slope_term_solves_under_default() {
    // firm level 2 is observed exactly once (obs 0); the slope sits on the
    // second (non-first) term. n_dofs = 3 (worker) + 3 + 3 (firm intercept +
    // slope) = 9, and the firm-slope level-2 column (global index 8) is last,
    // so an omitted preconditioner column shows up as a shape mismatch.
    let worker = [0u32, 0, 1, 1, 2, 2];
    let firm = [2u32, 0, 0, 1, 1, 0];
    let x = [0.5, -1.0, 0.3, 2.0, -0.7, 1.1];
    let y = [1.0, 2.0, 3.0, 1.5, 0.4, -0.2];

    let effects = || {
        vec![
            Effect::new(&worker[..], true, []).expect("plain effect"),
            Effect::new(&firm[..], true, [&x[..]]).expect("slope effect"),
        ]
    };

    let solver = Solver::new(effects(), None, PreconditionerConfig::default())
        .expect("default preconditioner builds");

    // The preconditioner must match the operator's column count, including the
    // uncovered structural-zero singleton-slope direction.
    let precond = solver
        .preconditioner()
        .expect("default has a preconditioner");
    assert_eq!(precond.ncols(), solver.n_dofs());
    assert_eq!(precond.nrows(), solver.n_dofs());

    let r = solver
        .solve(&y, &LsmrOptions::default())
        .expect("default solve");
    assert!(r.converged);
    assert_eq!(
        r.unidentified
            .iter()
            .map(|d| (d.term, d.level, d.column))
            .collect::<Vec<_>>(),
        vec![(1, 2, 1)],
    );

    // Same identified fit as Off/Diagonal: the demeaned residual y - Dx is
    // gauge-invariant, so it agrees even though the raw coefficient vectors
    // (with within-factor gauge freedom) need not.
    for cfg in [PreconditionerConfig::Off, PreconditionerConfig::Diagonal] {
        let alt = Solver::new(effects(), None, cfg)
            .expect("alt preconditioner builds")
            .solve(&y, &LsmrOptions::default())
            .expect("alt solve");
        for (i, (&d, &a)) in r.demeaned.iter().zip(&alt.demeaned).enumerate() {
            assert!(
                (d - a).abs() < 1e-8,
                "fit mismatch at obs {i}: default {d} vs alt {a}"
            );
        }
    }

    // Reuse-safe: the preconditioner round-trips through postcard and the
    // reloaded copy carries the full dimension, so a fresh solver accepts it
    // (a dropped n_dofs would trip Solver::new's dimension check).
    let bytes = postcard::to_stdvec(precond).expect("serialize");
    let restored: Preconditioner = postcard::from_bytes(&bytes).expect("deserialize");
    assert_eq!(restored.ncols(), solver.n_dofs());
    let reused = Solver::new(effects(), None, restored)
        .expect("reused preconditioner accepted")
        .solve(&y, &LsmrOptions::default())
        .expect("reused solve");
    for (i, (&a, &b)) in r.x.iter().zip(&reused.x).enumerate() {
        assert!(
            (a - b).abs() < 1e-9,
            "reuse coefficient drift at {i}: {a} vs {b}"
        );
    }
}
