//! Varying slopes on one factor (#59, #60): coefficient recovery in the
//! user's parametrization, fitted-value invariance under the internal
//! reparametrization, and deterministic rank-drop reporting.

use within::{Effect, LsmrOptions, PreconditionerConfig, SolveResult, Solver};

const TOL: f64 = 1e-6;

fn assert_close(actual: f64, expected: f64, what: &str) {
    assert!(
        (actual - expected).abs() <= TOL * (1.0 + expected.abs()),
        "{what}: got {actual}, expected {expected}"
    );
}

/// Per-level weighted OLS of `y` on `[1, z]` — the un-reparametrized
/// reference fit for an `f[z]` term. Levels with degenerate `z` are the
/// caller's concern.
fn per_level_intercept_slope(
    levels: &[u32],
    z: &[f64],
    y: &[f64],
    w: Option<&[f64]>,
    n_levels: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut s = vec![[0.0f64; 5]; n_levels];
    for i in 0..levels.len() {
        let wi = w.map_or(1.0, |w| w[i]);
        let acc = &mut s[levels[i] as usize];
        acc[0] += wi;
        acc[1] += wi * z[i];
        acc[2] += wi * z[i] * z[i];
        acc[3] += wi * y[i];
        acc[4] += wi * z[i] * y[i];
    }
    let mut a = vec![0.0; n_levels];
    let mut b = vec![0.0; n_levels];
    for l in 0..n_levels {
        let [sw, sz, szz, sy, szy] = s[l];
        let denom = sw * szz - sz * sz;
        b[l] = (sw * szy - sz * sy) / denom;
        a[l] = (sy - b[l] * sz) / sw;
    }
    (a, b)
}

fn synthetic_y(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| (i as f64 * 0.9 - 1.0).sin() * 2.0 + 0.5)
        .collect()
}

fn solve_with(
    levels: &[u32],
    z: &[f64],
    intercept: bool,
    weights: Option<Vec<f64>>,
    y: &[f64],
    config: &PreconditionerConfig,
) -> SolveResult {
    Solver::new(
        vec![Effect::new(levels, intercept, [z]).expect("effect")],
        weights,
        config,
    )
    .expect("solver")
    .solve(y, &LsmrOptions::default())
    .expect("solve")
}

fn solve_single(
    levels: &[u32],
    z: &[f64],
    intercept: bool,
    weights: Option<Vec<f64>>,
    y: &[f64],
) -> SolveResult {
    solve_with(
        levels,
        z,
        intercept,
        weights,
        y,
        &PreconditionerConfig::default(),
    )
}

/// `unidentified` as comparable `(term, level, column)` triples.
fn drops(r: &SolveResult) -> Vec<(usize, usize, usize)> {
    r.unidentified
        .iter()
        .map(|d| (d.channel.term, d.level, d.channel.column))
        .collect()
}

#[test]
fn single_slope_recovers_per_level_ols_and_fitted_values() {
    // Non-monotonic factor so the locality sort is genuinely exercised.
    let levels: Vec<u32> = vec![2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1];
    let z: Vec<f64> = vec![
        0.5, -1.2, 3.3, 1.5, 0.7, -2.1, 2.5, 1.9, 0.4, -0.5, -1.0, 1.1,
    ];
    let y = synthetic_y(levels.len());

    let r = solve_single(&levels, &z, true, None, &y);
    assert!(r.converged);
    assert!(r.unidentified.is_empty());

    let (a, b) = per_level_intercept_slope(&levels, &z, &y, None, 3);
    for l in 0..3 {
        assert_close(r.x[l], a[l], &format!("intercept of level {l}"));
        assert_close(r.x[3 + l], b[l], &format!("slope of level {l}"));
    }
    // Fitted values must equal the un-reparametrized fit.
    for i in 0..levels.len() {
        let l = levels[i] as usize;
        assert_close(
            y[i] - r.demeaned[i],
            a[l] + b[l] * z[i],
            &format!("fitted value {i}"),
        );
    }
}

#[test]
fn slope_only_term_recovers_per_level_projection() {
    let levels: Vec<u32> = vec![0, 0, 0, 1, 1, 1, 2, 2, 2];
    let z: Vec<f64> = vec![1.0, 2.0, -1.5, 0.5, 3.0, 1.0, -2.0, 0.7, 1.3];
    let y = synthetic_y(levels.len());

    let r = solve_single(&levels, &z, false, None, &y);
    assert!(r.converged);
    assert!(r.unidentified.is_empty());
    assert_eq!(r.x.len(), 3);

    // Reference: per-level projection onto z alone, b = Σzy / Σz².
    for l in 0..3 {
        let (szy, szz) = levels
            .iter()
            .zip(z.iter().zip(y.iter()))
            .filter(|(&li, _)| li as usize == l)
            .fold((0.0, 0.0), |(szy, szz), (_, (&zi, &yi))| {
                (szy + zi * yi, szz + zi * zi)
            });
        assert_close(r.x[l], szy / szz, &format!("slope of level {l}"));
    }
    for i in 0..levels.len() {
        let l = levels[i] as usize;
        assert_close(
            y[i] - r.demeaned[i],
            r.x[l] * z[i],
            &format!("fitted value {i}"),
        );
    }
}

#[test]
fn rank_drops_report_deterministically_with_exact_zeros() {
    // Level 1 never occurs; level 2's slope is constant, so only its slope drops.
    let levels: Vec<u32> = vec![0, 0, 0, 2, 2, 2, 3, 3, 3];
    let z: Vec<f64> = vec![1.0, 2.0, -1.5, 2.5, 2.5, 2.5, -2.0, 0.7, 1.3];
    let y = synthetic_y(levels.len());
    let run = |config: &PreconditionerConfig| solve_with(&levels, &z, true, None, &y, config);

    let r = run(&PreconditionerConfig::default());
    assert!(r.converged);
    // Ascending (level, column) order.
    assert_eq!(drops(&r), [(0, 1, 0), (0, 1, 1), (0, 2, 1)]);

    // Minimal-norm 0 at exactly the dropped slots; everything else finite.
    for slot in [1, 4 + 1, 4 + 2] {
        assert_eq!(r.x[slot], 0.0);
    }
    assert!(r.x.iter().all(|v| v.is_finite()));

    // The constant level degrades to intercept-only: a = mean(y within level).
    assert_close(
        r.x[2],
        (y[3] + y[4] + y[5]) / 3.0,
        "constant level intercept",
    );
    let (a, b) = per_level_intercept_slope(&levels, &z, &y, None, 4);
    for l in [0usize, 3] {
        assert_close(r.x[l], a[l], &format!("intercept of level {l}"));
        assert_close(r.x[4 + l], b[l], &format!("slope of level {l}"));
    }

    // The explicit diagonal preconditioner agrees with the default path.
    let diag = run(&PreconditionerConfig::Diagonal);
    assert_eq!(diag.unidentified, r.unidentified);
    for (i, (d, j)) in r.x.iter().zip(diag.x.iter()).enumerate() {
        assert_close(*j, *d, &format!("coefficient {i}"));
    }
}

#[test]
fn weighted_solve_matches_closed_form_and_masks_zero_weight_rows() {
    // Identification is judged over positive-weight rows, so level 2's slope drops.
    let levels: Vec<u32> = vec![0, 1, 0, 1, 0, 1, 2, 2, 2];
    let z: Vec<f64> = vec![0.5, -1.2, 3.3, 1.5, 0.7, -2.1, 5.0, 5.0, 7.0];
    let w: Vec<f64> = vec![1.0, 0.5, 2.0, 1.5, 3.0, 1.0, 1.0, 1.0, 0.0];
    let y = synthetic_y(levels.len());

    let r = solve_single(&levels, &z, true, Some(w.clone()), &y);
    assert!(r.converged);
    assert_eq!(drops(&r), [(0, 2, 1)]);
    assert_eq!(r.x[3 + 2], 0.0);

    let (a, b) = per_level_intercept_slope(&levels, &z, &y, Some(&w), 3);
    for l in 0..2 {
        assert_close(r.x[l], a[l], &format!("intercept of level {l}"));
        assert_close(r.x[3 + l], b[l], &format!("slope of level {l}"));
    }
    // The masked level's intercept is the weighted mean over its w>0 rows.
    assert_close(r.x[2], (y[6] + y[7]) / 2.0, "masked level intercept");
}

#[test]
fn zero_weight_garbage_does_not_poison_identification() {
    let levels: Vec<u32> = vec![0, 0, 0, 1, 1, 1];
    // Pins the left-to-right `w * z * z`: `0 * z` kills it before the square overflows.
    let z: Vec<f64> = vec![1.0, 2.0, 3.0, 5.0, 7.0, 1e300];
    let w: Vec<f64> = vec![1.0, 1.0, 1.0, 1.0, 1.0, 0.0];
    let y = synthetic_y(levels.len());
    let run =
        |config: &PreconditionerConfig| solve_with(&levels, &z, true, Some(w.clone()), &y, config);

    let r = run(&PreconditionerConfig::default());
    assert!(r.unidentified.is_empty());
    let (a, b) = per_level_intercept_slope(&levels, &z, &y, Some(&w), 2);
    for l in 0..2 {
        assert_close(r.x[l], a[l], &format!("intercept of level {l}"));
        assert_close(r.x[2 + l], b[l], &format!("slope of level {l}"));
    }

    let diag = run(&PreconditionerConfig::Diagonal);
    for (i, (d, j)) in r.x.iter().zip(diag.x.iter()).enumerate() {
        assert_close(*j, *d, &format!("coefficient {i}"));
    }
}

#[test]
fn batch_solve_shares_unidentified_and_back_transforms_each_rhs() {
    let levels: Vec<u32> = vec![0, 0, 0, 1, 1, 1];
    let z: Vec<f64> = vec![1.0, 2.0, 3.0, 2.5, 2.5, 2.5];
    let y1 = synthetic_y(levels.len());
    let y2: Vec<f64> = y1.iter().map(|v| v * -1.5 + 0.3).collect();

    let solver = Solver::new(
        vec![Effect::new(&levels, true, [&z[..]]).expect("effect")],
        None,
        PreconditionerConfig::default(),
    )
    .expect("solver");
    let opts = LsmrOptions::default();
    let batch = solver.solve_batch(&[&y1, &y2], &opts).expect("batch");

    let batch_drops: Vec<_> = batch
        .unidentified
        .iter()
        .map(|d| (d.channel.term, d.level, d.channel.column))
        .collect();
    assert_eq!(batch_drops, [(0, 1, 1)]);
    // Each RHS block is bit-identical to its single solve, back-transform included.
    let single1 = solver.solve(&y1, &opts).expect("solve y1");
    let single2 = solver.solve(&y2, &opts).expect("solve y2");
    let n = single1.x.len();
    assert_eq!(&batch.x[..n], &single1.x[..]);
    assert_eq!(&batch.x[n..], &single2.x[..]);
}

#[test]
fn three_slopes_solve_bounded_with_exact_rank_drops() {
    // Strongly correlated slopes, and level 1's z2/z3 are exactly collinear on top.
    let n = 24;
    let levels: Vec<u32> = (0..n).map(|i| (i % 3) as u32).collect();
    let z1: Vec<f64> = (0..n).map(|i| (i as f64 * 0.37).sin() * 2.0).collect();
    let z2: Vec<f64> = (0..n)
        .map(|i| 0.95 * z1[i] + 0.1 * (i as f64 * 0.9 + 0.2).cos())
        .collect();
    let z3: Vec<f64> = (0..n)
        .map(|i| {
            if i % 3 == 1 {
                2.0 * z2[i]
            } else {
                -0.8 * z1[i] + 0.15 * (i as f64 * 1.7).sin() + 0.1
            }
        })
        .collect();
    let y = synthetic_y(n);

    let r = Solver::new(
        vec![Effect::new(&levels, true, [&z1[..], &z2, &z3]).expect("effect")],
        None,
        PreconditionerConfig::default(),
    )
    .expect("solver")
    .solve(&y, &LsmrOptions::default())
    .expect("solve");
    assert!(r.converged);
    assert!(
        r.iterations <= 30,
        "conditioning cliff: {} iterations",
        r.iterations
    );

    // The pivot keeps the larger-variance z3, so z2's column reads exactly 0.
    assert_eq!(drops(&r), [(0, 1, 2)]);
    assert_eq!(r.x[2 * 3 + 1], 0.0);

    // The exact projection and the reported coefficients together pin them with no oracle.
    for l in 0..3 {
        for col in [None, Some(&z1), Some(&z2), Some(&z3)] {
            let dot: f64 = (0..n)
                .filter(|&i| levels[i] as usize == l)
                .map(|i| r.demeaned[i] * col.map_or(1.0, |z| z[i]))
                .sum();
            assert!(dot.abs() < 1e-6, "level {l}: residual·column = {dot}");
        }
    }
    for i in 0..n {
        let l = levels[i] as usize;
        let fitted = r.x[l] + r.x[3 + l] * z1[i] + r.x[6 + l] * z2[i] + r.x[9 + l] * z3[i];
        assert_close(y[i] - r.demeaned[i], fitted, &format!("fitted value {i}"));
    }
}
