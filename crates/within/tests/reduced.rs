use rstest::rstest;
use within::{
    BatchSolveResult, BuildError, Channel, CoefficientAddress, Design, Effect, IntoDesign,
    LsmrOptions, PreconditionerConfig, PreconditionerInput, SolveResult, Solver, WithinError,
};

#[derive(Debug, Clone, Copy)]
enum System {
    Full,
    Reduced,
}

fn solver<'a>(
    design: impl IntoDesign<'a>,
    weights: Option<&[f64]>,
    preconditioner: impl Into<PreconditionerInput>,
    system: System,
) -> Result<Solver<'a>, BuildError> {
    match system {
        System::Full => Solver::new(design, weights, preconditioner),
        System::Reduced => Solver::reduced(design, weights, preconditioner),
    }
}

fn solve<'a, 'o>(
    design: impl IntoDesign<'a>,
    y: &[f64],
    weights: Option<&[f64]>,
    lsmr: impl Into<Option<&'o LsmrOptions>>,
    preconditioner: impl Into<PreconditionerInput>,
    system: System,
) -> Result<SolveResult, WithinError> {
    solver(design, weights, preconditioner, system)?.solve(y, lsmr)
}

fn solve_batch<'a, 'o>(
    design: impl IntoDesign<'a>,
    ys: &[&[f64]],
    weights: Option<&[f64]>,
    lsmr: impl Into<Option<&'o LsmrOptions>>,
    preconditioner: impl Into<PreconditionerInput>,
    system: System,
) -> Result<BatchSolveResult, WithinError> {
    solver(design, weights, preconditioner, system)?.solve_batch(ys, lsmr)
}

fn noise(k: usize) -> f64 {
    let mut h = (k as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x2545_F491_4F6C_DD1D;
    h ^= h >> 29;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 32;
    (h >> 11) as f64 / (1u64 << 53) as f64 - 0.5
}

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

const N: usize = 3000;

/// Worker-, firm- and year-like factors; the 300-level first factor is eliminated.
fn factors(n_factors: usize) -> Vec<Vec<u32>> {
    let worker = (0..N).map(|i| (i % 300) as u32).collect();
    let firm = (0..N).map(|i| ((i * 7 + i / 300) % 40) as u32).collect();
    let year = (0..N).map(|i| ((i / 7) % 10) as u32).collect();
    [worker, firm, year].into_iter().take(n_factors).collect()
}

fn effects(factors: &[Vec<u32>]) -> Vec<Effect<'_>> {
    factors
        .iter()
        .map(|f| Effect::new(f, true, []).expect("effect"))
        .collect()
}

/// Row `i` belongs to worker 11 or firm 5, the levels whose weights are all zero.
fn unidentified_row(factors: &[Vec<u32>], i: usize) -> bool {
    factors[0][i] == 11 || factors.get(1).is_some_and(|f| f[i] == 5)
}

/// Zero on every seventh row and on every row of worker 11 and firm 5, which leaves both
/// an eliminated and a kept level unidentified.
fn weights(factors: &[Vec<u32>]) -> Vec<f64> {
    (0..N)
        .map(|i| {
            if i % 7 == 0 || unidentified_row(factors, i) {
                0.0
            } else {
                0.5 + noise(N + i).abs()
            }
        })
        .collect()
}

fn response(k: usize) -> Vec<f64> {
    (0..N).map(|i| noise(k * N + i)).collect()
}

fn lsmr() -> LsmrOptions {
    LsmrOptions {
        tol: 1e-12,
        maxiter: 20_000,
        ..LsmrOptions::default()
    }
}

/// Rows touching a level whose weights are all zero have gauge-dependent fits in either system.
fn identified_rows(factors: &[Vec<u32>], weighted: bool) -> Vec<bool> {
    (0..N)
        .map(|i| !weighted || !unidentified_row(factors, i))
        .collect()
}

/// One response solved on the reduced system and on the full one.
fn reduced_and_full(
    factors: &[Vec<u32>],
    y: &[f64],
    w: Option<&[f64]>,
    options: &LsmrOptions,
    preconditioner: &PreconditionerConfig,
) -> (SolveResult, SolveResult) {
    let solve_on =
        |system| solve(effects(factors), y, w, options, preconditioner, system).expect("solve");
    (solve_on(System::Reduced), solve_on(System::Full))
}

fn assert_same_fit(reduced: &[f64], full: &[f64], identified: &[bool], y: &[f64]) {
    let gap: Vec<f64> = (reduced.iter().zip(full).zip(identified))
        .map(|((r, f), &keep)| if keep { r - f } else { 0.0 })
        .collect();
    assert!(
        norm(&gap) <= 1e-8 * norm(y),
        "‖demeaned gap‖ = {:.2e} of ‖y‖ = {:.2e}",
        norm(&gap),
        norm(y)
    );
}

#[rstest]
fn reduced_and_full_fits_agree(
    #[values(2, 3)] n_factors: usize,
    #[values(false, true)] weighted: bool,
    #[values(PreconditionerConfig::Off, PreconditionerConfig::Diagonal)]
    preconditioner: PreconditionerConfig,
) {
    let factors = factors(n_factors);
    let w = weighted.then(|| weights(&factors));
    let y = response(1);
    let solve_on = |system, preconditioner: &PreconditionerConfig| {
        let effects = effects(&factors);
        solve(effects, &y, w.as_deref(), &lsmr(), preconditioner, system).expect("solve")
    };
    let reduced = solve_on(System::Reduced, &preconditioner);
    let full = solve_on(System::Full, &PreconditionerConfig::Diagonal);
    assert!(reduced.converged && full.converged, "{reduced:?}\n{full:?}");
    assert_eq!(reduced.unidentified, full.unidentified);
    let identified = identified_rows(&factors, weighted);
    assert_same_fit(&reduced.demeaned, &full.demeaned, &identified, &y);
    if weighted {
        // The eliminated level with no weight is unidentified, and recovery leaves it at 0.
        let worker_11 = CoefficientAddress {
            channel: Channel { term: 0, column: 0 },
            level: 11,
        };
        assert_eq!(
            reduced.x[reduced.layout.index(worker_11).expect("slot")],
            0.0
        );
    }
}

#[rstest]
fn reduced_and_full_batches_agree(#[values(2, 3)] n_factors: usize) {
    let factors = factors(n_factors);
    let w = weights(&factors);
    let ys = [response(2), response(3)];
    let ys: Vec<&[f64]> = ys.iter().map(Vec::as_slice).collect();
    let batch = |system| {
        let effects = effects(&factors);
        let preconditioner = PreconditionerConfig::Diagonal;
        solve_batch(effects, &ys, Some(&w), &lsmr(), &preconditioner, system).expect("batch")
    };
    let (reduced, full) = (batch(System::Reduced), batch(System::Full));
    assert_eq!(reduced.unidentified, full.unidentified);
    let identified = identified_rows(&factors, true);
    for (k, y) in ys.iter().enumerate() {
        assert!(reduced.converged[k] && full.converged[k]);
        assert_same_fit(reduced.demeaned(k), full.demeaned(k), &identified, y);
    }
}

#[rstest]
fn one_factor_is_solved_by_elimination_alone(#[values(false, true)] weighted: bool) {
    let factors = factors(1);
    let w = weighted.then(|| weights(&factors));
    let y = response(4);
    // With no kept factor, the Schwarz default settles like any one-factor system.
    let result = solve(
        effects(&factors),
        &y,
        w.as_deref(),
        None,
        None,
        System::Reduced,
    )
    .expect("solve");
    assert!(result.converged);
    assert_eq!(result.iterations, 0);

    let weight = |i: usize| w.as_ref().map_or(1.0, |w| w[i]);
    let mut sums = vec![(0.0, 0.0); 300];
    for (i, &level) in factors[0].iter().enumerate() {
        sums[level as usize].0 += weight(i) * y[i];
        sums[level as usize].1 += weight(i);
    }
    let mean = |l: usize| {
        let (wy, w) = sums[l];
        if w > 0.0 {
            wy / w
        } else {
            0.0
        }
    };
    for (l, &x) in result.x.iter().enumerate() {
        assert!(
            (x - mean(l)).abs() <= 1e-14,
            "level {l}: {x} vs {}",
            mean(l)
        );
    }
    for (i, &d) in result.demeaned.iter().enumerate() {
        let expected = y[i] - mean(factors[0][i] as usize);
        assert!((d - expected).abs() <= 1e-14, "row {i}: {d} vs {expected}");
    }
}

/// With no kept factor nothing is left to precondition: Schwarz settles, as on one full factor.
#[rstest]
fn schwarz_on_one_factor_matches_its_plain_counterpart(
    #[values(
        (PreconditionerConfig::default(), PreconditionerConfig::Diagonal),
        (PreconditionerConfig::Additive { local_solver: Default::default() }, PreconditionerConfig::Off)
    )]
    (schwarz, plain): (PreconditionerConfig, PreconditionerConfig),
) {
    let factors = factors(1);
    let w = weights(&factors);
    let y = response(7);
    let solve_with = |preconditioner: &PreconditionerConfig| {
        let effects = effects(&factors);
        solve(
            effects,
            &y,
            Some(&w),
            &lsmr(),
            preconditioner,
            System::Reduced,
        )
        .expect("solve")
    };
    let (schwarz, plain) = (solve_with(&schwarz), solve_with(&plain));
    assert!(schwarz.converged);
    assert_eq!(schwarz.iterations, plain.iterations);
    assert_eq!(schwarz.demeaned, plain.demeaned);
}

#[test]
fn a_prebuilt_preconditioner_must_fit_the_reduced_system() {
    let factors = factors(1);
    let full = solver(
        effects(&factors),
        None,
        &PreconditionerConfig::Diagonal,
        System::Full,
    )
    .expect("full");
    let prebuilt = full.preconditioner().expect("diagonal").clone();
    let reduced = solver(effects(&factors), None, prebuilt, System::Reduced);
    assert!(
        matches!(
            reduced,
            Err(BuildError::PreconditionerDimensionMismatch { expected: 0, .. })
        ),
        "{reduced:?}"
    );
}

/// A kept factor's levels are coupled through the eliminated one, and the reduced system does
/// not build Schwarz yet.
#[rstest]
fn schwarz_is_not_yet_built_on_the_reduced_system(
    #[values(2, 3)] n_factors: usize,
    #[values(
        None,
        Some(PreconditionerConfig::Additive { local_solver: Default::default() }),
        Some(PreconditionerConfig::default())
    )]
    preconditioner: Option<PreconditionerConfig>,
) {
    let factors = factors(n_factors);
    let y = response(5);
    let result = solve(
        effects(&factors),
        &y,
        None,
        None,
        preconditioner.as_ref(),
        System::Reduced,
    );
    assert!(
        matches!(
            result,
            Err(WithinError::Build(BuildError::SchwarzOnReducedSystem))
        ),
        "{result:?}"
    );
}

/// A Schwarz map built on another design's full system is refused even when its size fits.
#[test]
fn a_prebuilt_schwarz_map_is_refused_on_the_reduced_system() {
    let n = 200;
    let pair = vec![
        (0..n).map(|i| (i % 3) as u32).collect::<Vec<_>>(),
        (0..n).map(|i| ((i / 3) % 4) as u32).collect(),
    ];
    let additive = PreconditionerConfig::Additive {
        local_solver: Default::default(),
    };
    let full = solver(effects(&pair), None, &additive, System::Full).expect("full");
    let schwarz = full.preconditioner().expect("schwarz").clone();
    // An eliminated 20-level factor beside kept factors of 3 and 4 levels: 7 kept columns.
    let three = vec![
        (0..n).map(|i| (i % 20) as u32).collect::<Vec<_>>(),
        (0..n).map(|i| ((i * 7) % 3) as u32).collect(),
        (0..n).map(|i| ((i / 5) % 4) as u32).collect(),
    ];
    let reduced = solver(effects(&three), None, schwarz, System::Reduced);
    assert!(
        matches!(reduced, Err(BuildError::SchwarzOnReducedSystem)),
        "{reduced:?}"
    );
}

#[test]
fn local_solver_settings_are_validated_without_a_rung() {
    let factors = factors(1);
    let mut local_solver = within::LocalSolverConfig::default();
    local_solver.scaling.tolerance = f64::NAN;
    let additive = PreconditionerConfig::Additive { local_solver };
    for system in [System::Full, System::Reduced] {
        let result = solver(effects(&factors), None, &additive, system);
        assert!(
            matches!(result, Err(BuildError::InvalidScalingTolerance { .. })),
            "{system:?}: {result:?}"
        );
    }
}

#[test]
fn an_unsorted_design_is_refused_on_the_reduced_system() {
    let (e, f): (Vec<u32>, Vec<u32>) = (
        (0..12).map(|i| i % 6).collect(),
        (0..12).map(|i| i / 6).collect(),
    );
    let design = Design::new_unsorted([
        Effect::new(&f, true, []).expect("effect"),
        Effect::new(&e, true, []).expect("effect"),
    ])
    .expect("design");
    let solver = solver(
        design,
        None,
        &PreconditionerConfig::Diagonal,
        System::Reduced,
    );
    assert!(
        matches!(solver, Err(BuildError::UnsortedReducedSystem)),
        "{solver:?}"
    );
}

/// A kept level nested in one eliminated level is a null column of the reduced system; its
/// rounding noise must not be scaled up into a direction LSMR has to fit.
#[test]
fn a_kept_level_nested_in_an_eliminated_level_does_not_stall_the_diagonal() {
    let worker: Vec<u32> = (0..N).map(|i| (i % 300) as u32).collect();
    let firm: Vec<u32> = (0..N)
        .map(|i| match i % 300 {
            w @ 0..10 => w as u32,
            _ => 10 + ((i * 7 + i / 300) % 30) as u32,
        })
        .collect();
    let factors = vec![worker, firm];
    let w: Vec<f64> = (0..N).map(|i| 0.5 + 3.0 * noise(N + i).abs()).collect();
    let y = response(6);
    let options = LsmrOptions {
        maxiter: 500,
        ..lsmr()
    };
    let (reduced, full) = reduced_and_full(
        &factors,
        &y,
        Some(&w),
        &options,
        &PreconditionerConfig::Diagonal,
    );
    assert!(reduced.converged, "{} iterations", reduced.iterations);
    assert_same_fit(&reduced.demeaned, &full.demeaned, &[true; N], &y);
}

/// Each weight is finite but the level's sum `Σw` is not.
#[test]
fn weights_near_the_float_range_keep_the_fit() {
    let lsmr = LsmrOptions {
        tol: 1e-12,
        ..LsmrOptions::default()
    };
    let result = solve(
        effects(&[vec![0, 0]]),
        &[1.0, 1.0],
        Some(&[1e308, 1e308]),
        &lsmr,
        &PreconditionerConfig::Diagonal,
        System::Reduced,
    )
    .expect("solve");
    assert!(result.converged);
    assert!((result.x[0] - 1.0).abs() <= 1e-14, "{:?}", result.x);
    assert!(
        result.demeaned.iter().all(|d| d.abs() <= 1e-14),
        "{:?}",
        result.demeaned
    );
}
