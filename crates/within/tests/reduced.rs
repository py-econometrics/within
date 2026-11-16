use rstest::rstest;
use within::{
    BuildError, BuildWarning, Channel, CoarseBuild, CoarseCorrection, CoarseDecline, CoarseSpace,
    CoefficientAddress, Effect, LsmrOptions, PreconditionerConfig, Rung, SolveResult, Solver,
    Staleness, TwoLevelForm,
};

#[path = "common/orchestrate_helpers.rs"]
mod common;

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

fn solve_on(
    factors: &[Vec<u32>],
    y: &[f64],
    w: Option<&[f64]>,
    options: &LsmrOptions,
    preconditioner: &PreconditionerConfig,
) -> SolveResult {
    Solver::new(effects(factors), w, preconditioner)
        .expect("solver")
        .solve(y, options)
        .expect("solve")
}

/// `head`, then 1000 rows of `zero`.
fn padded<T: Clone>(head: &[T], zero: T) -> Vec<T> {
    let zeros = std::iter::repeat_n(zero, 1000);
    head.iter().cloned().chain(zeros).collect()
}

fn assert_same_fit(demeaned: &[f64], expected: &[f64], y: &[f64]) {
    let gap: Vec<f64> = demeaned.iter().zip(expected).map(|(d, e)| d - e).collect();
    assert!(
        norm(&gap) <= 1e-8 * norm(y),
        "‖demeaned gap‖ = {:.2e} of ‖y‖ = {:.2e}",
        norm(&gap),
        norm(y)
    );
}

#[rstest]
fn the_fit_satisfies_the_normal_equations(
    #[values(2, 3)] n_factors: usize,
    #[values(false, true)] weighted: bool,
    #[values(
        PreconditionerConfig::Off,
        PreconditionerConfig::Diagonal,
        PreconditionerConfig::Schwarz(Default::default())
    )]
    preconditioner: PreconditionerConfig,
) {
    let factors = factors(n_factors);
    let w = weighted.then(|| weights(&factors));
    let y = response(1);
    let reduced = solve_on(&factors, &y, w.as_deref(), &lsmr(), &preconditioner);
    assert!(reduced.converged, "{reduced:?}");
    common::assert_normal_equations_satisfied(&factors, w.as_deref(), &y, &reduced.demeaned, 1e-8);
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
fn each_fit_of_a_batch_satisfies_the_normal_equations(#[values(2, 3)] n_factors: usize) {
    let factors = factors(n_factors);
    let w = weights(&factors);
    let ys = [response(2), response(3)];
    let ys: Vec<&[f64]> = ys.iter().map(Vec::as_slice).collect();
    let batch = Solver::new(effects(&factors), Some(&w), &PreconditionerConfig::Diagonal)
        .expect("solver")
        .solve_batch(&ys, &lsmr())
        .expect("batch");
    for (k, y) in ys.iter().enumerate() {
        assert!(batch.converged[k]);
        common::assert_normal_equations_satisfied(&factors, Some(&w), y, batch.demeaned(k), 1e-8);
    }
}

#[rstest]
fn one_factor_is_solved_by_elimination_alone(#[values(false, true)] weighted: bool) {
    let factors = factors(1);
    let w = weighted.then(|| weights(&factors));
    let y = response(4);
    // With no kept factor, the Schwarz default settles like any one-factor system.
    let result = Solver::new(effects(&factors), w.as_deref(), None)
        .expect("solver")
        .solve(&y, None)
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
        (PreconditionerConfig::Schwarz(Default::default()), PreconditionerConfig::Off)
    )]
    (schwarz, plain): (PreconditionerConfig, PreconditionerConfig),
) {
    let factors = factors(1);
    let w = weights(&factors);
    let y = response(7);
    let solve_with = |p| solve_on(&factors, &y, Some(&w), &lsmr(), p);
    let (schwarz, plain) = (solve_with(&schwarz), solve_with(&plain));
    assert!(schwarz.converged);
    assert_eq!(schwarz.iterations, plain.iterations);
    assert_eq!(schwarz.demeaned, plain.demeaned);
}

/// A Schwarz build finds no subdomain only when the operator it would act on is zero; these
/// designs reach that case, and the solve is exact at `x = 0` without a map.
#[rstest]
#[case::every_column_zero([Some(0.0), Some(0.0)], 1.0)]
#[case::every_kept_column_zero([None, Some(0.0)], 1.0)]
#[case::every_weight_zero([None, None], 0.0)]
fn a_schwarz_build_without_a_subdomain_solves_at_zero(
    // `None` is an intercept term; `Some(v)` a term whose only column is the slope `v`.
    #[case] slopes: [Option<f64>; 2],
    #[case] weight: f64,
) {
    let levels: [Vec<u32>; 2] = [vec![0, 0, 1, 1, 2, 2], vec![0, 1, 0, 1, 0, 1]];
    let columns = slopes.map(|v| v.map(|v| vec![v; 6]));
    let effects: Vec<_> = levels
        .iter()
        .zip(&columns)
        .map(|(levels, slope)| {
            Effect::new(levels, slope.is_none(), slope.as_deref()).expect("effect")
        })
        .collect();
    let w = vec![weight; 6];
    let solver = Solver::new(
        effects,
        Some(&w),
        PreconditionerConfig::Schwarz(Default::default()),
    )
    .expect("solver");
    assert!(solver.preconditioner().is_none());
    let r = solver
        .solve(&[1.0, -2.0, 0.5, 3.0, -1.5, 2.5], &lsmr())
        .expect("solve");
    assert!(r.converged);
    assert_eq!(r.iterations, 0);
}

/// The ladder starts on the reduced system's own diagonal, and an easy panel finishes there.
#[rstest]
fn the_default_preconditioner_starts_on_the_reduced_diagonal(
    #[values(2, 3)] n_factors: usize,
    #[values(None, Some(PreconditionerConfig::default()))] preconditioner: Option<
        PreconditionerConfig,
    >,
) {
    let factors = factors(n_factors);
    let y = response(5);
    let solve_with = |preconditioner: Option<&PreconditionerConfig>| {
        Solver::new(effects(&factors), None, preconditioner)
            .expect("solver")
            .solve(&y, &lsmr())
            .expect("solve")
    };
    let ladder = solve_with(preconditioner.as_ref());
    let diagonal = solve_with(Some(&PreconditionerConfig::Diagonal));
    assert!(ladder.converged);
    assert_eq!(ladder.iterations, diagonal.iterations);
    assert_eq!(ladder.x, diagonal.x);
    common::assert_normal_equations_satisfied(&factors, None, &y, &ladder.demeaned, 1e-8);
}

/// With the eliminated term between the kept ones, its pairs keep their row side or their column
/// side; slopes shift every kept block. The fit is the one with the eliminated term first.
#[test]
fn schwarz_serves_an_eliminated_term_between_the_kept_ones() {
    let factors = factors(3);
    let z: Vec<Vec<f64>> = (0..3)
        .map(|k| (0..N).map(|i| noise((20 + k) * N + i)).collect())
        .collect();
    let term = |t: usize| Effect::new(&factors[t], true, [&z[t][..]]).expect("effect");
    let y = response(6);
    let additive = PreconditionerConfig::Schwarz(within::Schwarz::default());
    let solve = |order: [usize; 3]| {
        Solver::new(Vec::from(order.map(term)), None, &additive)
            .expect("solver")
            .solve(&y, &lsmr())
            .expect("solve")
    };
    let (between, first) = (solve([1, 0, 2]), solve([0, 1, 2]));
    assert!(between.converged && first.converged);
    assert_same_fit(&between.demeaned, &first.demeaned, &y);
}

/// A Schwarz map built on the reduced system solves the same when a second solver reuses it.
#[test]
fn a_reduced_schwarz_map_is_reused_on_the_reduced_system() {
    let factors = factors(3);
    let y = response(5);
    let additive = PreconditionerConfig::Schwarz(within::Schwarz::default());
    let built = Solver::new(effects(&factors), None, &additive).expect("built");
    let map = built.preconditioner().expect("schwarz").clone();
    let reused = Solver::new(effects(&factors), None, map).expect("reused");
    let built = built.solve(&y, &lsmr()).expect("solve");
    let reused = reused.solve(&y, &lsmr()).expect("solve");
    assert!(built.converged);
    assert_eq!(reused.iterations, built.iterations);
    assert_eq!(reused.x, built.x);
}

/// Workers staying in 40 firm clusters with a slope on both terms, and movers whose one row
/// elsewhere glues nothing: each cluster is a rigid component the one-level map cannot see.
struct Clustered {
    worker: Vec<u32>,
    firm: Vec<u32>,
    z_worker: Vec<f64>,
    z_firm: Vec<f64>,
}

impl Clustered {
    fn effects(&self) -> Vec<Effect<'_>> {
        vec![
            Effect::new(&self.worker, true, [&self.z_worker[..]]).expect("worker"),
            Effect::new(&self.firm, true, [&self.z_firm[..]]).expect("firm"),
        ]
    }

    fn y(&self) -> Vec<f64> {
        (0..self.worker.len()).map(noise).collect()
    }
}

fn clustered() -> Clustered {
    let (mut worker, mut firm) = (Vec::new(), Vec::new());
    let mut spell = |w: u32, f: u32, rows: usize| {
        worker.extend(std::iter::repeat_n(w, rows));
        firm.extend(std::iter::repeat_n(f, rows));
    };
    for k in 0..120 {
        spell(k, k / 3, 8);
    }
    for k in 0..30 {
        let a = (7 * k + 3) % 40;
        spell(120 + k, a, 7);
        spell(120 + k, (a + 1 + (13 * k) % 39) % 40, 1);
    }
    let n = worker.len();
    Clustered {
        z_worker: (0..n).map(|i| noise(2 * n + i)).collect(),
        z_firm: (0..n).map(|i| noise(3 * n + i)).collect(),
        worker,
        firm,
    }
}

fn coarse(build: CoarseBuild) -> CoarseCorrection {
    CoarseCorrection {
        space: CoarseSpace::RigidComponents,
        form: TwoLevelForm::Balanced,
        build,
    }
}

/// `schwarz` from construction on, or after a diagonal that stalls on its first iteration.
fn ladder(schwarz: within::Schwarz, adaptive: bool) -> PreconditionerConfig {
    match adaptive {
        false => PreconditionerConfig::Schwarz(schwarz),
        true => PreconditionerConfig::Adaptive {
            stall: Staleness::try_new(1, 0.0).expect("stall"),
            schwarz,
        },
    }
}

/// Built with Schwarz or on its first stall, the two-level map is where the solve ends.
#[rstest]
fn a_coarse_correction_lands_the_solve_on_the_two_level_map(
    #[values(false, true)] adaptive: bool,
    // Any contraction stalls: the solve leaves one-level Schwarz on its first iteration.
    #[values(
        CoarseBuild::WithSchwarz,
        CoarseBuild::OnStall(Staleness::try_new(1, 0.0).expect("stall"))
    )]
    build: CoarseBuild,
) {
    let panel = clustered();
    let y = panel.y();
    let solve = |coarse| {
        let config = ladder(
            within::Schwarz {
                coarse,
                ..Default::default()
            },
            adaptive,
        );
        let solver = Solver::new(panel.effects(), None, &config).expect("solver");
        let result = solver.solve(&y, &lsmr()).expect("solve");
        (result, solver.preconditioner().map(|p| p.rung()))
    };
    let (two_level, rung) = solve(Some(coarse(build)));
    let (one_level, _) = solve(None);
    assert_eq!(rung, Some(Rung::TwoLevel));
    assert!(two_level.converged && one_level.converged);
    assert_same_fit(&two_level.demeaned, &one_level.demeaned, &y);
}

/// A two-level preconditioner saved and loaded solves to the same bits.
#[rstest]
fn a_two_level_preconditioner_round_trips() {
    let panel = clustered();
    let y = panel.y();
    let config = PreconditionerConfig::Schwarz(within::Schwarz {
        coarse: Some(coarse(CoarseBuild::WithSchwarz)),
        ..Default::default()
    });
    let built = Solver::new(panel.effects(), None, &config).expect("solver");
    let original = built.preconditioner().expect("built");
    assert_eq!(original.rung(), Rung::TwoLevel);
    let bytes = postcard::to_stdvec(original).expect("serialize");
    let restored: within::Preconditioner = postcard::from_bytes(&bytes).expect("deserialize");
    let reused = Solver::new(panel.effects(), None, restored).expect("reused");
    let (a, b) = (built.solve(&y, &lsmr()), reused.solve(&y, &lsmr()));
    let (a, b) = (a.expect("solve"), b.expect("reused solve"));
    let bits = |x: &[f64]| x.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    assert!(a.converged);
    assert_eq!((a.iterations, bits(&a.x)), (b.iterations, bits(&b.x)));
}

/// Gauges the eliminated term absorbs to within 1e-6 of their energy keep the two-level fit:
/// `E`'s null modes stay at rounding, where its factor drops them, not amplified into the solve.
#[rstest]
fn nearly_absorbed_gauges_keep_the_two_level_fit() {
    let panel = clustered();
    // Movers' one row elsewhere is all the energy their firms' gauges keep.
    let short = |i: usize| i >= 960 && (i - 960) % 8 == 7;
    let w: Vec<f64> = (0..panel.worker.len())
        .map(|i| if short(i) { 1.0 } else { 1e6 })
        .collect();
    let y = panel.y();
    let solve = |coarse, options: &LsmrOptions| {
        let config = PreconditionerConfig::Schwarz(within::Schwarz {
            coarse,
            ..Default::default()
        });
        let solver = Solver::new(panel.effects(), Some(&w), &config).expect("solver");
        let result = solver.solve(&y, options).expect("solve");
        (result, solver.preconditioner().map(|p| p.rung()))
    };
    let (two_level, rung) = solve(Some(coarse(CoarseBuild::WithSchwarz)), &lsmr());
    // The light rows' fit is loose at 1e-12 of the weighted response: refer to a tighter one.
    let tight = LsmrOptions {
        tol: 1e-14,
        maxiter: 200_000,
        ..LsmrOptions::default()
    };
    let (reference, _) = solve(None, &tight);
    assert_eq!(rung, Some(Rung::TwoLevel));
    assert!(two_level.converged && reference.converged);
    assert_same_fit(&two_level.demeaned, &reference.demeaned, &y);
}

/// A design with no coarse space leaves the solve on one-level Schwarz, with the reason.
#[rstest]
fn a_declined_coarse_correction_leaves_the_solve_one_level(#[values(false, true)] adaptive: bool) {
    let factors = factors(3);
    let y = response(7);
    let solve = |coarse| {
        let config = ladder(
            within::Schwarz {
                coarse,
                ..Default::default()
            },
            adaptive,
        );
        let solver = Solver::new(effects(&factors), None, &config).expect("solver");
        let result = solver.solve(&y, &lsmr()).expect("solve");
        let rung = solver.preconditioner().map(|p| p.rung());
        (result, rung, solver.warnings())
    };
    let (declined, rung, warnings) = solve(Some(coarse(CoarseBuild::WithSchwarz)));
    let (one_level, ..) = solve(None);
    assert_eq!(rung, Some(Rung::Schwarz));
    assert_eq!(
        warnings,
        [BuildWarning::CoarseDeclined(CoarseDecline::NoCoarseSpace)]
    );
    assert_eq!(declined.iterations, one_level.iterations);
    assert_eq!(declined.x, one_level.x);
}

/// A response whose weighted value leaves the float range is a solve error.
#[test]
fn an_overflowing_weighted_response_is_refused() {
    let factors = factors(2);
    let y: Vec<f64> = (0..N).map(|i| 1e300 * noise(i)).collect();
    let w = vec![1e100; N];
    let diagonal = &PreconditionerConfig::Diagonal;
    let solver = Solver::new(effects(&factors), Some(&w), diagonal).expect("solver");
    assert!(solver.solve(&y, &lsmr()).is_err());
}

#[test]
fn local_solver_settings_are_validated_without_a_rung() {
    let factors = factors(1);
    let mut local_solver = within::LocalSolverConfig::default();
    local_solver.scaling.tolerance = f64::NAN;
    let additive = PreconditionerConfig::Schwarz(within::Schwarz {
        local_solver,
        ..Default::default()
    });
    let result = Solver::new(effects(&factors), None, &additive);
    assert!(
        matches!(result, Err(BuildError::InvalidScalingTolerance { .. })),
        "{result:?}"
    );
}

/// Level sums of a response near the float range overflow; the recovered effect must not.
#[rstest]
fn recovery_near_the_float_range_stays_finite(#[values(1, 2)] n_factors: usize) {
    let factors: Vec<Vec<u32>> = [vec![0, 0, 0], vec![0, 1, 0]]
        .into_iter()
        .take(n_factors)
        .collect();
    // ‖y‖ stays finite while the level sum does not; a constant `y` is fitted exactly.
    let y = [1e308; 3];
    let reduced = solve_on(&factors, &y, None, &lsmr(), &PreconditionerConfig::Diagonal);
    assert!(reduced.converged);
    assert!(reduced.x.iter().all(|x| x.is_finite()), "{:?}", reduced.x);
    assert!(
        reduced.demeaned.iter().all(|d| d.abs() <= 1e-12 * 1e308),
        "{:?}",
        reduced.demeaned
    );
}

/// Four-row designs at the edge of rounding, with their fits in closed form:
/// - each level's moment `Σ w·y` exceeds the float range, though `W^½ y` and the fit do not;
/// - rows of zero weight round nothing, so they must not raise the diagonal's rounding bound;
/// - a kept column's distinction from the eliminated span lies below that span's Gram rounding.
///
/// One factor leaves each level's deviation from its mean; two crossed factors on four rows leave
/// `c (cᵀ y) / 4` with `c = ±1` alternating over the cells, which the last two `y` make zero.
#[rstest]
#[case::moments_beyond_the_float_range_one_factor(
    vec![vec![0, 0, 1, 1]],
    vec![1e308; 4],
    vec![1e60, 1e60, -1e60, 0.5e60],
    vec![0.0, 0.0, -0.75e60, 0.75e60]
)]
#[case::moments_beyond_the_float_range_two_factors(
    vec![vec![0, 0, 1, 1], vec![0, 1, 0, 1]],
    vec![1e308; 4],
    vec![1e60, 1e60, -1e60, 0.5e60],
    vec![0.375e60, -0.375e60, -0.375e60, 0.375e60]
)]
#[case::zero_weight_rows_beside_a_tiny_direction(
    vec![padded(&[0, 1, 1, 0], 0), padded(&[0, 0, 1, 1], 0)],
    padded(&[1.0, 1e-26, 1.0, 1e-26], 0.0),
    padded(&[0.0, 1.0, 0.0, -1.0], 0.0),
    vec![0.0; 1004]
)]
#[case::a_column_below_the_eliminated_grams_rounding(
    vec![vec![0, 0, 1, 1], vec![0, 1, 1, 0]],
    vec![1.0, 1e-18, 1.0, 1e-18],
    vec![1.0, 1.0, 0.0, 0.0],
    vec![0.0; 4]
)]
fn an_edge_of_rounding_keeps_the_fit(
    #[case] factors: Vec<Vec<u32>>,
    #[case] w: Vec<f64>,
    #[case] y: Vec<f64>,
    #[case] expected: Vec<f64>,
    #[values(PreconditionerConfig::Off, PreconditionerConfig::Diagonal)]
    preconditioner: PreconditionerConfig,
) {
    let reduced = solve_on(&factors, &y, Some(&w), &lsmr(), &preconditioner);
    assert!(reduced.converged, "{reduced:?}");
    assert_same_fit(&reduced.demeaned, &expected, &y);
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
    let reduced = solve_on(
        &factors,
        &y,
        Some(&w),
        &options,
        &PreconditionerConfig::Diagonal,
    );
    assert!(reduced.converged, "{} iterations", reduced.iterations);
    common::assert_normal_equations_satisfied(&factors, Some(&w), &y, &reduced.demeaned, 1e-8);
}

/// Every column of a term that is a function of the eliminated one lies in its span: alone, it
/// leaves the reduced system nothing to iterate on, whatever the map, and the fit is the one
/// without it.
#[rstest]
fn a_kept_term_nested_in_the_eliminated_one_is_zero(
    // Slopes on (the eliminated term, the kept one); a kept slope needs the eliminated one's.
    #[values((false, false), (true, false), (true, true))] slopes: (bool, bool),
    // A live kept term beside the nested one.
    #[values(false, true)] firm: bool,
    #[values(
        PreconditionerConfig::Off,
        PreconditionerConfig::Diagonal,
        PreconditionerConfig::Schwarz(Default::default()),
        PreconditionerConfig::default()
    )]
    preconditioner: PreconditionerConfig,
) {
    let (eliminated_slope, kept_slope) = slopes;
    let mut factors = factors(2);
    let worker = factors.remove(0);
    let region: Vec<u32> = worker.iter().map(|&w| w / 10).collect();
    let z: Vec<f64> = (0..N).map(|i| noise(2 * N + i)).collect();
    let loadings = |on: bool| if on { vec![&z[..]] } else { vec![] };
    let effects = |nested: bool| {
        let mut effects = vec![Effect::new(&worker, true, loadings(eliminated_slope)).unwrap()];
        if nested {
            effects.push(Effect::new(&region, true, loadings(kept_slope)).unwrap());
        }
        if firm {
            effects.push(Effect::new(&factors[0], true, []).unwrap());
        }
        effects
    };
    let w: Vec<f64> = (0..N).map(|i| 0.5 + noise(N + i).abs()).collect();
    let y = response(7);
    let solve = |nested: bool, preconditioner: &PreconditionerConfig| {
        Solver::new(effects(nested), Some(&w), preconditioner)
            .unwrap()
            .solve(&y, &lsmr())
            .unwrap()
    };
    let (reduced, without) = (
        solve(true, &preconditioner),
        solve(false, &PreconditionerConfig::Diagonal),
    );
    assert!(
        reduced.converged && without.converged,
        "{reduced:?}\n{without:?}"
    );
    assert!(firm || reduced.iterations == 0, "{}", reduced.iterations);
    assert_same_fit(&reduced.demeaned, &without.demeaned, &y);
    assert!((0..reduced.x.len())
        .filter(|&j| reduced.layout.address(j).expect("slot").channel.term == 1)
        .all(|j| reduced.x[j] == 0.0));
}

/// Each weight is finite but the level's sum `Σw` is not.
#[test]
fn weights_near_the_float_range_keep_the_fit() {
    let lsmr = LsmrOptions {
        tol: 1e-12,
        ..LsmrOptions::default()
    };
    let result = solve_on(
        &[vec![0, 0]],
        &[1.0, 1.0],
        Some(&[1e308, 1e308]),
        &lsmr,
        &PreconditionerConfig::Diagonal,
    );
    assert!(result.converged);
    assert!((result.x[0] - 1.0).abs() <= 1e-14, "{:?}", result.x);
    assert!(
        result.demeaned.iter().all(|d| d.abs() <= 1e-14),
        "{:?}",
        result.demeaned
    );
}
