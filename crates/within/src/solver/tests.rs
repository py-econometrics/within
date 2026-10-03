use std::sync::Arc;

use rstest::rstest;

use super::{CoefficientAddress, CoefficientLayout};
use crate::channel::Channel;
use crate::config::{LocalSolverConfig, LsmrOptions, DEFAULT_DENSE_SCHUR_THRESHOLD};
use crate::domain::{build_local_domains, Design, Grounding, MatrixForm, PreparedDesign};
use crate::{Effect, PreconditionerConfig, Solver};

/// DGP kept in lockstep with `surplus_component_sampled_matches_exact_reduction`
/// in `tests/slopes_routing.rs`. A positive slope-only term is not centered by
/// whitening, so the signed pair stays all-positive — balanced — while generic
/// `z` keeps it strictly inside the PSD cone: genuine surplus, grounded.
fn at(term: usize, level: u32, column: usize) -> CoefficientAddress {
    CoefficientAddress {
        channel: Channel { term, column },
        level,
    }
}

fn positive_slope_only_panel() -> (Vec<u32>, Vec<u32>, Vec<f64>) {
    let n = 8000usize;
    let f: Vec<u32> = (0..n).map(|i| (i % 80) as u32).collect();
    let g: Vec<u32> = (0..n).map(|i| ((i / 80) % 40) as u32).collect();
    let z: Vec<f64> = (0..n)
        .map(|i| 0.5 + ((i * 13) % 100) as f64 / 100.0)
        .collect();
    (f, g, z)
}

#[test]
fn positive_slope_only_pair_grounds_beyond_dense_threshold() {
    let (f, g, z) = positive_slope_only_panel();
    let effects = vec![
        Effect::new(&f, false, [&z[..]]).expect("slope effect"),
        Effect::new(&g, true, []).expect("plain effect"),
    ];
    let prepared = PreparedDesign::unweighted_for_test(Design::new(effects).expect("design"));
    let (domains, warnings) =
        build_local_domains(&prepared, &LocalSolverConfig::default()).expect("domains");
    assert!(
        domains.iter().any(|ld| {
            let ct = &ld.component.matrix.cross_tab;
            ld.component.form == MatrixForm::Laplacian
                && ld.component.matrix.grounding == Grounding::Grounded
                && ct.n_rows().min(ct.n_cols()) > DEFAULT_DENSE_SCHUR_THRESHOLD
        }),
        "fixture must ground a component past the dense threshold (warnings: {warnings:?})"
    );
}

#[test]
fn coefficient_layout_translates_addresses_both_ways() {
    // term 0: plain 3-level factor; term 1: 2-level factor with intercept and one slope.
    let f = [0u32, 1, 2, 0, 1, 2];
    let g = [0u32, 0, 1, 1, 0, 1];
    let z = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let design = Design::new(vec![
        Effect::new(&f, true, []).expect("plain effect"),
        Effect::new(&g, true, [&z[..]]).expect("slope effect"),
    ])
    .expect("design");
    let layout = CoefficientLayout::from_design(&design);

    assert_eq!(layout.n_terms(), 2);
    assert_eq!(
        (layout.n_levels(0), layout.n_columns(0)),
        (Some(3), Some(1))
    );
    assert_eq!(
        (layout.n_levels(1), layout.n_columns(1)),
        (Some(2), Some(2))
    );
    assert_eq!(layout.n_levels(2), None);

    // Forward matches the documented `offset + column * n_levels + level`.
    assert_eq!(layout.index(at(0, 2, 0)), Some(2));
    assert_eq!(layout.index(at(1, 0, 0)), Some(3)); // term-1 intercept, level 0
    assert_eq!(layout.index(at(1, 1, 1)), Some(6)); // term-1 slope, level 1
    assert_eq!(layout.n_dofs(), 7);

    // Out-of-range coordinates are rejected, not silently wrapped.
    assert_eq!(layout.index(at(1, 2, 0)), None); // level past n_levels
    assert_eq!(layout.index(at(1, 0, 2)), None); // column past n_columns
    assert_eq!(layout.index(at(2, 0, 0)), None); // term past n_terms
    assert_eq!(layout.address(7), None);

    // `address` inverts `index` for every flat slot.
    for i in 0..layout.n_dofs() {
        assert_eq!(layout.index(layout.address(i).expect("in range")), Some(i));
    }
}

#[test]
fn solvers_built_from_a_borrowed_design_share_its_storage() {
    let (f, g, z) = positive_slope_only_panel();
    let design = Design::new(vec![
        Effect::new(&f, true, []).unwrap(),
        Effect::new(&g, false, [&z[..]]).unwrap(),
    ])
    .unwrap();
    let n = f.len();
    let w: Vec<f64> = (0..n).map(|i| 1.0 + (i % 5) as f64).collect();
    let y: Vec<f64> = (0..n).map(|i| (i % 7) as f64).collect();

    let unweighted = Solver::new(&design, None, &PreconditionerConfig::Diagonal).unwrap();
    let weighted = Solver::new(&design, Some(&w), &PreconditionerConfig::Diagonal).unwrap();

    assert!(Arc::ptr_eq(
        &unweighted.prepared.design.terms,
        &design.terms
    ));
    assert!(Arc::ptr_eq(&weighted.prepared.design.terms, &design.terms));
    let a = unweighted.solve(&y, None).unwrap();
    let b = weighted.solve(&y, None).unwrap();
    assert!(a.converged && b.converged);
    assert_ne!(a.x, b.x);
}

/// Worker/firm/year AKM panel; each worker is observed every year and moves firm
/// with probability `mobility`. `spec` picks how the worker's slope covariate relates
/// to the rest of the design.
#[derive(Clone, Copy, Debug)]
enum SlopeSpec {
    Independent,
    YearIndex,
    /// The year index perturbed off the year term's span by the given amount.
    NearYearIndex(f64),
    SharedWithFirm,
    /// Two worker slopes that both alias the year term, and each other to `1e-6`; whitening
    /// spends the first on the second, so only the second still proposes a direction.
    DuplicateYearIndex,
    YearIndexWithoutIntercept,
    /// Two independent aliases at once: the worker's slope is the year index and a fourth
    /// term's slope is the firm index, each reproduced by a different term.
    TwoIndependentAliases,
    /// Year plus a per-worker cohort: in the span of worker and year together, of neither alone.
    AgeCohort,
}

struct AkmPanel {
    worker: Vec<u32>,
    firm: Vec<u32>,
    year: Vec<u32>,
    z: Vec<f64>,
    /// The second slope, on the worker or the fourth term; empty unless the spec carries one.
    z2: Vec<f64>,
    /// A fourth factor, present only for [`SlopeSpec::TwoIndependentAliases`].
    region: Vec<u32>,
    y: Vec<f64>,
    spec: SlopeSpec,
}

impl AkmPanel {
    fn effects(&self) -> Vec<Effect<'_>> {
        let firm = match self.spec {
            SlopeSpec::SharedWithFirm => Effect::new(&self.firm, true, [&self.z[..]]),
            _ => Effect::new(&self.firm, true, []),
        };
        let worker = match self.spec {
            SlopeSpec::DuplicateYearIndex => {
                Effect::new(&self.worker, true, [&self.z[..], &self.z2[..]])
            }
            SlopeSpec::YearIndexWithoutIntercept => Effect::new(&self.worker, false, [&self.z[..]]),
            _ => Effect::new(&self.worker, true, [&self.z[..]]),
        };
        let mut effects = vec![
            worker.expect("worker term"),
            firm.expect("firm term"),
            Effect::new(&self.year, true, []).expect("year term"),
        ];
        if matches!(self.spec, SlopeSpec::TwoIndependentAliases) {
            effects.push(Effect::new(&self.region, true, [&self.z2[..]]).expect("region term"));
        }
        effects
    }
}

/// A fixed-seed xorshift stream of draws in `[0, 1)`.
fn uniform_draws() -> impl FnMut() -> f64 {
    let mut state = 0x2545_f491_4f6c_dd1du64;
    move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn akm_panel(
    n_workers: usize,
    n_firms: usize,
    n_years: usize,
    mobility: f64,
    spec: SlopeSpec,
) -> AkmPanel {
    let mut next = uniform_draws();
    let mut panel = AkmPanel {
        worker: Vec::new(),
        firm: Vec::new(),
        year: Vec::new(),
        z: Vec::new(),
        z2: Vec::new(),
        region: Vec::new(),
        y: Vec::new(),
        spec,
    };
    let worker_fe: Vec<f64> = (0..n_workers).map(|_| next()).collect();
    let firm_fe: Vec<f64> = (0..n_firms).map(|_| next()).collect();
    let year_fe: Vec<f64> = (0..n_years).map(|_| next()).collect();
    for (w, &w_fe) in worker_fe.iter().enumerate() {
        let mut current = (next() * n_firms as f64) as usize % n_firms;
        for (t, &t_fe) in year_fe.iter().enumerate() {
            if next() < mobility {
                current = (next() * n_firms as f64) as usize % n_firms;
            }
            let z = match spec {
                SlopeSpec::Independent | SlopeSpec::SharedWithFirm => next(),
                SlopeSpec::YearIndex
                | SlopeSpec::DuplicateYearIndex
                | SlopeSpec::YearIndexWithoutIntercept
                | SlopeSpec::TwoIndependentAliases => t as f64,
                SlopeSpec::NearYearIndex(delta) => t as f64 + delta * next(),
                SlopeSpec::AgeCohort => (t + 20 + w * 7919 % 40) as f64,
            };
            if matches!(spec, SlopeSpec::DuplicateYearIndex) {
                panel.z2.push(z + 1e-6 * z * z);
            }
            if matches!(spec, SlopeSpec::TwoIndependentAliases) {
                panel.region.push((w % 11) as u32);
                panel.z2.push(current as f64);
            }
            panel.worker.push(w as u32);
            panel.firm.push(current as u32);
            panel.year.push(t as u32);
            panel.z.push(z);
            panel
                .y
                .push(w_fe + firm_fe[current] + t_fe + 0.3 * z + next() - 0.5);
        }
    }
    panel
}

/// Largest absolute within-level mean of `demeaned`, over the levels of every term whose
/// normal equations force one: only an intercept makes the within-level sum a residual leg.
fn max_abs_group_mean(design: &Design<'_>, demeaned: &[f64]) -> f64 {
    let demeaned = design.permute_obs_in(demeaned);
    design
        .terms
        .iter()
        .filter(|t| t.intercept)
        .map(|t| {
            let levels = t.levels();
            let mut sums = vec![0.0f64; t.n_levels()];
            let mut counts = vec![0.0f64; t.n_levels()];
            for (obs, &level) in levels.iter().enumerate() {
                sums[level as usize] += demeaned[obs];
                counts[level as usize] += 1.0;
            }
            sums.iter()
                .zip(&counts)
                .filter(|&(_, &c)| c > 0.0)
                .map(|(&s, &c)| (s / c).abs())
                .fold(0.0f64, f64::max)
        })
        .fold(0.0f64, f64::max)
}

/// The spectral floor off, so an aliased solve rests on the local solves alone.
fn unfloored() -> PreconditionerConfig {
    PreconditionerConfig::Additive {
        local_solver: LocalSolverConfig {
            ridge: 0.0,
            ..Default::default()
        },
        reduction: Default::default(),
    }
}

fn solve_tight(solver: &Solver<'_>, y: &[f64]) -> crate::SolveResult {
    solver
        .solve(
            y,
            &LsmrOptions {
                tol: 1e-12,
                maxiter: 20_000,
                ..Default::default()
            },
        )
        .expect("solve")
}

#[rstest]
#[case::unrelated(SlopeSpec::Independent)]
#[case::exact_alias(SlopeSpec::YearIndex)]
#[case::shared_covariate(SlopeSpec::SharedWithFirm)]
#[case::deep_null(SlopeSpec::NearYearIndex(1e-10))]
#[case::recoverable_at_the_floor(SlopeSpec::NearYearIndex(1e-8))]
#[case::recoverable(SlopeSpec::NearYearIndex(1e-6))]
#[case::near_alias(SlopeSpec::NearYearIndex(1e-3))]
#[case::duplicate_aliases(SlopeSpec::DuplicateYearIndex)]
#[case::alias_without_intercept(SlopeSpec::YearIndexWithoutIntercept)]
#[case::two_independent_aliases(SlopeSpec::TwoIndependentAliases)]
#[case::alias_of_two_terms(SlopeSpec::AgeCohort)]
fn an_aliased_slope_converges(#[case] spec: SlopeSpec) {
    let panel = akm_panel(4_000, 200, 10, 0.15, spec);
    let solver = Solver::new(panel.effects(), None, unfloored()).expect("solver");
    let out = solve_tight(&solver, &panel.y);
    // A floating component grounded by mistake reports a false convergence at an O(1) mean.
    let group_mean = max_abs_group_mean(&solver.prepared.design, &out.demeaned);
    assert!(
        out.converged && group_mean < 1e-9,
        "converged={}, gm={group_mean:.3e}",
        out.converged
    );
}

/// Escalates after any single non-vanishing contraction, so a handoff is deterministic.
fn eager_ladder(local_solver: LocalSolverConfig) -> PreconditionerConfig {
    PreconditionerConfig::Adaptive {
        local_solver,
        reduction: Default::default(),
        stall: crate::Staleness::try_new(1, 0.0).expect("valid staleness"),
    }
}

/// Two crossed slope terms on one covariate: large enough for the diagonal to stall.
struct SharedCovariatePair {
    a: Vec<u32>,
    b: Vec<u32>,
    z: Vec<f64>,
    y: Vec<f64>,
}

impl SharedCovariatePair {
    fn new() -> Self {
        let n = 4000;
        let z: Vec<f64> = (0..n).map(|i| (i as f64 * 0.17 + 1.0).sin()).collect();
        Self {
            a: (0..n).map(|i| (i % 40) as u32).collect(),
            b: (0..n).map(|i| ((i / 40) % 25) as u32).collect(),
            y: (0..n).map(|i| z[i] + (i % 7) as f64).collect(),
            z,
        }
    }

    fn effects(&self) -> Vec<Effect<'_>> {
        vec![
            Effect::new(&self.a, true, [&self.z[..]]).expect("a"),
            Effect::new(&self.b, true, [&self.z[..]]).expect("b"),
        ]
    }
}

/// A deferred build that fails is settled like one that succeeds: the second solve reports the
/// same error without the ladder rebuilding, and re-failing, the same map.
#[test]
fn a_failed_deferred_build_is_kept_and_reported_again() {
    use super::ladder::PrecondSlot;
    use crate::config::{ScalingConfig, ScalingFailure};
    use crate::{BuildError, WithinError};

    // A zero-tolerance, zero-iteration certificate rejects the signed cross-block deterministically.
    let panel = SharedCovariatePair::new();
    let precond = eager_ladder(LocalSolverConfig {
        scaling: ScalingConfig {
            tolerance: 0.0,
            max_iterations: 0,
            on_failure: ScalingFailure::Error,
        },
        ..LocalSolverConfig::default()
    });
    let solver = Solver::new(panel.effects(), None, precond).unwrap();
    let unscalable = |r: &Result<crate::SolveResult, WithinError>| {
        matches!(
            r,
            Err(WithinError::Build(BuildError::UnscalableComponent { .. }))
        )
    };
    let first = solver.solve(&panel.y, None);
    assert!(unscalable(&first), "{first:?}");
    let PrecondSlot::Adaptive(a) = &solver.slot else {
        panic!("the ladder")
    };
    assert!(matches!(
        a.built.get(),
        Some(Err(BuildError::UnscalableComponent { .. }))
    ));
    assert!(unscalable(&solver.solve(&panel.y, None)));
    assert!(!solver.has_escalated());
}

/// Three mutually orthogonal, centered ±1 columns on eight observations.
fn walsh_columns() -> ([f64; 8], [f64; 8], [f64; 8]) {
    (
        [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
        [1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0],
        [1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0],
    )
}

/// Share of `y` left after residualizing on `effects` under the default preconditioner.
fn residual_share(effects: Vec<Effect<'_>>, y: &[f64]) -> f64 {
    let solver = Solver::new(effects, None, None).expect("solver");
    let out = solve_tight(&solver, y);
    let energy = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>();
    energy(&out.demeaned) / energy(y)
}

/// A near-duplicate slope whose remainder the other term does not span still residualizes fully.
#[test]
fn a_near_duplicate_slope_residualizes_its_span() {
    let (c, _, e) = walsh_columns();
    let near: [f64; 8] = std::array::from_fn(|i| 2.0 * c[i] + 1e-6 * e[i]);
    let level = [0u32; 8];
    let effects = vec![
        Effect::new(&level, true, [&c[..], &near[..]]).unwrap(),
        Effect::new(&level, true, [&c[..]]).unwrap(),
    ];
    // `e` lies in the design's span, so nothing of it may survive residualization.
    let share = residual_share(effects, &e);
    assert!(share < 1e-12, "share={share:.3e}");
}

/// A slope `c + 1e-3·d + 5e-11·e` beside a term carrying `c` and `d` still residualizes fully.
#[test]
fn a_near_combination_slope_residualizes_its_span() {
    let (c, d, e) = walsh_columns();
    let near: [f64; 8] = std::array::from_fn(|i| c[i] + 1e-3 * d[i] + 5e-11 * e[i]);
    let level = [0u32; 8];
    let effects = vec![
        Effect::new(&level, true, [&c[..], &near[..]]).unwrap(),
        Effect::new(&level, true, [&c[..], &d[..]]).unwrap(),
    ];
    let share = residual_share(effects, &e);
    assert!(share < 1e-12, "share={share:.3e}");
}

/// A mean-year noise loading must not tie a near-isolated year into the alias block.
#[test]
fn an_alias_through_a_zero_loading_still_converges() {
    let mut next = uniform_draws();
    let (n_workers, n_firms, n_years) = (500, 25, 11);
    let (mut worker, mut firm, mut year, mut z, mut y) = (vec![], vec![], vec![], vec![], vec![]);
    for w in 0..n_workers {
        let mut years: Vec<usize> = (0..n_years).collect();
        for k in (1..n_years).rev() {
            years.swap(k, (next() * (k + 1) as f64) as usize % (k + 1));
        }
        let mut current = (next() * n_firms as f64) as usize % n_firms;
        for t in years {
            if next() < 0.15 {
                current = (next() * n_firms as f64) as usize % n_firms;
            }
            worker.push(w as u32);
            firm.push(current as u32);
            year.push(t as u32);
            z.push(1e6 + t as f64);
            y.push(current as f64 + 0.3 * t as f64 + next() - 0.5);
        }
    }
    let effects = vec![
        Effect::new(&worker, true, [&z[..]]).unwrap(),
        Effect::new(&firm, true, []).unwrap(),
        Effect::new(&year, true, []).unwrap(),
    ];
    let schwarz = PreconditionerConfig::Additive {
        local_solver: Default::default(),
        reduction: Default::default(),
    };
    let solver = Solver::new(effects, None, schwarz).expect("solver");
    let out = solve_tight(&solver, &y);
    let group_mean = max_abs_group_mean(&solver.prepared.design, &out.demeaned);
    assert!(
        out.converged && group_mean < 1e-9,
        "converged={}, iterations={}, gm={group_mean:.3e}",
        out.converged,
        out.iterations
    );
}
