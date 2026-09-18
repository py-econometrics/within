use std::sync::Arc;

use rstest::rstest;

use super::{CoefficientAddress, CoefficientLayout};
use crate::channel::Channel;
use crate::config::{LocalSolverConfig, LsmrOptions, DEFAULT_DENSE_SCHUR_THRESHOLD};
use crate::domain::{build_local_domains, Design, Grounding, MatrixForm, PreparedDesign};
use crate::AliasVerdict::{self, Constrained, Kept};
use crate::{BuildWarning, Effect, PreconditionerConfig, Solver};

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
        &unweighted.prepared.design.frame,
        &design.frame
    ));
    assert!(Arc::ptr_eq(&weighted.prepared.design.frame, &design.frame));
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

fn akm_panel(
    n_workers: usize,
    n_firms: usize,
    n_years: usize,
    mobility: f64,
    spec: SlopeSpec,
) -> AkmPanel {
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
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
    (0..design.terms.len())
        .filter(|&term| design.terms[term].has_intercept())
        .map(|term| {
            let levels = design.frame.level_column(term);
            let mut sums = vec![0.0f64; design.terms[term].n_levels()];
            let mut counts = vec![0.0f64; design.terms[term].n_levels()];
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

fn constrained_rank(solver: &Solver<'_>) -> Option<usize> {
    solver
        .preconditioner()
        .and_then(|p| p.gauge.as_ref())
        .map(|gauge| gauge.rank())
}

fn verdicts(solver: &Solver<'_>) -> Vec<AliasVerdict> {
    solver
        .warnings()
        .iter()
        .filter_map(|w| match w {
            BuildWarning::CollinearSlopeCovariate { verdict, .. } => Some(*verdict),
            _ => None,
        })
        .collect()
}

/// The spectral floor off, so only the gauge constraint can save an aliased solve.
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
#[case::unrelated(SlopeSpec::Independent, &[], None)]
#[case::exact_alias(SlopeSpec::YearIndex, &[Constrained], Some(1))]
#[case::shared_covariate(SlopeSpec::SharedWithFirm, &[Constrained, Constrained], Some(1))]
#[case::deep_null(SlopeSpec::NearYearIndex(1e-10), &[Constrained], Some(1))]
#[case::recoverable_at_the_floor(SlopeSpec::NearYearIndex(1e-8), &[Kept], None)]
#[case::recoverable(SlopeSpec::NearYearIndex(1e-6), &[Kept], None)]
#[case::near_alias(SlopeSpec::NearYearIndex(1e-3), &[Kept], None)]
#[case::duplicate_aliases(SlopeSpec::DuplicateYearIndex, &[Kept, Constrained], Some(1))]
#[case::alias_without_intercept(SlopeSpec::YearIndexWithoutIntercept, &[Constrained], Some(1))]
#[case::two_independent_aliases(SlopeSpec::TwoIndependentAliases, &[Constrained, Constrained], Some(2))]
fn a_warned_direction_is_removed_only_when_it_carries_nothing(
    #[case] spec: SlopeSpec,
    #[case] expected: &[AliasVerdict],
    #[case] rank: Option<usize>,
) {
    let panel = akm_panel(4_000, 200, 10, 0.15, spec);
    let solver = Solver::new(panel.effects(), None, unfloored()).expect("solver");
    let out = solve_tight(&solver, &panel.y);
    // Without the constraint an aliased solve reports a false convergence at an O(1) mean.
    let group_mean = max_abs_group_mean(&solver.prepared.design, &out.demeaned);
    assert!(
        out.converged && group_mean < 1e-9,
        "converged={}, gm={group_mean:.3e}",
        out.converged
    );
    assert_eq!(verdicts(&solver), expected, "{:?}", solver.warnings());
    // Two proposals can name one direction; the duplicate is spent against the first row.
    assert_eq!(constrained_rank(&solver), rank);

    // The gauge is the design's, not the factorization's: a deserialized preconditioner gets it
    // back from the solver it is attached to.
    let bytes = postcard::to_stdvec(solver.preconditioner().expect("built")).expect("serialize");
    let prebuilt: crate::Preconditioner = postcard::from_bytes(&bytes).expect("deserialize");
    assert!(prebuilt.gauge.is_none());
    let reattached = Solver::new(panel.effects(), None, prebuilt).expect("solver");
    assert_eq!(constrained_rank(&reattached), rank);
}

/// The escalated rung is built after the gauge, so it has to inherit it: an aliased design that
/// hands off would otherwise run its second rung unconstrained.
#[test]
fn an_escalated_rung_keeps_the_constrained_directions_out() {
    let panel = akm_panel(4_000, 200, 10, 0.15, SlopeSpec::YearIndex);
    let ladder = PreconditionerConfig::Adaptive {
        local_solver: LocalSolverConfig {
            ridge: 0.0,
            ..Default::default()
        },
        reduction: Default::default(),
        stall: crate::Staleness::try_new(1, 0.0).expect("valid staleness"),
    };
    let solver = Solver::new(panel.effects(), None, ladder).expect("solver");
    let out = solve_tight(&solver, &panel.y);

    assert!(solver.has_escalated(), "eager stall must hand off");
    assert_eq!(constrained_rank(&solver), Some(1));
    let group_mean = max_abs_group_mean(&solver.prepared.design, &out.demeaned);
    assert!(
        out.converged && group_mean < 1e-9,
        "converged={}, gm={group_mean:.3e}",
        out.converged
    );
}

/// A deferred build that fails is settled like one that succeeds: the second solve reports the
/// same error without the ladder rebuilding, and re-failing, the same map.
#[test]
fn a_failed_deferred_build_is_kept_and_reported_again() {
    use super::ladder::PrecondSlot;
    use crate::config::{ScalingConfig, ScalingFailure};
    use crate::{BuildError, WithinError};

    // Two crossed slope pairs large enough for the diagonal to stall; a zero-tolerance,
    // zero-iteration certificate then rejects the signed cross-block deterministically.
    let n = 4000;
    let a: Vec<u32> = (0..n).map(|i| (i % 40) as u32).collect();
    let b: Vec<u32> = (0..n).map(|i| ((i / 40) % 25) as u32).collect();
    let z: Vec<f64> = (0..n).map(|i| (i as f64 * 0.17 + 1.0).sin()).collect();
    let effects = vec![
        Effect::new(&a, true, [&z[..]]).unwrap(),
        Effect::new(&b, true, [&z[..]]).unwrap(),
    ];
    let precond = PreconditionerConfig::Adaptive {
        local_solver: LocalSolverConfig {
            scaling: ScalingConfig {
                tolerance: 0.0,
                max_iterations: 0,
                on_failure: ScalingFailure::Error,
            },
            ..LocalSolverConfig::default()
        },
        reduction: Default::default(),
        stall: crate::Staleness::try_new(1, 0.0).unwrap(),
    };
    let solver = Solver::new(effects, None, precond).unwrap();
    let y: Vec<f64> = (0..n).map(|i| z[i] + (i % 7) as f64).collect();
    let unscalable = |r: &Result<crate::SolveResult, WithinError>| {
        matches!(
            r,
            Err(WithinError::Build(BuildError::UnscalableComponent { .. }))
        )
    };
    let first = solver.solve(&y, None);
    assert!(unscalable(&first), "{first:?}");
    let PrecondSlot::Adaptive(a) = &solver.slot else {
        panic!("the ladder")
    };
    assert!(matches!(
        a.built.get(),
        Some(Err(BuildError::UnscalableComponent { .. }))
    ));
    assert!(unscalable(&solver.solve(&y, None)));
    assert!(!solver.has_escalated());
}

/// A batch runs its right-hand sides on rayon workers that also execute the base apply's
/// subdomain jobs; the constraint's scratch must never be held across that apply.
#[test]
fn a_constrained_batch_solve_does_not_deadlock() {
    let panel = akm_panel(4_000, 200, 10, 0.15, SlopeSpec::YearIndex);
    let solver = Solver::new(panel.effects(), None, unfloored()).expect("solver");
    assert_eq!(constrained_rank(&solver), Some(1));
    let ys: Vec<Vec<f64>> = (0..64)
        .map(|k| panel.y.iter().map(|v| v * (k + 1) as f64).collect())
        .collect();
    let refs: Vec<&[f64]> = ys.iter().map(Vec::as_slice).collect();
    let out = solver
        .solve_batch(
            &refs,
            &LsmrOptions {
                tol: 1e-12,
                maxiter: 20_000,
                ..Default::default()
            },
        )
        .expect("batch");
    assert!(out.converged.iter().all(|&c| c), "{:?}", out.converged);
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
fn residual_share<'a>(effects: Vec<Effect<'a>>, y: &[f64]) -> (Solver<'a>, f64) {
    let solver = Solver::new(effects, None, None).expect("solver");
    let out = solve_tight(&solver, y);
    let energy = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>();
    let share = energy(&out.demeaned) / energy(y);
    (solver, share)
}

/// Whitening spent the carrying term's own `c` direction on a near-duplicate slope whose remainder
/// the other term does not span, so the proposed difference of fits is not a null.
#[test]
fn a_covariate_its_own_term_no_longer_carries_is_not_a_null() {
    let (c, _, e) = walsh_columns();
    let near: [f64; 8] = std::array::from_fn(|i| 2.0 * c[i] + 1e-6 * e[i]);
    let level = [0u32; 8];
    let effects = vec![
        Effect::new(&level, true, [&c[..], &near[..]]).unwrap(),
        Effect::new(&level, true, [&c[..]]).unwrap(),
    ];
    // `e` lies in the design's span, so nothing of it may survive residualization.
    let (solver, share) = residual_share(effects, &e);
    assert_eq!(
        verdicts(&solver),
        [Kept, Kept, Kept],
        "{:?}",
        solver.warnings()
    );
    assert_eq!(constrained_rank(&solver), None);
    assert!(share < 1e-12, "share={share:.3e}");
}

/// Two certified proposals `c` and `c + 1e-3·d + 5e-11·e` name one null and a contrast whose
/// unexplained `e` part is divided by the contrast's `1e-3` share: the contrast is not certified.
#[test]
fn a_contrast_of_certified_proposals_is_not_itself_certified() {
    let (c, d, e) = walsh_columns();
    let near: [f64; 8] = std::array::from_fn(|i| c[i] + 1e-3 * d[i] + 5e-11 * e[i]);
    let level = [0u32; 8];
    let effects = vec![
        Effect::new(&level, true, [&c[..], &near[..]]).unwrap(),
        Effect::new(&level, true, [&c[..], &d[..]]).unwrap(),
    ];
    let (solver, share) = residual_share(effects, &e);
    assert_eq!(
        constrained_rank(&solver),
        Some(1),
        "{:?}",
        solver.warnings()
    );
    assert!(share < 1e-12, "share={share:.3e}");
}
