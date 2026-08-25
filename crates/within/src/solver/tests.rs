use std::ops::Range;

use rstest::rstest;

use super::reparam::SlopeReparam;
use super::{CoefficientAddress, CoefficientLayout};
use crate::channel::Channel;
use crate::config::{
    LocalSolverConfig, LsmrOptions, PreconditionerConfig, DEFAULT_DENSE_SCHUR_THRESHOLD,
};
use crate::domain::level_moments::TermMoments;
use crate::domain::{build_local_domains, Design, Grounding, MatrixForm};
use crate::Effect;
use crate::Solver;

/// DGP kept in lockstep with `surplus_component_sampled_matches_exact_reduction`
/// in `tests/slopes_routing.rs`. A positive slope-only term is not centered by
/// whitening, so the signed pair stays all-positive — balanced — while generic
/// `z` keeps it strictly inside the PSD cone: genuine surplus, grounded.
fn at(term: usize, level: u32, column: usize) -> CoefficientAddress {
    CoefficientAddress {
        channel: Channel { term, column },
        level: level.into(),
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
    let mut design = Design::new(effects).expect("design");
    let moments = TermMoments::build(&design, None).expect("slopes");
    let _reparam = SlopeReparam::build(&mut design, &moments, None, &mut []);
    let (domains, warnings) =
        build_local_domains(&design, None, &LocalSolverConfig::default()).expect("domains");
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
    assert_eq!(layout.index(&at(0, 2, 0)), Some(2));
    assert_eq!(layout.index(&at(1, 0, 0)), Some(3)); // term-1 intercept, level 0
    assert_eq!(layout.index(&at(1, 1, 1)), Some(6)); // term-1 slope, level 1
    assert_eq!(layout.n_dofs(), 7);

    // Out-of-range coordinates are rejected, not silently wrapped.
    assert_eq!(layout.index(&at(1, 2, 0)), None); // level past n_levels
    assert_eq!(layout.index(&at(1, 0, 2)), None); // column past n_columns
    assert_eq!(layout.index(&at(2, 0, 0)), None); // term past n_terms
    assert_eq!(layout.address(7), None);

    // `address` inverts `index` for every flat slot.
    for i in 0..layout.n_dofs() {
        assert_eq!(layout.index(&layout.address(i).expect("in range")), Some(i));
    }
}

/// Worker/firm/year AKM panel; each worker is observed every year and moves firm
/// with probability `mobility`. `spec` picks how the worker's slope covariate relates
/// to the rest of the design.
#[derive(Clone, Copy, Debug)]
enum SlopeSpec {
    /// No relationship: the slope is its own variation.
    Independent,
    /// Exactly the year index, which term `year` reproduces per level.
    YearIndex,
    /// The year index perturbed off the year term's span by the given amount.
    NearYearIndex(f64),
    /// The same covariate carried by both the worker and the firm term.
    SharedWithFirm,
}

struct AkmPanel {
    worker: Vec<u32>,
    firm: Vec<u32>,
    year: Vec<u32>,
    z: Vec<f64>,
    y: Vec<f64>,
    spec: SlopeSpec,
}

impl AkmPanel {
    fn effects(&self) -> Vec<Effect<'_>> {
        let worker = Effect::new(&self.worker, true, [&self.z[..]]);
        let firm = match self.spec {
            SlopeSpec::SharedWithFirm => Effect::new(&self.firm, true, [&self.z[..]]),
            _ => Effect::new(&self.firm, true, []),
        };
        vec![
            worker.expect("worker term"),
            firm.expect("firm term"),
            Effect::new(&self.year, true, []).expect("year term"),
        ]
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
                SlopeSpec::YearIndex => t as f64,
                SlopeSpec::NearYearIndex(delta) => t as f64 + delta * next(),
            };
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

/// Largest absolute within-level mean of `demeaned`, over every term's levels.
fn max_abs_group_mean(design: &Design<'_>, demeaned: &[f64]) -> f64 {
    let demeaned = design.permute_obs_in(demeaned);
    (0..design.terms.len())
        .map(|term| {
            let levels = design.level_column(term);
            let n_levels = design.terms[term].n_levels();
            let mut sums = vec![0.0f64; n_levels];
            let mut counts = vec![0.0f64; n_levels];
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

/// Rows the solve space excludes beyond whitening's drops, `None` when there are none.
fn constrained_rank(solver: &Solver<'_>) -> Option<usize> {
    solver
        .reparam
        .as_ref()
        .map(|rp| rp.null.rank())
        .filter(|&k| k > 0)
}

/// Every collinearity warning's verdict, in the order the screen raised them.
fn verdicts(solver: &Solver<'_>) -> Vec<crate::AliasVerdict> {
    solver
        .warnings()
        .iter()
        .filter_map(|w| match w {
            crate::BuildWarning::CollinearSlopeCovariate { verdict, .. } => Some(*verdict),
            _ => None,
        })
        .collect()
}

/// Every collinearity warning's residual, in the order the screen raised them.
fn residuals(solver: &Solver<'_>) -> Vec<f64> {
    solver
        .warnings()
        .iter()
        .filter_map(|w| match w {
            crate::BuildWarning::CollinearSlopeCovariate {
                relative_residual, ..
            } => Some(*relative_residual),
            _ => None,
        })
        .collect()
}

/// Solve `panel` with the spectral floor off, so only the gauge constraint can save it.
fn solve_unfloored(panel: &AkmPanel) -> (Solver<'_>, crate::SolveResult) {
    let solver = Solver::new(panel.effects(), None, unfloored()).expect("solver");
    let out = solve_tight(&solver, &panel.y);
    (solver, out)
}

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

fn rss(r: &[f64]) -> f64 {
    r.iter().map(|x| x * x).sum()
}

/// What the screen's warnings decide, per relationship between the covariate and the design.
#[rstest]
#[case::unrelated(SlopeSpec::Independent, 0, None, 0.0..0.0)]
#[case::exact_alias(SlopeSpec::YearIndex, 1, Some(1), 0.0..1e-20)]
#[case::shared_covariate(SlopeSpec::SharedWithFirm, 2, Some(1), 0.0..1e-20)]
#[case::deep_null(SlopeSpec::NearYearIndex(1e-12), 1, Some(1), 0.0..1e-20)]
#[case::recoverable(SlopeSpec::NearYearIndex(1e-6), 1, None, 1e-20..f64::INFINITY)]
#[case::near_alias(SlopeSpec::NearYearIndex(1e-3), 1, None, 1e-20..f64::INFINITY)]
fn a_warned_direction_is_removed_only_when_it_carries_nothing(
    #[case] spec: SlopeSpec,
    #[case] warned: usize,
    #[case] rank: Option<usize>,
    #[case] residual: Range<f64>,
) {
    let panel = akm_panel(4_000, 200, 10, 0.15, spec);
    let (solver, out) = solve_unfloored(&panel);
    // Without the constraint an aliased solve reports a false convergence at an O(1) mean.
    let group_mean = max_abs_group_mean(&solver.design, &out.demeaned);
    assert!(
        out.converged && group_mean < 1e-9,
        "converged={}, gm={group_mean:.3e}",
        out.converged
    );
    let residuals = residuals(&solver);
    assert_eq!(residuals.len(), warned, "{residuals:?}");
    assert!(
        residuals.iter().all(|r| residual.contains(r)),
        "{residuals:?} outside {residual:?}"
    );
    // Two warnings can name one direction, so the second is absorbed by the first.
    assert_eq!(constrained_rank(&solver), rank);
    let expected = match rank {
        Some(_) => crate::AliasVerdict::Constrained,
        None => crate::AliasVerdict::Kept,
    };
    assert!(
        verdicts(&solver).iter().all(|&v| v == expected),
        "{:?}",
        verdicts(&solver)
    );
}

/// A direction an aligned response could still recover must not be constrained away, so
/// shrinking the perturbation cannot move the fit.
#[rstest]
#[case::loose(1e-3)]
#[case::tight(3e-5)]
#[case::tighter(1e-6)]
fn shrinking_the_perturbation_does_not_move_the_fit(#[case] delta: f64) {
    let reference = akm_panel(4_000, 200, 10, 0.15, SlopeSpec::NearYearIndex(1e-3));
    let reference = rss(&solve_unfloored(&reference).1.demeaned);

    let panel = akm_panel(4_000, 200, 10, 0.15, SlopeSpec::NearYearIndex(delta));
    let (solver, out) = solve_unfloored(&panel);
    let fit = rss(&out.demeaned);
    assert!(
        out.converged && (fit - reference).abs() <= 1e-6 * reference,
        "converged={}, rss={fit:.12e} against {reference:.12e}",
        out.converged
    );
    assert!(
        constrained_rank(&solver).is_none(),
        "{:?}",
        residuals(&solver)
    );
}

/// The covariate is the response most aligned with the cancellation, and its unexplained
/// share scales as the perturbation squared. Constraining the direction away breaks that,
/// so the tolerance must stay under what an aligned fit still recovers.
#[test]
fn an_aligned_response_is_still_recovered() {
    let share = |delta: f64| {
        let panel = akm_panel(4_000, 200, 10, 0.15, SlopeSpec::NearYearIndex(delta));
        let solver = Solver::new(panel.effects(), None, unfloored()).expect("solver");
        assert!(
            constrained_rank(&solver).is_none(),
            "{:?}",
            residuals(&solver)
        );
        rss(&solve_tight(&solver, &panel.z).demeaned) / rss(&panel.z)
    };
    let (coarse, fine) = (share(1e-6), share(1e-8));
    let ratio = fine / coarse;
    assert!(
        (ratio / 1e-4 - 1.0).abs() < 0.2,
        "share {fine:.6e} against {coarse:.6e} is {ratio:.3e} of the expected 1e-4"
    );
}

/// The verdict reads the design alone, so no preconditioner may change it.
#[rstest]
#[case::exact_alias(SlopeSpec::YearIndex, Some(1))]
#[case::recoverable(SlopeSpec::NearYearIndex(1e-6), None)]
fn the_verdict_does_not_depend_on_the_preconditioner(
    #[case] spec: SlopeSpec,
    #[case] rank: Option<usize>,
) {
    let panel = akm_panel(4_000, 200, 10, 0.15, spec);
    let reference = residuals(&Solver::new(panel.effects(), None, unfloored()).expect("solver"));
    for config in [PreconditionerConfig::Diagonal, unfloored()] {
        let solver = Solver::new(panel.effects(), None, &config).expect("solver");
        assert_eq!(residuals(&solver), reference, "{config:?}");
        assert_eq!(constrained_rank(&solver), rank, "{config:?}");
    }
}

/// Uniform weights cancel out of a relative residual, so they cannot move the verdict either.
#[rstest]
#[case::exact_alias(SlopeSpec::YearIndex, Some(1))]
#[case::recoverable(SlopeSpec::NearYearIndex(1e-6), None)]
fn the_verdict_survives_weight_scaling(#[case] spec: SlopeSpec, #[case] rank: Option<usize>) {
    let panel = akm_panel(4_000, 200, 10, 0.15, spec);
    for beta in [1e-8f64, 1e-4, 1.0, 1e4, 1e8] {
        let w = vec![beta; panel.y.len()];
        let solver = Solver::new(panel.effects(), Some(w), unfloored()).expect("solver");
        assert_eq!(
            constrained_rank(&solver),
            rank,
            "beta={beta:e}, {:?}",
            residuals(&solver)
        );
    }
}

/// The cancellation floor of an exact alias grows with `n_obs`; the tolerance must outrun it.
#[test]
#[ignore = "8M observations"]
fn the_exact_alias_floor_stays_under_the_tolerance_at_scale() {
    let panel = akm_panel(200_000, 10_000, 40, 0.15, SlopeSpec::YearIndex);
    let solver = Solver::new(
        panel.effects(),
        Some(vec![1e-4; panel.y.len()]),
        unfloored(),
    )
    .expect("solver");
    assert_eq!(
        constrained_rank(&solver),
        Some(1),
        "{:?}",
        residuals(&solver)
    );

    // The same panel's recoverable neighbour must survive where the shipped product did not.
    let panel = akm_panel(200_000, 10_000, 40, 0.15, SlopeSpec::NearYearIndex(3e-5));
    let solver = Solver::new(panel.effects(), None, unfloored()).expect("solver");
    assert!(
        constrained_rank(&solver).is_none(),
        "{:?}",
        residuals(&solver)
    );
}
