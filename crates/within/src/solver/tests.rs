use super::reparam::SlopeReparam;
use super::{CoefficientAddress, CoefficientLayout};
use crate::channel::Channel;
use crate::config::{LocalSolverConfig, DEFAULT_DENSE_SCHUR_THRESHOLD};
use crate::domain::level_moments::TermMoments;
use crate::domain::{build_local_domains, Design, Grounding, MatrixForm};
use crate::test_rng::{pseudo_noise, Lcg};
use crate::Effect;

/// DGP kept in lockstep with `surplus_component_sampled_matches_exact_reduction`
/// in `tests/slopes_routing.rs`. A positive slope-only term is not centered by
/// whitening, so the signed pair stays all-positive — balanced — while generic
/// `z` keeps it strictly inside the PSD cone: genuine surplus, grounded.
fn at(term: usize, level: usize, column: usize) -> CoefficientAddress {
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
    let mut design = Design::new(effects).expect("design");
    let moments = TermMoments::build(&design, None).expect("slopes");
    let _reparam = SlopeReparam::build(&mut design, &moments);
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

/// The second term's slope covariate is `z + eps * noise`; `eps = 0.0` is exact sharing.
struct SharedCovariatePanel {
    a: Vec<u32>,
    b: Vec<u32>,
    z: Vec<f64>,
    z2: Vec<f64>,
    y: Vec<f64>,
}

fn shared_covariate_panel(eps: f64) -> SharedCovariatePanel {
    let n = 20_000usize;
    let a: Vec<u32> = (0..n).map(|i| (i % 200) as u32).collect();
    let b: Vec<u32> = (0..n).map(|i| ((i * 7 / 200) % 100) as u32).collect();
    let z = pseudo_noise(n, 3);
    let noise = pseudo_noise(n, 17);
    let z2: Vec<f64> = z.iter().zip(&noise).map(|(&v, &e)| v + eps * e).collect();
    let y: Vec<f64> = (0..n)
        .map(|i| (a[i] as f64) * 0.1 + (b[i] as f64) * z[i] * 0.05 + noise[i])
        .collect();
    SharedCovariatePanel { a, b, z, z2, y }
}

fn assert_fused_solve_matches_reference(eps: f64) {
    use super::Solver;
    use crate::config::{LsmrOptions, PreconditionerConfig};

    let SharedCovariatePanel { a, b, z, z2, y } = shared_covariate_panel(eps);
    let effects = || {
        vec![
            crate::Effect::new(&a, true, [&z[..]]).unwrap(),
            crate::Effect::new(&b, true, [&z2[..]]).unwrap(),
        ]
    };
    let opts = LsmrOptions {
        tol: 1e-12,
        maxiter: 20_000,
        ..Default::default()
    };

    let solver = Solver::new(effects(), None, None).unwrap();
    assert!(
        !solver.fused.is_empty(),
        "screen must arm the fused block (eps = {eps:e})"
    );
    let got = solver.solve(&y, &opts).unwrap();
    assert!(got.converged);

    let reference = Solver::new(effects(), None, PreconditionerConfig::Off)
        .unwrap()
        .solve(&y, &opts)
        .unwrap();
    assert!(reference.converged);

    let scale = y.iter().map(|&v| v * v).sum::<f64>().sqrt();
    let diff = got
        .demeaned
        .iter()
        .zip(&reference.demeaned)
        .map(|(&p, &q)| (p - q) * (p - q))
        .sum::<f64>()
        .sqrt();
    assert!(
        diff <= 1e-6 * scale,
        "demeaned mismatch: {diff:e} vs scale {scale:e} (eps = {eps:e})"
    );
}

#[test]
fn fused_solve_matches_unpreconditioned_reference() {
    assert_fused_solve_matches_reference(1e-4);
}

#[test]
fn fused_solve_grounds_exact_sharing() {
    assert_fused_solve_matches_reference(0.0);
}

#[test]
fn independent_covariates_build_no_fused_block() {
    use super::Solver;

    let SharedCovariatePanel { a, b, .. } = shared_covariate_panel(0.0);
    let z = pseudo_noise(a.len(), 3);
    let z_other = pseudo_noise(a.len(), 29);
    let solver = Solver::new(
        vec![
            crate::Effect::new(&a, true, [&z[..]]).unwrap(),
            crate::Effect::new(&b, true, [&z_other[..]]).unwrap(),
        ],
        None,
        None,
    )
    .unwrap();
    assert!(solver.fused.is_empty());
}

/// AKM panel with a near-shared firm slope covariate (`fz = z1 + 1e-3 * z2`).
struct AkmPanel {
    worker: Vec<u32>,
    firm: Vec<u32>,
    year: Vec<u32>,
    z1: Vec<f64>,
    fz: Vec<f64>,
    y: Vec<f64>,
}

fn akm_panel(move_prob: f64) -> AkmPanel {
    let (n_workers, n_firms, n_time) = (50_000usize, 21_000usize, 10usize);
    let n = n_workers * n_time;
    let mut rng = Lcg(0xC0FFEE);
    let mut worker = Vec::with_capacity(n);
    let mut firm = Vec::with_capacity(n);
    let mut year = Vec::with_capacity(n);
    for wi in 0..n_workers {
        let mut current = (rng.next_u64() as usize) % n_firms;
        for t in 0..n_time {
            if t > 0 && rng.uniform() < move_prob {
                current = (rng.next_u64() as usize) % n_firms;
            }
            worker.push(wi as u32);
            firm.push(current as u32);
            year.push(t as u32);
        }
    }
    let z1: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
    let z2: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
    let fz: Vec<f64> = z1.iter().zip(&z2).map(|(&a, &b)| a + 1e-3 * b).collect();
    let y: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
    AkmPanel {
        worker,
        firm,
        year,
        z1,
        fz,
        y,
    }
}

/// Middle-band stress through the real gate: fill 539M declines exact, the FSAI rung engages.
#[test]
#[ignore]
fn fused_block_middle_band_stress() {
    use super::Solver;
    use crate::config::LsmrOptions;

    let AkmPanel {
        worker,
        firm,
        year,
        z1,
        fz,
        y,
    } = akm_panel(0.2);
    let solver = Solver::new(
        vec![
            Effect::new(&worker, true, [&z1[..]]).unwrap(),
            Effect::new(&firm, true, [&fz[..]]).unwrap(),
            Effect::new(&year, true, []).unwrap(),
        ],
        None,
        None,
    )
    .unwrap();
    assert!(!solver.fused.is_empty(), "screen must arm the ladder");
    assert!(
        solver.fused.iter().all(|b| !b.is_exact_for_test()),
        "fill gate must route to FSAI"
    );
    let opts = LsmrOptions {
        tol: 1e-10,
        maxiter: 3000,
        ..Default::default()
    };
    let r = solver.solve(&y, &opts).unwrap();
    eprintln!(
        "fsai middle-band stress: it={} conv={} setup={:.2}s solve={:.2}s",
        r.iterations, r.converged, r.time_setup, r.time_solve
    );
    assert!(r.converged);
    assert!(
        r.iterations < 200,
        "expected healthy iteration count, got {}",
        r.iterations
    );
}

/// Low-mobility AKM stress: unfused this exhausts 3000 iterations; fused must stay healthy.
#[test]
#[ignore]
fn fused_block_low_mobility_stress() {
    use super::Solver;
    use crate::config::LsmrOptions;

    let AkmPanel {
        worker,
        firm,
        year,
        z1,
        fz,
        y,
    } = akm_panel(0.05);

    let solver = Solver::new(
        vec![
            Effect::new(&worker, true, [&z1[..]]).unwrap(),
            Effect::new(&firm, true, [&fz[..]]).unwrap(),
            Effect::new(&year, true, []).unwrap(),
        ],
        None,
        None,
    )
    .unwrap();
    assert!(!solver.fused.is_empty(), "screen must arm the fused block");
    let opts = LsmrOptions {
        tol: 1e-10,
        maxiter: 3000,
        ..Default::default()
    };
    let r = solver.solve(&y, &opts).unwrap();
    eprintln!(
        "fused low-mobility stress: it={} conv={} setup={:.2}s solve={:.2}s",
        r.iterations, r.converged, r.time_setup, r.time_solve
    );
    assert!(r.converged);
    assert!(
        r.iterations < 100,
        "expected healthy iteration count, got {}",
        r.iterations
    );
}

#[test]
fn fused_block_restores_healthy_iteration_counts() {
    use super::Solver;
    use crate::config::LsmrOptions;

    let SharedCovariatePanel { a, b, z, z2, y } = shared_covariate_panel(1e-4);
    let solver = Solver::new(
        vec![
            crate::Effect::new(&a, true, [&z[..]]).unwrap(),
            crate::Effect::new(&b, true, [&z2[..]]).unwrap(),
        ],
        None,
        None,
    )
    .unwrap();
    let opts = LsmrOptions {
        tol: 1e-10,
        maxiter: 3000,
        ..Default::default()
    };
    let r = solver.solve(&y, &opts).unwrap();
    assert!(r.converged);
    // Without the fused block this design sits orders of magnitude higher.
    assert!(
        r.iterations < 100,
        "expected healthy iteration count, got {}",
        r.iterations
    );
}

/// Runs mlsmr with an explicit preconditioner, replicating `solve_rhs`'s data path.
fn probe_run<M: schwarz_precond::Operator>(
    solver: &super::Solver<'_>,
    y: &[f64],
    m: &M,
    tol: f64,
    maxiter: usize,
) -> (schwarz_precond::LsmrResult, f64) {
    use schwarz_precond::{mlsmr, MlsmrOptions};

    let y_int = solver.design.permute_obs_in(y);
    let rect_op =
        crate::operator::DesignOperator::new(&solver.design, solver.sqrt_weights.as_deref());
    let b = rect_op.weighted_rhs(&y_int);
    let t = std::time::Instant::now();
    let r = mlsmr(&rect_op, &b, m, tol, maxiter, MlsmrOptions::default()).unwrap();
    (r, t.elapsed().as_secs_f64())
}

fn demeaned_from(solver: &super::Solver<'_>, x: &[f64], y: &[f64]) -> Vec<f64> {
    let y_int = solver.design.permute_obs_in(y);
    let mut d = vec![0.0; solver.design.n_obs];
    crate::operator::design::gather_apply(&solver.design, x, &mut d, None);
    for (di, &yi) in d.iter_mut().zip(y_int.iter()) {
        *di = yi - *di;
    }
    solver.design.permute_obs_out(d)
}

/// Forces the FSAI rung with a zero fill budget on a panel whose real gate arms the exact rung.
fn assert_fsai_fallback_matches_reference(eps: f64) {
    use super::Solver;
    use crate::config::{LsmrOptions, PreconditionerConfig};
    use crate::operator::fused::{FusedBlockSolve, FusedPreconditioner};

    let SharedCovariatePanel { a, b, z, z2, y } = shared_covariate_panel(eps);
    let effects = || {
        vec![
            crate::Effect::new(&a, true, [&z[..]]).unwrap(),
            crate::Effect::new(&b, true, [&z2[..]]).unwrap(),
        ]
    };
    let solver = Solver::new(effects(), None, None).unwrap();
    let blocks = vec![FusedBlockSolve::build_for_test(&solver.design, &[0, 1], 0)];
    assert!(!blocks[0].is_exact_for_test());
    let op = FusedPreconditioner::new(solver.preconditioner.as_ref().unwrap(), &blocks);
    let (r, _) = probe_run(&solver, &y, &op, 1e-12, 20_000);
    assert!(r.converged);

    let reference = Solver::new(effects(), None, PreconditionerConfig::Off)
        .unwrap()
        .solve(
            &y,
            &LsmrOptions {
                tol: 1e-12,
                maxiter: 20_000,
                ..Default::default()
            },
        )
        .unwrap();
    assert!(reference.converged);
    let demeaned = demeaned_from(&solver, &r.x, &y);
    let scale = y.iter().map(|&v| v * v).sum::<f64>().sqrt();
    let diff = demeaned
        .iter()
        .zip(&reference.demeaned)
        .map(|(&p, &q)| (p - q) * (p - q))
        .sum::<f64>()
        .sqrt();
    assert!(
        diff <= 1e-6 * scale,
        "demeaned mismatch: {diff:e} vs scale {scale:e} (eps = {eps:e})"
    );
}

#[test]
fn fsai_fallback_matches_unpreconditioned_reference() {
    assert_fsai_fallback_matches_reference(1e-4);
}

#[test]
fn fsai_fallback_grounds_exact_sharing() {
    assert_fsai_fallback_matches_reference(0.0);
}

#[test]
fn fsai_fallback_restores_healthy_iteration_counts() {
    use super::Solver;
    use crate::operator::fused::{FusedBlockSolve, FusedPreconditioner};

    let SharedCovariatePanel { a, b, z, z2, y } = shared_covariate_panel(1e-4);
    let solver = Solver::new(
        vec![
            crate::Effect::new(&a, true, [&z[..]]).unwrap(),
            crate::Effect::new(&b, true, [&z2[..]]).unwrap(),
        ],
        None,
        None,
    )
    .unwrap();
    let blocks = vec![FusedBlockSolve::build_for_test(&solver.design, &[0, 1], 0)];
    let op = FusedPreconditioner::new(solver.preconditioner.as_ref().unwrap(), &blocks);
    let (r, _) = probe_run(&solver, &y, &op, 1e-10, 3000);
    assert!(r.converged);
    assert!(
        r.iterations < 100,
        "expected healthy iteration count, got {}",
        r.iterations
    );
}
