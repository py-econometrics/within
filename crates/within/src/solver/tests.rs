use super::reparam::SlopeReparam;
use super::{CoefficientAddress, CoefficientLayout};
use crate::channel::Channel;
use crate::config::{LocalSolverConfig, ScalingConfig, DEFAULT_DENSE_SCHUR_THRESHOLD};
use crate::domain::{build_local_domains, Design, Grounding, MatrixForm};
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
    let _reparam = SlopeReparam::build(&mut design, None);
    let (domains, warnings) =
        build_local_domains(&design, None, &ScalingConfig::default()).expect("domains");
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

/// Probe (#o1-demean follow-up): what does the additive Schwarz M⁻¹ amplify on a shared-slope
/// AKM panel, and which factor-pair subdomain contributes it? Reads WITHIN_PROBE_CSV
/// ("n W F" header, rows "w f x1 x2"), mirrors Solver::new's build sequence exactly.
#[test]
#[ignore]
fn probe_schwarz_amplification_shared_slope() {
    use crate::domain::CoordinateMap;
    use crate::operator::schwarz::build_entry;

    let path = std::env::var("WITHIN_PROBE_CSV").expect("set WITHIN_PROBE_CSV");
    let text = std::fs::read_to_string(&path).expect("read panel");
    let mut lines = text.lines();
    let _ = lines.next();
    let (mut w, mut f, mut x1, mut x2) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for line in lines {
        let mut t = line.split_whitespace();
        w.push(t.next().unwrap().parse::<u32>().unwrap());
        f.push(t.next().unwrap().parse::<u32>().unwrap());
        x1.push(t.next().unwrap().parse::<f64>().unwrap());
        x2.push(t.next().unwrap().parse::<f64>().unwrap());
    }
    let effects = vec![
        Effect::new(&w, true, [&x1[..]]).expect("worker effect"),
        Effect::new(&f, true, [&x2[..]]).expect("firm effect"),
    ];
    let mut design = Design::new(effects).expect("design");
    let _reparam = if std::env::var_os("WITHIN_PROBE_NO_REPARAM").is_some() {
        None
    } else {
        SlopeReparam::build(&mut design, None)
    };
    let layout = CoefficientLayout::from_design(&design);
    let n_dofs = design.n_dofs;

    let (meta_domains, warnings) =
        build_local_domains(&design, None, &ScalingConfig::default()).expect("domains");
    eprintln!(
        "n_dofs={n_dofs} subdomains={} warnings={warnings:?}",
        meta_domains.len()
    );
    struct Meta {
        channels: Vec<(usize, usize)>,
        n_elim: usize,
        n_kept: usize,
        form: MatrixForm,
        grounding: Grounding,
        scale_range: (f64, f64),
        surplus_min: f64,
        surplus_max: f64,
        diag_max: f64,
    }
    let metas: Vec<Meta> = meta_domains
        .iter()
        .map(|ld| {
            let mut channels: Vec<(usize, usize)> = ld
                .core
                .global_indices()
                .iter()
                .filter_map(|&gi| layout.address(gi as usize))
                .map(|a| (a.channel.term, a.channel.column))
                .collect();
            channels.dedup();
            channels.sort_unstable();
            channels.dedup();
            let m = &ld.component.matrix;
            let scale_range = match &ld.component.coordinates {
                CoordinateMap::Canonical => (1.0, 1.0),
                CoordinateMap::Scaled(s) => {
                    s.iter().fold((f64::INFINITY, 0.0f64), |(lo, hi), &v| {
                        (lo.min(v.abs()), hi.max(v.abs()))
                    })
                }
            };
            Meta {
                channels,
                n_elim: m.n_eliminated(),
                n_kept: m.n_kept(),
                form: ld.component.form,
                grounding: m.grounding,
                scale_range,
                surplus_min: m.ground_edges.iter().copied().fold(f64::INFINITY, f64::min),
                surplus_max: m.ground_edges.iter().copied().fold(0.0, f64::max),
                diag_max: m.diagonal.iter().copied().fold(0.0, f64::max),
            }
        })
        .collect();

    let (domains, _) =
        build_local_domains(&design, None, &ScalingConfig::default()).expect("domains again");
    let config = LocalSolverConfig::default();
    let entries: Vec<_> = domains
        .into_iter()
        .map(|d| build_entry(d, &config).expect("entry"))
        .collect();
    let max_scratch = entries.iter().map(|e| e.scratch_size()).max().unwrap();
    let mut rs = vec![0.0; max_scratch];
    let mut zs = vec![0.0; max_scratch];

    let apply = |x: &[f64], y: &mut [f64], rs: &mut [f64], zs: &mut [f64]| {
        y.fill(0.0);
        for e in &entries {
            e.apply_weighted_into_with_scratch(x, y, rs, zs, false)
                .expect("local solve");
        }
    };

    let mut seed = 0x9E3779B97F4A7C15u64;
    let mut rand = move || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 11) as f64 / (1u64 << 53) as f64) - 0.5
    };
    let normalize = |v: &mut [f64]| {
        let n = v.iter().map(|a| a * a).sum::<f64>().sqrt();
        for a in v.iter_mut() {
            *a /= n;
        }
        n
    };

    // Controls first: Rayleigh quotient of M⁻¹ on random unit directions.
    for trial in 0..3 {
        let mut v: Vec<f64> = (0..n_dofs).map(|_| rand()).collect();
        normalize(&mut v);
        let mut y = vec![0.0; n_dofs];
        apply(&v, &mut y, &mut rs, &mut zs);
        let q: f64 = v.iter().zip(&y).map(|(a, b)| a * b).sum();
        eprintln!("random control {trial}: vᵀM⁻¹v = {q:.3e}");
    }

    // Power iteration on M⁻¹ finds the amplified direction without assuming its form.
    let mut v: Vec<f64> = (0..n_dofs).map(|_| rand()).collect();
    normalize(&mut v);
    let mut y = vec![0.0; n_dofs];
    let mut q = 0.0;
    for it in 0..80 {
        apply(&v, &mut y, &mut rs, &mut zs);
        q = v.iter().zip(&y).map(|(a, b)| a * b).sum();
        v.copy_from_slice(&y);
        normalize(&mut v);
        if it % 10 == 9 {
            eprintln!("power it {}: rayleigh vᵀM⁻¹v = {q:.3e}", it + 1);
        }
    }

    // Where does the top direction live? Mass per channel.
    let mut mass = std::collections::BTreeMap::new();
    for (i, &vi) in v.iter().enumerate() {
        if let Some(a) = layout.address(i) {
            *mass
                .entry((a.channel.term, a.channel.column))
                .or_insert(0.0) += vi * vi;
        }
    }
    eprintln!("top-direction channel mass: {mass:?}");

    // Attribution: contribution of each subdomain to vᵀM⁻¹v.
    let mut contribs: Vec<(usize, f64, f64)> = entries
        .iter()
        .enumerate()
        .map(|(i, e)| {
            let mut out = vec![0.0; n_dofs];
            e.apply_weighted_into_with_scratch(&v, &mut out, &mut rs, &mut zs, false)
                .expect("local solve");
            let c: f64 = v.iter().zip(&out).map(|(a, b)| a * b).sum();
            let norm = out.iter().map(|a| a * a).sum::<f64>().sqrt();
            (i, c, norm)
        })
        .collect();
    contribs.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
    eprintln!("total rayleigh {q:.3e}; top subdomain contributions:");
    for &(i, c, norm) in contribs.iter().take(8) {
        let m = &metas[i];
        eprintln!(
            "  sub {i}: c={c:.3e} ‖z‖={norm:.3e} channels={:?} elim/kept={}/{} form={:?} grounding={:?} scales=[{:.2e},{:.2e}] surplus=[{:.2e},{:.2e}] diag_max={:.2e}",
            m.channels, m.n_elim, m.n_kept, m.form, m.grounding,
            m.scale_range.0, m.scale_range.1, m.surplus_min, m.surplus_max, m.diag_max
        );
    }

    // λmax(M⁻¹A) separates an exact-arithmetic blowup from a finite-precision-only hazard.
    if std::env::var_os("WITHIN_PROBE_GRAM").is_some() {
        use schwarz_precond::Operator;
        let gram = crate::operator::DesignOperator::new(&design, None);
        let mut obs = vec![0.0; design.n_obs];
        let mut av = vec![0.0; n_dofs];
        gram.apply(&v, &mut obs).expect("D v");
        gram.apply_adjoint(&obs, &mut av).expect("Dᵀ D v");
        let a_energy: f64 = v.iter().zip(&av).map(|(a, b)| a * b).sum();
        eprintln!("amplified direction: vᵀAv = {a_energy:.3e} vs vᵀM⁻¹v = {q:.3e}");

        let mut vg: Vec<f64> = (0..n_dofs).map(|_| rand()).collect();
        normalize(&mut vg);
        let mut lam = 0.0;
        for it in 0..200 {
            gram.apply(&vg, &mut obs).expect("D x");
            gram.apply_adjoint(&obs, &mut av).expect("Dᵀ D x");
            apply(&av, &mut y, &mut rs, &mut zs);
            lam = vg.iter().zip(&y).map(|(a, b)| a * b).sum();
            vg.copy_from_slice(&y);
            let growth = normalize(&mut vg);
            if it < 3 || it % 20 == 19 {
                eprintln!(
                    "gram power it {}: vᵀM⁻¹Av = {lam:.3e} growth = {growth:.3e}",
                    it + 1
                );
            }
        }
        eprintln!("λmax(M⁻¹A) ≈ {lam:.3e} over {} subdomains", entries.len());
    }
}

/// End-to-end LSMR on a shared-slope CSV panel: iterations, honesty of `converged`,
/// and the true NE residual. Run with RAYON_NUM_THREADS=1 for the deterministic trajectory.
#[test]
#[ignore]
fn probe_lsmr_shared_slope_end_to_end() {
    let path = std::env::var("WITHIN_PROBE_CSV").expect("set WITHIN_PROBE_CSV");
    let text = std::fs::read_to_string(&path).expect("read panel");
    let mut lines = text.lines();
    let _ = lines.next();
    let (mut w, mut f, mut x1, mut x2) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for line in lines {
        let mut t = line.split_whitespace();
        w.push(t.next().unwrap().parse::<u32>().unwrap());
        f.push(t.next().unwrap().parse::<u32>().unwrap());
        x1.push(t.next().unwrap().parse::<f64>().unwrap());
        x2.push(t.next().unwrap().parse::<f64>().unwrap());
    }
    let n = w.len();
    let mut seed = 42u64;
    let mut rand = move || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 11) as f64 / (1u64 << 53) as f64) - 0.5
    };
    let y: Vec<f64> = (0..n)
        .map(|i| (w[i] % 7) as f64 + (f[i] % 5) as f64 + 0.1 * x1[i] + rand())
        .collect();
    let effects = vec![
        Effect::new(&w, true, [&x1[..]]).expect("worker effect"),
        Effect::new(&f, true, [&x2[..]]).expect("firm effect"),
    ];
    let solver = crate::Solver::new(effects, None, None).expect("solver");
    let opts = crate::config::LsmrOptions {
        tol: 1e-12,
        maxiter: 20_000,
        ..Default::default()
    };
    let start = std::time::Instant::now();
    let r = solver.solve(&y, &opts).expect("solve");
    // True normal-equations residual from the reported demeaned vector.
    let effects2 = vec![
        Effect::new(&w, true, [&x1[..]]).expect("worker effect"),
        Effect::new(&f, true, [&x2[..]]).expect("firm effect"),
    ];
    let design = Design::new(effects2).expect("design");
    let layout = CoefficientLayout::from_design(&design);
    let mut dtr = vec![0.0f64; layout.n_dofs()];
    let mut dty = vec![0.0f64; layout.n_dofs()];
    for i in 0..n {
        for (term, (level, xval)) in [(w[i], x1[i]), (f[i], x2[i])].iter().enumerate() {
            for (col, load) in [1.0, *xval].iter().enumerate() {
                if let Some(idx) = layout.index(CoefficientAddress {
                    channel: Channel { term, column: col },
                    level: *level as usize,
                }) {
                    dtr[idx] += load * r.demeaned[i];
                    dty[idx] += load * y[i];
                }
            }
        }
    }
    let norm = |v: &[f64]| v.iter().map(|a| a * a).sum::<f64>().sqrt();
    eprintln!(
        "converged={} iters={} time={:.1}s true ‖Dᵀr‖/‖Dᵀy‖ = {:.3e}",
        r.converged,
        r.iterations,
        start.elapsed().as_secs_f64(),
        norm(&dtr) / norm(&dty)
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
