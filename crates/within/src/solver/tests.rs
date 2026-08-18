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

/// Middle-band probes (#281): inexact fused-block candidates, fill scaling, BLR rank decay.
mod middle_band_probe {
    use std::collections::HashMap;
    use std::ops::Range;
    use std::sync::Mutex;
    use std::time::Instant;

    use faer::sparse::linalg::cholesky::{
        factorize_symbolic_cholesky, CholeskySymbolicParams, SymmetricOrdering,
    };
    use faer::sparse::{SparseColMat, Triplet};
    use faer::Side;
    use schwarz_precond::{mlsmr, LsmrResult, MlsmrOptions, Operator};

    use super::{akm_panel, shared_covariate_panel, AkmPanel, SharedCovariatePanel};
    use crate::config::PreconditionerConfig;
    use crate::operator::design::gather_apply;
    use crate::operator::fused::{assemble_gram, FusedGram};
    use crate::operator::schwarz::Preconditioner;
    use crate::operator::DesignOperator;
    use crate::solver::Solver;
    use crate::Effect;

    const NONE: u32 = u32::MAX;

    /// Prescaled lower CSC of the Gram in elimination order (`order[new] = old`), diagonal first.
    struct ScaledLower {
        n: usize,
        scale: Vec<f64>,
        order: Vec<u32>,
        new_of_old: Vec<u32>,
        col_ptr: Vec<usize>,
        row_idx: Vec<u32>,
        values: Vec<f64>,
    }

    fn scaled_lower(gram: &FusedGram, order: &[u32]) -> ScaledLower {
        let n = gram.n_local;
        let scale: Vec<f64> = gram
            .diag
            .iter()
            .map(|&d| if d > 0.0 { 1.0 / d.sqrt() } else { 1.0 })
            .collect();
        let mut new_of_old = vec![0u32; n];
        for (new, &old) in order.iter().enumerate() {
            new_of_old[old as usize] = new as u32;
        }
        let mut col_ptr = vec![0usize; n + 2];
        for &(r, c) in gram.off.keys() {
            let (nr, nc) = (new_of_old[r as usize], new_of_old[c as usize]);
            col_ptr[nr.min(nc) as usize + 2] += 1;
        }
        for j in 0..n {
            col_ptr[j + 2] += 1;
        }
        for j in 2..n + 2 {
            col_ptr[j] += col_ptr[j - 1];
        }
        let nnz = col_ptr[n + 1];
        let mut row_idx = vec![0u32; nnz];
        let mut values = vec![0.0f64; nnz];
        for j in 0..n {
            let p = col_ptr[j + 1];
            col_ptr[j + 1] += 1;
            row_idx[p] = j as u32;
            values[p] = if gram.diag[order[j] as usize] > 0.0 {
                1.0
            } else {
                0.0
            };
        }
        for (&(r, c), &v) in &gram.off {
            let (nr, nc) = (new_of_old[r as usize], new_of_old[c as usize]);
            let (col, row) = (nr.min(nc) as usize, nr.max(nc));
            let p = col_ptr[col + 1];
            col_ptr[col + 1] += 1;
            row_idx[p] = row;
            values[p] = v * scale[r as usize] * scale[c as usize];
        }
        col_ptr.pop();
        let mut pairs: Vec<(u32, f64)> = Vec::new();
        for j in 0..n {
            let (s, e) = (col_ptr[j], col_ptr[j + 1]);
            pairs.clear();
            pairs.extend(
                row_idx[s..e]
                    .iter()
                    .copied()
                    .zip(values[s..e].iter().copied()),
            );
            pairs.sort_unstable_by_key(|&(i, _)| i);
            for (k, &(i, v)) in pairs.iter().enumerate() {
                row_idx[s + k] = i;
                values[s + k] = v;
            }
        }
        ScaledLower {
            n,
            scale,
            order: order.to_vec(),
            new_of_old,
            col_ptr,
            row_idx,
            values,
        }
    }

    /// Left-looking IC(tau) with a global diagonal shift retried on breakdown.
    struct IcFactor {
        col_ptr: Vec<usize>,
        row_idx: Vec<u32>,
        values: Vec<f64>,
        shift: f64,
    }

    const IC_NNZ_CAP: usize = 150_000_000;

    fn ic_factor(a: &ScaledLower, tau: f64) -> Option<IcFactor> {
        let n = a.n;
        let mut shift = 0.0f64;
        for _attempt in 0..30 {
            let mut col_ptr = Vec::with_capacity(n + 1);
            let mut row_idx: Vec<u32> = Vec::new();
            let mut values: Vec<f64> = Vec::new();
            let mut head = vec![NONE; n];
            let mut next = vec![NONE; n];
            let mut cur = vec![0usize; n];
            let mut w = vec![0.0f64; n];
            let mut in_pat = vec![false; n];
            let mut pattern: Vec<u32> = Vec::new();
            let mut kept: Vec<(u32, f64)> = Vec::new();
            col_ptr.push(0);
            let mut broke_down = false;
            for j in 0..n {
                for p in a.col_ptr[j]..a.col_ptr[j + 1] {
                    let i = a.row_idx[p] as usize;
                    w[i] = a.values[p];
                    in_pat[i] = true;
                    pattern.push(i as u32);
                }
                w[j] += shift;
                let mut k = head[j];
                head[j] = NONE;
                while k != NONE {
                    let ku = k as usize;
                    let kn = next[ku];
                    let kp = cur[ku];
                    let kend = col_ptr[ku + 1];
                    let ljk = values[kp];
                    for p in kp..kend {
                        let i = row_idx[p] as usize;
                        if !in_pat[i] {
                            in_pat[i] = true;
                            pattern.push(i as u32);
                            w[i] = 0.0;
                        }
                        w[i] -= ljk * values[p];
                    }
                    cur[ku] = kp + 1;
                    if kp + 1 < kend {
                        let r = row_idx[kp + 1] as usize;
                        next[ku] = head[r];
                        head[r] = k;
                    }
                    k = kn;
                }
                let d = w[j];
                if d <= 1e-10 {
                    shift = if shift == 0.0 { 1e-3 } else { shift * 4.0 };
                    broke_down = true;
                    for &i in &pattern {
                        w[i as usize] = 0.0;
                        in_pat[i as usize] = false;
                    }
                    pattern.clear();
                    break;
                }
                let sd = d.sqrt();
                let col_start = values.len();
                row_idx.push(j as u32);
                values.push(sd);
                kept.clear();
                for &iu in &pattern {
                    let i = iu as usize;
                    if i > j {
                        let v = w[i] / sd;
                        if v.abs() > tau {
                            kept.push((iu, v));
                        }
                    }
                    w[i] = 0.0;
                    in_pat[i] = false;
                }
                pattern.clear();
                kept.sort_unstable_by_key(|&(i, _)| i);
                for &(i, v) in &kept {
                    row_idx.push(i);
                    values.push(v);
                }
                col_ptr.push(values.len());
                if !kept.is_empty() {
                    cur[j] = col_start + 1;
                    let r = kept[0].0 as usize;
                    next[j] = head[r];
                    head[r] = j as u32;
                }
                if values.len() > IC_NNZ_CAP {
                    eprintln!(
                        "  ic(tau={tau:.0e}): nnz cap {IC_NNZ_CAP} exceeded at column {j}/{n}"
                    );
                    return None;
                }
            }
            if !broke_down {
                return Some(IcFactor {
                    col_ptr,
                    row_idx,
                    values,
                    shift,
                });
            }
        }
        None
    }

    /// Static-pattern FSAI (pattern of the prescaled Gram): `A^-1 ≈ G^T G`.
    struct FsaiFactor {
        row_ptr: Vec<usize>,
        col_idx: Vec<u32>,
        values: Vec<f64>,
    }

    fn fsai_factor(gram: &FusedGram, scale: &[f64]) -> FsaiFactor {
        let n = gram.n_local;
        let mut entries: HashMap<(u32, u32), f64> = HashMap::with_capacity(gram.off.len());
        let mut lower_adj: Vec<Vec<u32>> = vec![Vec::new(); n];
        for (&(r, c), &v) in &gram.off {
            let sv = v * scale[r as usize] * scale[c as usize];
            entries.insert((r.min(c), r.max(c)), sv);
            lower_adj[r.max(c) as usize].push(r.min(c));
        }
        let unit_diag: Vec<f64> = gram
            .diag
            .iter()
            .map(|&d| if d > 0.0 { 1.0 } else { 0.0 })
            .collect();
        let mut row_ptr = Vec::with_capacity(n + 1);
        let mut col_idx: Vec<u32> = Vec::new();
        let mut values: Vec<f64> = Vec::new();
        row_ptr.push(0);
        let mut small = vec![0.0f64; 64 * 64];
        let mut rhs = vec![0.0f64; 64];
        for (i, adj) in lower_adj.iter_mut().enumerate() {
            adj.sort_unstable();
            let mut idx: Vec<u32> = adj.clone();
            idx.push(i as u32);
            let m = idx.len();
            if small.len() < m * m {
                small.resize(m * m, 0.0);
                rhs.resize(m, 0.0);
            }
            for a in 0..m {
                for b in 0..=a {
                    let (r, c) = (idx[b].min(idx[a]), idx[b].max(idx[a]));
                    let v = if r == c {
                        unit_diag[r as usize] + 1e-10
                    } else {
                        entries.get(&(r, c)).copied().unwrap_or(0.0)
                    };
                    small[a * m + b] = v;
                    small[b * m + a] = v;
                }
            }
            for r in rhs.iter_mut().take(m) {
                *r = 0.0;
            }
            rhs[m - 1] = 1.0;
            assert!(
                dense_chol_solve(&mut small[..m * m], m, &mut rhs[..m]),
                "fsai row {i}: dense solve broke down"
            );
            let g_ii = rhs[m - 1].max(1e-30);
            let inv_sqrt = 1.0 / g_ii.sqrt();
            for (a, &col) in idx.iter().enumerate() {
                let v = rhs[a] * inv_sqrt;
                if v != 0.0 {
                    col_idx.push(col);
                    values.push(v);
                }
            }
            row_ptr.push(values.len());
        }
        FsaiFactor {
            row_ptr,
            col_idx,
            values,
        }
    }

    fn dense_chol_solve(a: &mut [f64], m: usize, rhs: &mut [f64]) -> bool {
        for j in 0..m {
            let mut d = a[j * m + j];
            for k in 0..j {
                d -= a[j * m + k] * a[j * m + k];
            }
            if d <= 1e-14 {
                d = 1e-12;
            }
            let sd = d.sqrt();
            a[j * m + j] = sd;
            for i in j + 1..m {
                let mut v = a[i * m + j];
                for k in 0..j {
                    v -= a[i * m + k] * a[j * m + k];
                }
                a[i * m + j] = v / sd;
            }
        }
        for i in 0..m {
            let mut v = rhs[i];
            for k in 0..i {
                v -= a[i * m + k] * rhs[k];
            }
            rhs[i] = v / a[i * m + i];
        }
        for i in (0..m).rev() {
            let mut v = rhs[i];
            for k in i + 1..m {
                v -= a[k * m + i] * rhs[k];
            }
            rhs[i] = v / a[i * m + i];
        }
        true
    }

    /// In-place approximate `A_fused^-1` on a span-gathered local vector (old order).
    trait FusedApprox: Sync {
        fn solve(&self, local: &mut [f64], work: &mut [f64]);
    }

    struct IcApprox<'a> {
        a: &'a ScaledLower,
        f: &'a IcFactor,
    }

    impl FusedApprox for IcApprox<'_> {
        fn solve(&self, local: &mut [f64], work: &mut [f64]) {
            let n = self.a.n;
            for old in 0..n {
                work[self.a.new_of_old[old] as usize] = local[old] * self.a.scale[old];
            }
            for j in 0..n {
                let (p0, p1) = (self.f.col_ptr[j], self.f.col_ptr[j + 1]);
                let zj = work[j] / self.f.values[p0];
                work[j] = zj;
                for p in p0 + 1..p1 {
                    work[self.f.row_idx[p] as usize] -= self.f.values[p] * zj;
                }
            }
            for j in (0..n).rev() {
                let (p0, p1) = (self.f.col_ptr[j], self.f.col_ptr[j + 1]);
                let mut s = work[j];
                for p in p0 + 1..p1 {
                    s -= self.f.values[p] * work[self.f.row_idx[p] as usize];
                }
                work[j] = s / self.f.values[p0];
            }
            for (new, &old) in self.a.order.iter().enumerate() {
                local[old as usize] = work[new] * self.a.scale[old as usize];
            }
        }
    }

    struct FsaiApprox<'a> {
        scale: &'a [f64],
        f: &'a FsaiFactor,
    }

    impl FusedApprox for FsaiApprox<'_> {
        fn solve(&self, local: &mut [f64], work: &mut [f64]) {
            let n = local.len();
            for (l, &s) in local.iter_mut().zip(self.scale) {
                *l *= s;
            }
            for (i, wi) in work[..n].iter_mut().enumerate() {
                let mut z = 0.0;
                for p in self.f.row_ptr[i]..self.f.row_ptr[i + 1] {
                    z += self.f.values[p] * local[self.f.col_idx[p] as usize];
                }
                *wi = z;
            }
            local[..n].fill(0.0);
            for (i, &zi) in work[..n].iter().enumerate() {
                for p in self.f.row_ptr[i]..self.f.row_ptr[i + 1] {
                    local[self.f.col_idx[p] as usize] += self.f.values[p] * zi;
                }
            }
            for (l, &s) in local.iter_mut().zip(self.scale) {
                *l *= s;
            }
        }
    }

    /// Probe stand-in for `FusedPreconditioner`: base Schwarz plus an approximate fused solve.
    struct AdditiveProbe<'a, F: FusedApprox + ?Sized> {
        base: &'a Preconditioner,
        spans: Vec<Range<usize>>,
        factor: &'a F,
        scratch: Mutex<(Vec<f64>, Vec<f64>)>,
    }

    impl<'a, F: FusedApprox + ?Sized> AdditiveProbe<'a, F> {
        fn new(base: &'a Preconditioner, spans: &[Range<usize>], factor: &'a F) -> Self {
            let n_local: usize = spans.iter().map(|s| s.len()).sum();
            Self {
                base,
                spans: spans.to_vec(),
                factor,
                scratch: Mutex::new((vec![0.0; n_local], vec![0.0; n_local])),
            }
        }

        fn solve_add(&self, x: &[f64], y: &mut [f64]) {
            let mut guard = self.scratch.lock().unwrap();
            let (local, work) = &mut *guard;
            let mut base = 0;
            for span in &self.spans {
                local[base..base + span.len()].copy_from_slice(&x[span.clone()]);
                base += span.len();
            }
            self.factor.solve(local, work);
            let mut base = 0;
            for span in &self.spans {
                for (yi, &li) in y[span.clone()].iter_mut().zip(&local[base..]) {
                    *yi += li;
                }
                base += span.len();
            }
        }
    }

    impl<F: FusedApprox + ?Sized> Operator for AdditiveProbe<'_, F> {
        fn nrows(&self) -> usize {
            self.base.nrows()
        }

        fn ncols(&self) -> usize {
            self.base.ncols()
        }

        fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), schwarz_precond::SolveError> {
            self.base.apply(x, y)?;
            self.solve_add(x, y);
            Ok(())
        }

        fn apply_adjoint(
            &self,
            x: &[f64],
            y: &mut [f64],
        ) -> Result<(), schwarz_precond::SolveError> {
            <Preconditioner as Operator>::apply_adjoint(self.base, x, y)?;
            self.solve_add(x, y);
            Ok(())
        }
    }

    fn probe_run<M: Operator>(
        solver: &Solver<'_>,
        y: &[f64],
        m: &M,
        tol: f64,
        maxiter: usize,
    ) -> (LsmrResult, f64) {
        let y_int = solver.design.permute_obs_in(y);
        let rect_op = DesignOperator::new(&solver.design, solver.sqrt_weights.as_deref());
        let b = rect_op.weighted_rhs(&y_int);
        let t = Instant::now();
        let r = mlsmr(&rect_op, &b, m, tol, maxiter, MlsmrOptions::default()).unwrap();
        (r, t.elapsed().as_secs_f64())
    }

    fn demeaned_from(solver: &Solver<'_>, x: &[f64], y: &[f64]) -> Vec<f64> {
        let y_int = solver.design.permute_obs_in(y);
        let mut d = vec![0.0; solver.design.n_obs];
        gather_apply(&solver.design, x, &mut d, None);
        for (di, &yi) in d.iter_mut().zip(y_int.iter()) {
            *di = yi - *di;
        }
        solver.design.permute_obs_out(d)
    }

    /// AMD order of the fused Gram (via faer's symbolic pass) plus the exact-fill count.
    fn amd_order(gram: &FusedGram) -> (Vec<u32>, Vec<u32>, usize) {
        let n = gram.n_local;
        let mut triplets = Vec::with_capacity(n + gram.off.len());
        for d in 0..n {
            triplets.push(Triplet::new(d, d, 1.0f64));
        }
        for &(r, c) in gram.off.keys() {
            triplets.push(Triplet::new(r as usize, c as usize, 1.0f64));
        }
        let a_upper = SparseColMat::<usize, f64>::try_new_from_triplets(n, n, &triplets).unwrap();
        let symbolic = factorize_symbolic_cholesky(
            a_upper.symbolic(),
            Side::Upper,
            SymmetricOrdering::Amd,
            CholeskySymbolicParams::default(),
        )
        .unwrap();
        let perm = symbolic.perm().expect("AMD ordering must produce a perm");
        let (fwd, inv) = perm.arrays();
        let fwd: Vec<u32> = fwd.iter().map(|&i| i as u32).collect();
        let inv: Vec<u32> = inv.iter().map(|&i| i as u32).collect();
        (fwd, inv, symbolic.len_val())
    }

    /// 20k-panel leg: the corrected solve must reproduce the reference, not just claim convergence.
    fn small_correctness(eps: f64, use_fsai: bool) {
        let SharedCovariatePanel { a, b, z, z2, y } = shared_covariate_panel(eps);
        let effects = || {
            vec![
                Effect::new(&a, true, [&z[..]]).unwrap(),
                Effect::new(&b, true, [&z2[..]]).unwrap(),
            ]
        };
        let solver = Solver::new(effects(), None, None).unwrap();
        let base = solver.preconditioner.as_ref().unwrap();
        let gram = assemble_gram(&solver.design, None, &[0, 1]);
        let scale_vec: Vec<f64> = gram
            .diag
            .iter()
            .map(|&d| if d > 0.0 { 1.0 / d.sqrt() } else { 1.0 })
            .collect();
        let (fwd, _inv, fill) = amd_order(&gram);
        let a_scaled = scaled_lower(&gram, &fwd);
        let ic;
        let fsai;
        let ic_approx;
        let fsai_approx;
        let (label, op): (&str, AdditiveProbe<'_, _>) = if use_fsai {
            fsai = fsai_factor(&gram, &scale_vec);
            eprintln!(
                "small fsai: exact fill {} | nnz {}",
                fill,
                fsai.values.len()
            );
            fsai_approx = FsaiApprox {
                scale: &scale_vec,
                f: &fsai,
            };
            (
                "fsai",
                AdditiveProbe::new(base, &gram.spans, &fsai_approx as &dyn FusedApprox),
            )
        } else {
            ic = ic_factor(&a_scaled, 1e-3).expect("IC must not break down here");
            eprintln!(
                "small ic: exact fill {} | nnz {} shift {:.1e}",
                fill,
                ic.values.len(),
                ic.shift
            );
            ic_approx = IcApprox {
                a: &a_scaled,
                f: &ic,
            };
            (
                "ic",
                AdditiveProbe::new(base, &gram.spans, &ic_approx as &dyn FusedApprox),
            )
        };
        let (r, secs) = probe_run(&solver, &y, &op, 1e-12, 20_000);
        eprintln!(
            "small {label} (eps={eps:e}): it={} conv={} {:.2}s",
            r.iterations, r.converged, secs
        );
        assert!(r.converged);

        let reference = Solver::new(effects(), None, PreconditionerConfig::Off)
            .unwrap()
            .solve(
                &y,
                &crate::config::LsmrOptions {
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
        eprintln!("small {label}: demeaned diff {diff:e} vs scale {scale:e}");
        assert!(diff <= 1e-6 * scale, "demeaned mismatch: {diff:e}");
    }

    #[test]
    #[ignore]
    fn probe_ic_small_correctness() {
        small_correctness(1e-4, false);
    }

    #[test]
    #[ignore]
    fn probe_ic_small_exact_sharing() {
        small_correctness(0.0, false);
    }

    #[test]
    #[ignore]
    fn probe_fsai_small_correctness() {
        small_correctness(1e-4, true);
    }

    #[test]
    #[ignore]
    fn probe_fsai_small_exact_sharing() {
        small_correctness(0.0, true);
    }

    /// FSAI at delta=0.05: does a local approximate inverse survive the long-range continuum?
    #[test]
    #[ignore]
    fn probe_low_mobility_fsai() {
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
        assert!(
            !solver.fused.is_empty(),
            "exact path must arm here (fill 11.4M < cap)"
        );
        let base = solver.preconditioner.as_ref().unwrap();
        let gram = assemble_gram(&solver.design, None, &[0, 1]);
        let scale: Vec<f64> = gram
            .diag
            .iter()
            .map(|&d| if d > 0.0 { 1.0 / d.sqrt() } else { 1.0 })
            .collect();
        let t = Instant::now();
        let fsai = fsai_factor(&gram, &scale);
        let factor_secs = t.elapsed().as_secs_f64();
        let approx = FsaiApprox {
            scale: &scale,
            f: &fsai,
        };
        let op = AdditiveProbe::new(base, &gram.spans, &approx);
        let (r, secs) = probe_run(&solver, &y, &op, 1e-10, 3000);
        eprintln!(
            "delta=0.05 fsai: nnz={} factor {:.2}s | it={} conv={} {:.2}s (exact fused: 26 it)",
            fsai.values.len(),
            factor_secs,
            r.iterations,
            r.converged,
            secs
        );

        // Certify against the production exact-fused solve (reference-matched elsewhere).
        let exact = solver
            .solve(
                &y,
                &crate::config::LsmrOptions {
                    tol: 1e-10,
                    maxiter: 3000,
                    ..Default::default()
                },
            )
            .unwrap();
        assert!(exact.converged);
        let fsai_demeaned = demeaned_from(&solver, &r.x, &y);
        let norm = y.iter().map(|&v| v * v).sum::<f64>().sqrt();
        let diff = fsai_demeaned
            .iter()
            .zip(&exact.demeaned)
            .map(|(&p, &q)| (p - q) * (p - q))
            .sum::<f64>()
            .sqrt();
        eprintln!("delta=0.05 fsai: demeaned diff vs exact {diff:e} (scale {norm:e})");
        assert!(
            diff <= 1e-5 * norm,
            "fsai solution disagrees with exact: {diff:e}"
        );
    }

    /// FSAI at delta=0.5 (gate also refuses): must not regress the healthy Schwarz baseline.
    #[test]
    #[ignore]
    fn probe_well_connected_fsai() {
        let AkmPanel {
            worker,
            firm,
            year,
            z1,
            fz,
            y,
        } = akm_panel(0.5);
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
        assert!(
            solver.fused.is_empty(),
            "fill gate must refuse at delta=0.5"
        );
        let base = solver.preconditioner.as_ref().unwrap();

        let (r, secs) = probe_run(&solver, &y, base, 1e-10, 3000);
        eprintln!(
            "delta=0.5 baseline: it={} conv={} {:.2}s",
            r.iterations, r.converged, secs
        );

        let gram = assemble_gram(&solver.design, None, &[0, 1]);
        let scale: Vec<f64> = gram
            .diag
            .iter()
            .map(|&d| if d > 0.0 { 1.0 / d.sqrt() } else { 1.0 })
            .collect();
        let t = Instant::now();
        let fsai = fsai_factor(&gram, &scale);
        let factor_secs = t.elapsed().as_secs_f64();
        let approx = FsaiApprox {
            scale: &scale,
            f: &fsai,
        };
        let op = AdditiveProbe::new(base, &gram.spans, &approx);
        let (r, secs) = probe_run(&solver, &y, &op, 1e-10, 3000);
        eprintln!(
            "delta=0.5 fsai: nnz={} factor {:.2}s | it={} conv={} {:.2}s",
            fsai.values.len(),
            factor_secs,
            r.iterations,
            r.converged,
            secs
        );
    }

    /// Fill-per-row scaling at fixed delta=0.05 (whitening changes values, not the pattern).
    #[test]
    #[ignore]
    fn probe_fill_scaling_low_mobility() {
        use crate::domain::Design;
        use crate::test_rng::Lcg;

        for scale in [1usize, 2, 4, 8] {
            let (n_workers, n_firms, n_time) = (50_000 * scale, 21_000 * scale, 10usize);
            let n = n_workers * n_time;
            let mut rng = Lcg(0xC0FFEE ^ scale as u64);
            let mut worker = Vec::with_capacity(n);
            let mut firm = Vec::with_capacity(n);
            for wi in 0..n_workers {
                let mut current = (rng.next_u64() as usize) % n_firms;
                for t in 0..n_time {
                    if t > 0 && rng.uniform() < 0.05 {
                        current = (rng.next_u64() as usize) % n_firms;
                    }
                    worker.push(wi as u32);
                    firm.push(current as u32);
                }
            }
            let z1: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
            let z2: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
            let fz: Vec<f64> = z1.iter().zip(&z2).map(|(&a, &b)| a + 1e-3 * b).collect();
            let design = Design::new(vec![
                Effect::new(&worker, true, [&z1[..]]).unwrap(),
                Effect::new(&firm, true, [&fz[..]]).unwrap(),
            ])
            .unwrap();
            let t = Instant::now();
            let gram = assemble_gram(&design, None, &[0, 1]);
            let (_fwd, _inv, fill) = amd_order(&gram);
            eprintln!(
                "scale {scale}x: n_local={} nnz(off)={} fill={} fill/row={:.1} bytes~{}MB ({:.1}s)",
                gram.n_local,
                gram.off.len(),
                fill,
                fill as f64 / gram.n_local as f64,
                fill * 8 / 1_000_000,
                t.elapsed().as_secs_f64()
            );
        }
    }

    /// BLR feasibility: eps-ranks of frontal blocks of the exact factor; median@1e-8 <= 64 reopens.
    #[test]
    #[ignore]
    fn probe_blr_rank_decay() {
        use crate::domain::Design;
        use crate::test_rng::Lcg;

        let (n_workers, n_firms, n_time) = (100_000usize, 42_000usize, 10usize);
        let n_obs = n_workers * n_time;
        let mut rng = Lcg(0xC0FFEE ^ 2);
        let mut worker = Vec::with_capacity(n_obs);
        let mut firm = Vec::with_capacity(n_obs);
        for wi in 0..n_workers {
            let mut current = (rng.next_u64() as usize) % n_firms;
            for t in 0..n_time {
                if t > 0 && rng.uniform() < 0.05 {
                    current = (rng.next_u64() as usize) % n_firms;
                }
                worker.push(wi as u32);
                firm.push(current as u32);
            }
        }
        let z1: Vec<f64> = (0..n_obs).map(|_| rng.normal()).collect();
        let z2: Vec<f64> = (0..n_obs).map(|_| rng.normal()).collect();
        let fz: Vec<f64> = z1.iter().zip(&z2).map(|(&a, &b)| a + 1e-3 * b).collect();
        let design = Design::new(vec![
            Effect::new(&worker, true, [&z1[..]]).unwrap(),
            Effect::new(&firm, true, [&fz[..]]).unwrap(),
        ])
        .unwrap();
        let gram = assemble_gram(&design, None, &[0, 1]);
        let (fwd, _inv, fill) = amd_order(&gram);
        let mut a_scaled = scaled_lower(&gram, &fwd);
        let n = a_scaled.n;
        eprintln!("n={n} exact fill={fill}");
        for j in 0..n {
            a_scaled.values[a_scaled.col_ptr[j]] += 1e-3;
        }
        let t = Instant::now();
        let ic = ic_factor(&a_scaled, 0.0).expect("exact factor");
        eprintln!(
            "exact factor: nnz={} extra shift={:.1e} ({:.1}s)",
            ic.values.len(),
            ic.shift,
            t.elapsed().as_secs_f64()
        );

        for d in 0..10 {
            let (s, e) = (n * d / 10, n * (d + 1) / 10);
            let nnz = ic.col_ptr[e] - ic.col_ptr[s];
            eprintln!(
                "  decile {d}: avg col nnz {:.1}",
                nnz as f64 / (e - s) as f64
            );
        }

        // BLR blocks live on a front's scattered row support, not on contiguous windows.
        const B: usize = 512;
        let (c_star, c_nnz) = (0..n)
            .map(|j| (j, ic.col_ptr[j + 1] - ic.col_ptr[j]))
            .max_by_key(|&(_, l)| l)
            .unwrap();
        eprintln!(
            "densest column {c_star} ({:.1}% into order) nnz {c_nnz}",
            100.0 * c_star as f64 / n as f64
        );
        let support: Vec<u32> = ic.row_idx[ic.col_ptr[c_star] + 1..ic.col_ptr[c_star + 1]].to_vec();
        let m = support.len();
        let pos: std::collections::HashMap<u32, usize> =
            support.iter().enumerate().map(|(i, &r)| (r, i)).collect();
        let k = (4 * B).min(n - c_star - 1);
        let mut panel = faer::Mat::<f64>::zeros(m, k);
        let (mut covered, mut total) = (0usize, 0usize);
        for (cj, c) in (c_star + 1..c_star + 1 + k).enumerate() {
            for p in ic.col_ptr[c] + 1..ic.col_ptr[c + 1] {
                total += 1;
                if let Some(&i) = pos.get(&ic.row_idx[p]) {
                    panel[(i, cj)] = ic.values[p];
                    covered += 1;
                }
            }
        }
        eprintln!(
            "front panel {m}x{k}: support coverage {covered}/{total} ({:.1}%)",
            100.0 * covered as f64 / total.max(1) as f64
        );

        let rank_profile = |mv: faer::MatRef<'_, f64>| -> (f64, [usize; 3]) {
            let nnz = (0..mv.ncols())
                .flat_map(|c| (0..mv.nrows()).map(move |r| (r, c)))
                .filter(|&(r, c)| mv[(r, c)] != 0.0)
                .count();
            let sv = mv.singular_values().unwrap();
            let s1 = sv[0].max(1e-300);
            let ranks = [1e-2, 1e-4, 1e-8].map(|tol| sv.iter().filter(|&&v| v > tol * s1).count());
            (nnz as f64 / (mv.nrows() * mv.ncols()) as f64, ranks)
        };

        // BLR-style 512x512 blocks on the front's index sets.
        let mut ranks8 = Vec::new();
        for rb in 0..(m / B).min(6) {
            for cb in 0..k / B {
                let mv = panel.as_ref().submatrix(rb * B, cb * B, B, B);
                let (density, ranks) = rank_profile(mv);
                ranks8.push(ranks[2]);
                eprintln!(
                    "front block[{rb},{cb}] ({B}x{B}): density {:.2} rank@1e-2/1e-4/1e-8 = {}/{}/{}",
                    density, ranks[0], ranks[1], ranks[2]
                );
            }
        }
        ranks8.sort_unstable();
        eprintln!(
            "front-block median rank@1e-8 = {} of {B} (reopen threshold: <=64)",
            ranks8[ranks8.len() / 2]
        );

        // HSS-style: the whole tall panel per column chunk.
        for cb in 0..k / B {
            let mv = panel.as_ref().submatrix(0, cb * B, m, B);
            let (density, ranks) = rank_profile(mv);
            eprintln!(
                "front panel[:, {cb}] ({m}x{B}): density {:.2} rank@1e-2/1e-4/1e-8 = {}/{}/{}",
                density, ranks[0], ranks[1], ranks[2]
            );
        }
    }

    /// The measurement: delta=0.2 mid-scale AKM (exact fill 539M, gate refuses).
    #[test]
    #[ignore]
    fn probe_middle_band_ic_fsai() {
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
        assert!(!solver.warnings().is_empty(), "screen must fire");
        assert!(
            solver.fused.is_empty(),
            "fill gate must refuse at delta=0.2"
        );
        let base = solver.preconditioner.as_ref().unwrap();
        let (tol, maxiter) = (1e-10, 3000);

        let (r, secs) = probe_run(&solver, &y, base, tol, maxiter);
        eprintln!(
            "baseline schwarz: it={} conv={} {:.2}s",
            r.iterations, r.converged, secs
        );

        let gram = assemble_gram(&solver.design, None, &[0, 1]);
        let t = Instant::now();
        let (fwd, inv, fill) = amd_order(&gram);
        eprintln!(
            "fused block: n={} nnz(off)={} exact fill={} (symbolic {:.2}s)",
            gram.n_local,
            gram.off.len(),
            fill,
            t.elapsed().as_secs_f64()
        );

        // Perm orientation settled empirically: factor both at a cheap tau, keep the lower fill.
        let t = Instant::now();
        let a_fwd = scaled_lower(&gram, &fwd);
        let a_inv = scaled_lower(&gram, &inv);
        let nnz_fwd = ic_factor(&a_fwd, 1e-2).map(|f| f.values.len());
        let nnz_inv = ic_factor(&a_inv, 1e-2).map(|f| f.values.len());
        eprintln!(
            "orientation: fwd nnz {nnz_fwd:?} | inv nnz {nnz_inv:?} ({:.2}s)",
            t.elapsed().as_secs_f64()
        );
        let a_scaled = if nnz_fwd.unwrap_or(usize::MAX) <= nnz_inv.unwrap_or(usize::MAX) {
            a_fwd
        } else {
            a_inv
        };

        let mut ic_demeaned: Option<Vec<f64>> = None;
        for tau in [1e-2, 3e-3, 1e-3, 3e-4] {
            let t = Instant::now();
            let Some(ic) = ic_factor(&a_scaled, tau) else {
                eprintln!("ic(tau={tau:.0e}): declined (cap or breakdown)");
                continue;
            };
            let factor_secs = t.elapsed().as_secs_f64();
            let approx = IcApprox {
                a: &a_scaled,
                f: &ic,
            };
            let op = AdditiveProbe::new(base, &gram.spans, &approx);
            let (r, secs) = probe_run(&solver, &y, &op, tol, maxiter);
            eprintln!(
                "ic(tau={tau:.0e}): nnz={} shift={:.1e} factor {:.2}s | it={} conv={} {:.2}s",
                ic.values.len(),
                ic.shift,
                factor_secs,
                r.iterations,
                r.converged,
                secs
            );
            if ic_demeaned.is_none() && r.converged {
                ic_demeaned = Some(demeaned_from(&solver, &r.x, &y));
            }
        }

        let t = Instant::now();
        let scale: Vec<f64> = gram
            .diag
            .iter()
            .map(|&d| if d > 0.0 { 1.0 / d.sqrt() } else { 1.0 })
            .collect();
        let fsai = fsai_factor(&gram, &scale);
        let factor_secs = t.elapsed().as_secs_f64();
        let approx = FsaiApprox {
            scale: &scale,
            f: &fsai,
        };
        let op = AdditiveProbe::new(base, &gram.spans, &approx);
        let (r, secs) = probe_run(&solver, &y, &op, tol, maxiter);
        eprintln!(
            "fsai(pattern A): nnz={} factor {:.2}s | it={} conv={} {:.2}s",
            fsai.values.len(),
            factor_secs,
            r.iterations,
            r.converged,
            secs
        );

        // Two independent preconditioners agreeing on demeaned = neither faked convergence.
        let fsai_demeaned = demeaned_from(&solver, &r.x, &y);
        let ic_demeaned = ic_demeaned.expect("at least one IC run must converge");
        let scale = y.iter().map(|&v| v * v).sum::<f64>().sqrt();
        let diff = fsai_demeaned
            .iter()
            .zip(&ic_demeaned)
            .map(|(&p, &q)| (p - q) * (p - q))
            .sum::<f64>()
            .sqrt();
        eprintln!("cross-check: fsai-vs-ic demeaned diff {diff:e} vs scale {scale:e}");
        assert!(diff <= 1e-5 * scale, "solutions disagree: {diff:e}");
    }
}
