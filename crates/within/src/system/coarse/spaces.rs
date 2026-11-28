#![allow(clippy::needless_range_loop, clippy::type_complexity)] // PROTOTYPE(#612)
//! PROTOTYPE(#612): coarse spaces on the assembled `S`, chosen by `PROBE_SPACE`.

use super::assembled::AssembledS;
use super::assembled::{
    coarse_from_columns, dense_columns, lower_triplets, small_unknowns, LevelGrams,
};
use super::rigid::{AnchorGauges, SmallColumns};
use super::{CoarseDecline, CoarseSolve};
use crate::system::ReducedDesign;

/// `PROBE_SPACE`: which coarse space the coarse rung builds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Space {
    /// #607's rigid glue and lift.
    Rigid,
    /// One column per connected component and channel of the anchor term, on the assembled `S`.
    Components,
    /// The same columns through the rigid lift, from the intercept channel's graph.
    ComponentsLift,
    /// The rigid columns, recomputed on the assembled `S`.
    RigidAssembled,
    /// The rigid columns plus every small kept term's unknowns, on the assembled `S`.
    RigidSmall,
    /// The same through the rigid lift.
    RigidSmallLift,
    /// Smoothed aggregation: aggregates of anchor levels by connection strength in `S`, the `p`
    /// gauges on each, one damped-Jacobi step.
    Sa,
    /// Unsmoothed aggregation through the rigid lift: aggregates of anchor levels by the
    /// intercept channel's connection strength in `S`, the year-like small terms as columns.
    Aggregates,
    /// Algebraic GenEO on aggregates: each aggregate's eigenvectors of the pencil
    /// `(Σ_g Schur(S_g → aggregate), S_aggregate)` below `τ`, the small terms as columns.
    Geneo,
}

pub(crate) fn space() -> Space {
    static SPACE: std::sync::OnceLock<Space> = std::sync::OnceLock::new();
    *SPACE.get_or_init(|| match std::env::var("PROBE_SPACE").as_deref() {
        Err(_) | Ok("") | Ok("rigid") => Space::Rigid,
        Ok("components") => Space::Components,
        Ok("components-lift") => Space::ComponentsLift,
        Ok("rigid-small-lift") => Space::RigidSmallLift,
        Ok("rigid-assembled") => Space::RigidAssembled,
        Ok("rigid-small") => Space::RigidSmall,
        Ok("sa") => Space::Sa,
        Ok("agg") => Space::Aggregates,
        Ok("geneo") => Space::Geneo,
        Ok(other) => panic!("PROBE_SPACE={other}"),
    })
}

fn verbose(what: &str, started: std::time::Instant) {
    if std::env::var_os("PROBE_COARSE").is_some() {
        eprintln!(
            "[probe-space] {what} secs={:.3}",
            started.elapsed().as_secs_f64()
        );
    }
}

/// The coarse solve of `space` over `design`'s unknowns.
pub(super) fn build(
    design: &ReducedDesign<'_>,
    space: Space,
) -> Result<CoarseSolve, CoarseDecline> {
    let started = std::time::Instant::now();
    let budget = design.split.n_obs();
    if space == Space::Rigid {
        return super::rigid::build(&design.split);
    }
    if matches!(space, Space::Aggregates | Space::ComponentsLift) {
        let solve = aggregates(design, space == Space::ComponentsLift);
        verbose("aggregates", started);
        return solve;
    }
    if space == Space::RigidSmallLift {
        let small = small_columns(design);
        let solve = super::rigid::build_with(&design.split, Some(&small));
        verbose("rigid-small-lift", started);
        return solve;
    }
    let s = AssembledS::assemble(design);
    verbose(&format!("assembled {s:?}"), started);
    let columns = match space {
        Space::Rigid | Space::Aggregates | Space::ComponentsLift | Space::RigidSmallLift => {
            unreachable!()
        }
        Space::Components => components(&s, design)?,
        Space::Sa => smoothed_aggregation(&s, design)?,
        Space::Geneo => geneo(&s, design)?,
        Space::RigidAssembled | Space::RigidSmall => {
            let rigid = super::rigid::build(&design.split)?;
            let mut columns = rigid.z_columns();
            if space == Space::RigidSmall {
                columns.extend(s.small.iter().map(|&u| vec![(u, 1.0)]));
            }
            columns
        }
    };
    verbose(&format!("columns={}", columns.len()), started);
    let solve = coarse_from_columns(&s, columns, budget);
    verbose("coarse", started);
    solve
}

fn find(parent: &mut [u32], mut x: u32) -> u32 {
    while parent[x as usize] != x {
        let g = parent[parent[x as usize] as usize];
        parent[x as usize] = g;
        x = g;
    }
    x
}

/// The anchor's gauges, one column per connected component of `S`'s large block and channel,
/// each scaled to unit norm over its component.
fn components(
    s: &AssembledS,
    design: &ReducedDesign<'_>,
) -> Result<Vec<Vec<(u32, f64)>>, CoarseDecline> {
    let anchor = AnchorGauges::of(&design.split).ok_or(CoarseDecline::NoCoarseSpace)?;
    let (nl, p) = (anchor.n_levels, anchor.p);
    // Nodes: the anchor's levels, then every other unknown on its own.
    let node = |u: usize| -> usize {
        if u >= anchor.start && u < anchor.start + p * nl {
            (u - anchor.start) % nl
        } else {
            nl + u
        }
    };
    let mut parent: Vec<u32> = (0..(nl + s.n) as u32).collect();
    for u in 0..s.n {
        for k in s.row_ptr[u]..s.row_ptr[u + 1] {
            let (a, b) = (
                find(&mut parent, node(u) as u32),
                find(&mut parent, node(s.col_idx[k] as usize) as u32),
            );
            if a != b {
                parent[a.max(b) as usize] = a.min(b);
            }
        }
    }
    let mut number = vec![u32::MAX; nl + s.n];
    let mut count = 0usize;
    let mut of_level = vec![0u32; nl];
    for l in 0..nl {
        let r = find(&mut parent, l as u32) as usize;
        if number[r] == u32::MAX {
            number[r] = count as u32;
            count += 1;
        }
        of_level[l] = number[r];
    }
    let mut columns: Vec<Vec<(u32, f64)>> = vec![Vec::new(); count * p];
    for l in 0..nl {
        let k = of_level[l] as usize;
        for j in 0..p {
            let g = &anchor.gauge[(l * p + j) * p..][..p];
            for (q, &v) in g.iter().enumerate() {
                let u = anchor.unknown(q, l);
                if v != 0.0 && s.diag[u] != 0.0 {
                    columns[k * p + j].push((u as u32, v));
                }
            }
        }
    }
    for column in &mut columns {
        let norm = column.iter().map(|e| e.1 * e.1).sum::<f64>().sqrt();
        if norm > 0.0 {
            column.iter_mut().for_each(|e| e.1 /= norm);
        }
    }
    columns.retain(|c| !c.is_empty());
    if std::env::var_os("PROBE_COARSE").is_some() {
        eprintln!("[probe-space] components={count} p={p}");
    }
    Ok(columns)
}

fn knob(name: &str, default: f64) -> f64 {
    std::env::var(name)
        .ok()
        .map_or(default, |v| v.parse().expect(name))
}

/// Anchor levels as nodes of `p` unknowns: `S`'s Jacobi-scaled block between two levels.
struct Nodes<'a> {
    s: &'a AssembledS,
    anchor: AnchorGauges,
}

impl Nodes<'_> {
    /// The anchor level of unknown `u`, if it is the anchor's.
    fn level(&self, u: usize) -> Option<usize> {
        let (start, nl, p) = (self.anchor.start, self.anchor.n_levels, self.anchor.p);
        (u >= start && u < start + p * nl).then(|| (u - start) % nl)
    }

    /// Per level, its neighbours `(level, strength)`, strength the Frobenius norm of the
    /// Jacobi-scaled block, ascending by level.
    fn graph(&self) -> Vec<Vec<(u32, f64)>> {
        use rayon::prelude::*;
        let (nl, p, s) = (self.anchor.n_levels, self.anchor.p, self.s);
        (0..nl)
            .into_par_iter()
            .map(|l| {
                let mut acc: Vec<(u32, f64)> = Vec::new();
                let channels = if knob("PROBE_SA_STRENGTH", 0.0) > 0.0 {
                    1
                } else {
                    p
                };
                for q in 0..channels {
                    let u = self.anchor.unknown(q, l);
                    let du = s.diag[u];
                    if du == 0.0 {
                        continue;
                    }
                    for k in s.row_ptr[u]..s.row_ptr[u + 1] {
                        let v = s.col_idx[k] as usize;
                        if let Some(m) = self.level(v) {
                            let same = (v - self.anchor.start) / nl;
                            if m != l && s.diag[v] != 0.0 && (channels == p || same == 0) {
                                acc.push((m as u32, s.val[k] * s.val[k] / (du * s.diag[v])));
                            }
                        }
                    }
                }
                acc.sort_by_key(|e| e.0);
                (acc.chunk_by(|a, b| a.0 == b.0))
                    .map(|run| (run[0].0, run.iter().map(|e| e.1).sum::<f64>().sqrt()))
                    .collect()
            })
            .collect()
    }

    /// Unit-norm, orthogonal gauge columns over each group of levels; `p` per group.
    fn tentative(&self, groups: &[Vec<u32>]) -> Vec<Vec<(u32, f64)>> {
        use rayon::prelude::*;
        let (p, s) = (self.anchor.p, self.s);
        groups
            .par_iter()
            .flat_map_iter(|levels| {
                let mut cols: Vec<Vec<(u32, f64)>> = (0..p)
                    .map(|j| {
                        let mut col = Vec::new();
                        for &l in levels {
                            let g = &self.anchor.gauge[(l as usize * p + j) * p..][..p];
                            for (q, &v) in g.iter().enumerate() {
                                let u = self.anchor.unknown(q, l as usize);
                                if v != 0.0 && s.diag[u] != 0.0 {
                                    col.push((u as u32, v));
                                }
                            }
                        }
                        col.sort_by_key(|e| e.0);
                        col
                    })
                    .collect();
                // Modified Gram–Schmidt over the group's channels.
                for j in 0..p {
                    for i in 0..j {
                        let (a, b) = cols.split_at_mut(j);
                        let d = dot_sorted(&a[i], &b[0]);
                        axpy_sorted(&mut b[0], -d, &a[i]);
                    }
                    let norm = cols[j].iter().map(|e| e.1 * e.1).sum::<f64>().sqrt();
                    if norm > 1e-8 {
                        cols[j].iter_mut().for_each(|e| e.1 /= norm);
                    } else {
                        cols[j].clear();
                    }
                }
                cols.into_iter().filter(|c| !c.is_empty())
            })
            .collect()
    }
}

fn dot_sorted(a: &[(u32, f64)], b: &[(u32, f64)]) -> f64 {
    let (mut i, mut j, mut acc) = (0, 0, 0.0);
    while i < a.len() && j < b.len() {
        match a[i].0.cmp(&b[j].0) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                acc += a[i].1 * b[j].1;
                i += 1;
                j += 1;
            }
        }
    }
    acc
}

/// `b += alpha a`, both sorted by index.
fn axpy_sorted(b: &mut Vec<(u32, f64)>, alpha: f64, a: &[(u32, f64)]) {
    let mut out = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0, 0);
    while i < a.len() || j < b.len() {
        if j == b.len() || (i < a.len() && a[i].0 < b[j].0) {
            out.push((a[i].0, alpha * a[i].1));
            i += 1;
        } else if i == a.len() || b[j].0 < a[i].0 {
            out.push(b[j]);
            j += 1;
        } else {
            out.push((b[j].0, b[j].1 + alpha * a[i].1));
            i += 1;
            j += 1;
        }
    }
    *b = out;
}

/// Vaněk–Mandel–Brezina's three passes over the strong graph, in level order.
fn aggregate(strong: &[Vec<(u32, f64)>]) -> Vec<Vec<u32>> {
    let n = strong.len();
    let mut of = vec![u32::MAX; n];
    let mut groups: Vec<Vec<u32>> = Vec::new();
    for i in 0..n {
        if of[i] == u32::MAX && strong[i].iter().all(|&(j, _)| of[j as usize] == u32::MAX) {
            let k = groups.len() as u32;
            let mut g = vec![i as u32];
            of[i] = k;
            for &(j, _) in &strong[i] {
                of[j as usize] = k;
                g.push(j);
            }
            groups.push(g);
        }
    }
    let first = of.clone();
    for i in 0..n {
        if of[i] == u32::MAX {
            let best = (strong[i].iter())
                .filter(|&&(j, _)| first[j as usize] != u32::MAX)
                .max_by(|a, b| a.1.total_cmp(&b.1).then(b.0.cmp(&a.0)));
            if let Some(&(j, _)) = best {
                of[i] = first[j as usize];
                groups[first[j as usize] as usize].push(i as u32);
            }
        }
    }
    for i in 0..n {
        if of[i] == u32::MAX {
            let k = groups.len() as u32;
            let mut g = vec![i as u32];
            of[i] = k;
            for &(j, _) in &strong[i] {
                if of[j as usize] == u32::MAX {
                    of[j as usize] = k;
                    g.push(j);
                }
            }
            groups.push(g);
        }
    }
    groups.iter_mut().for_each(|g| g.sort_unstable());
    groups
}

/// `ρ(D⁻¹S)` by power iteration from a fixed start.
fn spectral_radius(s: &AssembledS) -> f64 {
    let n = s.n;
    let inv: Vec<f64> = s
        .diag
        .iter()
        .map(|&d| if d > 0.0 { 1.0 / d } else { 0.0 })
        .collect();
    let mut x: Vec<f64> = (0..n)
        .map(|i| {
            if inv[i] > 0.0 {
                1.0 + (i % 7) as f64 / 7.0
            } else {
                0.0
            }
        })
        .collect();
    let mut y = vec![0.0; n];
    let mut rho = 0.0;
    for _ in 0..30 {
        let norm = x.iter().map(|v| v * v).sum::<f64>().sqrt();
        x.iter_mut().for_each(|v| *v /= norm);
        s.mul(&x, &mut y);
        y.iter_mut().zip(&inv).for_each(|(v, i)| *v *= i);
        rho = y.iter().map(|v| v * v).sum::<f64>().sqrt();
        std::mem::swap(&mut x, &mut y);
    }
    rho
}

/// Smoothed aggregation: `P = (I − ω D⁻¹S) T`, `ω = 4/(3ρ(D⁻¹S))`, `T` the gauges on each
/// aggregate of anchor levels strongly connected in `S` (`PROBE_SA_THETA`, default 0.08).
fn smoothed_aggregation(
    s: &AssembledS,
    design: &ReducedDesign<'_>,
) -> Result<Vec<Vec<(u32, f64)>>, CoarseDecline> {
    use rayon::prelude::*;
    let anchor = AnchorGauges::of(&design.split).ok_or(CoarseDecline::NoCoarseSpace)?;
    let nodes = Nodes { s, anchor };
    let theta = knob("PROBE_SA_THETA", 0.08);
    let graph = nodes.graph();
    let strong: Vec<Vec<(u32, f64)>> = (graph.into_iter())
        .map(|nb| nb.into_iter().filter(|&(_, w)| w >= theta).collect())
        .collect();
    let groups = aggregate(&strong);
    let tentative = nodes.tentative(&groups);
    let smooth = knob("PROBE_SA_SMOOTH", 1.0) as usize;
    let rho = spectral_radius(s);
    let omega = 4.0 / (3.0 * rho);
    if std::env::var_os("PROBE_COARSE").is_some() {
        let sizes: Vec<usize> = groups.iter().map(Vec::len).collect();
        eprintln!(
            "[probe-space] sa theta={theta} aggregates={} max_size={} singletons={} tentative={} rho={rho:.3} smooth={smooth}",
            groups.len(),
            sizes.iter().max().unwrap_or(&0),
            sizes.iter().filter(|&&k| k == 1).count(),
            tentative.len()
        );
    }
    let columns = (tentative.into_par_iter())
        .map(|mut t| {
            for _ in 0..smooth {
                let st = s.mul_sparse(&t);
                let mut out: Vec<(u32, f64)> = t.clone();
                for (u, v) in st {
                    let d = s.diag[u as usize];
                    // The small terms take their own coarse columns; smoothing into them would
                    // make every column of `SZ` dense.
                    if d > 0.0 && !s.is_small(u as usize) {
                        out.push((u, -omega * v / d));
                    }
                }
                out.sort_by_key(|e| e.0);
                t = (out.chunk_by(|a, b| a.0 == b.0))
                    .map(|run| (run[0].0, run.iter().fold(0.0, |v, e| v + e.1)))
                    .filter(|e| e.1 != 0.0)
                    .collect();
            }
            t
        })
        .collect::<Vec<_>>();
    let mut columns = columns;
    if knob("PROBE_SA_SMALL", 1.0) > 0.0 {
        columns.extend(s.small.iter().map(|&u| vec![(u, 1.0)]));
    }
    Ok(columns)
}

/// Unsmoothed aggregation on the rigid lift: the anchor's levels joined where the intercept
/// channel's `|S_ab| / √(S_aa S_bb)` reaches `PROBE_SA_THETA` (default 0.02), by
/// Vaněk–Mandel–Brezina's passes; `PROBE_SA_SMALL=0` leaves out the small terms' columns.
///
/// With `components`, every nonzero coupling joins: the groups are the connected components, and
/// the small terms stay out unless `PROBE_SA_SMALL=1`.
fn aggregates(design: &ReducedDesign<'_>, components: bool) -> Result<CoarseSolve, CoarseDecline> {
    let started = std::time::Instant::now();
    let anchor = AnchorGauges::of(&design.split).ok_or(CoarseDecline::NoCoarseSpace)?;
    let ic = anchor.intercept.ok_or(CoarseDecline::NoCoarseSpace)?;
    let (nl, lo) = (anchor.n_levels, anchor.unknown(ic, 0));
    let diag = design.diag_s.values();
    let lower = lower_triplets(design, |u| u >= lo && u < lo + nl && diag[u] != 0.0);
    verbose(&format!("graph triplets={}", lower.len()), started);
    let theta = if components {
        0.0
    } else {
        knob("PROBE_SA_THETA", 0.02)
    };
    let mut d = vec![0.0; nl];
    for &(c, r, v) in &lower {
        if c == r {
            d[c as usize - lo] = v;
        }
    }
    let mut strong: Vec<Vec<(u32, f64)>> = vec![Vec::new(); nl];
    for &(c, r, v) in &lower {
        let (a, b) = (c as usize - lo, r as usize - lo);
        if a != b && d[a] > 0.0 && d[b] > 0.0 {
            let w = v.abs() / (d[a] * d[b]).sqrt();
            if w >= theta && w > 0.0 {
                strong[a].push((b as u32, w));
                strong[b].push((a as u32, w));
            }
        }
    }
    drop(lower);
    strong.iter_mut().for_each(|nb| nb.sort_by_key(|e| e.0));
    let groups = if components {
        connected(&strong)
    } else {
        aggregate(&strong)
    };
    drop(strong);
    let mut of_anchor = vec![u32::MAX; nl];
    for (k, g) in groups.iter().enumerate() {
        for &l in g {
            of_anchor[l as usize] = k as u32;
        }
    }
    if std::env::var_os("PROBE_COARSE").is_some() {
        let sizes: Vec<usize> = groups.iter().map(Vec::len).collect();
        eprintln!(
            "[probe-space] agg theta={theta} aggregates={} max_size={} singletons={} secs={:.3}",
            groups.len(),
            sizes.iter().max().unwrap_or(&0),
            sizes.iter().filter(|&&k| k == 1).count(),
            started.elapsed().as_secs_f64()
        );
    }
    let default = if components { 0.0 } else { 1.0 };
    let small = (knob("PROBE_SA_SMALL", default) > 0.0).then(|| small_columns(design));
    super::rigid::build_grouped(&design.split, of_anchor, groups.len(), small.as_ref())
}

fn small_columns(design: &ReducedDesign<'_>) -> SmallColumns {
    let unknowns = small_unknowns(design);
    let dense = dense_columns(design, &unknowns);
    SmallColumns { unknowns, dense }
}

/// The connected components of `graph`, each ascending, numbered by their least node.
fn connected(graph: &[Vec<(u32, f64)>]) -> Vec<Vec<u32>> {
    let n = graph.len();
    let mut parent: Vec<u32> = (0..n as u32).collect();
    for (a, nb) in graph.iter().enumerate() {
        for &(b, _) in nb {
            let (x, y) = (find(&mut parent, a as u32), find(&mut parent, b));
            if x != y {
                parent[x.max(y) as usize] = x.min(y);
            }
        }
    }
    let mut number = vec![u32::MAX; n];
    let mut groups: Vec<Vec<u32>> = Vec::new();
    for l in 0..n {
        let r = find(&mut parent, l as u32) as usize;
        if number[r] == u32::MAX {
            number[r] = groups.len() as u32;
            groups.push(Vec::new());
        }
        groups[number[r] as usize].push(l as u32);
    }
    groups
}

/// `M⁺` of a symmetric `t × t` row-major `m`, eigenvalues below `1e-12` of the largest dropped.
fn pseudo_inverse(m: &[f64], t: usize) -> Vec<f64> {
    let mat = faer::Mat::from_fn(t, t, |i, j| m[i * t + j]);
    let Ok(eig) = mat.self_adjoint_eigen(faer::Side::Lower) else {
        return vec![0.0; t * t];
    };
    let (u, lam) = (eig.U(), eig.S().column_vector());
    let top = (0..t).map(|i| lam[i].abs()).fold(0.0, f64::max);
    let mut out = vec![0.0; t * t];
    for k in 0..t {
        if lam[k] > 1e-12 * top {
            let inv = 1.0 / lam[k];
            for i in 0..t {
                for j in 0..t {
                    out[i * t + j] += u[(i, k)] * inv * u[(j, k)];
                }
            }
        }
    }
    out
}

/// Algebraic GenEO (Al Daas, Jolivet, Scott 2022, for normal equations), each eliminated level a
/// term of the splitting `S = Σ_g S_g`: per aggregate `Ω` (Frobenius strength `PROBE_GENEO_THETA`,
/// default 0.04), `N = Σ_{g touching Ω} Schur(S_g → Ω)` against `A = S_ΩΩ`; the eigenvectors of
/// `N u = λ A u` with `λ < τ` (`PROBE_GENEO_TAU`, default 0.1), at most `PROBE_GENEO_MAX` (8) per
/// aggregate, as columns, and the small terms' unknowns.
fn geneo(
    s: &AssembledS,
    design: &ReducedDesign<'_>,
) -> Result<Vec<Vec<(u32, f64)>>, CoarseDecline> {
    use rayon::prelude::*;
    let started = std::time::Instant::now();
    let anchor = AnchorGauges::of(&design.split).ok_or(CoarseDecline::NoCoarseSpace)?;
    let nodes = Nodes { s, anchor };
    let theta = knob("PROBE_GENEO_THETA", 0.04);
    let tau = knob("PROBE_GENEO_TAU", 0.1);
    let kmax = knob("PROBE_GENEO_MAX", 8.0) as usize;
    let graph = nodes.graph();
    let strong: Vec<Vec<(u32, f64)>> = (graph.into_iter())
        .map(|nb| nb.into_iter().filter(|&(_, w)| w >= theta).collect())
        .collect();
    let groups = aggregate(&strong);
    drop(strong);
    let nl = nodes.anchor.n_levels;
    let mut of_level = vec![u32::MAX; nl];
    for (k, g) in groups.iter().enumerate() {
        for &l in g {
            of_level[l as usize] = k as u32;
        }
    }
    let group_of = |u: usize| -> Option<u32> { nodes.level(u).map(|l| of_level[l]) };
    let large = |u: usize| !s.is_small(u) && s.diag[u] != 0.0;
    let levels = LevelGrams::new(design);
    // Each level's `Schur(S_g → Ω ∩ supp)` for each aggregate `Ω` it touches.
    let mut parts: Vec<(u32, u32, u32, f64)> = (0..levels.n_levels())
        .into_par_iter()
        .flat_map_iter(|g| {
            let (cols, gram) = levels.gram(g, &large);
            let t = cols.len();
            let mut out = Vec::new();
            let mut touched: Vec<u32> = cols.iter().filter_map(|&u| group_of(u as usize)).collect();
            touched.sort_unstable();
            touched.dedup();
            for &k in &touched {
                let inside: Vec<usize> = (0..t)
                    .filter(|&i| group_of(cols[i] as usize) == Some(k))
                    .collect();
                let outside: Vec<usize> = (0..t)
                    .filter(|&i| group_of(cols[i] as usize) != Some(k))
                    .collect();
                let (ni, no) = (inside.len(), outside.len());
                let mut schur: Vec<f64> = (0..ni * ni)
                    .map(|x| gram[inside[x / ni] * t + inside[x % ni]])
                    .collect();
                if no > 0 {
                    let jj: Vec<f64> = (0..no * no)
                        .map(|x| gram[outside[x / no] * t + outside[x % no]])
                        .collect();
                    let pinv = pseudo_inverse(&jj, no);
                    for a in 0..ni {
                        for b in 0..ni {
                            let mut acc = 0.0;
                            for x in 0..no {
                                let sax = gram[inside[a] * t + outside[x]];
                                if sax == 0.0 {
                                    continue;
                                }
                                for y in 0..no {
                                    acc +=
                                        sax * pinv[x * no + y] * gram[outside[y] * t + inside[b]];
                                }
                            }
                            schur[a * ni + b] -= acc;
                        }
                    }
                }
                for a in 0..ni {
                    for b in 0..ni {
                        out.push((k, cols[inside[a]], cols[inside[b]], schur[a * ni + b]));
                    }
                }
            }
            out
        })
        .collect();
    parts.par_sort_by_key(|e| (e.0, e.1, e.2));
    let by_group: Vec<&[(u32, u32, u32, f64)]> = {
        let mut v = Vec::with_capacity(groups.len());
        let mut start = 0;
        for k in 0..groups.len() as u32 {
            let end = start + parts[start..].partition_point(|e| e.0 == k);
            v.push(&parts[start..end]);
            start = end;
        }
        v
    };
    let counts: Vec<(usize, Vec<Vec<(u32, f64)>>)> = (0..groups.len())
        .into_par_iter()
        .map(|k| {
            let mut omega: Vec<u32> = Vec::new();
            for &l in &groups[k] {
                for q in 0..nodes.anchor.p {
                    let u = nodes.anchor.unknown(q, l as usize);
                    if large(u) {
                        omega.push(u as u32);
                    }
                }
            }
            omega.sort_unstable();
            let m = omega.len();
            if m == 0 {
                return (0, Vec::new());
            }
            let index = |u: u32| omega.binary_search(&u).ok();
            let mut nmat = vec![0.0; m * m];
            for &(_, a, b, v) in by_group[k] {
                if let (Some(i), Some(j)) = (index(a), index(b)) {
                    nmat[i * m + j] += v;
                }
            }
            let mut amat = vec![0.0; m * m];
            for (i, &u) in omega.iter().enumerate() {
                let u = u as usize;
                for q in s.row_ptr[u]..s.row_ptr[u + 1] {
                    if let Some(j) = index(s.col_idx[q]) {
                        amat[i * m + j] = s.val[q];
                    }
                }
            }
            // A^{-1/2} N A^{-1/2}, A's nulls dropped.
            let a = faer::Mat::from_fn(m, m, |i, j| amat[i * m + j]);
            let Ok(ea) = a.self_adjoint_eigen(faer::Side::Lower) else {
                return (0, Vec::new());
            };
            let (ua, la) = (ea.U(), ea.S().column_vector());
            let top = (0..m).map(|i| la[i]).fold(0.0, f64::max);
            let keep: Vec<usize> = (0..m).filter(|&i| la[i] > 1e-12 * top).collect();
            let r = keep.len();
            let half = faer::Mat::from_fn(m, r, |i, c| ua[(i, keep[c])] / la[keep[c]].sqrt());
            let nm = faer::Mat::from_fn(m, m, |i, j| nmat[i * m + j]);
            let reduced = half.transpose() * &nm * &half;
            let sym = faer::Mat::from_fn(r, r, |i, j| 0.5 * (reduced[(i, j)] + reduced[(j, i)]));
            let Ok(en) = sym.self_adjoint_eigen(faer::Side::Lower) else {
                return (0, Vec::new());
            };
            let (un, ln) = (en.U(), en.S().column_vector());
            let mut cols = Vec::new();
            for c in 0..r.min(kmax) {
                if ln[c] >= tau {
                    break;
                }
                let y = un.col(c);
                let u = &half * y;
                let norm = (0..m).map(|i| u[i] * u[i]).sum::<f64>().sqrt();
                if norm > 0.0 {
                    cols.push(
                        (0..m)
                            .map(|i| (omega[i], u[i] / norm))
                            .filter(|e| e.1 != 0.0)
                            .collect(),
                    );
                }
            }
            (cols.len(), cols)
        })
        .collect();
    let per: Vec<usize> = counts.iter().map(|c| c.0).collect();
    let mut columns: Vec<Vec<(u32, f64)>> = counts.into_iter().flat_map(|c| c.1).collect();
    if std::env::var_os("PROBE_COARSE").is_some() {
        let mut hist = [0usize; 10];
        for &c in &per {
            hist[c.min(9)] += 1;
        }
        eprintln!(
            "[probe-space] geneo theta={theta} tau={tau} aggregates={} max_size={} columns={} per-aggregate histogram={hist:?} secs={:.3}",
            groups.len(),
            groups.iter().map(Vec::len).max().unwrap_or(0),
            columns.len(),
            started.elapsed().as_secs_f64()
        );
    }
    columns.extend(s.small.iter().map(|&u| vec![(u, 1.0)]));
    Ok(columns)
}
