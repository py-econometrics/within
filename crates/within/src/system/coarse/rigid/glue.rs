//! Gluing: spells whose Gram is close to both levels' own join their levels into components; each
//! component's anchor-side gauges, normalized over it, span one coarse column per channel.

use faer::{Mat, Side};
use rayon::prelude::*;

use super::super::{CoarseDecline, DiagonalGramBasis};
use super::{term_gram, Rows};
use crate::design::Column;
use crate::system::basis::OrthogonalSlopes;
use crate::system::{SplitDesign, TermBlock};

/// The directions whose Gram diagonal `g` is above rounding of the largest.
fn active(g: &[f64]) -> Vec<usize> {
    let max = g.iter().copied().fold(0.0, f64::max);
    (0..g.len()).filter(|&i| g[i] > 1e-14 * max).collect()
}

/// A spell glues its two levels when its Gram is within this ratio of both levels' own.
const GLUE_RATIO: f64 = 20.0;

/// Whether a level touching `comps` of `n_comps` components is wide: its dense `t × t` block of
/// `E` would outweigh augmenting `E` with its `p` rows `[G s]`.
fn is_wide(comps: usize, p: usize, n_comps: usize) -> bool {
    (comps * p).pow(2) > 2 * p * (n_comps * p)
}

/// Declines when the levels' blocks of `E`, before merging, far exceed `n_obs` values.
fn check_budget(
    touched: &[Vec<u32>],
    p: usize,
    n_comps: usize,
    n_obs: usize,
) -> Result<(), CoarseDecline> {
    let bound: usize = (touched.iter())
        .map(|comps| {
            let t = comps.len() * p;
            match is_wide(comps.len(), p, n_comps) {
                true => t * (p + 1) / 2 + p * t + p * (p + 1) / 2,
                false => t * (t + 1) / 2,
            }
        })
        .sum();
    // The bound overcounts `E` by 3–50×, so it only turns away factors far past the budget.
    match bound > 2 * n_obs {
        true => Err(CoarseDecline::OverBudget),
        false => Ok(()),
    }
}

/// Condition number of the pencil `(s, diag(g))` on the directions where `g` is positive.
fn pencil_condition(s: &[f64], g: &[f64]) -> f64 {
    let (p, act) = (g.len(), active(g));
    if act.is_empty() {
        return f64::INFINITY;
    }
    let scaled = Mat::from_fn(act.len(), act.len(), |i, j| {
        let (a, b) = (act[i], act[j]);
        s[a * p + b] / (g[a] * g[b]).sqrt()
    });
    let Ok(eigenvalues) = scaled.self_adjoint_eigenvalues(Side::Lower) else {
        return f64::INFINITY;
    };
    let (lo, hi) = (eigenvalues[0], eigenvalues[act.len() - 1]);
    if lo <= 1e-12 * hi {
        f64::INFINITY
    } else {
        hi / lo
    }
}

fn find(parent: &mut [u32], mut x: u32) -> u32 {
    while parent[x as usize] != x {
        let g = parent[parent[x as usize] as usize];
        parent[x as usize] = g;
        x = g;
    }
    x
}

/// `Σ a aᵀ` over a spell's `(position, row)`s, `a(at, r, j)` the weighted entry in column `j`.
fn spell_gram(
    p: usize,
    spell: &[(usize, usize, usize)],
    a: impl Fn(usize, usize, usize) -> f64,
    out: &mut [f64],
) {
    out.fill(0.0);
    for &(_, at, r) in spell {
        for i in 0..p {
            let ai = a(at, r, i);
            for j in 0..p {
                out[i * p + j] += ai * a(at, r, j);
            }
        }
    }
}

/// Per eliminated level, the anchor levels its weighted rows reach and, of those, the ones whose
/// spell's Gram is within [`GLUE_RATIO`] of both levels' own; each ascending.
fn spells(rows: &Rows<'_>, anchor: &Anchor<'_>) -> (Vec<Vec<u32>>, Vec<Vec<u32>>) {
    let p = rows.width();
    (0..rows.n_levels())
        .into_par_iter()
        .map(|lw| {
            let basis = rows.level(lw);
            let level: Vec<f64> = (0..p).map(|a| basis.norm2(a)).collect();
            let mut gram = vec![0.0; p * p];
            let mut by_anchor: Vec<(usize, usize, usize)> = (basis.weighted())
                .map(|(at, r)| (anchor.level(r), at, r))
                .collect();
            by_anchor.sort_unstable();
            let (mut reached, mut glued) = (Vec::new(), Vec::new());
            for spell in by_anchor.chunk_by(|a, b| a.0 == b.0) {
                let lf = spell[0].0 as u32;
                reached.push(lf);
                spell_gram(p, spell, |at, _, j| basis.entry(at, j), &mut gram);
                let eliminated = pencil_condition(&gram, &level);
                let anchored = |_, r, j| rows.root(r) * anchor.loading(r, j);
                spell_gram(p, spell, anchored, &mut gram);
                let own = anchor.gram(spell[0].0, p);
                if eliminated.max(pencil_condition(&gram, &own)) <= GLUE_RATIO {
                    glued.push(lf);
                }
            }
            (reached, glued)
        })
        .unzip()
}

/// The kept term with the eliminated term's column layout and the most levels.
struct Anchor<'p> {
    term: TermBlock<'p>,
    /// Its slopes' change of basis, for the caller-coordinate gauges.
    slopes: &'p OrthogonalSlopes,
    /// Its block start among the unknowns.
    start: usize,
    /// Its `diag(D̃ᵀWD̃)` block, each level's whole Gram after whitening, column-major.
    gram: Vec<f64>,
}

impl<'p> Anchor<'p> {
    fn find(split: &'p SplitDesign<'_>) -> Option<Self> {
        let layout = split.layout();
        let columns = |t: usize| {
            let term = layout.term(t);
            (0..term.n_columns()).map(move |c| term.column(c) == Column::Intercept)
        };
        let e = split.eliminated();
        let (t, block) = (split.unknowns().blocks())
            .filter(|(t, _)| columns(*t).eq(columns(e)))
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .max_by_key(|(t, _)| layout.term(*t).n_levels())?;
        Some(Self {
            term: split.term(t),
            slopes: &split.slopes[t],
            start: block.start,
            gram: term_gram(split, t),
        })
    }

    fn n_levels(&self) -> usize {
        self.term.layout().n_levels()
    }

    fn level(&self, i: usize) -> usize {
        self.term.levels()[i] as usize
    }

    /// Row `i`'s loading in column `q`.
    fn loading(&self, i: usize, q: usize) -> f64 {
        self.term.loading(q).map_or(1.0, |z| z[i])
    }

    /// The unknown of column `q` at level `l`.
    fn unknown(&self, q: usize, l: usize) -> usize {
        self.start + q * self.n_levels() + l
    }

    /// Level `l`'s Gram diagonal.
    fn gram(&self, l: usize, p: usize) -> Vec<f64> {
        (0..p).map(|q| self.gram[q * self.n_levels() + l]).collect()
    }

    /// Per level and caller column `j`, the solve coefficients reproducing it.
    ///
    /// PROTOTYPE(#607): main keeps no raw slopes, so a caller slope is read back through the
    /// level's `u = F z − shift`: its intercept coefficient is its level mean, its slope
    /// coefficients the least-squares `F_Kᵀ x = e_s` over `F`'s nonzero rows `K`. Exact where the
    /// level keeps every slope; where it drops one, exact only if the dropped slope is constant
    /// on the level.
    fn gauges(&self, p: usize) -> Vec<f64> {
        let layout = self.term.layout();
        let v = layout.n_slopes();
        let mut gauge = vec![0.0; self.n_levels() * p * p];
        let mut coefs = vec![0.0; v * v];
        for (l, gauge) in gauge.chunks_exact_mut(p * p).enumerate() {
            let means = if v > 0 {
                let (factor, _, means) = self.slopes.level_map(l);
                slope_coefficients(factor, v, &mut coefs);
                means
            } else {
                &[][..]
            };
            for (j, gauge) in gauge.chunks_exact_mut(p).enumerate() {
                match layout.column(j) {
                    Column::Intercept => gauge[j] = 1.0,
                    Column::Slope(s) => {
                        if let Some(ic) = layout.intercept_column() {
                            gauge[ic] = means[s];
                        }
                        for q in 0..v {
                            gauge[layout.slope_column(q)] = coefs[s * v + q];
                        }
                    }
                }
            }
            let g = self.gram(l, p);
            let active = active(&g);
            for gauge in gauge.chunks_exact_mut(p) {
                for (q, x) in gauge.iter_mut().enumerate() {
                    if !active.contains(&q) {
                        *x = 0.0;
                    }
                }
            }
        }
        gauge
    }
}

/// `out[s · v + q]`: caller slope `s`'s coefficient on solve slope `q`, from a level's row-major
/// `v × v` factor `F` (`u = F z − shift`, rows past the rank zero): `x = (F_K F_Kᵀ)⁻¹ F_K e_s`.
fn slope_coefficients(factor: &[f64], v: usize, out: &mut [f64]) {
    out.fill(0.0);
    let k = (0..v)
        .take_while(|&q| factor[q * v..][..v].iter().any(|&f| f != 0.0))
        .count();
    if k == 0 {
        return;
    }
    // M = F_K F_Kᵀ, Cholesky in place (lower, row-major k × k).
    let mut m = vec![0.0; k * k];
    for a in 0..k {
        for b in 0..k {
            m[a * k + b] = (0..v).map(|c| factor[a * v + c] * factor[b * v + c]).sum();
        }
    }
    for a in 0..k {
        for b in 0..=a {
            let mut sum = m[a * k + b];
            for c in 0..b {
                sum -= m[a * k + c] * m[b * k + c];
            }
            m[a * k + b] = if a == b {
                sum.max(0.0).sqrt()
            } else if m[b * k + b] > 0.0 {
                sum / m[b * k + b]
            } else {
                0.0
            };
        }
    }
    let mut x = vec![0.0; k];
    for s in 0..v {
        for (a, x) in x.iter_mut().enumerate() {
            *x = factor[a * v + s];
        }
        for a in 0..k {
            let sum = x[a] - (0..a).map(|c| m[a * k + c] * x[c]).sum::<f64>();
            x[a] = if m[a * k + a] > 0.0 {
                sum / m[a * k + a]
            } else {
                0.0
            };
        }
        for a in (0..k).rev() {
            let sum = x[a] - (a + 1..k).map(|c| m[c * k + a] * x[c]).sum::<f64>();
            x[a] = if m[a * k + a] > 0.0 {
                sum / m[a * k + a]
            } else {
                0.0
            };
        }
        out[s * v..][..k].copy_from_slice(&x);
    }
}

/// The coarse columns: one per glued component `k` and channel `j`, numbered `k p + j`, its gauge
/// on the anchor's levels in `k` scaled to unit norm over `k`.
pub(super) struct Gauges<'a> {
    rows: &'a Rows<'a>,
    anchor: Anchor<'a>,
    /// The component of each anchor level, numbered in anchor-level order; `u32::MAX` without a
    /// weighted row.
    of_anchor: Vec<u32>,
    count: usize,
    /// Per eliminated level, the components its weighted rows touch, ascending.
    touched: Vec<Vec<u32>>,
    /// Per anchor level and channel `j`, its gauge's `p` solve coefficients.
    gauge: Vec<f64>,
}

impl<'a> Gauges<'a> {
    /// Declines without an anchor, when the components cover more than half the weighted levels,
    /// when no eliminated level touches two, or when the levels' blocks of `E` far exceed `n_obs`
    /// values.
    pub(super) fn glue(
        split: &'a SplitDesign<'_>,
        rows: &'a Rows<'a>,
    ) -> Result<Self, CoarseDecline> {
        let anchor = Anchor::find(split).ok_or(CoarseDecline::NoCoarseSpace)?;
        let (reached, glued) = spells(rows, &anchor);
        let (na, nb, p) = (rows.n_levels(), anchor.n_levels(), rows.width());
        let mut parent: Vec<u32> = (0..(na + nb) as u32).collect();
        for (lw, glued) in glued.into_iter().enumerate() {
            for lf in glued {
                let (a, b) = (
                    find(&mut parent, lw as u32),
                    find(&mut parent, na as u32 + lf),
                );
                if a != b {
                    parent[a.max(b) as usize] = a.min(b);
                }
            }
        }
        // A level with no weighted row is a removed level: it joins and counts nothing.
        let mut weighted: Vec<bool> = reached.iter().map(|levels| !levels.is_empty()).collect();
        weighted.resize(na + nb, false);
        for &lf in reached.iter().flatten() {
            weighted[na + lf as usize] = true;
        }
        let roots: Vec<u32> = (0..(na + nb) as u32)
            .map(|x| find(&mut parent, x))
            .collect();
        let n_roots = (roots.iter().enumerate())
            .filter(|&(x, &r)| weighted[x] && x == r as usize)
            .count();
        // Components covering most levels glue nothing worth a coarse column.
        if 2 * n_roots > weighted.iter().filter(|&&w| w).count() {
            return Err(CoarseDecline::NoCoarseSpace);
        }
        let mut number = vec![u32::MAX; na + nb];
        let mut count = 0;
        let of_anchor: Vec<u32> = (roots[na..].iter().zip(&weighted[na..]))
            .map(|(&r, &weighted)| {
                if weighted && number[r as usize] == u32::MAX {
                    number[r as usize] = count;
                    count += 1;
                }
                number[r as usize]
            })
            .collect();
        let touched: Vec<Vec<u32>> = (reached.into_par_iter())
            .map(|levels| {
                let mut comps: Vec<u32> = (levels.into_iter())
                    .map(|l| of_anchor[l as usize])
                    .collect();
                comps.sort_unstable();
                comps.dedup();
                comps
            })
            .collect();
        // A level inside one component absorbs the gauges on its rows exactly.
        if touched.iter().all(|comps| comps.len() < 2) {
            return Err(CoarseDecline::NoCoarseSpace);
        }
        check_budget(&touched, p, count as usize, split.n_obs())?;
        let mut gauge = anchor.gauges(p);
        let mut norm2 = vec![0.0f64; count as usize * p];
        for (gauge, &comp) in gauge.chunks_exact(p * p).zip(&of_anchor) {
            if comp == u32::MAX {
                continue;
            }
            for (j, g) in gauge.chunks_exact(p).enumerate() {
                norm2[comp as usize * p + j] += g.iter().map(|v| v * v).sum::<f64>();
            }
        }
        for (gauge, &comp) in gauge.chunks_exact_mut(p * p).zip(&of_anchor) {
            if comp == u32::MAX {
                continue;
            }
            let comp = comp as usize;
            for (j, g) in gauge.chunks_exact_mut(p).enumerate() {
                let n2 = norm2[comp * p + j];
                let inv = if n2 > 0.0 { n2.sqrt().recip() } else { 0.0 };
                g.iter_mut().for_each(|v| *v *= inv);
            }
        }
        Ok(Self {
            rows,
            anchor,
            of_anchor,
            count: count as usize,
            touched,
            gauge,
        })
    }

    pub(super) fn n_components(&self) -> usize {
        self.count
    }

    /// Per eliminated level, the components its weighted rows touch, ascending.
    pub(super) fn touched(&self) -> &[Vec<u32>] {
        &self.touched
    }

    /// Whether eliminated level `lw` is wide; see [`is_wide`].
    pub(super) fn is_wide(&self, lw: usize) -> bool {
        is_wide(self.touched[lw].len(), self.rows.width(), self.count)
    }

    /// The component of row `r`.
    pub(super) fn component(&self, r: usize) -> u32 {
        self.of_anchor[self.anchor.level(r)]
    }

    /// `√w (D Z)` at row `r` in coarse column `c`: nonzero only on the row's component.
    pub(super) fn lifted(&self, r: usize, c: usize) -> f64 {
        let p = self.rows.width();
        let l = self.anchor.level(r);
        if self.of_anchor[l] as usize != c / p {
            return 0.0;
        }
        let g = &self.gauge[(l * p + c % p) * p..][..p];
        let val: f64 = (0..p).map(|q| self.anchor.loading(r, q) * g[q]).sum();
        self.rows.root(r) * val
    }

    /// `Z` on the anchor's unknowns: its nonzero `(coarse column, unknown, value)`s.
    pub(super) fn entries(&self) -> impl Iterator<Item = (u32, u32, f64)> + '_ {
        let p = self.rows.width();
        (self.gauge.chunks_exact(p * p).enumerate())
            .filter(|&(l, _)| self.of_anchor[l] != u32::MAX)
            .flat_map(move |(l, gauge)| {
                let comp = self.of_anchor[l] as usize;
                (gauge.chunks_exact(p).enumerate()).flat_map(move |(j, g)| {
                    (g.iter().enumerate())
                        .filter(|&(_, &v)| v != 0.0)
                        .map(move |(q, &v)| {
                            ((comp * p + j) as u32, self.anchor.unknown(q, l) as u32, v)
                        })
                })
            })
    }
}
