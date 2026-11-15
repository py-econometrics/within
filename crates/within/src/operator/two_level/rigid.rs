//! The rigid-component coarse space. A spell (one eliminated level × one anchor level) whose Gram
//! is within a ratio of both levels' own Grams glues the two levels; each glued component's
//! anchor-side gauges span one coarse column per channel. With the eliminated term fitted exactly,
//! `U = ÃZ` is the gauges' energy-minimal extension; on the full system `Z` carries that fit `−β`.

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::ldlt::factor::LdltRegularization;
use faer::sparse::linalg::cholesky::{
    factorize_symbolic_cholesky, CholeskySymbolicParams, SymmetricOrdering,
};
use faer::sparse::{SparseColMat, SymbolicSparseColMat};
use faer::{Mat, Par, Side};
use rayon::prelude::*;

use super::{CoarseSolve, HubCorrection, HubRows, TwoWay};
use crate::domain::{
    cut, row_weight, Column, LevelRows, PreparedDesign, PreparedTerm, TermRows, Unknowns,
};
use crate::linalg::{add_moments, remove_fit, solve, DiagonalGramBasis};
use crate::CoarseDecline;

/// The gluing ratios, tried in turn while the factor is over budget.
const RATIOS: [f64; 4] = [20.0, 40.0, 80.0, 160.0];

/// The coarse solve over `unknowns`, at the first gluing ratio whose factor fits the budget.
pub(super) fn build(
    prepared: &PreparedDesign<'_>,
    unknowns: Unknowns,
) -> Result<CoarseSolve, CoarseDecline> {
    let rows = Rows::new(prepared, unknowns);
    let anchor = Anchor::find(prepared, unknowns).ok_or(CoarseDecline::NoCoarseSpace)?;
    let gauge = anchor.gauges(&rows);
    let spells = spells(&rows, &anchor);
    let mut over_budget = None;
    for ratio in RATIOS {
        let glued: Vec<&Spell> = spells.iter().filter(|s| s.condition <= ratio).collect();
        // The same glued spells build the same over-budget factor.
        if over_budget == Some(glued.len()) {
            break;
        }
        match build_glued(prepared, unknowns, &rows, &anchor, &gauge, &glued) {
            Err(CoarseDecline::OverBudget) => over_budget = Some(glued.len()),
            // Fewer glued spells had a coarse space, only one over budget.
            Err(CoarseDecline::NoCoarseSpace) if over_budget.is_some() => break,
            built => return built,
        }
    }
    Err(CoarseDecline::OverBudget)
}

/// The rows grouped by eliminated level, read in the solve basis.
struct Rows<'p> {
    by_level: &'p LevelRows,
    eliminated: TermRows<'p>,
    /// `diag(A_eᵀA_e)`, each level's whole Gram block after whitening, column-major.
    gram: &'p [f64],
    sqrt_weights: Option<&'p [f64]>,
    /// Per kept term, its columns and its block start among the unknowns.
    kept: Vec<(TermRows<'p>, usize)>,
}

impl<'p> Rows<'p> {
    fn new(prepared: &'p PreparedDesign<'_>, unknowns: Unknowns) -> Self {
        let design = &prepared.design;
        let dominant = design.dominant;
        let kept = (0..design.n_factors())
            .filter(|&t| t != dominant)
            .map(|t| {
                let start = design.block_start(unknowns, t).expect("a kept term");
                (TermRows::new(&prepared.term(t)), start)
            })
            .collect();
        Self {
            by_level: prepared.level_rows(dominant),
            eliminated: TermRows::new(&prepared.term(dominant)),
            gram: prepared.diagonal(dominant),
            sqrt_weights: prepared.sqrt_weights(),
            kept,
        }
    }

    fn n_levels(&self) -> usize {
        self.eliminated.n_levels
    }

    fn width(&self) -> usize {
        self.eliminated.width()
    }

    fn rows(&self, level: usize) -> impl Iterator<Item = usize> + Clone + '_ {
        self.by_level.rows(level)
    }

    fn root(&self, i: usize) -> f64 {
        self.sqrt_weights.map_or(1.0, |s| s[i])
    }

    fn weight(&self, i: usize) -> f64 {
        row_weight(self.sqrt_weights, i)
    }

    fn level(&self, level: usize) -> EliminatedLevel<'_> {
        EliminatedLevel {
            rows: self,
            level,
            start: self.by_level.positions(level).start,
        }
    }

    /// `f(unknown, loading)` over row `i`'s kept coefficients.
    fn for_each_kept(&self, i: usize, mut f: impl FnMut(usize, f64)) {
        for (term, start) in &self.kept {
            term.for_each_column(*start, i, &mut f);
        }
    }
}

/// One level's columns of `A_e`, `W½ D_e`; their row `k` is the level's `k`-th row.
struct EliminatedLevel<'r> {
    rows: &'r Rows<'r>,
    level: usize,
    start: usize,
}

impl DiagonalGramBasis for EliminatedLevel<'_> {
    fn entry(&self, k: usize, c: usize) -> f64 {
        let r = self.rows.by_level.row(self.start + k);
        self.rows.root(r) * self.rows.eliminated.column(r, c)
    }

    fn norm2(&self, c: usize) -> f64 {
        self.rows.gram[c * self.rows.n_levels() + self.level]
    }
}

/// The kept term with the eliminated term's column layout and the most levels.
struct Anchor<'p> {
    term: PreparedTerm<'p>,
    /// Its block start among the unknowns.
    start: usize,
    /// Its `diag(AᵀA)` block, each level's whole Gram after whitening, column-major.
    gram: &'p [f64],
}

impl<'p> Anchor<'p> {
    fn find(prepared: &'p PreparedDesign<'_>, unknowns: Unknowns) -> Option<Self> {
        let design = &prepared.design;
        let layout = |t: usize| {
            let term = &design.terms[t];
            (0..term.n_columns()).map(move |c| term.column(c) == Column::Intercept)
        };
        let t = (0..design.n_factors())
            .filter(|&t| t != design.dominant && layout(t).eq(layout(design.dominant)))
            .rev()
            .max_by_key(|&t| design.terms[t].n_levels())?;
        Some(Self {
            term: prepared.term(t),
            start: design.block_start(unknowns, t).expect("a kept term"),
            gram: prepared.diagonal(t),
        })
    }

    fn n_levels(&self) -> usize {
        self.term.term.n_levels()
    }

    fn level(&self, i: usize) -> usize {
        self.term.term.levels()[i] as usize
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
    fn gauges(&self, rows: &Rows<'_>) -> Vec<f64> {
        let p = rows.width();
        let term = self.term.term;
        let raw: Vec<&[f64]> = term.raw_slopes().collect();
        let caller = |c: usize, r: usize| match term.column(c) {
            Column::Intercept => 1.0,
            Column::Slope(j) => raw[j][r],
        };
        let mut gauge = vec![0.0; self.n_levels() * p * p];
        for (r, &l) in term.levels().iter().enumerate() {
            let w = rows.weight(r);
            let at = l as usize * p * p;
            for q in 0..p {
                let wq = w * self.loading(r, q);
                for j in 0..p {
                    gauge[at + j * p + q] += wq * caller(j, r);
                }
            }
        }
        for (l, gauge) in gauge.chunks_exact_mut(p * p).enumerate() {
            let g = self.gram(l, p);
            let active = active(&g);
            for gauge in gauge.chunks_exact_mut(p) {
                for (q, x) in gauge.iter_mut().enumerate() {
                    *x = if active.contains(&q) { *x / g[q] } else { 0.0 };
                }
            }
        }
        gauge
    }
}

/// The directions whose Gram diagonal `g` is above rounding of the largest.
fn active(g: &[f64]) -> Vec<usize> {
    let max = g.iter().copied().fold(0.0, f64::max);
    (0..g.len()).filter(|&i| g[i] > 1e-14 * max).collect()
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

/// `Σ w a aᵀ` over `rows`, `a(i, j)` row `i`'s loading in column `j`.
fn row_gram(
    grouped: &Rows<'_>,
    rows: impl Iterator<Item = usize>,
    a: impl Fn(usize, usize) -> f64,
    out: &mut [f64],
) {
    let p = grouped.width();
    out.fill(0.0);
    for r in rows {
        let w = grouped.weight(r);
        for i in 0..p {
            let wa = w * a(r, i);
            for j in 0..p {
                out[i * p + j] += wa * a(r, j);
            }
        }
    }
}

/// One eliminated level × one anchor level sharing rows.
struct Spell {
    eliminated: u32,
    anchor: u32,
    /// The larger condition of the spell's Gram against either level's own.
    condition: f64,
}

fn spells(rows: &Rows<'_>, anchor: &Anchor<'_>) -> Vec<Spell> {
    let p = rows.width();
    (0..rows.n_levels())
        .into_par_iter()
        .flat_map_iter(|lw| {
            let basis = rows.level(lw);
            let level: Vec<f64> = (0..p).map(|a| basis.norm2(a)).collect();
            let mut gram = vec![0.0; p * p];
            let mut by_anchor: Vec<(usize, usize)> =
                rows.rows(lw).map(|r| (anchor.level(r), r)).collect();
            by_anchor.sort_unstable();
            let spells: Vec<Spell> = by_anchor
                .chunk_by(|a, b| a.0 == b.0)
                .map(|spell| {
                    let lf = spell[0].0;
                    let spell_rows = || spell.iter().map(|&(_, r)| r);
                    row_gram(
                        rows,
                        spell_rows(),
                        |r, j| rows.eliminated.column(r, j),
                        &mut gram,
                    );
                    let eliminated = pencil_condition(&gram, &level);
                    row_gram(rows, spell_rows(), |r, j| anchor.loading(r, j), &mut gram);
                    let anchored = pencil_condition(&gram, &anchor.gram(lf, p));
                    Spell {
                        eliminated: lw as u32,
                        anchor: lf as u32,
                        condition: eliminated.max(anchored),
                    }
                })
                .collect();
            spells
        })
        .collect()
}

/// One eliminated level's share of `U = ÃZ`: the coarse columns its rows touch and their fit `β`.
struct GroupFit {
    /// Touched components, ascending; their channels are the level's coarse columns.
    comps: Vec<u32>,
    /// Per column, its `β = G⁻ Eᵀ W½ (W½ D Z)` over the `width` eliminated columns.
    beta: Vec<f64>,
}

impl GroupFit {
    /// The level's coarse columns, every channel of each touched component.
    fn columns(&self, p: usize) -> impl Iterator<Item = u32> + '_ {
        (self.comps.iter()).flat_map(move |&c| (0..p as u32).map(move |j| c * p as u32 + j))
    }

    /// The local index of coarse column `c`.
    fn local(&self, c: usize, p: usize) -> usize {
        let k = (self.comps)
            .binary_search(&((c / p) as u32))
            .expect("touched column");
        k * p + c % p
    }
}

/// A level's share of the column energies and of `E = UᵀU`.
struct LevelGram {
    /// `‖W½DZ‖²` per touched column, in the fit's column order.
    pre: Vec<f64>,
    /// `‖U‖²` per touched column, in the fit's column order.
    energy: Vec<f64>,
    /// The lower triangle of `UᵀU` over the touched columns; a hub's own blocks only.
    entries: Vec<(u32, u32, f64)>,
    /// A hub's `s = EᵀW DZ`, `width` per column: rows `[G s]` augmenting `E₀`.
    hub: Option<Vec<f64>>,
}

/// What the coarse columns are made of, to evaluate `U = ÃZ` one entry at a time.
struct Lift<'a> {
    rows: &'a Rows<'a>,
    anchor: &'a Anchor<'a>,
    anchor_comp: &'a [u32],
    gauge: &'a [f64],
    inv_norm: &'a [f64],
}

impl Lift<'_> {
    /// `√w (D Z)` at row `r` in coarse column `c`: nonzero only on the row's component.
    fn lifted(&self, r: usize, c: usize) -> f64 {
        let p = self.rows.width();
        let l = self.anchor.level(r);
        if self.anchor_comp[l] as usize != c / p {
            return 0.0;
        }
        let g = &self.gauge[(l * p + c % p) * p..][..p];
        let val: f64 = (0..p).map(|q| self.anchor.loading(r, q) * g[q]).sum();
        self.rows.root(r) * val * self.inv_norm[c]
    }

    /// Level `lw`'s fit, `‖W½DZ‖²` and the lower triangle of `UᵀU` over its columns.
    ///
    /// `UᵀU = (DZ)ᵀW(DZ) − sᵀG⁻s` with `s = EᵀW DZ`: a row touches only its own component's
    /// columns, so the first term is per-component `p × p` and no row-by-column buffer is needed.
    fn fit(&self, lw: usize, comps: Vec<u32>, hub: bool) -> (GroupFit, LevelGram) {
        let (rows, p) = (self.rows, self.rows.width());
        let basis = rows.level(lw);
        let t = comps.len() * p;
        let (mut pre, mut s) = (vec![0.0; t], vec![0.0; t * p]);
        let mut blocks = vec![0.0; comps.len() * p * p];
        let mut lifted = vec![0.0; p];
        for (at, r) in rows.rows(lw).enumerate() {
            let c = self.anchor_comp[self.anchor.level(r)];
            let k = comps.binary_search(&c).expect("touched component");
            for (j, v) in lifted.iter_mut().enumerate() {
                *v = self.lifted(r, c as usize * p + j);
                pre[k * p + j] += *v * *v;
            }
            let block = &mut blocks[k * p * p..][..p * p];
            for (i, &vi) in lifted.iter().enumerate() {
                for (j, &vj) in lifted.iter().enumerate() {
                    block[i * p + j] += vi * vj;
                }
            }
            for (j, &v) in lifted.iter().enumerate() {
                add_moments(&basis, at, v, &mut s[(k * p + j) * p..][..p]);
            }
        }
        let mut beta = s.clone();
        beta.chunks_exact_mut(p).for_each(|b| solve(&basis, b));
        let own = |a: usize, b: usize| match a / p == b / p {
            true => blocks[(a / p) * p * p + (a % p) * p + b % p],
            false => 0.0,
        };
        let fitted =
            |a: usize, b: usize| -> f64 { (0..p).map(|k| s[a * p + k] * beta[b * p + k]).sum() };
        let energy = (0..t).map(|a| own(a, a) - fitted(a, a)).collect();
        let column = |a: usize| comps[a / p] * p as u32 + (a % p) as u32;
        let mut entries = Vec::new();
        for a in 0..t {
            match hub {
                true => (a..(a / p + 1) * p)
                    .for_each(|b| entries.push((column(a), column(b), own(a, b)))),
                false => (a..t)
                    .for_each(|b| entries.push((column(a), column(b), own(a, b) - fitted(a, b)))),
            }
        }
        let gram = LevelGram {
            pre,
            energy,
            entries,
            hub: hub.then_some(s),
        };
        (GroupFit { comps, beta }, gram)
    }
}

/// `(column, row, value)` triplets by column then row, duplicates summed in input order, zero
/// sums dropped.
fn merged(mut triplets: Vec<(u32, u32, f64)>) -> Vec<(u32, u32, f64)> {
    triplets.par_sort_by_key(|e| (e.0, e.1));
    (triplets.chunk_by(|a, b| (a.0, a.1) == (b.0, b.1)))
        .filter_map(|run| {
            let v = run.iter().fold(0.0, |v, e| v + e.2);
            (v != 0.0).then_some((run[0].0, run[0].1, v))
        })
        .collect()
}

impl HubRows {
    /// Appends wide level `lw`'s rows: its `G` and `s` against the live columns.
    fn push(&mut self, rows: &Rows<'_>, lw: usize, s: &[f64], fit: &GroupFit, renumber: &[u32]) {
        let (p, basis) = (rows.width(), rows.level(lw));
        for k in 0..p {
            for (i, c) in fit.columns(p).enumerate() {
                let c = renumber[c as usize];
                if c != u32::MAX {
                    self.cols.push(c);
                    self.vals.push(s[i * p + k]);
                }
            }
            self.ptr.push(self.cols.len());
            self.gram.push(basis.norm2(k));
        }
    }
}

/// Declines when the components cover more than half the levels, when no eliminated level
/// splits one, or when the factor exceeds `n_obs` values.
fn build_glued(
    prepared: &PreparedDesign<'_>,
    unknowns: Unknowns,
    rows: &Rows<'_>,
    anchor: &Anchor<'_>,
    gauge: &[f64],
    glued: &[&Spell],
) -> Result<CoarseSolve, CoarseDecline> {
    let design = &prepared.design;
    let n_obs = design.n_obs;
    let p = rows.width();
    let (na, nb) = (rows.n_levels(), anchor.n_levels());

    let mut parent: Vec<u32> = (0..(na + nb) as u32).collect();
    for spell in glued {
        let a = find(&mut parent, spell.eliminated);
        let b = find(&mut parent, na as u32 + spell.anchor);
        if a != b {
            parent[a.max(b) as usize] = a.min(b);
        }
    }
    let roots: Vec<u32> = (0..(na + nb) as u32)
        .map(|x| find(&mut parent, x))
        .collect();
    let n_components = (roots.iter().enumerate())
        .filter(|&(x, &r)| x == r as usize)
        .count();
    // Components covering most levels glue nothing worth a coarse column.
    if 2 * n_components > na + nb {
        return Err(CoarseDecline::NoCoarseSpace);
    }
    let mut fc_of_root = vec![u32::MAX; na + nb];
    let mut n_fc = 0u32;
    let anchor_comp: Vec<u32> = (0..nb)
        .map(|l| {
            let r = roots[na + l] as usize;
            if fc_of_root[r] == u32::MAX {
                fc_of_root[r] = n_fc;
                n_fc += 1;
            }
            fc_of_root[r]
        })
        .collect();
    let touched: Vec<(Vec<u32>, bool)> = (0..na)
        .into_par_iter()
        .map(|lw| {
            let comp = |r: usize| anchor_comp[anchor.level(r)];
            // Zero-weight rows carry no energy, so they cannot split a level.
            let mut positive = rows.rows(lw).filter(|&r| rows.weight(r) > 0.0).map(comp);
            let first = positive.next();
            let split = positive.any(|c| Some(c) != first);
            let mut comps: Vec<u32> = rows.rows(lw).map(comp).collect();
            comps.sort_unstable();
            comps.dedup();
            (comps, split)
        })
        .collect();
    // An eliminated level inside one anchor component absorbs the gauges on its rows exactly.
    if !touched.iter().any(|t| t.1) {
        return Err(CoarseDecline::NoCoarseSpace);
    }
    let m = n_fc as usize * p;
    // A wide level (`t² > 2pm`) augments `E` with its rows `[G s]` rather than a dense block.
    let hub: Vec<bool> = (touched.iter())
        .map(|(comps, _)| (comps.len() * p).pow(2) > 2 * p * m)
        .collect();
    // `E` gathers each level's block before merging: bound it like the factor.
    let e_bound: usize = (touched.iter().zip(&hub))
        .map(|((comps, _), &hub)| match hub {
            true => comps.len() * p * (p + 1) / 2 + p * comps.len() * p + p * (p + 1) / 2,
            false => comps.len() * p * (comps.len() * p + 1) / 2,
        })
        .sum();
    // `E` overcounts by 3–50×, so this only turns away factors far past the budget below.
    if e_bound > 2 * n_obs {
        return Err(CoarseDecline::OverBudget);
    }

    let mut norm2 = vec![0.0f64; m];
    for l in 0..nb {
        for j in 0..p {
            norm2[anchor_comp[l] as usize * p + j] += gauge[(l * p + j) * p..][..p]
                .iter()
                .map(|v| v * v)
                .sum::<f64>();
        }
    }
    let inv_norm: Vec<f64> = (norm2.iter())
        .map(|&v| if v > 0.0 { v.sqrt().recip() } else { 0.0 })
        .collect();
    let lift = Lift {
        rows,
        anchor,
        anchor_comp: &anchor_comp,
        gauge,
        inv_norm: &inv_norm,
    };
    let per_level: Vec<(GroupFit, LevelGram)> = (touched.into_par_iter())
        .enumerate()
        .map(|(lw, (comps, _))| lift.fit(lw, comps, hub[lw]))
        .collect();
    // Exact nulls of S (the eliminated term absorbs the gauge) need no lift; keep the rest.
    let (mut pre_energy, mut energy) = (vec![0.0f64; m], vec![0.0f64; m]);
    for (fit, gram) in &per_level {
        for (c, v) in fit.columns(p).zip(&gram.pre) {
            pre_energy[c as usize] += v;
        }
        for (c, v) in fit.columns(p).zip(&gram.energy) {
            energy[c as usize] += v;
        }
    }
    let mut renumber = vec![u32::MAX; m];
    let mut live = 0u32;
    for c in 0..m {
        if energy[c] > 1e-10 * pre_energy[c] {
            renumber[c] = live;
            live += 1;
        }
    }
    let m = live as usize;
    if m == 0 {
        return Err(CoarseDecline::NoCoarseSpace);
    }
    let n = design.n_unknowns(unknowns);
    let mut z_entries = Vec::new();
    for l in 0..nb {
        let comp = anchor_comp[l] as usize;
        for j in 0..p {
            let c = comp * p + j;
            if renumber[c] == u32::MAX {
                continue;
            }
            for (q, &v) in gauge[(l * p + j) * p..][..p].iter().enumerate() {
                if v != 0.0 {
                    let unknown = anchor.unknown(q, l) as u32;
                    z_entries.push((renumber[c], unknown, v * inv_norm[c]));
                }
            }
        }
    }
    // Over the eliminated term too, `Z` is `[−β; Z]`, so that `A Z = U`.
    if let Some(start) = design.block_start(unknowns, design.dominant) {
        for (lw, (fit, _)) in per_level.iter().enumerate() {
            for a in 0..p {
                let unknown = (start + a * na + lw) as u32;
                for (i, c) in fit.columns(p).enumerate() {
                    let v = fit.beta[i * p + a];
                    if renumber[c as usize] != u32::MAX && v != 0.0 {
                        z_entries.push((renumber[c as usize], unknown, -v));
                    }
                }
            }
        }
    }
    z_entries.par_sort_unstable_by_key(|t| (t.0, t.1));
    let z = TwoWay::from_parts(n, m, vec![z_entries]);
    let (mut fits, mut e_triplets) = (Vec::with_capacity(na), Vec::new());
    let mut hubs = HubRows {
        ptr: vec![0],
        cols: Vec::new(),
        vals: Vec::new(),
        gram: Vec::new(),
    };
    for (lw, (fit, gram)) in per_level.into_iter().enumerate() {
        e_triplets.extend(gram.entries.into_iter().filter_map(|(a, b, v)| {
            let (a, b) = (renumber[a as usize], renumber[b as usize]);
            (a != u32::MAX && b != u32::MAX).then_some((a, b, v))
        }));
        if let Some(s) = gram.hub {
            hubs.push(rows, lw, &s, &fit, &renumber);
        }
        fits.push(fit);
    }
    let rank = hubs.rank();
    for (i, &g) in hubs.gram.iter().enumerate() {
        let row = (m + i) as u32;
        e_triplets.push((row, row, g));
        let span = hubs.ptr[i]..hubs.ptr[i + 1];
        e_triplets.extend(
            (hubs.cols[span.clone()].iter().zip(&hubs.vals[span])).map(|(&c, &v)| (c, row, v)),
        );
    }
    let dim = m + rank;
    let lower = merged(e_triplets);
    let mut diagonal = vec![0.0; dim];
    for &(c, r, v) in &lower {
        if c == r {
            diagonal[c as usize] = v;
        }
    }
    // Unit diagonal: null pivots then sit at rounding level, far below any live mode.
    let scale: Vec<f64> = (diagonal.iter())
        .map(|&d| if d > 0.0 { d.sqrt().recip() } else { 1.0 })
        .collect();
    let lower: Vec<(u32, u32, f64)> = (lower.into_iter())
        .map(|(c, r, v)| (c, r, v * (scale[c as usize] * scale[r as usize])))
        .collect();
    let d_max = (lower.iter())
        .filter(|e| e.0 == e.1)
        .map(|e| e.2)
        .fold(0.0, f64::max);
    let mut col_ptr = vec![0u32; dim + 1];
    for &(c, ..) in &lower {
        col_ptr[c as usize + 1] += 1;
    }
    for c in 0..dim {
        col_ptr[c + 1] += col_ptr[c];
    }
    let (row_idx, values) = lower.into_iter().map(|(_, r, v)| (r, v)).unzip();
    let pattern = SymbolicSparseColMat::new_checked(dim, dim, col_ptr, None, row_idx);
    let e_lower = SparseColMat::new(pattern, values);
    let symbolic = factorize_symbolic_cholesky(
        e_lower.symbolic(),
        Side::Lower,
        SymmetricOrdering::Amd,
        CholeskySymbolicParams::default(),
    )
    .map_err(|_| CoarseDecline::NoCoarseSpace)?;
    if symbolic.len_val() > n_obs {
        return Err(CoarseDecline::OverBudget);
    }
    // Null pivots become LARGE, as in the fused blocks: tiny ones amplify nulls.
    let signs = vec![1i8; dim];
    let regularization = LdltRegularization {
        dynamic_regularization_signs: Some(&signs),
        dynamic_regularization_delta: d_max,
        dynamic_regularization_epsilon: 1e-10 * d_max,
    };
    let mut l_values = vec![0.0; symbolic.len_val()];
    let mut mem = MemBuffer::new(
        symbolic.factorize_numeric_ldlt_scratch::<f64>(Par::Seq, Default::default()),
    );
    symbolic
        .factorize_numeric_ldlt(
            &mut l_values,
            e_lower.as_ref(),
            Side::Lower,
            regularization,
            Par::Seq,
            MemStack::new(&mut mem),
            Default::default(),
        )
        .map_err(|_| CoarseDecline::NoCoarseSpace)?;
    if !l_values.iter().all(|v| v.is_finite()) {
        return Err(CoarseDecline::NoCoarseSpace);
    }
    drop((e_lower, mem));
    let sz = projected_columns(rows, &lift, &fits, &hub, &renumber, n, m);
    let hub_levels: Vec<usize> = (0..na).filter(|&lw| hub[lw]).collect();
    let f_parts: Vec<Vec<(u32, u32, f64)>> = (hub_levels.par_iter().enumerate())
        .map(|(h, &lw)| {
            let mut entries = Vec::new();
            for r in rows.rows(lw) {
                let w = rows.weight(r);
                rows.for_each_kept(r, |d, z| {
                    for a in 0..p {
                        entries.push((
                            (h * p + a) as u32,
                            d as u32,
                            w * z * rows.eliminated.column(r, a),
                        ));
                    }
                });
            }
            merged(entries)
        })
        .collect();
    let f = TwoWay::from_parts(n, rank, f_parts);
    Ok(CoarseSolve {
        z,
        sz,
        hubs: (rank > 0).then_some(HubCorrection { rows: hubs, f }),
        symbolic,
        l_values,
        scale,
    })
}

/// `SZ₀ = D_keptᵀ W½ U`, one coarse column at a time over the levels touching its component; a
/// wide level contributes only its own component's rows, its projection being the hub correction.
fn projected_columns(
    rows: &Rows<'_>,
    lift: &Lift<'_>,
    fits: &[GroupFit],
    hub: &[bool],
    renumber: &[u32],
    n: usize,
    m: usize,
) -> TwoWay {
    let (p, n_all) = (rows.width(), renumber.len());
    let n_fc = n_all / p;
    let mut comp_ptr = vec![0usize; n_fc + 1];
    for fit in fits {
        for &c in &fit.comps {
            comp_ptr[c as usize + 1] += 1;
        }
    }
    for c in 0..n_fc {
        comp_ptr[c + 1] += comp_ptr[c];
    }
    let mut comp_levels = vec![0u32; comp_ptr[n_fc]];
    let mut cursor = comp_ptr.clone();
    for (lw, fit) in fits.iter().enumerate() {
        for &c in &fit.comps {
            let k = &mut cursor[c as usize];
            comp_levels[*k] = lw as u32;
            *k += 1;
        }
    }
    let hub_rows: Vec<Option<Vec<(u32, u32)>>> = (0..fits.len())
        .map(|lw| {
            hub[lw].then(|| {
                let mut by_comp: Vec<(u32, u32)> = (rows.rows(lw))
                    .map(|r| (lift.anchor_comp[lift.anchor.level(r)], r as u32))
                    .collect();
                by_comp.sort_unstable();
                by_comp
            })
        })
        .collect();
    let own_rows = |lw: usize, comp: usize| -> &[(u32, u32)] {
        let by_comp = hub_rows[lw].as_deref().unwrap_or(&[]);
        let lo = by_comp.partition_point(|e| (e.0 as usize) < comp);
        let hi = by_comp.partition_point(|e| (e.0 as usize) <= comp);
        &by_comp[lo..hi]
    };
    let levels_of = |comp: usize| &comp_levels[comp_ptr[comp]..comp_ptr[comp + 1]];
    let live_columns: Vec<usize> = (0..n_all).filter(|&c| renumber[c] != u32::MAX).collect();
    let work = |c: usize| -> usize {
        (levels_of(c / p).iter())
            .map(|&lw| match hub[lw as usize] {
                true => own_rows(lw as usize, c / p).len(),
                false => rows.by_level.positions(lw as usize).len(),
            })
            .sum()
    };
    let total: usize = live_columns.iter().map(|&c| work(c)).sum();
    let per = total
        .div_ceil(4 * rayon::current_num_threads().max(1))
        .max(1);
    let windows = cut(live_columns.len(), per, |i| work(live_columns[i]));
    let parts: Vec<Vec<(u32, u32, f64)>> = windows
        .par_iter()
        .map_init(
            || (vec![0.0f64; n], vec![u32::MAX; n], Vec::<u32>::new()),
            |(acc, seen, touched), window| {
                let mut out = Vec::new();
                for &c in &live_columns[window.clone()] {
                    let col = renumber[c];
                    let mut visit = |r: usize, u: f64| {
                        if u == 0.0 {
                            return;
                        }
                        let val = rows.root(r) * u;
                        rows.for_each_kept(r, |d, z| {
                            if seen[d] != col {
                                seen[d] = col;
                                touched.push(d as u32);
                            }
                            acc[d] += z * val;
                        });
                    };
                    for &lw in levels_of(c / p) {
                        let lw = lw as usize;
                        if hub[lw] {
                            for &(_, r) in own_rows(lw, c / p) {
                                visit(r as usize, lift.lifted(r as usize, c));
                            }
                            continue;
                        }
                        let (fit, basis) = (&fits[lw], rows.level(lw));
                        let beta = &fit.beta[fit.local(c, p) * p..][..p];
                        for (at, r) in rows.rows(lw).enumerate() {
                            visit(r, remove_fit(&basis, at, lift.lifted(r, c), beta));
                        }
                    }
                    touched.sort_unstable();
                    for &d in touched.iter() {
                        let v = std::mem::take(&mut acc[d as usize]);
                        if v != 0.0 {
                            out.push((col, d, v));
                        }
                    }
                    touched.clear();
                }
                out
            },
        )
        .collect();
    TwoWay::from_parts(n, m, parts)
}
