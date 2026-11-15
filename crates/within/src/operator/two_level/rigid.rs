//! The rigid-component coarse space. A spell (one eliminated level × one anchor level) whose Gram
//! is within a ratio of both levels' own Grams glues the two levels; each glued component's
//! anchor-side gauges span one coarse column per channel. With the eliminated term fitted exactly,
//! `U = ÃZ` is the gauges' energy-minimal extension; on the full system `Z` carries that fit `−β`.

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::ldlt::factor::LdltRegularization;
use faer::linalg::solvers::Solve;
use faer::linalg::triangular_solve::solve_lower_triangular_in_place;
use faer::sparse::linalg::cholesky::{
    factorize_symbolic_cholesky, CholeskySymbolicParams, SymmetricOrdering,
};
use faer::sparse::{SparseColMat, SymbolicSparseColMat};
use faer::{Mat, Par, Side};
use rayon::prelude::*;

use super::{CoarseSolve, HubCorrection, TwoWay};
use crate::domain::{row_weight, Column, LevelRows, PreparedDesign, PreparedTerm, Unknowns};
use crate::CoarseDecline;

/// The gluing ratio tried first; doubled up to 8× while the factor is over budget.
const FIRST_RATIO: f64 = 20.0;

/// Why one gluing ratio built nothing; a larger ratio can only cure `OverBudget`.
enum Declined {
    NoCoarseSpace,
    /// With the spells glued, which a larger ratio must raise to help.
    OverBudget {
        glued: usize,
    },
}

/// The coarse solve over `unknowns`, at the first gluing ratio whose factor fits the budget.
pub(super) fn build(
    prepared: &PreparedDesign<'_>,
    unknowns: Unknowns,
) -> Result<CoarseSolve, CoarseDecline> {
    let rows = Rows::new(prepared, unknowns);
    let anchor = Anchor::find(prepared, unknowns).ok_or(CoarseDecline::NoCoarseSpace)?;
    let (mut ratio, mut glued) = (FIRST_RATIO, None);
    loop {
        match build_glued(prepared, unknowns, &rows, &anchor, ratio) {
            Ok(coarse) => return Ok(coarse),
            Err(Declined::OverBudget { glued: now })
                if ratio < 8.0 * FIRST_RATIO && glued != Some(now) =>
            {
                (ratio, glued) = (2.0 * ratio, Some(now));
            }
            Err(Declined::OverBudget { .. }) => return Err(CoarseDecline::OverBudget),
            Err(Declined::NoCoarseSpace) => return Err(CoarseDecline::NoCoarseSpace),
        }
    }
}

/// The rows grouped by eliminated level, read in the solve basis.
struct Rows<'p> {
    by_level: &'p LevelRows,
    n_levels: usize,
    /// Per eliminated column, its loading; `None` is the intercept.
    eliminated: Vec<Option<&'p [f64]>>,
    /// `diag(A_eᵀA_e)`, each level's whole Gram block after whitening, column-major.
    gram: &'p [f64],
    sqrt_weights: Option<&'p [f64]>,
    kept: Vec<KeptTerm<'p>>,
}

struct KeptTerm<'p> {
    levels: &'p [u32],
    n_levels: usize,
    loadings: Vec<Option<&'p [f64]>>,
    /// Its block start among the unknowns.
    start: usize,
}

fn loadings<'p>(term: &PreparedTerm<'p>) -> Vec<Option<&'p [f64]>> {
    (0..term.term.n_columns())
        .map(|c| term.loading(c))
        .collect()
}

impl<'p> Rows<'p> {
    fn new(prepared: &'p PreparedDesign<'_>, unknowns: Unknowns) -> Self {
        let design = &prepared.design;
        let dominant = design.dominant;
        let kept = (0..design.n_factors())
            .filter(|&t| t != dominant)
            .map(|t| {
                let term = prepared.term(t);
                KeptTerm {
                    levels: term.term.levels(),
                    n_levels: term.term.n_levels(),
                    loadings: loadings(&term),
                    start: design.block_start(unknowns, t).expect("a kept term"),
                }
            })
            .collect();
        Self {
            by_level: prepared.level_rows(dominant),
            n_levels: design.terms[dominant].n_levels(),
            eliminated: loadings(&prepared.term(dominant)),
            gram: prepared.diagonal(dominant),
            sqrt_weights: prepared.sqrt_weights(),
            kept,
        }
    }

    fn width(&self) -> usize {
        self.eliminated.len()
    }

    fn rows(&self, level: usize) -> impl Iterator<Item = usize> + Clone + '_ {
        self.by_level.rows(level)
    }

    fn len(&self, level: usize) -> usize {
        self.by_level.positions(level).len()
    }

    fn root(&self, i: usize) -> f64 {
        row_weight(self.sqrt_weights, i)
    }

    fn weight(&self, i: usize) -> f64 {
        self.root(i) * self.root(i)
    }

    /// Row `i`'s loading in eliminated column `a`.
    fn eliminated(&self, i: usize, a: usize) -> f64 {
        self.eliminated[a].map_or(1.0, |z| z[i])
    }

    /// Level `l`'s `G`, `p × p` row-major and diagonal.
    fn level_gram(&self, l: usize) -> Vec<f64> {
        let p = self.width();
        let mut g = vec![0.0; p * p];
        for a in 0..p {
            g[a * p + a] = self.gram[a * self.n_levels + l];
        }
        g
    }

    /// Level `l`'s `G⁻`, `p × p` row-major: the pseudo-inverse of its diagonal.
    fn level_inverse(&self, l: usize) -> Vec<f64> {
        let mut g = self.level_gram(l);
        g.iter_mut()
            .filter(|g| **g > 0.0)
            .for_each(|g| *g = g.recip());
        g
    }

    /// `f(unknown, loading)` over row `i`'s kept coefficients.
    fn for_each_kept(&self, i: usize, mut f: impl FnMut(usize, f64)) {
        for term in &self.kept {
            let level = term.levels[i] as usize;
            for (c, z) in term.loadings.iter().enumerate() {
                f(
                    term.start + c * term.n_levels + level,
                    z.map_or(1.0, |z| z[i]),
                );
            }
        }
    }
}

/// The kept term with the eliminated term's column layout and the most levels.
struct Anchor<'p> {
    term: PreparedTerm<'p>,
    /// Its block start among the unknowns.
    start: usize,
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

    /// Per level, its Gram and per caller column `j` the solve coefficients reproducing it.
    fn gauges(&self, sqrt_weights: Option<&[f64]>, p: usize) -> (Vec<f64>, Vec<f64>) {
        let term = self.term.term;
        let raw: Vec<&[f64]> = term.raw_slopes().collect();
        let caller = |c: usize, r: usize| match term.column(c) {
            Column::Intercept => 1.0,
            Column::Slope(j) => raw[j][r],
        };
        let nb = self.n_levels();
        let (mut gram, mut rhs) = (vec![0.0; nb * p * p], vec![0.0; nb * p * p]);
        for (r, &l) in term.levels().iter().enumerate() {
            let w = row_weight(sqrt_weights, r).powi(2);
            let at = l as usize * p * p;
            for q in 0..p {
                let wq = w * self.loading(r, q);
                for b in 0..p {
                    gram[at + q * p + b] += wq * self.loading(r, b);
                }
                for j in 0..p {
                    rhs[at + j * p + q] += wq * caller(j, r);
                }
            }
        }
        let mut gauge = vec![0.0; nb * p * p];
        for l in 0..nb {
            let g = &gram[l * p * p..][..p * p];
            for j in 0..p {
                let x = active_solve(g, &rhs[(l * p + j) * p..][..p], p);
                gauge[(l * p + j) * p..][..p].copy_from_slice(&x);
            }
        }
        (gram, gauge)
    }
}

/// The directions of `l` whose diagonal is above rounding of the largest.
fn active(l: &[f64], p: usize) -> Vec<usize> {
    let lmax = (0..p).map(|i| l[i * p + i]).fold(0.0, f64::max);
    (0..p).filter(|&i| l[i * p + i] > 1e-14 * lmax).collect()
}

fn restricted(a: &[f64], p: usize, act: &[usize]) -> Mat<f64> {
    Mat::from_fn(act.len(), act.len(), |i, j| a[act[i] * p + act[j]])
}

/// Condition number of the pencil `(s, l)` on the directions where `l` is positive.
fn pencil_condition(s: &[f64], l: &[f64], p: usize) -> f64 {
    let act = active(l, p);
    if act.is_empty() {
        return f64::INFINITY;
    }
    let Ok(llt) = restricted(l, p, &act).llt(Side::Lower) else {
        return f64::INFINITY;
    };
    let mut m = restricted(s, p, &act);
    solve_lower_triangular_in_place(llt.L(), m.as_mut(), Par::Seq);
    let mut m = m.transpose().to_owned();
    solve_lower_triangular_in_place(llt.L(), m.as_mut(), Par::Seq);
    let Ok(eigenvalues) = m.self_adjoint_eigenvalues(Side::Lower) else {
        return f64::INFINITY;
    };
    let (lo, hi) = (eigenvalues[0], eigenvalues[act.len() - 1]);
    if lo <= 1e-12 * hi {
        f64::INFINITY
    } else {
        hi / lo
    }
}

/// Solves `l x = b` on `l`'s positive directions, zero elsewhere.
fn active_solve(l: &[f64], b: &[f64], p: usize) -> Vec<f64> {
    let mut x = vec![0.0; p];
    let act = active(l, p);
    let Ok(llt) = restricted(l, p, &act).llt(Side::Lower) else {
        return x;
    };
    let mut y = Mat::from_fn(act.len(), 1, |i, _| b[act[i]]);
    llt.solve_in_place(y.as_mut());
    for (k, &i) in act.iter().enumerate() {
        x[i] = y[(k, 0)];
    }
    x
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

/// Spells whose Gram is within `ratio` of both levels' own Grams, as `(eliminated, anchor)` levels.
fn glued_spells(
    rows: &Rows<'_>,
    anchor: &Anchor<'_>,
    anchor_gram: &[f64],
    ratio: f64,
) -> Vec<(u32, u32)> {
    let p = rows.width();
    let eliminated = |r: usize, j: usize| rows.eliminated(r, j);
    let anchored = |r: usize, j: usize| anchor.loading(r, j);
    (0..rows.n_levels)
        .into_par_iter()
        .flat_map_iter(|lw| {
            let (mut level, mut gram) = (vec![0.0; p * p], vec![0.0; p * p]);
            row_gram(rows, rows.rows(lw), eliminated, &mut level);
            let mut by_anchor: Vec<(usize, usize)> =
                rows.rows(lw).map(|r| (anchor.level(r), r)).collect();
            by_anchor.sort_unstable();
            let mut out = Vec::new();
            for spell in by_anchor.chunk_by(|a, b| a.0 == b.0) {
                let lf = spell[0].0;
                let spell_rows = || spell.iter().map(|&(_, r)| r);
                row_gram(rows, spell_rows(), eliminated, &mut gram);
                let mut worst = pencil_condition(&gram, &level, p);
                row_gram(rows, spell_rows(), anchored, &mut gram);
                let own = &anchor_gram[lf * p * p..][..p * p];
                worst = worst.max(pencil_condition(&gram, own, p));
                if worst <= ratio {
                    out.push((lw as u32, lf as u32));
                }
            }
            out
        })
        .collect()
}

/// One eliminated level's share of `U = ÃZ`: the coarse columns its rows touch and their fit `β`.
struct GroupFit {
    /// Touched coarse columns, ascending: every channel of each touched component.
    columns: Vec<u32>,
    /// `β = G⁻ Eᵀ W½ (W½ D Z)`, `width × columns`, row-major.
    beta: Vec<f64>,
}

/// A level's share of the column energies and of `E = UᵀU`.
struct LevelGram {
    /// `‖W½DZ‖²` per touched column, in the fit's column order.
    pre: Vec<f64>,
    /// `‖U‖²` per touched column, in the fit's column order.
    energy: Vec<f64>,
    /// The lower triangle of `UᵀU` over the touched columns; a hub's own blocks only.
    entries: Vec<(u32, u32, f64)>,
    /// A hub's `s = EᵀW DZ` (`width × columns`): rows `[G s]` augmenting `E₀`.
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

    /// `U` at row `r` in local column `i` of `fit`, from its lifted value `v`.
    fn projected(&self, r: usize, mut v: f64, fit: &GroupFit, i: usize) -> f64 {
        let (p, t) = (self.rows.width(), fit.columns.len());
        for a in 0..p {
            v -= self.rows.root(r) * self.rows.eliminated(r, a) * fit.beta[a * t + i];
        }
        v
    }

    /// Level `lw`'s fit, `‖W½DZ‖²` and the lower triangle of `UᵀU` over its columns.
    ///
    /// `UᵀU = (DZ)ᵀW(DZ) − sᵀG⁻s` with `s = EᵀW DZ`: a row touches only its own component's
    /// columns, so the first term is per-component `p × p` and no row-by-column buffer is needed.
    fn fit(&self, lw: usize, comps: &[u32], hub: bool) -> (GroupFit, LevelGram) {
        let (rows, p) = (self.rows, self.rows.width());
        let columns: Vec<u32> = (comps.iter())
            .flat_map(|&c| (0..p as u32).map(move |j| c * p as u32 + j))
            .collect();
        let t = columns.len();
        let (mut pre, mut s) = (vec![0.0; t], vec![0.0; p * t]);
        let mut blocks = vec![0.0; comps.len() * p * p];
        let mut lifted = vec![0.0; p];
        for r in rows.rows(lw) {
            let c = self.anchor_comp[self.anchor.level(r)];
            let k = comps.binary_search(&c).expect("touched component");
            for (j, v) in lifted.iter_mut().enumerate() {
                *v = self.lifted(r, columns[k * p + j] as usize);
                pre[k * p + j] += *v * *v;
            }
            let block = &mut blocks[k * p * p..][..p * p];
            for (i, &vi) in lifted.iter().enumerate() {
                for (j, &vj) in lifted.iter().enumerate() {
                    block[i * p + j] += vi * vj;
                }
            }
            for a in 0..p {
                let wa = rows.root(r) * rows.eliminated(r, a);
                for (j, &v) in lifted.iter().enumerate() {
                    s[a * t + k * p + j] += wa * v;
                }
            }
        }
        let inv = rows.level_inverse(lw);
        let mut beta = vec![0.0; p * t];
        for a in 0..p {
            for b in 0..p {
                for c in 0..t {
                    beta[a * t + c] += inv[a * p + b] * s[b * t + c];
                }
            }
        }
        let own = |a: usize, b: usize| match a / p == b / p {
            true => blocks[(a / p) * p * p + (a % p) * p + b % p],
            false => 0.0,
        };
        let fitted =
            |a: usize, b: usize| -> f64 { (0..p).map(|k| s[k * t + a] * beta[k * t + b]).sum() };
        let energy = (0..t).map(|a| own(a, a) - fitted(a, a)).collect();
        let mut entries = Vec::new();
        for a in 0..t {
            match hub {
                true => (a..(a / p + 1) * p)
                    .for_each(|b| entries.push((columns[a], columns[b], own(a, b)))),
                false => (a..t)
                    .for_each(|b| entries.push((columns[a], columns[b], own(a, b) - fitted(a, b)))),
            }
        }
        let gram = LevelGram {
            pre,
            energy,
            entries,
            hub: hub.then_some(s),
        };
        (GroupFit { columns, beta }, gram)
    }
}

/// Lower `(column, row, value)` triplets as a column-major lower triangle, duplicates summed in
/// input order.
fn lower_csc(m: usize, triplets: Vec<(u32, u32, f64)>) -> (Vec<u32>, Vec<u32>, Vec<f64>) {
    let mut col_ptr = vec![0usize; m + 1];
    for &(c, ..) in &triplets {
        col_ptr[c as usize + 1] += 1;
    }
    for c in 0..m {
        col_ptr[c + 1] += col_ptr[c];
    }
    let mut sorted = vec![(0u32, 0.0); triplets.len()];
    let mut cursor = col_ptr.clone();
    for (c, r, v) in triplets {
        sorted[cursor[c as usize]] = (r, v);
        cursor[c as usize] += 1;
    }
    let (mut ptr, mut rows, mut values) = (vec![0u32], Vec::new(), Vec::new());
    for c in 0..m {
        let column = &mut sorted[col_ptr[c]..col_ptr[c + 1]];
        column.sort_by_key(|e| e.0);
        for run in column.chunk_by(|a, b| a.0 == b.0) {
            let v: f64 = run.iter().map(|e| e.1).sum();
            if v != 0.0 {
                rows.push(run[0].0);
                values.push(v);
            }
        }
        ptr.push(rows.len() as u32);
    }
    (ptr, rows, values)
}

/// Wide levels' `s = EᵀW DZ` rows over the live columns, their `G` and `G⁻`, in level order.
struct HubRows {
    /// The live coarse columns, which the hub rows follow in the factor.
    m: usize,
    ptr: Vec<usize>,
    cols: Vec<u32>,
    vals: Vec<f64>,
    c: Vec<f64>,
    p: usize,
}

impl HubRows {
    fn new(m: usize, p: usize) -> Self {
        Self {
            m,
            ptr: vec![0],
            cols: Vec::new(),
            vals: Vec::new(),
            c: Vec::new(),
            p,
        }
    }

    /// Appends wide level `lw`'s rows `m + rank..`: `G`'s lower triangle and `s` against `columns`.
    fn push(
        &mut self,
        rows: &Rows<'_>,
        lw: usize,
        s: &[f64],
        columns: &[u32],
        renumber: &[u32],
        lower: &mut Vec<(u32, u32, f64)>,
    ) {
        let (t, p) = (columns.len(), self.p);
        let g = rows.level_gram(lw);
        let base = self.m + self.rank();
        // Rows scaled to a unit diagonal: the Schur complement onto `E₀` is unchanged.
        let scale: Vec<f64> = (0..p)
            .map(|k| match g[k * p + k] > 0.0 {
                true => g[k * p + k].sqrt().recip(),
                false => 1.0,
            })
            .collect();
        for k in 0..p {
            let row = (base + k) as u32;
            for j in 0..=k {
                lower.push(((base + j) as u32, row, scale[k] * scale[j] * g[k * p + j]));
            }
            for (i, &c) in columns.iter().enumerate() {
                if renumber[c as usize] != u32::MAX {
                    lower.push((renumber[c as usize], row, scale[k] * s[k * t + i]));
                    self.cols.push(renumber[c as usize]);
                    self.vals.push(s[k * t + i]);
                }
            }
            self.ptr.push(self.cols.len());
        }
        self.c.extend(rows.level_inverse(lw));
    }

    fn rank(&self) -> usize {
        self.ptr.len() - 1
    }

    fn into_correction(self, f: TwoWay) -> HubCorrection {
        HubCorrection {
            ptr: self.ptr,
            cols: self.cols,
            vals: self.vals,
            c: self.c,
            p: self.p,
            f,
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
    ratio: f64,
) -> Result<CoarseSolve, Declined> {
    let design = &prepared.design;
    let n_obs = design.n_obs;
    let p = rows.width();
    let (na, nb) = (rows.n_levels, anchor.n_levels());
    let (anchor_gram, gauge) = anchor.gauges(prepared.sqrt_weights(), p);
    let glued = glued_spells(rows, anchor, &anchor_gram, ratio);

    let mut parent: Vec<u32> = (0..(na + nb) as u32).collect();
    for &(a, b) in &glued {
        let (ra, rb) = (find(&mut parent, a), find(&mut parent, na as u32 + b));
        if ra != rb {
            parent[ra.max(rb) as usize] = ra.min(rb);
        }
    }
    let roots: Vec<u32> = (0..(na + nb) as u32)
        .map(|x| find(&mut parent, x))
        .collect();
    let n_components = {
        let mut seen = vec![false; na + nb];
        (roots.iter())
            .filter(|&&r| !std::mem::replace(&mut seen[r as usize], true))
            .count()
    };
    // Components covering most levels glue nothing worth a coarse column.
    if 2 * n_components > na + nb {
        return Err(Declined::NoCoarseSpace);
    }
    // Coarse index per component touching the anchor side.
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
    // Per eliminated level, the anchor components its rows touch, and whether it splits one.
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
        return Err(Declined::NoCoarseSpace);
    }
    let m = n_fc as usize * p;
    // A level touching `t` columns adds a dense `t × t` block to `E`; a wide one (`t² > 2pm`)
    // instead augments the coarse system with its `p` rows `[G s]`, leaving `E₀` sparse.
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
        return Err(Declined::OverBudget { glued: glued.len() });
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
        gauge: &gauge,
        inv_norm: &inv_norm,
    };
    let per_level: Vec<(GroupFit, LevelGram)> = (touched.par_iter())
        .enumerate()
        .map(|(lw, (comps, _))| lift.fit(lw, comps, hub[lw]))
        .collect();
    drop(touched);
    // Exact nulls of S (the eliminated term absorbs the gauge) need no lift; keep the rest.
    let (mut pre_energy, mut energy) = (vec![0.0f64; m], vec![0.0f64; m]);
    for (fit, gram) in &per_level {
        for (&c, v) in fit.columns.iter().zip(&gram.pre) {
            pre_energy[c as usize] += v;
        }
        for (&c, v) in fit.columns.iter().zip(&gram.energy) {
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
        return Err(Declined::NoCoarseSpace);
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
            let t = fit.columns.len();
            for a in 0..p {
                let unknown = (start + a * na + lw) as u32;
                for (i, &c) in fit.columns.iter().enumerate() {
                    let v = fit.beta[a * t + i];
                    if renumber[c as usize] != u32::MAX && v != 0.0 {
                        z_entries.push((renumber[c as usize], unknown, -v));
                    }
                }
            }
        }
    }
    z_entries.par_sort_unstable_by_key(|t| (t.0, t.1));
    let z = TwoWay::from_parts(n, m, vec![z_entries]);
    // E = UᵀU from each level's block, lower triangle by column.
    let (mut fits, mut e_triplets) = (Vec::with_capacity(na), Vec::new());
    let mut hubs = HubRows::new(m, p);
    for (lw, (fit, gram)) in per_level.into_iter().enumerate() {
        e_triplets.extend(gram.entries.into_iter().filter_map(|(a, b, v)| {
            let (a, b) = (renumber[a as usize], renumber[b as usize]);
            (a != u32::MAX && b != u32::MAX).then_some((a, b, v))
        }));
        if let Some(s) = gram.hub {
            hubs.push(rows, lw, &s, &fit.columns, &renumber, &mut e_triplets);
        }
        fits.push(fit);
    }
    let rank = hubs.rank();
    let dim = m + rank;
    let (col_ptr, row_idx, mut values) = lower_csc(dim, e_triplets);
    let diagonal = |c: usize, values: &[f64]| {
        let first = col_ptr[c] as usize;
        (first < col_ptr[c + 1] as usize && row_idx[first] as usize == c).then(|| values[first])
    };
    // Unit diagonal: null pivots then sit at rounding level, far below any live mode.
    let scale: Vec<f64> = (0..dim)
        .map(|c| match diagonal(c, &values) {
            Some(d) if d > 0.0 => d.sqrt().recip(),
            _ => 1.0,
        })
        .collect();
    for c in 0..dim {
        for k in col_ptr[c] as usize..col_ptr[c + 1] as usize {
            values[k] *= scale[c] * scale[row_idx[k] as usize];
        }
    }
    let d_max = (0..dim)
        .filter_map(|c| diagonal(c, &values))
        .fold(0.0, f64::max);
    let pattern = SymbolicSparseColMat::new_checked(dim, dim, col_ptr, None, row_idx);
    let e_lower = SparseColMat::new(pattern, values);
    let symbolic = factorize_symbolic_cholesky(
        e_lower.symbolic(),
        Side::Lower,
        SymmetricOrdering::Amd,
        CholeskySymbolicParams::default(),
    )
    .map_err(|_| Declined::NoCoarseSpace)?;
    if symbolic.len_val() > n_obs {
        return Err(Declined::OverBudget { glued: glued.len() });
    }
    let sz = projected_columns(rows, &lift, &fits, &hub, &renumber, n, m);
    let mut f_entries = Vec::new();
    for (h, lw) in (0..na).filter(|&lw| hub[lw]).enumerate() {
        let mut acc = vec![0.0; n * p];
        for r in rows.rows(lw) {
            let w = rows.weight(r);
            rows.for_each_kept(r, |d, z| {
                for a in 0..p {
                    acc[d * p + a] += w * z * rows.eliminated(r, a);
                }
            });
        }
        for a in 0..p {
            (0..n)
                .filter(|&d| acc[d * p + a] != 0.0)
                .for_each(|d| f_entries.push(((h * p + a) as u32, d as u32, acc[d * p + a])));
        }
    }
    let f = TwoWay::from_parts(n, rank, vec![f_entries]);
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
        .map_err(|_| Declined::NoCoarseSpace)?;
    if !l_values.iter().all(|v| v.is_finite()) {
        return Err(Declined::NoCoarseSpace);
    }
    Ok(CoarseSolve {
        m,
        z,
        sz,
        hubs: (rank > 0).then(|| hubs.into_correction(f)),
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
        for &c in fit.columns.iter().step_by(p) {
            comp_ptr[c as usize / p + 1] += 1;
        }
    }
    for c in 0..n_fc {
        comp_ptr[c + 1] += comp_ptr[c];
    }
    let mut comp_levels = vec![0u32; comp_ptr[n_fc]];
    let mut cursor = comp_ptr.clone();
    for (lw, fit) in fits.iter().enumerate() {
        for &c in fit.columns.iter().step_by(p) {
            let k = &mut cursor[c as usize / p];
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
                false => rows.len(lw as usize),
            })
            .sum()
    };
    let total: usize = live_columns.iter().map(|&c| work(c)).sum();
    let per = total
        .div_ceil(4 * rayon::current_num_threads().max(1))
        .max(1);
    let (mut windows, mut first, mut load) = (Vec::new(), 0, 0);
    for (i, &c) in live_columns.iter().enumerate() {
        load += work(c);
        if load >= per {
            windows.push(first..i + 1);
            (first, load) = (i + 1, 0);
        }
    }
    if first < live_columns.len() {
        windows.push(first..live_columns.len());
    }
    let parts: Vec<Vec<(u32, u32, f64)>> = windows
        .par_iter()
        .map(|window| {
            let mut acc = vec![0.0f64; n];
            let mut seen = vec![u32::MAX; n];
            let mut touched: Vec<u32> = Vec::new();
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
                    let fit = &fits[lw];
                    let i = (fit.columns)
                        .binary_search(&(c as u32))
                        .expect("touched column");
                    for r in rows.rows(lw) {
                        visit(r, lift.projected(r, lift.lifted(r, c), fit, i));
                    }
                }
                touched.sort_unstable();
                for &d in &touched {
                    let v = std::mem::take(&mut acc[d as usize]);
                    if v != 0.0 {
                        out.push((col, d, v));
                    }
                }
                touched.clear();
            }
            out
        })
        .collect();
    TwoWay::from_parts(n, m, parts)
}
