//! Rigid-component coarse space on the reduced system (prototype, `WITHIN_REDUCED_RIGID=K`).
//!
//! Spells (one eliminated level × one anchor level) whose Gram is within `K` of both levels' own
//! Grams glue the two levels; each glued component's anchor-side gauges span one coarse column per
//! channel. With the eliminated term solved exactly, `Ã Z` is the energy-minimal extension.

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::ldlt::factor::LdltRegularization;
use faer::linalg::solvers::SolveCore;
use faer::sparse::linalg::cholesky::{
    factorize_symbolic_cholesky, CholeskySymbolicParams, LdltRef, SymbolicCholesky,
    SymmetricOrdering,
};
use faer::sparse::{SparseColMat, SymbolicSparseColMat};
use faer::{Conj, MatMut, Par, Side};
use rayon::prelude::*;
use schwarz_precond::{Operator, SolveError};

use super::coarse::CoarseRung;
use crate::domain::elimination::{Elimination, GroupedRows};
use crate::domain::{row_weight, Column, PreparedTerm, Term};

/// Below this many entries a product is faster on one thread than fanned out over rayon.
const PARALLEL_NNZ: usize = 1 << 18;

/// Sparse `n × m` held both ways: by column for `Mᵀx`, by nonempty row for `M c`.
struct TwoWay {
    col_ptr: Vec<usize>,
    col_idx: Vec<u32>,
    col_val: Vec<f64>,
    rows: Vec<u32>,
    row_ptr: Vec<usize>,
    row_idx: Vec<u32>,
    row_val: Vec<f64>,
}

/// Keep only the nonempty rows of a filled `n + 1` row pointer.
fn nonempty_rows(row_ptr: &[usize]) -> (Vec<u32>, Vec<usize>) {
    let rows: Vec<u32> = (0..row_ptr.len() - 1)
        .filter(|&r| row_ptr[r + 1] > row_ptr[r])
        .map(|r| r as u32)
        .collect();
    let ptr = rows
        .iter()
        .map(|&r| row_ptr[r as usize])
        .chain(row_ptr.last().copied())
        .collect();
    (rows, ptr)
}

impl TwoWay {
    /// From `(column, row, value)` triplets sorted by column then row, duplicates merged.
    fn from_sorted(n: usize, m: usize, entries: &[(u32, u32, f64)]) -> Self {
        let mut col_ptr = vec![0usize; m + 1];
        let mut row_ptr = vec![0usize; n + 1];
        for &(c, r, _) in entries {
            col_ptr[c as usize + 1] += 1;
            row_ptr[r as usize + 1] += 1;
        }
        for k in 0..m {
            col_ptr[k + 1] += col_ptr[k];
        }
        for k in 0..n {
            row_ptr[k + 1] += row_ptr[k];
        }
        let mut cursor = row_ptr.clone();
        let mut row_idx = vec![0u32; entries.len()];
        let mut row_val = vec![0.0; entries.len()];
        for &(c, r, v) in entries {
            let k = &mut cursor[r as usize];
            row_idx[*k] = c;
            row_val[*k] = v;
            *k += 1;
        }
        let (rows, row_ptr) = nonempty_rows(&row_ptr);
        Self {
            col_ptr,
            col_idx: entries.iter().map(|e| e.1).collect(),
            col_val: entries.iter().map(|e| e.2).collect(),
            rows,
            row_ptr,
            row_idx,
            row_val,
        }
    }

    /// From column-ordered parts (each sorted by column then row), consumed one at a time.
    fn from_parts(n: usize, m: usize, parts: Vec<Vec<(u32, u32, f64)>>) -> Self {
        let total: usize = parts.iter().map(Vec::len).sum();
        let mut col_ptr = vec![0usize; m + 1];
        let mut row_ptr = vec![0usize; n + 1];
        for &(c, r, _) in parts.iter().flatten() {
            col_ptr[c as usize + 1] += 1;
            row_ptr[r as usize + 1] += 1;
        }
        for k in 0..m {
            col_ptr[k + 1] += col_ptr[k];
        }
        for k in 0..n {
            row_ptr[k + 1] += row_ptr[k];
        }
        let (mut col_idx, mut col_val) = (Vec::with_capacity(total), Vec::with_capacity(total));
        let (mut row_idx, mut row_val) = (vec![0u32; total], vec![0.0; total]);
        let mut cursor = row_ptr.clone();
        for part in parts {
            for (c, r, v) in part {
                col_idx.push(r);
                col_val.push(v);
                let k = &mut cursor[r as usize];
                row_idx[*k] = c;
                row_val[*k] = v;
                *k += 1;
            }
        }
        let (rows, row_ptr) = nonempty_rows(&row_ptr);
        Self {
            col_ptr,
            col_idx,
            col_val,
            rows,
            row_ptr,
            row_idx,
            row_val,
        }
    }

    fn nnz(&self) -> usize {
        self.col_idx.len()
    }

    fn t_mul(&self, x: &[f64], out: &mut [f64]) {
        let column = |(c, o): (usize, &mut f64)| {
            let span = self.col_ptr[c]..self.col_ptr[c + 1];
            *o = (self.col_idx[span.clone()].iter().zip(&self.col_val[span]))
                .map(|(&r, &v)| v * x[r as usize])
                .sum();
        };
        match self.nnz() < PARALLEL_NNZ {
            true => out.iter_mut().enumerate().for_each(column),
            false => out.par_iter_mut().enumerate().for_each(column),
        }
    }

    fn mul_add(&self, c: &[f64], alpha: f64, out: &mut [f64]) {
        const BLOCK: usize = 4096;
        let row = |i: usize| -> f64 {
            let span = self.row_ptr[i]..self.row_ptr[i + 1];
            (self.row_idx[span.clone()].iter().zip(&self.row_val[span]))
                .map(|(&k, &v)| v * c[k as usize])
                .sum()
        };
        if self.nnz() < PARALLEL_NNZ {
            for (i, &r) in self.rows.iter().enumerate() {
                out[r as usize] += alpha * row(i);
            }
            return;
        }
        out.par_chunks_mut(BLOCK)
            .enumerate()
            .for_each(|(j, block)| {
                let base = j * BLOCK;
                let lo = self.rows.partition_point(|&r| (r as usize) < base);
                let hi = self
                    .rows
                    .partition_point(|&r| (r as usize) < base + block.len());
                for i in lo..hi {
                    block[self.rows[i] as usize - base] += alpha * row(i);
                }
            });
    }
}

/// Cyclic Jacobi eigenvalues of a small symmetric matrix (row-major, overwritten).
fn small_eigenvalues(a: &mut [f64], q: usize) -> Vec<f64> {
    for _ in 0..64 {
        let off: f64 = (0..q)
            .flat_map(|i| (0..q).filter(move |&j| j != i).map(move |j| (i, j)))
            .map(|(i, j)| a[i * q + j] * a[i * q + j])
            .sum();
        if off <= 1e-30 * (0..q).map(|i| a[i * q + i] * a[i * q + i]).sum::<f64>() {
            break;
        }
        for i in 0..q {
            for j in i + 1..q {
                if a[i * q + j] == 0.0 {
                    continue;
                }
                let theta = (a[j * q + j] - a[i * q + i]) / (2.0 * a[i * q + j]);
                let t = if theta == 0.0 {
                    1.0
                } else {
                    theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt())
                };
                let (c, s) = (1.0 / (t * t + 1.0).sqrt(), t / (t * t + 1.0).sqrt());
                for k in 0..q {
                    let (aki, akj) = (a[k * q + i], a[k * q + j]);
                    a[k * q + i] = c * aki - s * akj;
                    a[k * q + j] = s * aki + c * akj;
                }
                for k in 0..q {
                    let (aik, ajk) = (a[i * q + k], a[j * q + k]);
                    a[i * q + k] = c * aik - s * ajk;
                    a[j * q + k] = s * aik + c * ajk;
                }
            }
        }
    }
    (0..q).map(|i| a[i * q + i]).collect()
}

/// Lower Cholesky of the directions of `l` with a positive diagonal; `None` if indefinite.
fn active_cholesky(l: &[f64], p: usize) -> Option<(Vec<usize>, Vec<f64>)> {
    let lmax = (0..p).map(|i| l[i * p + i]).fold(0.0, f64::max);
    let act: Vec<usize> = (0..p).filter(|&i| l[i * p + i] > 1e-14 * lmax).collect();
    let q = act.len();
    let mut c = vec![0.0; q * q];
    for i in 0..q {
        for j in 0..=i {
            let mut v = l[act[i] * p + act[j]];
            for k in 0..j {
                v -= c[i * q + k] * c[j * q + k];
            }
            if i == j {
                if v <= 0.0 {
                    return None;
                }
                c[i * q + i] = v.sqrt();
            } else {
                c[i * q + j] = v / c[j * q + j];
            }
        }
    }
    Some((act, c))
}

/// Condition number of the pencil `(s, l)` on the directions where `l` is positive.
fn pencil_condition(s: &[f64], l: &[f64], p: usize) -> f64 {
    let Some((act, c)) = active_cholesky(l, p) else {
        return f64::INFINITY;
    };
    let q = act.len();
    if q == 0 {
        return f64::INFINITY;
    }
    let mut m = vec![0.0; q * q];
    for i in 0..q {
        for j in 0..q {
            m[i * q + j] = s[act[i] * p + act[j]];
        }
    }
    for col in 0..q {
        for i in 0..q {
            let mut v = m[i * q + col];
            for k in 0..i {
                v -= c[i * q + k] * m[k * q + col];
            }
            m[i * q + col] = v / c[i * q + i];
        }
    }
    for row in 0..q {
        for i in 0..q {
            let mut v = m[row * q + i];
            for k in 0..i {
                v -= c[i * q + k] * m[row * q + k];
            }
            m[row * q + i] = v / c[i * q + i];
        }
    }
    let ev = small_eigenvalues(&mut m, q);
    let (lo, hi) = (ev.iter()).fold((f64::INFINITY, 0.0f64), |(lo, hi), &e| {
        (lo.min(e), hi.max(e))
    });
    if lo <= 1e-12 * hi {
        f64::INFINITY
    } else {
        hi / lo
    }
}

/// Solves `l x = b` on `l`'s positive directions, zero elsewhere.
fn active_solve(l: &[f64], b: &[f64], p: usize) -> Vec<f64> {
    let mut x = vec![0.0; p];
    let Some((act, c)) = active_cholesky(l, p) else {
        return x;
    };
    let q = act.len();
    let mut y: Vec<f64> = act.iter().map(|&i| b[i]).collect();
    for i in 0..q {
        for k in 0..i {
            y[i] -= c[i * q + k] * y[k];
        }
        y[i] /= c[i * q + i];
    }
    for i in (0..q).rev() {
        for k in i + 1..q {
            y[i] -= c[k * q + i] * y[k];
        }
        y[i] /= c[i * q + i];
    }
    for (k, &i) in act.iter().enumerate() {
        x[i] = y[k];
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

/// The kept term sharing the eliminated term's column layout, with the most levels.
struct Anchor<'a> {
    term: PreparedTerm<'a>,
    /// Its first column's index among the kept channels.
    channel: usize,
    /// Its first coefficient among the kept coefficients.
    base: usize,
}

impl<'a> Anchor<'a> {
    fn find(e: &Elimination<'a>) -> Option<Self> {
        let design = &e.prepared().design;
        let layout = |t: &Term<'_>| {
            (0..t.n_columns())
                .map(|c| matches!(t.column(c), Column::Intercept))
                .collect::<Vec<_>>()
        };
        let target = layout(&design.terms[e.term()]);
        let (mut channel, mut base) = (0, 0);
        let mut anchor: Option<(usize, usize, usize)> = None;
        for (t, term) in design.terms.iter().enumerate() {
            if t == e.term() {
                continue;
            }
            if layout(term) == target
                && anchor.is_none_or(|(a, ..)| design.terms[a].n_levels() < term.n_levels())
            {
                anchor = Some((t, channel, base));
            }
            channel += term.n_columns();
            base += term.n_dofs();
        }
        let (t, channel, base) = anchor?;
        Some(Self {
            term: e.prepared().term(t),
            channel,
            base,
        })
    }

    fn n_levels(&self) -> usize {
        self.term.term.n_levels()
    }

    /// Grouped row `p`'s anchor level.
    fn level(&self, rows: &GroupedRows, p: usize) -> usize {
        rows.kept.entry(p, self.channel).0 as usize - self.base
    }

    /// Grouped row `p`'s anchor loading in column `q`.
    fn loading(&self, rows: &GroupedRows, p: usize, q: usize) -> f64 {
        rows.kept.entry(p, self.channel + q).1
    }

    /// Per anchor level, its Gram and per user channel `j` the solve coefficients reproducing it.
    fn gauges(&self, sqrt_weights: Option<&[f64]>, p: usize) -> (Vec<f64>, Vec<f64>) {
        let term = self.term.term;
        let raw: Vec<&[f64]> = term.raw_slopes().collect();
        let user = |c: usize, r: usize| match term.column(c) {
            Column::Intercept => 1.0,
            Column::Slope(j) => raw[j][r],
        };
        let coef = |q: usize, r: usize| self.term.loading(q).map_or(1.0, |z| z[r]);
        let nb = self.n_levels();
        let (mut gram, mut rhs) = (vec![0.0; nb * p * p], vec![0.0; nb * p * p]);
        for (r, &l) in term.levels().iter().enumerate() {
            let wr = row_weight(sqrt_weights, r);
            let at = l as usize * p * p;
            for q in 0..p {
                let wq = wr * coef(q, r);
                for b in 0..p {
                    gram[at + q * p + b] += wq * coef(b, r);
                }
                for j in 0..p {
                    rhs[at + j * p + q] += wq * user(j, r);
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

/// `Σ w a aᵀ` over grouped rows `span`, `a(p, j)` row `p`'s loading in column `j`.
fn row_gram(
    rows: &GroupedRows,
    span: impl Iterator<Item = usize>,
    a: impl Fn(usize, usize) -> f64,
    out: &mut [f64],
) {
    let p = rows.eliminated.width;
    out.fill(0.0);
    for r in span {
        let wr = rows.weight(r);
        for i in 0..p {
            let wa = wr * a(r, i);
            for j in 0..p {
                out[i * p + j] += wa * a(r, j);
            }
        }
    }
}

/// Spells whose Gram is within `k` of both levels' own Grams, as `(eliminated, anchor)` levels.
fn glued_spells(
    rows: &GroupedRows,
    anchor: &Anchor<'_>,
    anchor_gram: &[f64],
    k: f64,
) -> Vec<(u32, u32)> {
    let p = rows.eliminated.width;
    let eliminated = |r: usize, j: usize| rows.eliminated.loading(r, j);
    let anchored = |r: usize, j: usize| anchor.loading(rows, r, j);
    (0..rows.n_levels())
        .into_par_iter()
        .flat_map_iter(|lw| {
            let span = rows.starts[lw]..rows.starts[lw + 1];
            let (mut level, mut gram) = (vec![0.0; p * p], vec![0.0; p * p]);
            row_gram(rows, span.clone(), eliminated, &mut level);
            let mut by_anchor: Vec<(usize, usize)> =
                span.map(|r| (anchor.level(rows, r), r)).collect();
            by_anchor.sort_unstable();
            let mut out = Vec::new();
            for spell in by_anchor.chunk_by(|a, b| a.0 == b.0) {
                let lf = spell[0].0;
                let spell_rows = || spell.iter().map(|&(_, r)| r);
                row_gram(rows, spell_rows(), eliminated, &mut gram);
                let mut worst = pencil_condition(&gram, &level, p);
                row_gram(rows, spell_rows(), anchored, &mut gram);
                worst = worst.max(pencil_condition(
                    &gram,
                    &anchor_gram[lf * p * p..][..p * p],
                    p,
                ));
                if worst <= k {
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
    /// A hub's `s = EᵀW DZ`, `width × columns`: the rank-`p` part left out of `entries`.
    correction: Option<Vec<f64>>,
}

/// What the coarse columns are made of, to evaluate `U = ÃZ` one entry at a time.
struct Lift<'a> {
    rows: &'a GroupedRows,
    anchor: &'a Anchor<'a>,
    anchor_comp: &'a [u32],
    gauge: &'a [f64],
    inv_norm: &'a [f64],
}

impl Lift<'_> {
    /// `√w (D Z)` at grouped row `r` in coarse column `c`: nonzero only on the row's component.
    fn lifted(&self, r: usize, c: usize) -> f64 {
        let p = self.rows.eliminated.width;
        let l = self.anchor.level(self.rows, r);
        if self.anchor_comp[l] as usize != c / p {
            return 0.0;
        }
        let g = &self.gauge[(l * p + c % p) * p..][..p];
        let val: f64 = (0..p)
            .map(|q| self.anchor.loading(self.rows, r, q) * g[q])
            .sum();
        self.root(r) * val * self.inv_norm[c]
    }

    fn root(&self, r: usize) -> f64 {
        self.rows.sqrt_weights.as_ref().map_or(1.0, |s| s[r])
    }

    /// `U` at grouped row `r` in local column `i` of `fit`, from its lifted value `v`.
    fn projected(&self, r: usize, mut v: f64, fit: &GroupFit, i: usize) -> f64 {
        let (p, t) = (self.rows.eliminated.width, fit.columns.len());
        for a in 0..p {
            let wa = self.root(r) * self.rows.eliminated.loading(r, a);
            v -= wa * fit.beta[a * t + i];
        }
        v
    }

    /// Level `lw`'s fit, `‖W½DZ‖²` and the lower triangle of `UᵀU` over its columns.
    ///
    /// `UᵀU = (DZ)ᵀW(DZ) − sᵀG⁻s` with `s = EᵀW DZ`: a row touches only its own component's
    /// columns, so the first term is per-component `p × p` and no row-by-column buffer is needed.
    fn fit(&self, lw: usize, comps: &[u32], inverse: &[f64], hub: bool) -> (GroupFit, LevelGram) {
        let (rows, p) = (self.rows, self.rows.eliminated.width);
        let columns: Vec<u32> = (comps.iter())
            .flat_map(|&c| (0..p as u32).map(move |j| c * p as u32 + j))
            .collect();
        let t = columns.len();
        let (mut pre, mut s) = (vec![0.0; t], vec![0.0; p * t]);
        let mut blocks = vec![0.0; comps.len() * p * p];
        let mut lifted = vec![0.0; p];
        for r in rows.starts[lw]..rows.starts[lw + 1] {
            let c = self.anchor_comp[self.anchor.level(rows, r)];
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
                let wa = self.root(r) * rows.eliminated.loading(r, a);
                for (j, &v) in lifted.iter().enumerate() {
                    s[a * t + k * p + j] += wa * v;
                }
            }
        }
        let inv = &inverse[lw * p * p..][..p * p];
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
            correction: hub.then_some(s),
        };
        (GroupFit { columns, beta }, gram)
    }
}

/// Lower `(column, row, value)` triplets as a column-major lower triangle, duplicates summed in input order.
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

/// The coarse space `Z` over the kept coefficients and the LDLᵀ of `ZᵀSZ = (ÃZ)ᵀ(ÃZ)`.
pub(crate) struct ReducedCoarse {
    m: usize,
    z: TwoWay,
    sz: TwoWay,
    symbolic: SymbolicCholesky<u32>,
    l_values: Vec<f64>,
    par: Par,
    /// Hub levels' rank-`p` corrections: the factor above is of `E₀ = E + BᵀCB`.
    low_rank: Option<LowRank>,
}

/// Hub levels' `B` rows (`s = EᵀW DZ` over live columns) and `C` blocks (`G⁻`), gathered in level order.
#[derive(Default)]
struct LowRankParts {
    ptr: Vec<usize>,
    cols: Vec<u32>,
    vals: Vec<f64>,
    c: Vec<f64>,
    p: usize,
}

impl LowRankParts {
    fn push(&mut self, s: &[f64], columns: &[u32], renumber: &[u32], inverse: &[f64]) {
        let (t, p) = (columns.len(), s.len() / columns.len());
        if self.ptr.is_empty() {
            self.ptr.push(0);
        }
        self.p = p;
        for k in 0..p {
            for (i, &c) in columns.iter().enumerate() {
                if renumber[c as usize] != u32::MAX {
                    self.cols.push(renumber[c as usize]);
                    self.vals.push(s[k * t + i]);
                }
            }
            self.ptr.push(self.cols.len());
        }
        self.c.extend_from_slice(inverse);
    }

    fn rank(&self) -> usize {
        self.ptr.len().saturating_sub(1)
    }

    /// `W = E₀⁻¹Bᵀ` and the LU of `K = I − C B W`; `None` when `K` is numerically singular.
    fn factor(self, f: TwoWay, solve: impl Fn(&mut [f64]), m: usize) -> Option<LowRank> {
        let (r, p) = (self.rank(), self.p);
        let mut w = vec![0.0; m * r];
        for (i, column) in w.chunks_exact_mut(m).enumerate() {
            for k in self.ptr[i]..self.ptr[i + 1] {
                column[self.cols[k] as usize] = self.vals[k];
            }
            solve(column);
        }
        let b_row = |i: usize, x: &[f64]| -> f64 {
            (self.ptr[i]..self.ptr[i + 1])
                .map(|k| self.vals[k] * x[self.cols[k] as usize])
                .sum()
        };
        let bw: Vec<f64> = (0..r)
            .flat_map(|j| (0..r).map(move |i| (i, j)))
            .map(|(i, j)| b_row(i, &w[j * m..][..m]))
            .collect();
        let k = faer::Mat::<f64>::from_fn(r, r, |i, j| {
            let (h, a) = (i / p, i % p);
            let cbw: f64 = (0..p)
                .map(|b| self.c[h * p * p + a * p + b] * bw[j * r + h * p + b])
                .sum();
            (i == j) as u8 as f64 - cbw
        });
        let low_rank = LowRank {
            ptr: self.ptr,
            cols: self.cols,
            vals: self.vals,
            c: self.c,
            p,
            w,
            k: k.partial_piv_lu(),
            f,
        };
        let mut probe = vec![1.0; m];
        low_rank.correct(&mut probe);
        probe.iter().all(|v| v.is_finite()).then_some(low_rank)
    }
}

/// `E⁻¹ = E₀⁻¹ + W K⁻¹ C B E₀⁻¹` for `E = E₀ − BᵀCB`, `W = E₀⁻¹Bᵀ`, `K = I − C B W`.
struct LowRank {
    ptr: Vec<usize>,
    cols: Vec<u32>,
    vals: Vec<f64>,
    c: Vec<f64>,
    p: usize,
    /// `E₀⁻¹Bᵀ`, column-major `m × r`.
    w: Vec<f64>,
    k: faer::linalg::solvers::PartialPivLu<f64>,
    /// `D_keptᵀ W E` over each hub's rows, `n × r`: `SZ = SZ₀ − F C B`.
    f: TwoWay,
}

impl LowRank {
    fn rank(&self) -> usize {
        self.ptr.len() - 1
    }

    /// `C q` blockwise: each hub's `G⁻`.
    fn c_mul(&self, q: &[f64]) -> Vec<f64> {
        let p = self.p;
        (0..q.len())
            .map(|i| {
                let (h, a) = (i / p, i % p);
                (0..p)
                    .map(|b| self.c[h * p * p + a * p + b] * q[h * p + b])
                    .sum()
            })
            .collect()
    }

    /// `C B v`: each hub's fit `β` of a coarse vector.
    fn beta(&self, v: &[f64]) -> Vec<f64> {
        let bv: Vec<f64> = (0..self.rank())
            .map(|i| {
                (self.ptr[i]..self.ptr[i + 1])
                    .map(|k| self.vals[k] * v[self.cols[k] as usize])
                    .sum()
            })
            .collect();
        self.c_mul(&bv)
    }

    /// `out += alpha (F C B) c`.
    fn add_sz(&self, c: &[f64], alpha: f64, out: &mut [f64]) {
        self.f.mul_add(&self.beta(c), alpha, out);
    }

    /// `out += alpha (F C B)ᵀ y = alpha Bᵀ C Fᵀ y`.
    fn add_sz_t(&self, y: &[f64], alpha: f64, out: &mut [f64]) {
        let mut fy = vec![0.0; self.rank()];
        self.f.t_mul(y, &mut fy);
        for (i, q) in self.c_mul(&fy).into_iter().enumerate() {
            for k in self.ptr[i]..self.ptr[i + 1] {
                out[self.cols[k] as usize] += alpha * self.vals[k] * q;
            }
        }
    }

    /// `v ← v + W K⁻¹ C B v`, turning `v = E₀⁻¹x` into `E⁻¹x`.
    fn correct(&self, v: &mut [f64]) {
        let (r, m) = (self.rank(), v.len());
        let mut q = self.beta(v);
        self.k
            .solve_in_place_with_conj(Conj::No, MatMut::from_column_major_slice_mut(&mut q, r, 1));
        for (j, &qj) in q.iter().enumerate() {
            for (vi, wi) in v.iter_mut().zip(&self.w[j * m..][..m]) {
                *vi += wi * qj;
            }
        }
    }
}

impl ReducedCoarse {
    /// `None` without a kept term sharing the eliminated term's column layout, when the
    /// components cover more than half the levels, or when `E` or its factor exceeds `8 · n_obs` values.
    pub(crate) fn build(e: &Elimination<'_>, k: f64) -> Option<Self> {
        let started = std::time::Instant::now();
        let stats = std::env::var_os("WITHIN_REDUCED_RIGID_STATS").is_some();
        let stage = |name: &str| {
            if stats {
                eprintln!(
                    "reduced_stage: {name} {:.3}",
                    started.elapsed().as_secs_f64()
                );
            }
        };
        let n_obs = e.prepared().design.n_obs();
        let p = e.width();
        let anchor = Anchor::find(e)?;
        let rows = &**e.rows();
        let (na, nb) = (rows.n_levels(), anchor.n_levels());
        let (anchor_gram, gauge) = anchor.gauges(e.prepared().sqrt_weights(), p);
        let glued = glued_spells(rows, &anchor, &anchor_gram, k);
        stage("spells");

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
            roots
                .iter()
                .filter(|&&r| !std::mem::replace(&mut seen[r as usize], true))
                .count()
        };
        // Components covering most levels glue nothing worth a coarse column.
        if 2 * n_components > na + nb {
            if stats {
                eprintln!(
                    "reduced_rigid: glued={} components={n_components} levels={} declined",
                    glued.len(),
                    na + nb
                );
            }
            return None;
        }
        stage("components");
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
                let span = rows.starts[lw]..rows.starts[lw + 1];
                let comp = |r: usize| anchor_comp[anchor.level(rows, r)];
                // Zero-weight rows carry no energy, so they cannot split a level.
                let mut positive = span.clone().filter(|&r| rows.weight(r) > 0.0).map(comp);
                let first = positive.next();
                let split = positive.any(|c| Some(c) != first);
                let mut comps: Vec<u32> = span.map(comp).collect();
                comps.sort_unstable();
                comps.dedup();
                (comps, split)
            })
            .collect();
        // An eliminated level inside one anchor component absorbs the gauges on its rows exactly.
        if !touched.iter().any(|t| t.1) {
            if stats {
                eprintln!("reduced_rigid: anchor_components={n_fc} none split, declined");
            }
            return None;
        }
        let m = n_fc as usize * p;
        // A level touching `t` columns adds a dense `t × t` block to `E`; past `t² > 2pm` its exact
        // rank-`p` Woodbury correction (`p × m`) is cheaper. The widest go first, rank capped at `m/4`.
        let mut hub = vec![false; na];
        let mut wide: Vec<usize> = (0..na)
            .filter(|&l| (touched[l].0.len() * p).pow(2) > 2 * p * m)
            .collect();
        wide.sort_unstable_by_key(|&l| std::cmp::Reverse(touched[l].0.len()));
        wide.iter()
            .take((m / 4 / p).max(1))
            .for_each(|&l| hub[l] = true);
        // `E` gathers each level's block before merging: bound it like the factor.
        let e_bound: usize = (touched.iter().zip(&hub))
            .map(|((comps, _), &hub)| match hub {
                true => comps.len() * p * (p + 1) / 2,
                false => comps.len() * p * (comps.len() * p + 1) / 2,
            })
            .sum();
        if e_bound > 8 * n_obs {
            if stats {
                eprintln!("reduced_rigid: E_bound={e_bound} n_obs={n_obs} declined");
            }
            return None;
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
        let inv_norm: Vec<f64> = norm2
            .iter()
            .map(|&v| if v > 0.0 { v.sqrt().recip() } else { 0.0 })
            .collect();
        stage("gauges");
        let lift = Lift {
            rows,
            anchor: &anchor,
            anchor_comp: &anchor_comp,
            gauge: &gauge,
            inv_norm: &inv_norm,
        };
        let inverse = e.inverse();
        let per_group: Vec<(GroupFit, LevelGram)> = (touched.par_iter())
            .enumerate()
            .map(|(lw, (comps, _))| lift.fit(lw, comps, inverse, hub[lw]))
            .collect();
        drop(touched);
        stage("U per group");
        // Exact nulls of S (the eliminated term absorbs the gauge) need no lift; keep the rest.
        let (mut pre_energy, mut energy) = (vec![0.0f64; m], vec![0.0f64; m]);
        for (fit, gram) in &per_group {
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
        let (n_all, m) = (m, live as usize);
        let dropped = n_all - m;
        if m == 0 {
            return None;
        }
        let n = e.n_kept();
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
                        let dof = anchor.base + q * nb + l;
                        z_entries.push((renumber[c], dof as u32, v * inv_norm[c]));
                    }
                }
            }
        }
        z_entries.par_sort_unstable_by_key(|t| (t.0, t.1));
        let z = TwoWay::from_sorted(n, m, &z_entries);
        drop(z_entries);
        // E = UᵀU from each level's block, lower triangle by column.
        let (mut fits, mut e_triplets) = (Vec::with_capacity(na), Vec::new());
        let mut corrections = LowRankParts::default();
        for (lw, (fit, gram)) in per_group.into_iter().enumerate() {
            e_triplets.extend(gram.entries.into_iter().filter_map(|(a, b, v)| {
                let (a, b) = (renumber[a as usize], renumber[b as usize]);
                (a != u32::MAX && b != u32::MAX).then_some((a, b, v))
            }));
            if let Some(s) = gram.correction {
                corrections.push(&s, &fit.columns, &renumber, &inverse[lw * p * p..][..p * p]);
            }
            fits.push(fit);
        }
        let (col_ptr, row_idx, values) = lower_csc(m, e_triplets);
        let d_max = (0..m)
            .filter_map(|c| {
                let first = col_ptr[c] as usize;
                (first < col_ptr[c + 1] as usize && row_idx[first] as usize == c)
                    .then(|| values[first])
            })
            .fold(0.0, f64::max);
        let e_nnz = values.len();
        let pattern = SymbolicSparseColMat::new_checked(m, m, col_ptr, None, row_idx);
        let e_lower = SparseColMat::new(pattern, values);
        stage("E");
        let symbolic = factorize_symbolic_cholesky(
            e_lower.symbolic(),
            Side::Lower,
            SymmetricOrdering::Amd,
            CholeskySymbolicParams::default(),
        )
        .ok()?;
        let rank = corrections.rank();
        let fits_budget = symbolic.len_val() + m * rank <= 8 * n_obs;
        if stats {
            eprintln!(
                "reduced_rigid: glued={} components={n_components} anchor_components={n_fc} m={m} hub_rank={rank} null_dropped={dropped} Z_nnz={} E_nnz={e_nnz} L_nnz={} fits={fits_budget} build_s={:.3}",
                glued.len(),
                z.nnz(),
                symbolic.len_val(),
                started.elapsed().as_secs_f64()
            );
        }
        if !fits_budget {
            return None;
        }
        // SZ = D_keptᵀ W½ U, one coarse column at a time over the levels touching its component.
        let mut comp_ptr = vec![0usize; n_fc as usize + 1];
        for fit in &fits {
            for &c in fit.columns.iter().step_by(p) {
                comp_ptr[c as usize / p + 1] += 1;
            }
        }
        for c in 0..n_fc as usize {
            comp_ptr[c + 1] += comp_ptr[c];
        }
        let mut comp_levels = vec![0u32; comp_ptr[n_fc as usize]];
        let mut cursor = comp_ptr.clone();
        for (lw, fit) in fits.iter().enumerate() {
            for &c in fit.columns.iter().step_by(p) {
                let k = &mut cursor[c as usize / p];
                comp_levels[*k] = lw as u32;
                *k += 1;
            }
        }
        drop(cursor);
        // A hub's projection is the rank-`p` `F C B` below, so `SZ₀` needs only its own-component rows.
        let hub_rows: Vec<Option<Vec<(u32, u32)>>> = (0..na)
            .map(|lw| {
                hub[lw].then(|| {
                    let span = rows.starts[lw]..rows.starts[lw + 1];
                    let mut by_comp: Vec<(u32, u32)> = span
                        .map(|r| (anchor_comp[anchor.level(rows, r)], r as u32))
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
        let live_columns: Vec<usize> = (0..n_all).filter(|&c| renumber[c] != u32::MAX).collect();
        let work = |c: usize| -> usize {
            let comp = c / p;
            (comp_levels[comp_ptr[comp]..comp_ptr[comp + 1]].iter())
                .map(|&lw| match hub[lw as usize] {
                    true => own_rows(lw as usize, comp).len(),
                    false => rows.starts[lw as usize + 1] - rows.starts[lw as usize],
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
        let sz_parts: Vec<Vec<(u32, u32, f64)>> = windows
            .par_iter()
            .map(|window| {
                let mut acc = vec![0.0f64; n];
                let mut seen = vec![u32::MAX; n];
                let mut touched: Vec<u32> = Vec::new();
                let mut out = Vec::new();
                for &c in &live_columns[window.clone()] {
                    let col = renumber[c];
                    let comp = c / p;
                    let mut visit = |r: usize, u: f64| {
                        if u == 0.0 {
                            return;
                        }
                        let val = lift.root(r) * u;
                        rows.kept.for_each_entry(r, |d, z| {
                            if seen[d as usize] != col {
                                seen[d as usize] = col;
                                touched.push(d);
                            }
                            acc[d as usize] += z * val;
                        });
                    };
                    for &lw in &comp_levels[comp_ptr[comp]..comp_ptr[comp + 1]] {
                        let lw = lw as usize;
                        if hub[lw] {
                            for &(_, r) in own_rows(lw, comp) {
                                visit(r as usize, lift.lifted(r as usize, c));
                            }
                            continue;
                        }
                        let fit = &fits[lw];
                        let i = fit
                            .columns
                            .binary_search(&(c as u32))
                            .expect("touched column");
                        for r in rows.starts[lw]..rows.starts[lw + 1] {
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
        let sz = TwoWay::from_parts(n, m, sz_parts);
        let mut f_entries = Vec::new();
        for (h, lw) in (0..na).filter(|&lw| hub[lw]).enumerate() {
            let mut acc = vec![0.0; n * p];
            for r in rows.starts[lw]..rows.starts[lw + 1] {
                let w = rows.weight(r);
                rows.kept.for_each_entry(r, |d, z| {
                    for a in 0..p {
                        acc[d as usize * p + a] += w * z * rows.eliminated.loading(r, a);
                    }
                });
            }
            for a in 0..p {
                (0..n)
                    .filter(|&d| acc[d * p + a] != 0.0)
                    .for_each(|d| f_entries.push(((h * p + a) as u32, d as u32, acc[d * p + a])));
            }
        }
        let f = TwoWay::from_sorted(n, rank, &f_entries);
        drop(f_entries);
        if stats {
            eprintln!(
                "reduced_rigid: SZ_nnz={} sz_s={:.3} n={n} Z_rows={} SZ_rows={}",
                sz.nnz(),
                started.elapsed().as_secs_f64(),
                z.rows.len(),
                sz.rows.len()
            );
        }
        // Null pivots become LARGE, as in the fused blocks: tiny ones amplify nulls.
        let signs = vec![1i8; m];
        let regularization = LdltRegularization {
            dynamic_regularization_signs: Some(&signs),
            dynamic_regularization_delta: d_max,
            dynamic_regularization_epsilon: 1e-12 * d_max,
        };
        // `WITHIN_REDUCED_COARSE_PAR` runs the factorization and solves on the rayon pool.
        let par = match std::env::var_os("WITHIN_REDUCED_COARSE_PAR") {
            Some(_) => Par::rayon(0),
            None => Par::Seq,
        };
        let mut l_values = vec![0.0; symbolic.len_val()];
        let mut mem =
            MemBuffer::new(symbolic.factorize_numeric_ldlt_scratch::<f64>(par, Default::default()));
        symbolic
            .factorize_numeric_ldlt(
                &mut l_values,
                e_lower.as_ref(),
                Side::Lower,
                regularization,
                par,
                MemStack::new(&mut mem),
                Default::default(),
            )
            .ok()?;
        stage("factored");
        if !l_values.iter().all(|v| v.is_finite()) {
            return None;
        }
        let mut coarse = Self {
            m,
            z,
            sz,
            symbolic,
            l_values,
            par,
            low_rank: None,
        };
        if rank > 0 {
            coarse.low_rank = Some(corrections.factor(f, |v| coarse.solve_sparse(v), m)?);
            stage("hub correction");
        }
        Some(coarse)
    }

    /// `out += alpha SZ c`.
    fn sz_mul_add(&self, c: &[f64], alpha: f64, out: &mut [f64]) {
        self.sz.mul_add(c, alpha, out);
        if let Some(low_rank) = &self.low_rank {
            low_rank.add_sz(c, -alpha, out);
        }
    }

    /// `out = SZᵀ y`.
    fn sz_t_mul(&self, y: &[f64], out: &mut [f64]) {
        self.sz.t_mul(y, out);
        if let Some(low_rank) = &self.low_rank {
            low_rank.add_sz_t(y, -1.0, out);
        }
    }

    fn solve_coarse(&self, v: &mut [f64]) {
        self.solve_sparse(v);
        if let Some(low_rank) = &self.low_rank {
            low_rank.correct(v);
        }
    }

    fn solve_sparse(&self, v: &mut [f64]) {
        let mut buf = MemBuffer::new(self.symbolic.solve_in_place_scratch::<f64>(1, self.par));
        LdltRef::new(&self.symbolic, &self.l_values).solve_in_place_with_conj(
            Conj::No,
            MatMut::from_column_major_slice_mut(v, self.m, 1),
            self.par,
            MemStack::new(&mut buf),
        );
    }
}

impl CoarseRung for ReducedCoarse {
    /// Balancing form around `inner`: `Z E⁻¹ Zᵀx + (I − Z E⁻¹ ZᵀS) B (I − S Z E⁻¹ Zᵀ) x`.
    fn apply_around(
        &self,
        inner: &dyn Operator,
        x: &[f64],
        y: &mut [f64],
    ) -> Result<(), SolveError> {
        let mut c = vec![0.0; self.m];
        self.z.t_mul(x, &mut c);
        self.solve_coarse(&mut c);
        let mut t = x.to_vec();
        self.sz_mul_add(&c, -1.0, &mut t);
        inner.apply(&t, y)?;
        let mut d = vec![0.0; self.m];
        self.sz_t_mul(y, &mut d);
        self.solve_coarse(&mut d);
        for (ci, di) in c.iter_mut().zip(&d) {
            *ci -= di;
        }
        self.z.mul_add(&c, 1.0, y);
        Ok(())
    }
}

#[cfg(test)]
mod hub {
    use super::*;
    use crate::domain::PreparedDesign;
    use crate::{Design, Effect, IntoDesign};

    /// `T` clusters of 4 firms × 20 workers glue internally; one hub worker spreads `R` rows over
    /// every cluster with two slopes `1e-4` apart per spell: full rank, too ill-conditioned to glue.
    #[test]
    #[ignore]
    fn hub_level_dense_buffer() {
        let var = |k: &str, d: usize| std::env::var(k).map_or(d, |v| v.parse().unwrap());
        let (t, r) = (var("HUB_T", 200), var("HUB_R", 10_000));
        let (mut worker, mut firm, mut x, mut z) = (vec![], vec![], vec![], vec![]);
        for k in 0..t {
            for w in 0..20 {
                for i in 0..5 {
                    worker.push((20 * k + w) as u32);
                    firm.push((4 * k + (w + (i >= 3) as usize) % 4) as u32);
                    x.push(i as f64);
                    z.push(((k * 131 + w * 17 + i * 7) % 23) as f64);
                }
            }
        }
        let hub = (20 * t) as u32;
        for j in 0..r {
            worker.push(hub);
            firm.push((4 * (j % t)) as u32);
            x.push(2.0 + (j % t) as f64 * 1e-3 + ((j / t) % 2) as f64 * 1e-4);
            z.push(((j * 13) % 29) as f64);
        }
        let n_obs = worker.len();
        let effects = vec![
            Effect::new(&worker, true, [&x[..]]).unwrap(),
            Effect::new(&firm, true, [&z[..]]).unwrap(),
        ];
        let design = Design::new_unsorted(effects).unwrap();
        let prepared = PreparedDesign::new(design.into_design().unwrap(), None).unwrap();
        let e = Elimination::new(&prepared);
        println!(
            "T={t} R={r} n_obs={n_obs} hub buffer R×2T = {:.1} MB, E bound 2T(2T+1)/2 = {} vs 8·n_obs = {}",
            (r * 2 * t * 8) as f64 / 1e6,
            2 * t * (2 * t + 1) / 2,
            8 * n_obs
        );
        if var("HUB_BUILD", 1) == 1 {
            let start = std::time::Instant::now();
            let coarse = ReducedCoarse::build(&e, 150.0);
            println!(
                "built={} m={} build_s={:.3}",
                coarse.is_some(),
                coarse.as_ref().map_or(0, |c| c.m),
                start.elapsed().as_secs_f64()
            );
            if let Some(c) = coarse {
                // `ZᵀSZ = E` from the separately built SZ: for `x = E z`, `E E⁻¹x` returns `x`.
                let e_mul = |v: &[f64]| {
                    let mut sv = vec![0.0; e.n_kept()];
                    c.sz_mul_add(v, 1.0, &mut sv);
                    let mut out = vec![0.0; c.m];
                    c.z.t_mul(&sv, &mut out);
                    out
                };
                let z: Vec<f64> = (0..c.m).map(|i| (i as f64 * 0.7).sin()).collect();
                let x = e_mul(&z);
                let mut y = x.clone();
                c.solve_coarse(&mut y);
                let norm = |v: &[f64]| v.iter().map(|a| a * a).sum::<f64>().sqrt();
                let r: Vec<f64> = e_mul(&y).iter().zip(&x).map(|(a, b)| a - b).collect();
                let dz: Vec<f64> = y.iter().zip(&z).map(|(a, b)| a - b).collect();
                println!(
                    "‖E E⁻¹x − x‖/‖x‖ = {:.1e}  ‖E⁻¹x − z‖/‖z‖ = {:.1e}",
                    norm(&r) / norm(&x),
                    norm(&dz) / norm(&z)
                );
                let w: Vec<f64> = (0..e.n_kept()).map(|i| (i as f64 * 0.3).cos()).collect();
                let (mut sz, mut szt) = (vec![0.0; e.n_kept()], vec![0.0; c.m]);
                c.sz_mul_add(&z, 1.0, &mut sz);
                c.sz_t_mul(&w, &mut szt);
                let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
                let (lhs, rhs) = (dot(&sz, &w), dot(&z, &szt));
                println!(
                    "adjoint |⟨SZz,w⟩ − ⟨z,SZᵀw⟩| / |⟨SZz,w⟩| = {:.1e}",
                    (lhs - rhs).abs() / lhs.abs()
                );
            }
        }
    }
}
