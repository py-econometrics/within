//! Rigid-component coarse space on the reduced system (prototype, `WITHIN_REDUCED_RIGID=K`).
//!
//! Spells (one eliminated level × one anchor level) whose Gram is within `K` of both levels' own
//! Grams glue the two levels; each glued component's anchor-side gauges span one coarse column per
//! channel. With the eliminated term solved exactly, `Ã Z` is the energy-minimal extension.

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::ldlt::factor::LdltRegularization;
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

/// Entries from which rayon beats one thread (`two_way_crossover`): the gather `Mᵀx` pays later
/// than the row-blocked `M c`.
const PARALLEL_T_MUL: usize = 1 << 18;
const PARALLEL_MUL_ADD: usize = 1 << 16;

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

    /// Probe: `out = Aᵀ x` for `k` interleaved columns.
    fn t_mul_k(&self, x: &[f64], out: &mut [f64], k: usize) {
        macro_rules! go { ($($k:literal)*) => { match k {
            $($k => self.t_mul_r::<$k>(x.as_chunks().0, out.as_chunks_mut().0),)*
            _ => panic!("k = {k}"),
        } } }
        go!(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)
    }

    fn t_mul_r<const K: usize>(&self, x: &[[f64; K]], out: &mut [[f64; K]]) {
        out.par_iter_mut().enumerate().with_min_len(64).for_each(|(c, o)| {
            let mut acc = [0.0; K];
            let span = self.col_ptr[c]..self.col_ptr[c + 1];
            for (&r, &v) in self.col_idx[span.clone()].iter().zip(&self.col_val[span]) {
                let xr = &x[r as usize];
                (0..K).for_each(|l| acc[l] += v * xr[l]);
            }
            *o = acc;
        });
    }

    /// Probe: `out += alpha A c` for `k` interleaved columns.
    fn mul_add_k(&self, c: &[f64], alpha: f64, out: &mut [f64], k: usize) {
        macro_rules! go { ($($k:literal)*) => { match k {
            $($k => self.mul_add_r::<$k>(c.as_chunks().0, alpha, out.as_chunks_mut().0),)*
            _ => panic!("k = {k}"),
        } } }
        go!(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)
    }

    fn mul_add_r<const K: usize>(&self, c: &[[f64; K]], alpha: f64, out: &mut [[f64; K]]) {
        const BLOCK: usize = 512;
        out.par_chunks_mut(BLOCK).enumerate().for_each(|(j, block)| {
            let base = j * BLOCK;
            let lo = self.rows.partition_point(|&r| (r as usize) < base);
            let hi = self.rows.partition_point(|&r| (r as usize) < base + block.len());
            for i in lo..hi {
                let mut acc = [0.0; K];
                let span = self.row_ptr[i]..self.row_ptr[i + 1];
                for (&kk, &v) in self.row_idx[span.clone()].iter().zip(&self.row_val[span]) {
                    let cr = &c[kk as usize];
                    (0..K).for_each(|l| acc[l] += v * cr[l]);
                }
                let o = &mut block[self.rows[i] as usize - base];
                (0..K).for_each(|l| o[l] += alpha * acc[l]);
            }
        });
    }

    fn t_mul(&self, x: &[f64], out: &mut [f64]) {
        self.t_mul_on(x, out, self.nnz() >= PARALLEL_T_MUL);
    }

    fn t_mul_on(&self, x: &[f64], out: &mut [f64], parallel: bool) {
        let column = |(c, o): (usize, &mut f64)| {
            let span = self.col_ptr[c]..self.col_ptr[c + 1];
            *o = (self.col_idx[span.clone()].iter().zip(&self.col_val[span]))
                .map(|(&r, &v)| v * x[r as usize])
                .sum();
        };
        match parallel {
            false => out.iter_mut().enumerate().for_each(column),
            true => out.par_iter_mut().enumerate().for_each(column),
        }
    }

    fn mul_add(&self, c: &[f64], alpha: f64, out: &mut [f64]) {
        self.mul_add_on(c, alpha, out, self.nnz() >= PARALLEL_MUL_ADD);
    }

    fn mul_add_on(&self, c: &[f64], alpha: f64, out: &mut [f64], parallel: bool) {
        const BLOCK: usize = 4096;
        let row = |i: usize| -> f64 {
            let span = self.row_ptr[i]..self.row_ptr[i + 1];
            (self.row_idx[span.clone()].iter().zip(&self.row_val[span]))
                .map(|(&k, &v)| v * c[k as usize])
                .sum()
        };
        if !parallel {
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

const PENCIL_EDGES: [f64; 12] = [
    2.0, 4.0, 8.0, 12.0, 14.0, 16.0, 18.0, 20.0, 30.0, 50.0, 150.0, 1e3,
];

/// Spells whose Gram is within `k` of both levels' own Grams, as `(eliminated, anchor)` levels.
fn glued_spells(
    rows: &GroupedRows,
    anchor: &Anchor<'_>,
    anchor_gram: &[f64],
    k: f64,
    histogram: Option<&[std::sync::atomic::AtomicUsize]>,
) -> Vec<(u32, u32)> {
    let bin = |v: f64| PENCIL_EDGES.iter().take_while(|&&e| v > e).count();
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
                if let Some(hist) = histogram {
                    hist[bin(worst)].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
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
    /// A hub's `s = EᵀW DZ` (`width × columns`) and level Gram `G`: rows `[G s]` augmenting `E₀`.
    correction: Option<(Vec<f64>, Vec<f64>)>,
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
            correction: hub.then(|| {
                let mut g = vec![0.0; p * p];
                for r in rows.starts[lw]..rows.starts[lw + 1] {
                    for a in 0..p {
                        let wa = rows.weight(r) * rows.eliminated.loading(r, a);
                        for b in 0..p {
                            g[a * p + b] += wa * rows.eliminated.loading(r, b);
                        }
                    }
                }
                (s, g)
            }),
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
/// Why no coarse rung was built; a larger gluing ratio can only cure `OverBudget`.
pub(crate) enum Declined {
    NoCoarseSpace,
    /// With the number of spells glued, which a larger ratio must raise to help.
    OverBudget {
        glued: usize,
    },
}

pub(crate) struct ReducedCoarse {
    m: usize,
    z: TwoWay,
    sz: TwoWay,
    symbolic: SymbolicCholesky<u32>,
    l_values: Vec<f64>,
    par: Par,
    /// Wide levels' rows: the factor is of `[G s; sᵀ E₀]`, whose Schur complement is `E`.
    hubs: Option<HubCorrection>,
    /// The factor is of `D E D` with `D` this diagonal.
    scale: Vec<f64>,
}

/// Wide levels' `s = EᵀW DZ` rows over the live columns, their `G` and `G⁻`, in level order.
#[derive(Default)]
struct HubRows {
    ptr: Vec<usize>,
    cols: Vec<u32>,
    vals: Vec<f64>,
    c: Vec<f64>,
    p: usize,
}

impl HubRows {
    /// Appends a wide level's rows `m + rank..`: `G`'s lower triangle and `s` against the columns.
    #[allow(clippy::too_many_arguments)]
    fn push(
        &mut self,
        s: &[f64],
        g: &[f64],
        inverse: &[f64],
        columns: &[u32],
        renumber: &[u32],
        m: usize,
        lower: &mut Vec<(u32, u32, f64)>,
    ) {
        let (t, p) = (columns.len(), g.len().isqrt());
        if self.ptr.is_empty() {
            self.ptr.push(0);
        }
        self.p = p;
        let base = m + self.rank();
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
        self.c.extend_from_slice(inverse);
    }

    fn rank(&self) -> usize {
        self.ptr.len().saturating_sub(1)
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

/// `SZ = SZ₀ − F C B` for the wide levels: `B` their `s` rows, `C` their `G⁻`, `F = D_keptᵀ W E`.
struct HubCorrection {
    ptr: Vec<usize>,
    cols: Vec<u32>,
    vals: Vec<f64>,
    c: Vec<f64>,
    p: usize,
    f: TwoWay,
}

impl HubCorrection {
    fn rank(&self) -> usize {
        self.ptr.len() - 1
    }

    fn b_mul(&self, v: &[f64]) -> Vec<f64> {
        (0..self.rank())
            .map(|i| {
                (self.ptr[i]..self.ptr[i + 1])
                    .map(|k| self.vals[k] * v[self.cols[k] as usize])
                    .sum()
            })
            .collect()
    }

    /// `C q`, each wide level's `G⁻` on its `p` entries.
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

    /// `out += alpha (F C B) c`.
    fn add_sz(&self, c: &[f64], alpha: f64, out: &mut [f64]) {
        self.f.mul_add(&self.c_mul(&self.b_mul(c)), alpha, out);
    }

    /// Probe: `C q` on interleaved rows.
    fn c_mul_r<const K: usize>(&self, q: &[[f64; K]]) -> Vec<[f64; K]> {
        let p = self.p;
        (0..q.len())
            .map(|i| {
                let (h, a) = (i / p, i % p);
                let mut acc = [0.0; K];
                for b in 0..p {
                    let (cv, qr) = (self.c[h * p * p + a * p + b], &q[h * p + b]);
                    (0..K).for_each(|l| acc[l] += cv * qr[l]);
                }
                acc
            })
            .collect()
    }

    /// Probe: [`Self::add_sz`] on interleaved rows.
    fn add_sz_r<const K: usize>(&self, c: &[[f64; K]], alpha: f64, out: &mut [[f64; K]]) {
        let bc: Vec<[f64; K]> = (0..self.rank())
            .map(|i| {
                let mut acc = [0.0; K];
                for k in self.ptr[i]..self.ptr[i + 1] {
                    let (v, cr) = (self.vals[k], &c[self.cols[k] as usize]);
                    (0..K).for_each(|l| acc[l] += v * cr[l]);
                }
                acc
            })
            .collect();
        self.f.mul_add_r(&self.c_mul_r(&bc), alpha, out);
    }

    /// Probe: [`Self::add_sz_t`] on interleaved rows.
    fn add_sz_t_r<const K: usize>(&self, y: &[[f64; K]], alpha: f64, out: &mut [[f64; K]]) {
        let mut fy = vec![[0.0; K]; self.rank()];
        self.f.t_mul_r(y, &mut fy);
        for (i, q) in self.c_mul_r(&fy).into_iter().enumerate() {
            for k in self.ptr[i]..self.ptr[i + 1] {
                let (o, v) = (&mut out[self.cols[k] as usize], alpha * self.vals[k]);
                (0..K).for_each(|l| o[l] += v * q[l]);
            }
        }
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
}

impl ReducedCoarse {
    /// Declines without a kept term sharing the eliminated term's column layout, when the
    /// components cover more than half the levels, or when its factor exceeds `n_obs` values.
    pub(crate) fn build(e: &Elimination<'_>, k: f64) -> Result<Self, Declined> {
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
        let anchor = Anchor::find(e).ok_or(Declined::NoCoarseSpace)?;
        let rows = &**e.rows();
        let (na, nb) = (rows.n_levels(), anchor.n_levels());
        let (anchor_gram, gauge) = anchor.gauges(e.prepared().sqrt_weights(), p);
        let histogram: Vec<std::sync::atomic::AtomicUsize> = (0..=PENCIL_EDGES.len())
            .map(|_| Default::default())
            .collect();
        let glued = glued_spells(
            rows,
            &anchor,
            &anchor_gram,
            k,
            stats.then_some(&histogram[..]),
        );
        if stats {
            let counts: Vec<String> = (histogram.iter().enumerate())
                .map(|(i, c)| match PENCIL_EDGES.get(i) {
                    Some(e) => format!("<={e}:{}", c.load(std::sync::atomic::Ordering::Relaxed)),
                    None => format!(">1e3:{}", c.load(std::sync::atomic::Ordering::Relaxed)),
                })
                .collect();
            eprintln!("reduced_rigid: pencil {}", counts.join(" "));
        }
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
                    "reduced_rigid: k={k} glued={} components={n_components} levels={} declined",
                    glued.len(),
                    na + nb
                );
            }
            return Err(Declined::NoCoarseSpace);
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
            return Err(Declined::NoCoarseSpace);
        }
        if let Some(path) = std::env::var_os("WITHIN_REDUCED_RIGID_COMPONENTS") {
            let bytes: Vec<u8> = anchor_comp.iter().flat_map(|c| c.to_le_bytes()).collect();
            std::fs::write(path, bytes).expect("component dump");
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
            if stats {
                eprintln!("reduced_rigid: E_bound={e_bound} n_obs={n_obs} declined");
            }
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
            return Err(Declined::NoCoarseSpace);
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
        let mut hubs = HubRows::default();
        for (lw, (fit, gram)) in per_group.into_iter().enumerate() {
            e_triplets.extend(gram.entries.into_iter().filter_map(|(a, b, v)| {
                let (a, b) = (renumber[a as usize], renumber[b as usize]);
                (a != u32::MAX && b != u32::MAX).then_some((a, b, v))
            }));
            if let Some((s, g)) = gram.correction {
                let inverse = &inverse[lw * p * p..][..p * p];
                hubs.push(&s, &g, inverse, &fit.columns, &renumber, m, &mut e_triplets);
            }
            fits.push(fit);
        }
        let rank = hubs.rank();
        let dim = m + rank;
        let (col_ptr, row_idx, mut values) = lower_csc(dim, e_triplets);
        if let Some(path) = std::env::var_os("WITHIN_REDUCED_RIGID_DUMP_E") {
            let mut out: Vec<u8> = (dim as u64).to_le_bytes().to_vec();
            for c in 0..dim {
                for k in col_ptr[c] as usize..col_ptr[c + 1] as usize {
                    out.extend((c as u64).to_le_bytes());
                    out.extend((row_idx[k] as u64).to_le_bytes());
                    out.extend(values[k].to_le_bytes());
                }
            }
            std::fs::write(path, out).expect("E dump");
        }
        // Unit diagonal: null pivots then sit at rounding level, far below any live mode.
        let mut scale = vec![1.0; dim];
        for c in 0..dim {
            let first = col_ptr[c] as usize;
            if first < col_ptr[c + 1] as usize
                && row_idx[first] as usize == c
                && values[first] > 0.0
            {
                scale[c] = values[first].sqrt().recip();
            }
        }
        for c in 0..dim {
            for k in col_ptr[c] as usize..col_ptr[c + 1] as usize {
                values[k] *= scale[c] * scale[row_idx[k] as usize];
            }
        }
        let d_max = (0..dim)
            .filter_map(|c| {
                let first = col_ptr[c] as usize;
                (first < col_ptr[c + 1] as usize && row_idx[first] as usize == c)
                    .then(|| values[first])
            })
            .fold(0.0, f64::max);
        let e_nnz = values.len();
        let pattern = SymbolicSparseColMat::new_checked(dim, dim, col_ptr, None, row_idx);
        let e_lower = SparseColMat::new(pattern, values);
        stage("E");
        let symbolic = factorize_symbolic_cholesky(
            e_lower.symbolic(),
            Side::Lower,
            SymmetricOrdering::Amd,
            CholeskySymbolicParams::default(),
        )
        .map_err(|_| Declined::NoCoarseSpace)?;
        let fits_budget = symbolic.len_val() <= n_obs;
        if stats {
            eprintln!(
                "reduced_rigid: k={k} e_bound={e_bound} glued={} components={n_components} anchor_components={n_fc} m={m} hub_rank={rank} null_dropped={dropped} Z_nnz={} E_nnz={e_nnz} L_nnz={} fits={fits_budget} build_s={:.3}",
                glued.len(),
                z.nnz(),
                symbolic.len_val(),
                started.elapsed().as_secs_f64()
            );
        }
        if !fits_budget {
            return Err(Declined::OverBudget { glued: glued.len() });
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
        let signs = vec![1i8; dim];
        let regularization = LdltRegularization {
            dynamic_regularization_signs: Some(&signs),
            dynamic_regularization_delta: d_max,
            dynamic_regularization_epsilon: 1e-10 * d_max,
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
            .map_err(|_| Declined::NoCoarseSpace)?;
        stage("factored");
        if !l_values.iter().all(|v| v.is_finite()) {
            return Err(Declined::NoCoarseSpace);
        }
        Ok(Self {
            m,
            z,
            sz,
            symbolic,
            l_values,
            par,
            hubs: (rank > 0).then(|| hubs.into_correction(f)),
            scale,
        })
    }

    /// `out += alpha SZ c`.
    fn sz_mul_add(&self, c: &[f64], alpha: f64, out: &mut [f64]) {
        self.sz.mul_add(c, alpha, out);
        if let Some(hubs) = &self.hubs {
            hubs.add_sz(c, -alpha, out);
        }
    }

    /// `out = SZᵀ y`.
    fn sz_t_mul(&self, y: &[f64], out: &mut [f64]) {
        self.sz.t_mul(y, out);
        if let Some(hubs) = &self.hubs {
            hubs.add_sz_t(y, -1.0, out);
        }
    }

    /// Probe: `E⁻¹` on `k` interleaved columns.
    fn solve_coarse_k(&self, v: &mut [f64], k: usize) {
        let dim = self.m + self.hubs.as_ref().map_or(0, HubCorrection::rank);
        let mut x = vec![0.0; dim * k];
        for i in 0..self.m {
            for j in 0..k {
                x[j * dim + i] = v[i * k + j] * self.scale[i];
            }
        }
        if dim > self.m {
            for j in 0..k {
                for i in self.m..dim {
                    x[j * dim + i] *= self.scale[i];
                }
            }
        }
        let mode = std::env::var("PROBE_ESOLVE").unwrap_or_else(|_| "split1".into());
        let solve = |cols: &mut [f64], par: Par| {
            let kk = cols.len() / dim;
            let mut buf = MemBuffer::new(self.symbolic.solve_in_place_scratch::<f64>(kk, par));
            LdltRef::new(&self.symbolic, &self.l_values).solve_in_place_with_conj(
                Conj::No,
                MatMut::from_column_major_slice_mut(cols, dim, kk),
                par,
                MemStack::new(&mut buf),
            );
        };
        match mode.as_str() {
            "par" => solve(&mut x, Par::rayon(rayon::current_num_threads())),
            m if m.starts_with("split") => {
                let g: usize = m[5..].parse().unwrap_or(1);
                x.par_chunks_mut(dim * g).for_each(|cols| solve(cols, Par::Seq));
            }
            _ => solve(&mut x, self.par),
        }
        for i in 0..self.m {
            for j in 0..k {
                v[i * k + j] = x[j * dim + i] * self.scale[i];
            }
        }
    }

    /// `E⁻¹v`: the wide levels' rows ride along with zero right-hand side.
    fn solve_coarse(&self, v: &mut [f64]) {
        let dim = self.m + self.hubs.as_ref().map_or(0, HubCorrection::rank);
        let mut x = v.to_vec();
        x.resize(dim, 0.0);
        x.iter_mut().zip(&self.scale).for_each(|(x, s)| *x *= s);
        let mut buf = MemBuffer::new(self.symbolic.solve_in_place_scratch::<f64>(1, self.par));
        LdltRef::new(&self.symbolic, &self.l_values).solve_in_place_with_conj(
            Conj::No,
            MatMut::from_column_major_slice_mut(&mut x, dim, 1),
            self.par,
            MemStack::new(&mut buf),
        );
        x.iter_mut().zip(&self.scale).for_each(|(x, s)| *x *= s);
        v.copy_from_slice(&x[..self.m]);
    }
}

static SZ_BENCHED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
pub(crate) static CTIME: [std::sync::atomic::AtomicU64; 5] = [const { std::sync::atomic::AtomicU64::new(0) }; 5];

impl ReducedCoarse {
    fn around_rows<const K: usize>(&self, inner: &(dyn Fn(&[f64], &mut [f64]) + Sync), x: &[f64], y: &mut [f64]) {
        use std::sync::atomic::Ordering::Relaxed;
        thread_local! {
            static T: std::cell::Cell<Vec<f64>> = const { std::cell::Cell::new(Vec::new()) };
        }
        let mut tick = std::time::Instant::now();
        let mut lap = |i: usize| {
            CTIME[i].fetch_add(tick.elapsed().as_nanos() as u64, Relaxed);
            tick = std::time::Instant::now();
        };
        let rows = |v: &[f64]| -> Vec<[f64; K]> { v.as_chunks::<K>().0.to_vec() };
        let mut c = vec![[0.0; K]; self.m];
        self.z.t_mul_r(x.as_chunks().0, &mut c);
        lap(0);
        self.solve_coarse_k(c.as_flattened_mut(), K);
        lap(1);
        let mut t = T.take();
        t.clear();
        t.extend_from_slice(x);
        let tr = t.as_chunks_mut::<K>().0;
        self.sz.mul_add_r(&c, -1.0, tr);
        lap(2);
        if let Some(h) = &self.hubs {
            h.add_sz_r(&c, 1.0, tr);
        }
        lap(3);
        inner(&t, y);
        T.set(t);
        lap(4);
        let yr = y.as_chunks::<K>().0;
        let mut d = vec![[0.0; K]; self.m];
        self.sz.t_mul_r(yr, &mut d);
        lap(2);
        if let Some(h) = &self.hubs {
            h.add_sz_t_r(yr, -1.0, &mut d);
        }
        lap(3);
        self.solve_coarse_k(d.as_flattened_mut(), K);
        lap(1);
        c.iter_mut().zip(&d).for_each(|(c, d)| (0..K).for_each(|l| c[l] -= d[l]));
        self.z.mul_add_r(&c, 1.0, y.as_chunks_mut::<K>().0);
        lap(0);
        let _ = rows;
    }
}

impl CoarseRung for ReducedCoarse {
    fn apply_around_block(
        &self,
        inner: &(dyn Fn(&[f64], &mut [f64]) + Sync),
        x: &[f64],
        y: &mut [f64],
        k: usize,
    ) {
        if std::env::var_os("PROBE_SZ_BENCH").is_some() && !SZ_BENCHED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            let n = x.len() / k;
            let best = |f: &mut dyn FnMut()| (0..5).map(|_| { let t = std::time::Instant::now(); f(); t.elapsed().as_secs_f64() }).fold(f64::INFINITY, f64::min) * 1e3;
            let cols: Vec<Vec<f64>> = (0..k).map(|j| (0..n).map(|i| x[i * k + j]).collect()).collect();
            let mut cm = vec![vec![0.0; self.m]; k];
            let mut bm = vec![0.0; self.m * k];
            let t_conc = best(&mut || cols.par_iter().zip(&mut cm).for_each(|(x, o)| self.sz.t_mul(x, o)));
            let t_seqpar = best(&mut || cols.iter().zip(&mut cm).for_each(|(x, o)| self.sz.t_mul_on(x, o, true)));
            let t_blk = best(&mut || self.sz.t_mul_k(x, &mut bm, k));
            let mut on = vec![vec![0.0; n]; k];
            let mut bn = vec![0.0; n * k];
            let m_conc = best(&mut || cm.par_iter().zip(&mut on).for_each(|(c, o)| self.sz.mul_add(c, 1.0, o)));
            let m_blk = best(&mut || self.sz.mul_add_k(&bm, 1.0, &mut bn, k));
            eprintln!("SZ nnz={} m={} n={n} k={k}: t_mul conc {t_conc:.2} seq-of-par {t_seqpar:.2} block {t_blk:.2} ms | mul_add conc {m_conc:.2} block {m_blk:.2} ms", self.sz.nnz(), self.m);
        }
        macro_rules! go { ($($k:literal)*) => { match k {
            $($k => self.around_rows::<$k>(inner, x, y),)*
            _ => panic!("k = {k}"),
        } } }
        go!(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)
    }
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
mod threshold {
    use super::*;
    use std::hint::black_box;
    use std::time::Instant;

    /// Sequential vs rayon products over `nnz`: columns of ~16 rows each, near their own index.
    #[test]
    #[ignore]
    fn two_way_crossover() {
        println!("threads={}", rayon::current_num_threads());
        for log in 13..=23 {
            let nnz = 1usize << log;
            let (m, n) = (nnz / 16, nnz / 4);
            let mut entries = Vec::with_capacity(nnz);
            for c in 0..m {
                let mut rows: Vec<u32> = (0..16)
                    .map(|k| ((c * 4 + (c * 7 + k * 131) % 64) % n) as u32)
                    .collect();
                rows.sort_unstable();
                rows.dedup();
                entries.extend(
                    rows.into_iter()
                        .map(|r| (c as u32, r, 1.0 + (r % 7) as f64)),
                );
            }
            let a = TwoWay::from_sorted(n, m, &entries);
            let (x, cv) = (vec![0.5; n], vec![0.25; m]);
            let (mut om, mut on) = (vec![0.0; m], vec![0.0; n]);
            let mut best = [f64::INFINITY; 4];
            let reps = (1 << 24) / nnz + 20;
            for _ in 0..reps {
                for (k, b) in best.iter_mut().enumerate() {
                    let start = Instant::now();
                    match k {
                        0 => a.t_mul_on(black_box(&x), &mut om, false),
                        1 => a.t_mul_on(black_box(&x), &mut om, true),
                        2 => a.mul_add_on(black_box(&cv), 1.0, &mut on, false),
                        _ => a.mul_add_on(black_box(&cv), 1.0, &mut on, true),
                    }
                    *b = b.min(start.elapsed().as_secs_f64() * 1e6);
                }
            }
            println!(
                "nnz=2^{log:<2} ({:>8}) t_mul seq {:>8.1} par {:>8.1} µs | mul_add seq {:>8.1} par {:>8.1} µs",
                a.nnz(), best[0], best[1], best[2], best[3]
            );
        }
    }
}

#[cfg(test)]
mod hub {
    use super::*;
    use crate::domain::PreparedDesign;
    use crate::{
        Design, Effect, IntoDesign, LocalSolverConfig, LsmrOptions, PreconditionerConfig, Solver,
    };

    /// `clusters` blocks of 4 firms × 20 workers glue internally; each of `hubs` workers spreads
    /// `rows` over `reach` clusters with slopes `1e-4` apart per spell: full rank, too ill-conditioned
    /// to glue. `flat` drops that spread (singular per component); `shared` gives firms the workers' slopes.
    #[derive(Clone, Copy, Debug)]
    struct HubPanel {
        clusters: usize,
        rows: usize,
        hubs: usize,
        reach: usize,
        slopes: usize,
        weighted: bool,
        flat: bool,
        shared: bool,
    }

    struct Data {
        worker: Vec<u32>,
        firm: Vec<u32>,
        x: Vec<Vec<f64>>,
        z: Vec<Vec<f64>>,
        weights: Option<Vec<f64>>,
        y: Vec<f64>,
    }

    impl HubPanel {
        fn base() -> Self {
            Self {
                clusters: 200,
                rows: 10_000,
                hubs: 1,
                reach: 200,
                slopes: 1,
                weighted: false,
                flat: false,
                shared: false,
            }
        }

        fn data(&self) -> Data {
            let s = self.slopes;
            let per_worker = 2 * (s + 1);
            let (mut worker, mut firm, mut w) = (vec![], vec![], vec![]);
            let (mut x, mut z) = (vec![vec![]; s], vec![vec![]; s]);
            for k in 0..self.clusters {
                for v in 0..20 {
                    for i in 0..per_worker {
                        worker.push((20 * k + v) as u32);
                        firm.push((4 * k + (v + i % 2) % 4) as u32);
                        for q in 0..s {
                            x[q].push((i as f64).powi(q as i32 + 1) / (q as f64 + 1.0));
                            z[q].push(((k * 131 + v * 17 + i * 7 + q * 5) % 23) as f64);
                        }
                        w.push(1.0 + (i % 3) as f64 * 0.5);
                    }
                }
            }
            let stride = (self.clusters / self.hubs).max(1);
            for h in 0..self.hubs {
                for j in 0..self.rows {
                    let cluster = (h * stride + j % self.reach) % self.clusters;
                    worker.push((20 * self.clusters + h) as u32);
                    firm.push((4 * cluster) as u32);
                    let alt = (j / self.reach) % (s + 1);
                    for q in 0..s {
                        let spread = if self.flat {
                            0.0
                        } else {
                            (alt == q + 1) as u8 as f64 * 1e-4
                        };
                        x[q].push(2.0 + q as f64 + cluster as f64 * 1e-3 + spread);
                        z[q].push(((j * 13 + q * 3) % 29) as f64);
                    }
                    w.push(if j % 7 == 0 {
                        0.0
                    } else {
                        0.5 + (j % 3) as f64
                    });
                }
            }
            if self.shared {
                z = x.clone();
            }
            let hash = |i: usize| ((i as f64 * 12.9898).sin() * 43758.5453).fract();
            let y = (0..worker.len())
                .map(|i| {
                    (worker[i] as f64 * 0.37).sin()
                        + (firm[i] as f64 * 0.11).cos()
                        + x[0][i] * (worker[i] as f64 * 0.7).sin()
                        + z[0][i] * (firm[i] as f64 * 0.3).cos()
                        + 0.1 * hash(i)
                })
                .collect();
            Data {
                worker,
                firm,
                x,
                z,
                weights: self.weighted.then_some(w),
                y,
            }
        }
    }

    impl Data {
        fn effects(&self) -> Vec<Effect<'_>> {
            let x: Vec<&[f64]> = self.x.iter().map(|v| &v[..]).collect();
            let z: Vec<&[f64]> = self.z.iter().map(|v| &v[..]).collect();
            vec![
                Effect::new(&self.worker, true, x).unwrap(),
                Effect::new(&self.firm, true, z).unwrap(),
            ]
        }
    }

    /// The built coarse operators against each other: `E = ZᵀSZ` inverted on its range, `SZ` vs `SZᵀ`.
    fn operators(panel: &HubPanel, data: &Data) -> String {
        let design = Design::new_unsorted(data.effects()).unwrap();
        let prepared =
            PreparedDesign::new(design.into_design().unwrap(), data.weights.as_deref()).unwrap();
        let e = Elimination::new(&prepared);
        let start = std::time::Instant::now();
        let Ok(c) = ReducedCoarse::build(&e, 150.0) else {
            return format!("declined in {:.3}s", start.elapsed().as_secs_f64());
        };
        let build_s = start.elapsed().as_secs_f64();
        let e_mul = |v: &[f64]| {
            let mut sv = vec![0.0; e.n_kept()];
            c.sz_mul_add(v, 1.0, &mut sv);
            let mut out = vec![0.0; c.m];
            c.z.t_mul(&sv, &mut out);
            out
        };
        let norm = |v: &[f64]| v.iter().map(|a| a * a).sum::<f64>().sqrt();
        let z: Vec<f64> = (0..c.m).map(|i| (i as f64 * 0.7).sin()).collect();
        let x = e_mul(&z);
        let mut y = x.clone();
        c.solve_coarse(&mut y);
        let r: Vec<f64> = e_mul(&y).iter().zip(&x).map(|(a, b)| a - b).collect();
        // Backward error: a stable solve leaves `‖E y − x‖ ~ ε ‖E‖ ‖y‖` whatever `κ(E)`.
        let mut v = z.clone();
        let mut e_norm = 0.0;
        for _ in 0..30 {
            let ev = e_mul(&v);
            e_norm = norm(&ev) / norm(&v);
            v = ev;
        }
        let residual = norm(&r) / (e_norm * norm(&y));
        let kappa = e_norm * norm(&y) / norm(&x);
        let w: Vec<f64> = (0..e.n_kept()).map(|i| (i as f64 * 0.3).cos()).collect();
        let (mut sz, mut szt) = (vec![0.0; e.n_kept()], vec![0.0; c.m]);
        c.sz_mul_add(&z, 1.0, &mut sz);
        c.sz_t_mul(&w, &mut szt);
        let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
        let adjoint = (dot(&sz, &w) - dot(&z, &szt)).abs() / dot(&sz, &w).abs();
        let rank = c.hubs.as_ref().map_or(0, HubCorrection::rank);
        assert!(
            residual < 1e-8 && adjoint < 1e-10,
            "{panel:?}: residual {residual:.1e} adjoint {adjoint:.1e}"
        );
        format!(
            "m={} rank={rank} L={} SZ={} build={build_s:.3}s backward={residual:.0e} κ≳{kappa:.0e} adjoint={adjoint:.0e}",
            c.m,
            c.l_values.len(),
            c.sz.nnz()
        )
    }

    /// Reduced LSMR with and without the rigid rung against the full solve.
    fn end_to_end(data: &Data) -> String {
        let options = LsmrOptions {
            tol: 1e-10,
            maxiter: 20_000,
            ..Default::default()
        };
        let config = LocalSolverConfig::default();
        let full = Solver::new(
            data.effects(),
            data.weights.as_deref(),
            PreconditionerConfig::Additive {
                local_solver: config.clone(),
                reduction: Default::default(),
            },
        )
        .unwrap();
        let reference = full.solve(&data.y, &options).unwrap();
        let norm = |v: &[f64]| v.iter().map(|a| a * a).sum::<f64>().sqrt();
        let mut out = format!("full {}", reference.iterations);
        for rigid in [false, true] {
            match rigid {
                true => std::env::set_var("WITHIN_REDUCED_RIGID", "150"),
                false => std::env::remove_var("WITHIN_REDUCED_RIGID"),
            }
            let reduced = full
                .pair_schur_prototype(&config, Default::default())
                .unwrap();
            let result = reduced.solve_lsmr(&data.y, &options).unwrap();
            let gap: Vec<f64> = (result.demeaned.iter().zip(&reference.demeaned))
                .map(|(a, b)| a - b)
                .collect();
            out += &format!(
                " | {} {}{} gap={:.0e}",
                if rigid { "rigid" } else { "reduced" },
                result.iterations,
                if result.converged { "" } else { "!" },
                norm(&gap) / norm(&reference.demeaned)
            );
        }
        std::env::remove_var("WITHIN_REDUCED_RIGID");
        out
    }

    /// Many hubs at scale: `HUB_T` clusters, `HUB_HUBS` hubs of `HUB_ROWS` rows over `HUB_REACH` clusters.
    #[test]
    #[ignore]
    fn hub_scale() {
        let var = |k: &str, d: usize| std::env::var(k).map_or(d, |v| v.parse().unwrap());
        let panel = HubPanel {
            clusters: var("HUB_T", 10_000),
            hubs: var("HUB_HUBS", 100),
            reach: var("HUB_REACH", 200),
            rows: var("HUB_ROWS", 2_000),
            ..HubPanel::base()
        };
        let data = panel.data();
        println!("{panel:?} n_obs={}", data.y.len());
        println!("{}", operators(&panel, &data));
        if std::env::var_os("HUB_SOLVE").is_some() {
            println!("{}", end_to_end(&data));
        }
    }

    #[test]
    #[ignore]
    fn hub_battle() {
        let base = HubPanel::base();
        let variants = [
            ("base", base),
            (
                "4 hubs disjoint",
                HubPanel {
                    hubs: 4,
                    reach: 50,
                    rows: 2_500,
                    ..base
                },
            ),
            (
                "4 hubs overlap",
                HubPanel {
                    hubs: 4,
                    rows: 2_500,
                    ..base
                },
            ),
            (
                "80 hubs past cap",
                HubPanel {
                    hubs: 80,
                    reach: 40,
                    rows: 400,
                    ..base
                },
            ),
            (
                "weighted, zeros",
                HubPanel {
                    weighted: true,
                    ..base
                },
            ),
            ("flat hub", HubPanel { flat: true, ..base }),
            (
                "shared slopes",
                HubPanel {
                    shared: true,
                    ..base
                },
            ),
            (
                "p=3",
                HubPanel {
                    slopes: 2,
                    rows: 15_000,
                    ..base
                },
            ),
            (
                "t=40 below",
                HubPanel {
                    reach: 20,
                    rows: 2_000,
                    ..base
                },
            ),
            (
                "t=42 above",
                HubPanel {
                    reach: 21,
                    rows: 2_100,
                    ..base
                },
            ),
            (
                "3 clusters",
                HubPanel {
                    clusters: 3,
                    reach: 3,
                    rows: 300,
                    ..base
                },
            ),
        ];
        let solve = std::env::var_os("HUB_SOLVE").is_some();
        for (name, panel) in variants {
            let data = panel.data();
            println!("{name:18} {}", operators(&panel, &data));
            if solve {
                println!("{:18} {}", "", end_to_end(&data));
            }
        }
    }
}
