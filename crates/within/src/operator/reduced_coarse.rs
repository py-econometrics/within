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
use crate::domain::elimination::Elimination;
use crate::domain::Column;

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
        out.par_iter_mut().enumerate().for_each(|(c, o)| {
            let span = self.col_ptr[c]..self.col_ptr[c + 1];
            *o = (self.col_idx[span.clone()].iter().zip(&self.col_val[span]))
                .map(|(&r, &v)| v * x[r as usize])
                .sum();
        });
    }

    fn mul_add(&self, c: &[f64], alpha: f64, out: &mut [f64]) {
        const BLOCK: usize = 4096;
        out.par_chunks_mut(BLOCK)
            .enumerate()
            .for_each(|(j, block)| {
                let base = j * BLOCK;
                let lo = self.rows.partition_point(|&r| (r as usize) < base);
                let hi = self
                    .rows
                    .partition_point(|&r| (r as usize) < base + block.len());
                for i in lo..hi {
                    let span = self.row_ptr[i]..self.row_ptr[i + 1];
                    let s: f64 = (self.row_idx[span.clone()].iter().zip(&self.row_val[span]))
                        .map(|(&k, &v)| v * c[k as usize])
                        .sum();
                    block[self.rows[i] as usize - base] += alpha * s;
                }
            });
    }
}

/// Merge `(a, b, v)` triplets with equal `(a, b)` after sorting.
fn sort_merge(mut t: Vec<(u32, u32, f64)>) -> Vec<(u32, u32, f64)> {
    t.par_sort_unstable_by_key(|e| (e.0, e.1));
    let mut out: Vec<(u32, u32, f64)> = Vec::with_capacity(t.len());
    for e in t {
        match out.last_mut() {
            Some(last) if last.0 == e.0 && last.1 == e.1 => last.2 += e.2,
            _ => out.push(e),
        }
    }
    out
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

/// One term's solve-basis columns at a row.
struct LevelSide<'a> {
    levels: &'a [u32],
    loadings: Vec<Option<&'a [f64]>>,
    n_levels: usize,
}

impl LevelSide<'_> {
    fn coef(&self, column: usize, row: usize) -> f64 {
        self.loadings[column].map_or(1.0, |z| z[row])
    }

    fn gram(&self, rows: impl Iterator<Item = usize>, w: impl Fn(usize) -> f64, out: &mut [f64]) {
        let p = self.loadings.len();
        out.fill(0.0);
        for r in rows {
            let wr = w(r);
            for a in 0..p {
                let wa = wr * self.coef(a, r);
                for b in 0..p {
                    out[a * p + b] += wa * self.coef(b, r);
                }
            }
        }
    }
}

/// The coarse space `Z` over the kept coefficients and the LDLᵀ of `ZᵀSZ = (ÃZ)ᵀ(ÃZ)`.
pub(crate) struct ReducedCoarse {
    m: usize,
    z: TwoWay,
    sz: TwoWay,
    symbolic: SymbolicCholesky<u32>,
    l_values: Vec<f64>,
    /// Nanoseconds per apply stage: Zᵀ, solve, SZ, inner, SZᵀ, solve, Z; and apply count.
    timers: [std::sync::atomic::AtomicU64; 8],
}

impl Drop for ReducedCoarse {
    fn drop(&mut self) {
        if std::env::var_os("WITHIN_REDUCED_RIGID_STATS").is_some() {
            let t: Vec<u64> = self
                .timers
                .iter()
                .map(|a| a.load(std::sync::atomic::Ordering::Relaxed))
                .collect();
            let calls = t[7].max(1) as f64;
            let ms = |i: usize| t[i] as f64 / calls / 1e6;
            eprintln!(
                "reduced_coarse_apply: calls={} ms/apply Zt={:.3} solve1={:.3} SZ={:.3} inner={:.3} SZt={:.3} solve2={:.3} Z={:.3}",
                t[7], ms(0), ms(1), ms(2), ms(3), ms(4), ms(5), ms(6)
            );
        }
    }
}

impl ReducedCoarse {
    /// `None` without a kept term sharing the eliminated term's column layout, or when the
    /// components cover more than half the levels or the coarse factor exceeds `8 · n_obs` values.
    pub(crate) fn build(e: &Elimination<'_>, k: f64) -> Option<Self> {
        let started = std::time::Instant::now();
        let prepared = e.prepared();
        let design = &prepared.design;
        let sw = prepared.sqrt_weights();
        let weight = |r: usize| sw.map_or(1.0, |s| s[r] * s[r]);
        let root = |r: usize| sw.map_or(1.0, |s| s[r]);
        let n_obs = design.n_obs();
        let p = e.width();
        let elim_term = &design.terms[e.term()];
        let layout = |t: &crate::domain::Term<'_>| {
            (0..t.n_columns())
                .map(|c| matches!(t.column(c), Column::Intercept))
                .collect::<Vec<_>>()
        };

        // Anchor: the kept term with the eliminated term's layout and the most levels.
        let mut base = 0usize;
        let mut anchor: Option<(usize, usize)> = None;
        for (t, term) in design.terms.iter().enumerate() {
            if t == e.term() {
                continue;
            }
            if layout(term) == layout(elim_term)
                && anchor.is_none_or(|(a, _)| design.terms[a].n_levels() < term.n_levels())
            {
                anchor = Some((t, base));
            }
            base += term.n_dofs();
        }
        let (anchor, anchor_base) = anchor?;
        let a_term = prepared.term(anchor);
        let sides = [
            LevelSide {
                levels: elim_term.levels(),
                loadings: e.eliminated_loadings(),
                n_levels: elim_term.n_levels(),
            },
            LevelSide {
                levels: a_term.term.levels(),
                loadings: (0..p).map(|c| a_term.loading(c)).collect(),
                n_levels: a_term.term.n_levels(),
            },
        ];
        let (na, nb) = (sides[0].n_levels, sides[1].n_levels);
        let groups = e.groups();

        // Level Grams of both sides.
        let level_grams = |s: &LevelSide<'_>| {
            let mut g = vec![0.0; s.n_levels * p * p];
            for (r, &l) in s.levels.iter().enumerate() {
                let block = &mut g[l as usize * p * p..][..p * p];
                let wr = weight(r);
                for a in 0..p {
                    let wa = wr * s.coef(a, r);
                    for b in 0..p {
                        block[a * p + b] += wa * s.coef(b, r);
                    }
                }
            }
            g
        };
        let lgram = [level_grams(&sides[0]), level_grams(&sides[1])];

        // Spells: within each eliminated level's group, rows sorted by anchor level.
        let glued: Vec<(u32, u32)> = (0..na)
            .into_par_iter()
            .flat_map_iter(|lw| {
                let mut rows: Vec<usize> =
                    (groups.order.rows(groups.starts[lw]..groups.starts[lw + 1]))
                        .map(|i| i as usize)
                        .collect();
                rows.sort_unstable_by_key(|&r| sides[1].levels[r]);
                let mut out = Vec::new();
                let mut gram = vec![0.0; p * p];
                for spell in rows.chunk_by(|&a, &b| sides[1].levels[a] == sides[1].levels[b]) {
                    let lf = sides[1].levels[spell[0]] as usize;
                    let mut worst = 0.0f64;
                    for (si, (s, level)) in sides.iter().zip([lw, lf]).enumerate() {
                        s.gram(spell.iter().copied(), weight, &mut gram);
                        worst = worst.max(pencil_condition(
                            &gram,
                            &lgram[si][level * p * p..][..p * p],
                            p,
                        ));
                    }
                    if worst <= k {
                        out.push((lw as u32, lf as u32));
                    }
                }
                out
            })
            .collect();

        if std::env::var_os("WITHIN_REDUCED_RIGID_STATS").is_some() {
            eprintln!(
                "reduced_stage: spells {:.3}",
                started.elapsed().as_secs_f64()
            );
        }
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
        let stats = std::env::var_os("WITHIN_REDUCED_RIGID_STATS").is_some();
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
        if stats {
            eprintln!(
                "reduced_stage: components {:.3}",
                started.elapsed().as_secs_f64()
            );
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
        let m = n_fc as usize * p;

        // Gauges: per anchor level, the solve coefficients reproducing each user channel.
        let raw: Vec<&[f64]> = a_term.term.raw_slopes().collect();
        let user = |c: usize, r: usize| match a_term.term.column(c) {
            Column::Intercept => 1.0,
            Column::Slope(j) => raw[j][r],
        };
        let mut rhs = vec![0.0; nb * p * p];
        for (r, &l) in sides[1].levels.iter().enumerate() {
            let wr = weight(r);
            for q in 0..p {
                let wq = wr * sides[1].coef(q, r);
                for j in 0..p {
                    rhs[(l as usize * p + j) * p + q] += wq * user(j, r);
                }
            }
        }
        let mut gauge = vec![0.0; nb * p * p];
        for l in 0..nb {
            let g = &lgram[1][l * p * p..][..p * p];
            for j in 0..p {
                let x = active_solve(g, &rhs[(l * p + j) * p..][..p], p);
                gauge[(l * p + j) * p..][..p].copy_from_slice(&x);
            }
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
        let n = e.n_kept();
        let mut z_entries = Vec::new();
        for l in 0..nb {
            let c = anchor_comp[l] as usize;
            for j in 0..p {
                for (q, &v) in gauge[(l * p + j) * p..][..p].iter().enumerate() {
                    if v != 0.0 {
                        let dof = anchor_base + q * nb + l;
                        z_entries.push(((c * p + j) as u32, dof as u32, v * inv_norm[c * p + j]));
                    }
                }
            }
        }
        if stats {
            eprintln!(
                "reduced_stage: gauges+Z {:.3}",
                started.elapsed().as_secs_f64()
            );
        }
        // U = ÃZ per grouped row, projected within each eliminated level's group.
        let inverse = e.inverse();
        let elim_load = e.eliminated_loadings();
        let ec = |a: usize, r: usize| elim_load[a].map_or(1.0, |z| z[r]);
        // Per group: (grouped row, coarse column, value) and the group's Gram contributions.
        type Group = (Vec<(u32, u32, f64)>, Vec<(u32, u32, f64)>, Vec<(u32, f64)>);
        let per_group: Vec<Group> = (0..na)
            .into_par_iter()
            .map(|lw| {
                let span = groups.starts[lw]..groups.starts[lw + 1];
                let rows: Vec<usize> = groups
                    .order
                    .rows(span.clone())
                    .map(|i| i as usize)
                    .collect();
                let mut cols: Vec<u32> = rows
                    .iter()
                    .flat_map(|&r| {
                        let c = anchor_comp[sides[1].levels[r] as usize];
                        (0..p as u32).map(move |j| c * p as u32 + j)
                    })
                    .collect();
                cols.sort_unstable();
                cols.dedup();
                let t = cols.len();
                let col_of = |c: u32| cols.binary_search(&c).expect("touched column");
                // v[row][t]: w½ · (D Z) at the row.
                let mut v = vec![0.0; rows.len() * t];
                for (i, &r) in rows.iter().enumerate() {
                    let l = sides[1].levels[r] as usize;
                    let c = anchor_comp[l] as usize;
                    for j in 0..p {
                        let g = &gauge[(l * p + j) * p..][..p];
                        let val: f64 = (0..p).map(|q| sides[1].coef(q, r) * g[q]).sum();
                        v[i * t + col_of((c * p + j) as u32)] = root(r) * val * inv_norm[c * p + j];
                    }
                }
                let pre: Vec<(u32, f64)> = (0..t)
                    .map(|c| {
                        (
                            cols[c],
                            (0..rows.len()).map(|i| v[i * t + c] * v[i * t + c]).sum(),
                        )
                    })
                    .collect();
                // Project off the eliminated level's columns: β = G⁻ Eᵀ W½ v.
                let inv = &inverse[lw * p * p..][..p * p];
                let mut s = vec![0.0; p * t];
                for (i, &r) in rows.iter().enumerate() {
                    for a in 0..p {
                        let wa = root(r) * ec(a, r);
                        for c in 0..t {
                            s[a * t + c] += wa * v[i * t + c];
                        }
                    }
                }
                let mut beta = vec![0.0; p * t];
                for a in 0..p {
                    for b in 0..p {
                        for c in 0..t {
                            beta[a * t + c] += inv[a * p + b] * s[b * t + c];
                        }
                    }
                }
                for (i, &r) in rows.iter().enumerate() {
                    for a in 0..p {
                        let wa = root(r) * ec(a, r);
                        for c in 0..t {
                            v[i * t + c] -= wa * beta[a * t + c];
                        }
                    }
                }
                let mut u = Vec::with_capacity(rows.len() * t);
                for (i, p_row) in span.enumerate() {
                    let vi = &v[i * t..][..t];
                    for c in 0..t {
                        if vi[c] != 0.0 {
                            u.push((p_row as u32, cols[c], vi[c]));
                        }
                    }
                }
                (u, Vec::new(), pre)
            })
            .collect();
        if stats {
            eprintln!(
                "reduced_stage: U per group {:.3}",
                started.elapsed().as_secs_f64()
            );
        }
        let mut pre_energy = vec![0.0f64; m];
        // Exact nulls of S (the eliminated term absorbs the gauge) need no lift; keep the rest.
        let mut energy = vec![0.0f64; m];
        let mut u = Vec::with_capacity(na);
        for (ug, _, pg) in per_group {
            for &(_, c, v) in &ug {
                energy[c as usize] += v * v;
            }
            u.push(ug);
            for (c, v) in pg {
                pre_energy[c as usize] += v;
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
        let dropped = m - live as usize;
        let m = live as usize;
        if m == 0 {
            return None;
        }
        let keep = |t: &(u32, u32, f64)| renumber[t.0 as usize] != u32::MAX;
        let z_entries: Vec<_> = (z_entries.into_iter().filter(keep))
            .map(|(c, r, v)| (renumber[c as usize], r, v))
            .collect();
        let z = TwoWay::from_sorted(n, m, &sort_merge(z_entries));
        if stats {
            eprintln!(
                "reduced_stage: renumber {:.3}",
                started.elapsed().as_secs_f64()
            );
        }
        // U renumbered in grouped-row order: segment `s` is one row's entries, columns ascending.
        let (mut seg_prow, mut seg_ptr) = (Vec::<u32>::new(), vec![0usize]);
        let (mut row_col, mut row_val) = (Vec::<u32>::new(), Vec::<f64>::new());
        for (p, c, v) in u.into_iter().flatten() {
            let c = renumber[c as usize];
            if c == u32::MAX {
                continue;
            }
            if seg_prow.last() != Some(&p) {
                if !seg_prow.is_empty() {
                    seg_ptr.push(row_col.len());
                }
                seg_prow.push(p);
            }
            row_col.push(c);
            row_val.push(v);
        }
        if !seg_prow.is_empty() {
            seg_ptr.push(row_col.len());
        }
        // The same entries by coarse column, each naming its row segment.
        let mut u_ptr = vec![0usize; m + 1];
        for &c in &row_col {
            u_ptr[c as usize + 1] += 1;
        }
        for k in 0..m {
            u_ptr[k + 1] += u_ptr[k];
        }
        let mut u_seg = vec![0u32; u_ptr[m]];
        let mut u_val = vec![0.0f64; u_ptr[m]];
        let mut cursor = u_ptr.clone();
        for s in 0..seg_prow.len() {
            for k in seg_ptr[s]..seg_ptr[s + 1] {
                let j = &mut cursor[row_col[k] as usize];
                u_seg[*j] = s as u32;
                u_val[*j] = row_val[k];
                *j += 1;
            }
        }
        drop(cursor);
        let ranges = 4 * rayon::current_num_threads().max(1);
        let per = u_ptr[m].div_ceil(ranges).max(1);
        let mut splits = vec![0usize];
        for (c, &start) in u_ptr.iter().enumerate().take(m).skip(1) {
            if start >= per * splits.len() {
                splits.push(c);
            }
        }
        splits.push(m);
        let windows: Vec<(usize, usize)> = splits.windows(2).map(|w| (w[0], w[1])).collect();
        if stats {
            eprintln!(
                "reduced_stage: U columns {:.3}",
                started.elapsed().as_secs_f64()
            );
        }
        // E = UᵀU by column, lower triangle: column c gathers the rows touching it, kept at c' ≥ c.
        let e_entries: Vec<(u32, u32, f64)> = windows
            .par_iter()
            .map(|&(lo, hi)| {
                let mut acc = vec![0.0f64; m];
                let mut seen = vec![u32::MAX; m];
                let mut touched: Vec<u32> = Vec::new();
                let mut out = Vec::new();
                for col in lo..hi {
                    for k in u_ptr[col]..u_ptr[col + 1] {
                        let s = u_seg[k] as usize;
                        let a = u_val[k];
                        let span = seg_ptr[s]..seg_ptr[s + 1];
                        let from = span.start
                            + row_col[span.clone()].partition_point(|&c| (c as usize) < col);
                        for j in from..span.end {
                            let c2 = row_col[j] as usize;
                            if seen[c2] != col as u32 {
                                seen[c2] = col as u32;
                                touched.push(c2 as u32);
                            }
                            acc[c2] += a * row_val[j];
                        }
                    }
                    touched.sort_unstable();
                    for &c2 in &touched {
                        let v = std::mem::take(&mut acc[c2 as usize]);
                        if v != 0.0 {
                            out.push((col as u32, c2, v));
                        }
                    }
                    touched.clear();
                }
                out
            })
            .collect::<Vec<_>>()
            .concat();
        drop((row_col, row_val, seg_ptr));

        if stats {
            eprintln!("reduced_stage: E {:.3}", started.elapsed().as_secs_f64());
        }
        let mut col_ptr = vec![0u32; m + 1];
        for &(c, _, _) in &e_entries {
            col_ptr[c as usize + 1] += 1;
        }
        for d in 0..m {
            col_ptr[d + 1] += col_ptr[d];
        }
        let d_max = e_entries
            .iter()
            .filter(|t| t.0 == t.1)
            .map(|t| t.2)
            .fold(0.0, f64::max);
        let pattern = SymbolicSparseColMat::new_checked(
            m,
            m,
            col_ptr,
            None,
            e_entries.iter().map(|t| t.1).collect(),
        );
        let e_lower = SparseColMat::new(pattern, e_entries.iter().map(|t| t.2).collect());
        let symbolic = factorize_symbolic_cholesky(
            e_lower.symbolic(),
            Side::Lower,
            SymmetricOrdering::Amd,
            CholeskySymbolicParams::default(),
        )
        .ok()?;
        let fits = symbolic.len_val() <= 8 * n_obs;
        if stats {
            eprintln!(
                "reduced_rigid: glued={} components={n_components} anchor_components={n_fc} m={m} null_dropped={dropped} Z_nnz={} E_nnz={} L_nnz={} fits={fits} build_s={:.3}",
                glued.len(),
                z.nnz(),
                e_entries.len(),
                symbolic.len_val(),
                started.elapsed().as_secs_f64()
            );
        }
        if !fits {
            return None;
        }
        // SZ = D_keptᵀ W½ U: every kept channel's level sums of w½·a·U.
        let channels = e.kept_channels();
        let order: Vec<usize> = groups
            .order
            .rows(0..groups.order.len())
            .map(|i| i as usize)
            .collect();
        // Each coarse column summed into a dense kept-coefficient buffer: no global sort.
        let sz_parts: Vec<Vec<(u32, u32, f64)>> = windows
            .par_iter()
            .map(|&(lo, hi)| {
                let mut acc = vec![0.0f64; n];
                let mut seen = vec![u32::MAX; n];
                let mut touched: Vec<u32> = Vec::new();
                let mut out = Vec::new();
                for col in lo..hi {
                    for k in u_ptr[col]..u_ptr[col + 1] {
                        let r = order[seg_prow[u_seg[k] as usize] as usize];
                        let val = root(r) * u_val[k];
                        for ch in &channels {
                            let d = ch.start + ch.levels[r] as usize;
                            if seen[d] != col as u32 {
                                seen[d] = col as u32;
                                touched.push(d as u32);
                            }
                            acc[d] += ch.loading.map_or(1.0, |z| z[r]) * val;
                        }
                    }
                    touched.sort_unstable();
                    for &d in &touched {
                        let v = std::mem::take(&mut acc[d as usize]);
                        if v != 0.0 {
                            out.push((col as u32, d, v));
                        }
                    }
                    touched.clear();
                }
                out
            })
            .collect();
        drop((u_ptr, u_seg, u_val, seg_prow));
        let sz = TwoWay::from_parts(n, m, sz_parts);
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
        let mut l_values = vec![0.0; symbolic.len_val()];
        let mut mem = MemBuffer::new(
            symbolic.factorize_numeric_ldlt_scratch::<f64>(coarse_par(), Default::default()),
        );
        symbolic
            .factorize_numeric_ldlt(
                &mut l_values,
                e_lower.as_ref(),
                Side::Lower,
                regularization,
                coarse_par(),
                MemStack::new(&mut mem),
                Default::default(),
            )
            .ok()?;
        if stats {
            eprintln!(
                "reduced_rigid: factored_s={:.3}",
                started.elapsed().as_secs_f64()
            );
        }
        l_values.iter().all(|v| v.is_finite()).then_some(Self {
            m,
            z,
            sz,
            symbolic,
            l_values,
            timers: Default::default(),
        })
    }

    fn solve_coarse(&self, v: &mut [f64]) {
        let mut buf = MemBuffer::new(self.symbolic.solve_in_place_scratch::<f64>(1, coarse_par()));
        LdltRef::new(&self.symbolic, &self.l_values).solve_in_place_with_conj(
            Conj::No,
            MatMut::from_column_major_slice_mut(v, self.m, 1),
            coarse_par(),
            MemStack::new(&mut buf),
        );
    }

    /// Balancing form around `inner`: `Z E⁻¹ Zᵀx + (I − Z E⁻¹ ZᵀS) B (I − S Z E⁻¹ Zᵀ) x`.
    fn apply_balanced(
        &self,
        inner: &dyn Operator,
        x: &[f64],
        y: &mut [f64],
    ) -> Result<(), SolveError> {
        use std::sync::atomic::Ordering::Relaxed;
        let mut clock = std::time::Instant::now();
        let mut lap = |i: usize| {
            let now = std::time::Instant::now();
            self.timers[i].fetch_add((now - clock).as_nanos() as u64, Relaxed);
            clock = now;
        };
        let mut c = vec![0.0; self.m];
        self.z.t_mul(x, &mut c);
        lap(0);
        self.solve_coarse(&mut c);
        lap(1);
        if std::env::var_os("WITHIN_REDUCED_COARSE_ADDITIVE").is_some() {
            inner.apply(x, y)?;
            lap(3);
            self.z.mul_add(&c, 1.0, y);
            lap(6);
            self.timers[7].fetch_add(1, Relaxed);
            return Ok(());
        }
        let mut t = x.to_vec();
        self.sz.mul_add(&c, -1.0, &mut t);
        lap(2);
        inner.apply(&t, y)?;
        lap(3);
        let mut d = vec![0.0; self.m];
        self.sz.t_mul(y, &mut d);
        lap(4);
        self.solve_coarse(&mut d);
        lap(5);
        for (ci, di) in c.iter_mut().zip(&d) {
            *ci -= di;
        }
        self.z.mul_add(&c, 1.0, y);
        lap(6);
        self.timers[7].fetch_add(1, Relaxed);
        Ok(())
    }
}

impl CoarseRung for ReducedCoarse {
    fn apply_around(
        &self,
        inner: &dyn Operator,
        x: &[f64],
        y: &mut [f64],
    ) -> Result<(), SolveError> {
        self.apply_balanced(inner, x, y)
    }
}

/// `WITHIN_REDUCED_COARSE_PAR=1` runs the coarse factorization and solves on the rayon pool.
fn coarse_par() -> Par {
    match std::env::var_os("WITHIN_REDUCED_COARSE_PAR") {
        Some(_) => Par::rayon(0),
        None => Par::Seq,
    }
}
