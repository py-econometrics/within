//! The reduced system: the kept terms with the eliminated term's span projected out, one sweep
//! per product over the rows grouped by eliminated level.

use std::ops::Range;

use rayon::prelude::*;
use schwarz_precond::Operator;

use crate::domain::{
    fixed_tree, row_weight, LevelRows, PreparedDesign, PreparedTerm, MIN_LEAF_ROWS,
};
use crate::linalg::{fit_twice, project, DenseBasis, DiagonalGramBasis};
use crate::BuildError;

/// Rows per parallel task, cut at an eliminated level's end so no level spans two tasks.
const TASK_ROWS: usize = 16_384;

/// One term's coefficient columns, read row by row.
struct Columns<'p> {
    levels: &'p [u32],
    n_levels: usize,
    /// Per column, its loading; `None` is the intercept.
    loadings: Vec<Option<&'p [f64]>>,
}

impl<'p> Columns<'p> {
    fn new(t: &PreparedTerm<'p>) -> Self {
        Self {
            levels: t.term.levels(),
            n_levels: t.term.n_levels(),
            loadings: (0..t.term.n_columns()).map(|c| t.loading(c)).collect(),
        }
    }

    /// `(D_t x_t)_i` with `x_t` at `offset` of `x`; `PLAIN` when every term is a lone intercept.
    #[inline(always)]
    fn row<const PLAIN: bool>(&self, x: &[f64], offset: usize, i: usize) -> f64 {
        let level = offset + self.levels[i] as usize;
        if PLAIN {
            return x[level];
        }
        (self.loadings.iter().enumerate()).fold(-0.0, |acc, (c, z)| {
            let b = x[c * self.n_levels + level];
            acc + z.map_or(b, |z| b * z[i])
        })
    }

    /// `x_t += v · D_tᵀ e_i` with `x_t` at `offset` of `x`.
    #[inline(always)]
    fn add_row<const PLAIN: bool>(&self, x: &mut [f64], offset: usize, i: usize, v: f64) {
        let level = offset + self.levels[i] as usize;
        if PLAIN {
            x[level] += v;
            return;
        }
        for (c, z) in self.loadings.iter().enumerate() {
            x[c * self.n_levels + level] += z.map_or(v, |z| z[i] * v);
        }
    }
}

/// `A_r = (I − P_e) A_k`, where `P_e` projects onto the eliminated term's columns `A_e`.
///
/// The kept coefficients are the full ones without the eliminated term's block.
pub(crate) struct ReducedDesign<'p, 'a> {
    prepared: &'p PreparedDesign<'a>,
    eliminated: Columns<'p>,
    /// `diag(A_eᵀA_e)`, the whole Gram block of each level after whitening.
    gram: &'p [f64],
    /// The rows of each eliminated level, contiguous: the design is sorted by it.
    rows: &'p LevelRows,
    /// Consecutive eliminated levels per parallel task.
    tasks: Vec<Range<usize>>,
    /// Consecutive tasks per adjoint accumulator, fixed by the data alone.
    leaves: Vec<Range<usize>>,
    /// Kept terms: design index, columns, offset in the kept coefficients.
    kept: Vec<(usize, Columns<'p>, usize)>,
    n_kept: usize,
    max_level_rows: usize,
    /// Every term is a lone intercept: the plain kernels.
    plain: bool,
}

/// Runs `$body` with `$plain` and the row weight `$w` fixed at compile time.
macro_rules! with_shape {
    ($self:ident, |$plain:ident, $w:ident| $body:expr) => {
        match ($self.plain, $self.prepared.sqrt_weights()) {
            (true, None) => {
                const $plain: bool = true;
                let $w = |_: usize| 1.0;
                $body
            }
            (true, Some(sw)) => {
                const $plain: bool = true;
                let $w = |i: usize| sw[i];
                $body
            }
            (false, None) => {
                const $plain: bool = false;
                let $w = |_: usize| 1.0;
                $body
            }
            (false, Some(sw)) => {
                const $plain: bool = false;
                let $w = |i: usize| sw[i];
                $body
            }
        }
    };
}

impl<'p, 'a> ReducedDesign<'p, 'a> {
    /// Needs `u32` row indices; refuses a design not sorted by its dominant term.
    pub(crate) fn new(prepared: &'p PreparedDesign<'a>) -> Result<Self, BuildError> {
        let design = &prepared.design;
        let e = prepared.term(design.dominant);
        let rows = prepared.level_rows(design.dominant);
        if rows.order().is_some() {
            return Err(BuildError::UnsortedReducedSystem);
        }
        let mut kept = Vec::new();
        let mut n_kept = 0;
        for t in (0..design.n_factors()).filter(|&t| t != design.dominant) {
            let term = prepared.term(t);
            kept.push((t, Columns::new(&term), n_kept));
            n_kept += term.term.n_dofs();
        }
        let tasks = cut(e.term.n_levels(), TASK_ROWS, |l| rows.positions(l).len());
        let task_rows =
            |t: usize| rows.positions(tasks[t].end - 1).end - rows.positions(tasks[t].start).start;
        // Merging costs ∝ n_kept per leaf, so a leaf spans at least 4·n_kept rows.
        let leaves = cut(tasks.len(), MIN_LEAF_ROWS.max(4 * n_kept), task_rows);
        let plain = (design.terms.iter()).all(|t| t.intercept && !t.has_slopes());
        let max_level_rows = (0..e.term.n_levels())
            .map(|l| rows.positions(l).len())
            .max()
            .unwrap_or(0);
        let eliminated = Columns::new(&e);
        let gram = prepared.diagonal(design.dominant);
        Ok(Self {
            plain,
            prepared,
            eliminated,
            gram,
            rows,
            tasks,
            leaves,
            kept,
            n_kept,
            max_level_rows,
        })
    }

    /// `(D_k x)_i`.
    #[inline(always)]
    fn kept_row<const PLAIN: bool>(&self, x: &[f64], i: usize) -> f64 {
        let row = |(_, columns, offset): &(usize, Columns<'_>, usize)| {
            columns.row::<PLAIN>(x, *offset, i)
        };
        // One and two kept terms skip the loop, which costs a 3FE product a fifth of its time.
        match &self.kept[..] {
            [a] => row(a),
            [a, b] => row(a) + row(b),
            kept => kept.iter().fold(-0.0, |acc, term| acc + row(term)),
        }
    }

    /// `acc += v · D_kᵀ e_i`.
    #[inline(always)]
    fn add_kept_row<const PLAIN: bool>(&self, acc: &mut [f64], i: usize, v: f64) {
        let mut add = |(_, columns, offset): &(usize, Columns<'_>, usize)| {
            columns.add_row::<PLAIN>(acc, *offset, i, v)
        };
        match &self.kept[..] {
            [a] => add(a),
            [a, b] => {
                add(a);
                add(b);
            }
            kept => kept.iter().for_each(add),
        }
    }

    /// Adds row `i`'s weighted value `wu` to its level's moments `D_e,lᵀ W^½ u`.
    #[inline(always)]
    fn accumulate(&self, moments: &mut [f64], i: usize, wu: f64) {
        for (m, z) in moments.iter_mut().zip(&self.eliminated.loadings) {
            *m += z.map_or(wu, |z| z[i] * wu);
        }
    }

    /// Level `l`'s moments become its fit `G_l⁺ A_e,lᵀ u`; 0 where unidentified.
    #[inline(always)]
    fn solve(&self, l: usize, fit: &mut [f64]) {
        let n_levels = self.eliminated.n_levels;
        for (c, f) in fit.iter_mut().enumerate() {
            let g = self.gram[c * n_levels + l];
            *f = if g > 0.0 { *f / g } else { 0.0 };
        }
    }

    /// Fills `level` with level `l`'s basis `W^½ z_c`, sets each row's value in `v` by `set`, and
    /// sets `level.fit` to their moments.
    #[inline(always)]
    fn level_basis(
        &self,
        l: usize,
        positions: Range<usize>,
        w: impl Fn(usize) -> f64,
        level: &mut Level<'_>,
        v: &mut [f64],
        set: impl Fn(usize, &mut f64),
    ) {
        let n_levels = self.eliminated.n_levels;
        for (c, g) in level.norms.iter_mut().enumerate() {
            *g = self.gram[c * n_levels + l];
        }
        level.fit.fill(0.0);
        let width = level.fit.len();
        let rows = level.entries.chunks_exact_mut(width).zip(v.iter_mut());
        for (i, (entries, x)) in positions.zip(rows) {
            let wi = w(i);
            set(i, x);
            for ((e, m), z) in
                (entries.iter_mut().zip(level.fit.iter_mut())).zip(&self.eliminated.loadings)
            {
                *e = z.map_or(wi, |z| wi * z[i]);
                *m += *e * *x;
            }
        }
    }

    /// `(D_e,l fit)_i`.
    #[inline(always)]
    fn fitted(&self, fit: &[f64], i: usize) -> f64 {
        (fit.iter().zip(&self.eliminated.loadings))
            .fold(-0.0, |acc, (f, z)| acc + z.map_or(*f, |z| f * z[i]))
    }

    /// `out ← (I − P_e) out` in one pass, `set` first writing each row's value; `w = W^½`.
    fn project<const PLAIN: bool>(
        &self,
        set: impl Fn(usize, &mut f64) + Sync,
        w: impl Fn(usize) -> f64 + Sync,
        out: &mut [f64],
    ) {
        let mut parts = Vec::with_capacity(self.tasks.len());
        let mut rest = out;
        for levels in &self.tasks {
            let start = self.rows.positions(levels.start).start;
            let end = self.rows.positions(levels.end - 1).end;
            let (head, tail) = std::mem::take(&mut rest).split_at_mut(end - start);
            parts.push((levels.clone(), start, head));
            rest = tail;
        }
        let width = self.eliminated.loadings.len();
        if PLAIN {
            parts.into_par_iter().for_each(|(levels, base, part)| {
                for l in levels {
                    let rows = self.rows.positions(l);
                    let v = &mut part[rows.start - base..rows.end - base];
                    let mut sum = 0.0;
                    for (x, i) in v.iter_mut().zip(rows.clone()) {
                        set(i, x);
                        sum += w(i) * *x;
                    }
                    let gram = self.gram[l];
                    let basis = WeightRun {
                        start: rows.start,
                        w: &w,
                        gram,
                    };
                    project(&basis, v, &mut [sum]);
                }
            });
            return;
        }
        parts.into_par_iter().for_each_init(
            || LevelScratch::new(width, self.max_level_rows),
            |scratch, (levels, base, part)| {
                for l in levels {
                    let positions = self.rows.positions(l);
                    let v = &mut part[positions.start - base..positions.end - base];
                    let (mut level, _) = scratch.level(v.len());
                    self.level_basis(l, positions, &w, &mut level, v, &set);
                    level.project(v);
                }
            },
        );
    }

    /// `(I − P_e) b`.
    pub(crate) fn project_rhs(&self, b: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; b.len()];
        with_shape!(self, |PLAIN, w| {
            self.project::<PLAIN>(|i, x| *x = b[i], w, &mut out);
            // One pass rounds away entries of heavy rows that light rows' weights make matter.
            self.project::<PLAIN>(|_, _| {}, w, &mut out);
        });
        out
    }

    /// `A_rᵀ v`, one accumulator per leaf summed in a fixed tree.
    fn adjoint<const PLAIN: bool>(&self, v: &[f64], w: impl Fn(usize) -> f64 + Sync) -> Vec<f64> {
        let width = self.eliminated.loadings.len();
        let leaf = |tasks: Range<usize>| {
            let mut acc = vec![0.0; self.n_kept];
            let mut scratch = LevelScratch::new(width, 0);
            let fit = scratch.level(0).0.fit;
            let levels = match tasks.len() {
                0 => 0..0,
                _ => self.tasks[tasks.start].start..self.tasks[tasks.end - 1].end,
            };
            if PLAIN {
                for l in levels {
                    let rows = self.rows.positions(l);
                    let g = self.gram;
                    let sum: f64 = rows.clone().map(|i| w(i) * v[i]).sum();
                    let mean = if g[l] > 0.0 { sum / g[l] } else { 0.0 };
                    for i in rows {
                        let projected = w(i) * (v[i] - w(i) * mean);
                        self.add_kept_row::<PLAIN>(&mut acc, i, projected);
                    }
                }
                return acc;
            }
            for l in levels {
                fit.fill(0.0);
                let rows = self.rows.positions(l);
                for i in rows.clone() {
                    self.accumulate(fit, i, w(i) * v[i]);
                }
                self.solve(l, fit);
                for i in rows {
                    let projected = w(i) * (v[i] - w(i) * self.fitted(fit, i));
                    self.add_kept_row::<PLAIN>(&mut acc, i, projected);
                }
            }
            acc
        };
        let add = |l: &mut Vec<f64>, r: &Vec<f64>| l.iter_mut().zip(r).for_each(|(l, r)| *l += r);
        let leaf = |mut k: Range<usize>| leaf(k.next().map_or(0..0, |k| self.leaves[k].clone()));
        fixed_tree(0..self.leaves.len(), 1, &leaf, &add)
    }

    /// The full coefficients: `x_k` plus the eliminated block fitted to `b − A_k x_k`.
    pub(crate) fn recover(&self, b: &[f64], x_k: &[f64]) -> Vec<f64> {
        let (n_levels, width) = (self.eliminated.n_levels, self.eliminated.loadings.len());
        let mut by_level = vec![0.0; n_levels * width];
        with_shape!(self, |PLAIN, w| {
            let levels = by_level.par_chunks_mut(width).enumerate().with_min_len(256);
            levels.for_each_init(
                || LevelScratch::new(width, self.max_level_rows),
                |scratch, (l, out)| {
                    let positions = self.rows.positions(l);
                    let (mut level, v) = scratch.level(positions.len());
                    let set = |i, x: &mut f64| *x = b[i] - w(i) * self.kept_row::<PLAIN>(x_k, i);
                    self.level_basis(l, positions, w, &mut level, v, set);
                    level.fit(v);
                    out.copy_from_slice(level.fit);
                },
            )
        });
        let dofs = self.prepared.design.terms[self.prepared.design.dominant].dofs();
        let mut x = Vec::with_capacity(self.prepared.design.n_dofs);
        x.extend_from_slice(&x_k[..dofs.start]);
        x.extend(column_major(&by_level, width));
        x.extend_from_slice(&x_k[dofs.start..]);
        x
    }

    /// `‖(I − P_e) a‖²` of column `c` of `columns`' `level`, summed row by row over the rows of
    /// `e_levels`; 0 when it lies within the rounding of its own evaluation.
    fn projected_energy(
        &self,
        columns: &Columns<'_>,
        level: usize,
        c: usize,
        e_levels: impl Iterator<Item = u32>,
    ) -> f64 {
        let sqrt_weights = self.prepared.sqrt_weights();
        let a = |i: usize| {
            if columns.levels[i] as usize == level {
                columns.loadings[c].map_or(1.0, |z| z[i])
            } else {
                0.0
            }
        };
        let width = self.eliminated.loadings.len();
        let (mut fit, mut fit_error) = (vec![0.0; width], vec![0.0; width]);
        // Running error bounds (N. J. Higham, *Accuracy and Stability of Numerical Algorithms*,
        // 2nd ed., SIAM, 2002, §3.3): a row whose weight is 0 rounds nothing.
        let (mut energy, mut bound) = (0.0, 0.0);
        let row_error = (width + 2) as f64 * f64::EPSILON;
        for e in e_levels.map(|e| e as usize) {
            fit.fill(0.0);
            fit_error.fill(0.0);
            let rows = self.rows.positions(e);
            for i in rows.clone() {
                let wa = row_weight(sqrt_weights, i) * a(i);
                for ((f, error), z) in fit
                    .iter_mut()
                    .zip(&mut fit_error)
                    .zip(&self.eliminated.loadings)
                {
                    let term = z.map_or(wa, |z| z[i] * wa);
                    if term != 0.0 {
                        *f += term;
                        *error += f.abs() + 2.0 * term.abs();
                    }
                }
            }
            self.solve(e, &mut fit);
            let n_levels = self.eliminated.n_levels;
            for (k, (error, f)) in fit_error.iter_mut().zip(&fit).enumerate() {
                let g = self.gram[k * n_levels + e];
                *error = if g > 0.0 {
                    f64::EPSILON * (*error / g + f.abs())
                } else {
                    0.0
                };
            }
            for i in rows {
                let s = sqrt_weights.map_or(1.0, |s| s[i]);
                let fitted = self.fitted(&fit, i);
                let r = s * (a(i) - fitted);
                energy += r * r;
                let spread = (fit_error.iter().zip(&self.eliminated.loadings))
                    .fold(0.0, |acc, (d, z)| acc + d * z.map_or(1.0, |z| z[i].abs()));
                let error = s * (row_error * (a(i).abs() + fitted.abs()) + spread);
                bound += error * error;
            }
        }
        if energy <= 4.0 * bound {
            0.0
        } else {
            energy
        }
    }

    /// `diag(A_rᵀ A_r)`: each kept column's squared norm less its projection onto `A_e`; 0 when
    /// that is rounding.
    pub(crate) fn gram_diagonal(&self) -> Vec<f64> {
        let (e_levels, e_n_levels) = (self.eliminated.levels, self.eliminated.n_levels);
        let sqrt_weights = self.prepared.sqrt_weights();
        let mut diagonal = Vec::with_capacity(self.n_kept);
        for (index, columns, _) in &self.kept {
            let (n_levels, width) = (columns.n_levels, columns.loadings.len());
            let gram = self.prepared.diagonal(*index);
            let rows = self.prepared.level_rows(*index);
            let mut by_level = vec![0.0; n_levels * width];
            // Each level owns its rows' sums, so no reduction depends on the thread count.
            by_level.par_chunks_mut(width).enumerate().for_each_init(
                Vec::new,
                |pairs: &mut Vec<(u32, usize)>, (level, entries)| {
                    pairs.clear();
                    rows.for_each_row(rows.positions(level), |i| pairs.push((e_levels[i], i)));
                    pairs.sort_unstable();
                    for (c, entry) in entries.iter_mut().enumerate() {
                        *entry = gram[c * n_levels + level];
                    }
                    for run in pairs.chunk_by(|a, b| a.0 == b.0) {
                        let e_level = run[0].0 as usize;
                        for (e_column, e_loading) in self.eliminated.loadings.iter().enumerate() {
                            let g = self.gram[e_column * e_n_levels + e_level];
                            if g <= 0.0 {
                                continue;
                            }
                            for (entry, loading) in entries.iter_mut().zip(&columns.loadings) {
                                let cross: f64 = run
                                    .iter()
                                    .map(|&(_, i)| {
                                        // Weight first: a zero weight kills a huge loading.
                                        let w = row_weight(sqrt_weights, i);
                                        let wz = loading.map_or(w, |z| w * z[i]);
                                        e_loading.map_or(wz, |z| wz * z[i])
                                    })
                                    .sum();
                                *entry -= cross / g * cross;
                            }
                        }
                    }
                    // The difference cancels below `n·ε` of the squared norm; there it is resummed.
                    let noise = 4.0 * pairs.len() as f64 * f64::EPSILON;
                    for (c, entry) in entries.iter_mut().enumerate() {
                        let norm = gram[c * n_levels + level];
                        if *entry <= noise * norm {
                            let e_levels = pairs.chunk_by(|a, b| a.0 == b.0).map(|r| r[0].0);
                            *entry = self.projected_energy(columns, level, c, e_levels);
                        }
                    }
                },
            );
            diagonal.extend(column_major(&by_level, width));
        }
        diagonal
    }
}

/// Level-major `by_level` (`width` entries per level) read column by column.
fn column_major(by_level: &[f64], width: usize) -> impl Iterator<Item = f64> + '_ {
    (0..width).flat_map(move |c| by_level.iter().skip(c).step_by(width).copied())
}

/// `0..n` cut into consecutive ranges whose `size`s sum to at least `min`, the last maybe less.
fn cut(n: usize, min: usize, size: impl Fn(usize) -> usize) -> Vec<Range<usize>> {
    let mut ranges = Vec::new();
    let (mut first, mut total) = (0, 0);
    for k in 0..n {
        total += size(k);
        if total >= min {
            ranges.push(first..k + 1);
            (first, total) = (k + 1, 0);
        }
    }
    if first < n {
        ranges.push(first..n);
    }
    ranges
}

/// Doubles in a cache line, 128 bytes on Apple silicon and 64 elsewhere.
const LINE: usize = 16;

/// One task's buffers for the eliminated term's levels, in one allocation padded by a cache line
/// at each end: small buffers of different threads that shared lines ran up to 2× slower.
struct LevelScratch {
    buf: Vec<f64>,
    width: usize,
}

impl LevelScratch {
    /// For levels of at most `rows` rows.
    fn new(width: usize, rows: usize) -> Self {
        Self {
            buf: vec![0.0; 2 * LINE + rows * (width + 1) + 3 * width],
            width,
        }
    }

    /// The buffers of a level of `len` rows, and room for its row values.
    fn level(&mut self, len: usize) -> (Level<'_>, &mut [f64]) {
        let width = self.width;
        let (entries, rest) = self.buf[LINE..].split_at_mut(len * width);
        let (values, rest) = rest.split_at_mut(len);
        let (norms, rest) = rest.split_at_mut(width);
        let (fit, rest) = rest.split_at_mut(width);
        let step = &mut rest[..width];
        let level = Level {
            entries,
            norms,
            fit,
            step,
        };
        (level, values)
    }
}

/// One level's basis, row-major, with its squared column norms and the fit of its values.
struct Level<'s> {
    entries: &'s mut [f64],
    norms: &'s mut [f64],
    fit: &'s mut [f64],
    step: &'s mut [f64],
}

impl Level<'_> {
    /// `v ← (I − P_l) v`.
    fn project(&mut self, v: &mut [f64]) {
        let basis = DenseBasis {
            entries: self.entries,
            norms: self.norms,
        };
        project(&basis, v, self.fit);
    }

    /// `fit` becomes the level's coefficients of `v`.
    fn fit(&mut self, v: &mut [f64]) {
        let basis = DenseBasis {
            entries: self.entries,
            norms: self.norms,
        };
        fit_twice(&basis, v, self.fit, self.step);
    }
}

/// A plain level: grouped rows from `start`, its one column `W^½` itself.
struct WeightRun<W> {
    start: usize,
    w: W,
    gram: f64,
}

impl<W: Fn(usize) -> f64> DiagonalGramBasis for WeightRun<W> {
    #[inline(always)]
    fn entry(&self, p: usize, _: usize) -> f64 {
        (self.w)(self.start + p)
    }

    #[inline(always)]
    fn norm2(&self, _: usize) -> f64 {
        self.gram
    }
}

impl Operator for ReducedDesign<'_, '_> {
    fn nrows(&self) -> usize {
        self.prepared.design.n_obs
    }

    fn ncols(&self) -> usize {
        self.n_kept
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), schwarz_precond::SolveError> {
        debug_assert_eq!(x.len(), self.n_kept);
        with_shape!(self, |PLAIN, w| {
            self.project::<PLAIN>(|i, y| *y = w(i) * self.kept_row::<PLAIN>(x, i), w, y)
        });
        Ok(())
    }

    fn apply_adjoint(&self, v: &[f64], y: &mut [f64]) -> Result<(), schwarz_precond::SolveError> {
        debug_assert_eq!(y.len(), self.n_kept);
        with_shape!(self, |PLAIN, w| y
            .copy_from_slice(&self.adjoint::<PLAIN>(v, w)));
        Ok(())
    }
}
