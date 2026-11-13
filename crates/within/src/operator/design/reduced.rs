//! The reduced system: the kept terms with the eliminated term's span projected out, one sweep
//! per product over the rows grouped by eliminated level.

use std::ops::Range;

use rayon::prelude::*;
use schwarz_precond::Operator;

use crate::domain::{
    fixed_tree, row_weight, LevelRows, PreparedDesign, PreparedTerm, MIN_LEAF_ROWS,
};
use crate::linalg::{add_moments, remove_fit, solve, DiagonalGramBasis};
use crate::BuildError;

/// Rows per parallel task, cut at an eliminated level's end so no level spans two tasks.
const TASK_ROWS: usize = 16_384;

/// One term's coefficient columns, read row by row.
struct TermRows<'p> {
    levels: &'p [u32],
    n_levels: usize,
    /// Per column, its loading; `None` is the intercept.
    loadings: Vec<Option<&'p [f64]>>,
}

impl<'p> TermRows<'p> {
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
    eliminated: TermRows<'p>,
    /// `diag(A_eᵀA_e)`, the whole Gram block of each level after whitening.
    gram: &'p [f64],
    /// The rows of each eliminated level, contiguous: the design is sorted by it.
    rows: &'p LevelRows,
    /// Consecutive eliminated levels per parallel task.
    tasks: Vec<Range<usize>>,
    /// Consecutive tasks per adjoint accumulator, fixed by the data alone.
    leaves: Vec<Range<usize>>,
    /// Kept terms: design index, columns, offset in the kept coefficients.
    kept: Vec<(usize, TermRows<'p>, usize)>,
    n_kept: usize,
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
            kept.push((t, TermRows::new(&term), n_kept));
            n_kept += term.term.n_dofs();
        }
        let tasks = cut(e.term.n_levels(), TASK_ROWS, |l| rows.positions(l).len());
        let task_rows =
            |t: usize| rows.positions(tasks[t].end - 1).end - rows.positions(tasks[t].start).start;
        // Merging costs ∝ n_kept per leaf, so a leaf spans at least 4·n_kept rows.
        let leaves = cut(tasks.len(), MIN_LEAF_ROWS.max(4 * n_kept), task_rows);
        let plain = (design.terms.iter()).all(|t| t.intercept && !t.has_slopes());
        let eliminated = TermRows::new(&e);
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
        })
    }

    pub(crate) fn prepared(&self) -> &'p PreparedDesign<'a> {
        self.prepared
    }

    /// `(D_k x)_i`.
    #[inline(always)]
    fn kept_row<const PLAIN: bool>(&self, x: &[f64], i: usize) -> f64 {
        let row = |(_, columns, offset): &(usize, TermRows<'_>, usize)| {
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
        let mut add = |(_, columns, offset): &(usize, TermRows<'_>, usize)| {
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

    /// Level `l`'s rows and its columns of `A_e`.
    #[inline(always)]
    fn level<const PLAIN: bool, W: Fn(usize) -> f64>(
        &self,
        l: usize,
        w: W,
    ) -> (Range<usize>, LevelBasis<'_, PLAIN, W>) {
        let rows = self.rows.positions(l);
        let basis = LevelBasis {
            start: rows.start,
            w,
            loadings: &self.eliminated.loadings,
            gram: self.gram,
            n_levels: self.eliminated.n_levels,
            level: l,
        };
        (rows, basis)
    }

    /// `k` moments per eliminated column, a cache line clear of other threads' buffers at each
    /// end: small buffers that shared a line ran up to 2× slower.
    fn moments(&self, k: usize) -> Vec<f64> {
        vec![0.0; 2 * LINE + k * self.eliminated.loadings.len()]
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
        parts.into_par_iter().for_each_init(
            || self.moments(1),
            |moments, (levels, base, part)| {
                // A plain level's one moment stays in a register; through the heap it cost 12%.
                let mut plain = [0.0];
                let fit = match PLAIN {
                    true => &mut plain[..],
                    false => &mut moments[LINE..][..width],
                };
                for l in levels {
                    let (rows, basis) = self.level::<PLAIN, _>(l, &w);
                    let v = &mut part[rows.start - base..rows.end - base];
                    fit.fill(0.0);
                    for (p, x) in v.iter_mut().enumerate() {
                        set(rows.start + p, x);
                        add_moments(&basis, p, *x, fit);
                    }
                    solve(&basis, fit);
                    for (p, x) in v.iter_mut().enumerate() {
                        *x = remove_fit(&basis, p, *x, fit);
                    }
                }
            },
        );
    }

    /// `(I − P_e) b`.
    pub(crate) fn project_rhs(&self, b: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; b.len()];
        with_shape!(self, |PLAIN, w| {
            self.project::<PLAIN>(|i, x| *x = b[i], w, &mut out)
        });
        out
    }

    /// `A_rᵀ v`, one accumulator per leaf summed in a fixed tree.
    fn adjoint<const PLAIN: bool>(&self, v: &[f64], w: impl Fn(usize) -> f64 + Sync) -> Vec<f64> {
        let width = self.eliminated.loadings.len();
        let leaf = |tasks: Range<usize>| {
            let mut acc = vec![0.0; self.n_kept];
            let mut moments = self.moments(1);
            let mut plain = [0.0];
            let fit = match PLAIN {
                true => &mut plain[..],
                false => &mut moments[LINE..][..width],
            };
            let levels = match tasks.len() {
                0 => 0..0,
                _ => self.tasks[tasks.start].start..self.tasks[tasks.end - 1].end,
            };
            for l in levels {
                let (rows, basis) = self.level::<PLAIN, _>(l, &w);
                let u = &v[rows.clone()];
                fit.fill(0.0);
                for (p, &x) in u.iter().enumerate() {
                    add_moments(&basis, p, x, fit);
                }
                solve(&basis, fit);
                for (p, &x) in u.iter().enumerate() {
                    let i = rows.start + p;
                    let projected = w(i) * remove_fit(&basis, p, x, fit);
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
                || (self.moments(1), Vec::new()),
                |(moments, v), (l, out)| {
                    let fit = &mut moments[LINE..][..width];
                    let (rows, basis) = self.level::<PLAIN, _>(l, w);
                    v.clear();
                    v.extend(rows.map(|i| b[i] - w(i) * self.kept_row::<PLAIN>(x_k, i)));
                    fit.fill(0.0);
                    for (p, &x) in v.iter().enumerate() {
                        add_moments(&basis, p, x, fit);
                    }
                    solve(&basis, fit);
                    out.copy_from_slice(fit);
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
                    // The difference cancels below `n·ε` of the squared norm.
                    let noise = 4.0 * pairs.len() as f64 * f64::EPSILON;
                    for (c, entry) in entries.iter_mut().enumerate() {
                        if *entry <= noise * gram[c * n_levels + level] {
                            *entry = 0.0;
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

/// One eliminated level's columns `W^½ D_e,l`, row `p` of them at design row `start + p`.
struct LevelBasis<'r, const PLAIN: bool, W> {
    start: usize,
    w: W,
    loadings: &'r [Option<&'r [f64]>],
    /// `diag(A_eᵀA_e)`, column-major over the term's `n_levels`.
    gram: &'r [f64],
    n_levels: usize,
    level: usize,
}

impl<const PLAIN: bool, W: Fn(usize) -> f64> DiagonalGramBasis for LevelBasis<'_, PLAIN, W> {
    #[inline(always)]
    fn entry(&self, p: usize, c: usize) -> f64 {
        let i = self.start + p;
        let w = (self.w)(i);
        if PLAIN {
            return w;
        }
        self.loadings[c].map_or(w, |z| w * z[i])
    }

    #[inline(always)]
    fn norm2(&self, c: usize) -> f64 {
        self.gram[c * self.n_levels + self.level]
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
