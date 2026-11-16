//! The reduced system: the kept terms with the eliminated term's span projected out, one sweep
//! per product over the rows grouped by eliminated level.

use std::borrow::Cow;
use std::ops::Range;

use rayon::prelude::*;
use schwarz_precond::Operator;

use crate::domain::{
    cut, fixed_tree, row_weight, Design, LevelRows, PreparedDesign, PreparedTerm, TermRows,
    Unknowns, MIN_LEAF_ROWS,
};
use crate::linalg::{add_moments, fit_twice, remove_fit, solve, solve_rescued, DiagonalGramBasis};
use crate::BuildError;

/// Rows per parallel task, cut at an eliminated level's end so no level spans two tasks.
const TASK_ROWS: usize = 16_384;

/// `A_r = (I − P_e) A_k`, where `P_e` projects onto the eliminated term's columns `A_e`.
///
/// The kept coefficients are the full ones without the eliminated term's block.
pub(crate) struct ReducedDesign<'a> {
    prepared: PreparedDesign<'a>,
    /// Consecutive eliminated levels per parallel task.
    tasks: Vec<Range<usize>>,
    /// Consecutive tasks per adjoint accumulator, fixed by the data alone.
    leaves: Vec<Range<usize>>,
    /// Every term is a lone intercept: the plain kernels.
    plain: bool,
}

/// `A_e` and `A_k` borrowed for one product.
struct ProjectedColumns<'r> {
    system: &'r ReducedDesign<'r>,
    eliminated: TermRows<'r>,
    /// `diag(A_eᵀA_e)`, the whole Gram block of each level after whitening.
    gram: &'r [f64],
    /// The rows of each eliminated level, contiguous: the design is sorted by it.
    rows: Cow<'r, LevelRows>,
    /// Per kept term, its columns and its block start in the kept coefficients.
    kept: Vec<(TermRows<'r>, usize)>,
}

/// Runs `$body` with `$plain` and the row weight `$w` fixed at compile time.
macro_rules! with_shape {
    ($self:ident, |$plain:ident, $w:ident| $body:expr) => {
        match ($self.system.plain, $self.system.prepared.sqrt_weights()) {
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

impl<'a> ReducedDesign<'a> {
    /// Refuses a design not sorted by its dominant term.
    pub(crate) fn new(design: Design<'a>, weights: Option<&[f64]>) -> Result<Self, BuildError> {
        let dominant = design.dominant;
        if !design.terms[dominant].sorted() {
            return Err(BuildError::UnsortedReducedSystem);
        }
        let prepared = PreparedDesign::new(design, weights)?;
        let design = &prepared.design;
        let rows = prepared.level_rows(dominant);
        let tasks = cut(design.terms[dominant].n_levels(), TASK_ROWS, |l| {
            rows.positions(l).len()
        });
        let task_rows =
            |t: usize| rows.positions(tasks[t].end - 1).end - rows.positions(tasks[t].start).start;
        // Merging costs ∝ the kept coefficients per leaf: 4× as many rows, capped at 8 leaves' worth.
        let n_kept = design.n_unknowns(Unknowns::WithoutDominant);
        let leaf_rows = (4 * n_kept).min(design.n_obs.div_ceil(8));
        let leaves = cut(tasks.len(), MIN_LEAF_ROWS.max(leaf_rows), task_rows);
        let plain = (design.terms.iter()).all(|t| t.intercept && !t.has_slopes());
        // Filled here: left to the first product, a solve's fan-out would wait on it.
        (0..design.n_factors()).into_par_iter().for_each(|t| {
            prepared.diagonal(t);
        });
        Ok(Self {
            prepared,
            tasks,
            leaves,
            plain,
        })
    }

    pub(crate) fn prepared(&self) -> &PreparedDesign<'a> {
        &self.prepared
    }

    fn columns(&self) -> ProjectedColumns<'_> {
        let design = &self.prepared.design;
        let kept = (0..design.n_factors()).filter_map(|t| {
            let start = design.block_start(Unknowns::WithoutDominant, t)?;
            Some((TermRows::new(&self.prepared.term(t)), start))
        });
        ProjectedColumns {
            system: self,
            eliminated: TermRows::new(&self.prepared.term(design.dominant)),
            gram: self.prepared.diagonal(design.dominant),
            rows: self.prepared.level_rows(design.dominant),
            kept: kept.collect(),
        }
    }

    /// `(I − P_e) b`.
    pub(crate) fn project_rhs(&self, b: &[f64]) -> Vec<f64> {
        let columns = self.columns();
        let mut out = vec![0.0; b.len()];
        with_shape!(columns, |PLAIN, w| {
            columns.project::<PLAIN>(|i, x| *x = b[i], w, &mut out);
            // One pass rounds away entries of heavy rows that light rows' weights make matter.
            columns.project::<PLAIN>(|_, _| {}, w, &mut out);
        });
        out
    }

    /// The full coefficients: `x_k` plus the eliminated block fitted to `b − A_k x_k`.
    pub(crate) fn recover(&self, b: &[f64], x_k: &[f64]) -> Vec<f64> {
        let columns = self.columns();
        let (n_levels, width) = (columns.eliminated.n_levels, columns.eliminated.width());
        let mut by_level = vec![0.0; n_levels * width];
        with_shape!(columns, |PLAIN, w| {
            let levels = by_level.par_chunks_mut(width).enumerate().with_min_len(256);
            levels.for_each_init(
                || (columns.moments(2), Vec::new()),
                |(moments, v), (l, out)| {
                    let (fit, step) = moments[LINE..][..2 * width].split_at_mut(width);
                    let (rows, basis) = columns.level::<PLAIN, _>(l, w);
                    v.clear();
                    fit.fill(0.0);
                    for (p, i) in rows.enumerate() {
                        let x = b[i] - w(i) * columns.kept_row::<PLAIN>(x_k, i);
                        v.push(x);
                        add_moments(&basis, p, x, fit);
                    }
                    fit_twice(&basis, v, fit, step);
                    out.copy_from_slice(fit);
                },
            )
        });
        let design = &self.prepared.design;
        let dofs = design.terms[design.dominant].dofs();
        let mut x = Vec::with_capacity(design.n_dofs);
        x.extend_from_slice(&x_k[..dofs.start]);
        x.extend(column_major(&by_level, width));
        x.extend_from_slice(&x_k[dofs.start..]);
        x
    }

    /// `‖(I − P_e) a‖²` of column `c` of `kept`'s `level`, summed row by row over the rows of
    /// `e_levels`; 0 when it lies within the rounding of its own evaluation.
    fn projected_energy(
        &self,
        kept: &PreparedTerm<'_>,
        level: usize,
        c: usize,
        e_levels: impl Iterator<Item = u32>,
    ) -> f64 {
        let dominant = self.prepared.design.dominant;
        let eliminated = self.prepared.term(dominant);
        let (gram, e_rows) = (
            self.prepared.diagonal(dominant),
            self.prepared.level_rows(dominant),
        );
        let sqrt_weights = self.prepared.sqrt_weights();
        let (levels, loading) = (kept.term.levels(), kept.loading(c));
        let a = |i: usize| {
            if levels[i] as usize == level {
                loading.map_or(1.0, |z| z[i])
            } else {
                0.0
            }
        };
        let width = eliminated.term.n_columns();
        let e_loadings: Vec<Option<&[f64]>> = (0..width).map(|k| eliminated.loading(k)).collect();
        let (mut fit, mut fit_error) = (vec![0.0; width], vec![0.0; width]);
        // Running error bounds (Higham, ASNA 2nd ed., §3.3); a row with weight 0 rounds nothing.
        let (mut energy, mut bound) = (0.0, 0.0);
        let row_error = (width + 2) as f64 * f64::EPSILON;
        for e in e_levels.map(|e| e as usize) {
            fit.fill(0.0);
            fit_error.fill(0.0);
            let rows = e_rows.positions(e);
            for i in rows.clone() {
                let wa = row_weight(sqrt_weights, i) * a(i);
                for ((f, error), z) in fit.iter_mut().zip(&mut fit_error).zip(&e_loadings) {
                    let term = z.map_or(wa, |z| z[i] * wa);
                    if term != 0.0 {
                        *f += term;
                        *error += f.abs() + 2.0 * term.abs();
                    }
                }
            }
            let n_levels = eliminated.term.n_levels();
            for (k, (error, f)) in fit_error.iter_mut().zip(&mut fit).enumerate() {
                let g = gram[k * n_levels + e];
                (*f, *error) = if g > 0.0 {
                    let fit = *f / g;
                    (fit, f64::EPSILON * (*error / g + fit.abs()))
                } else {
                    (0.0, 0.0)
                };
            }
            for i in rows {
                let s = sqrt_weights.map_or(1.0, |s| s[i]);
                let fitted = (fit.iter().zip(&e_loadings))
                    .fold(-0.0, |acc, (f, z)| acc + z.map_or(*f, |z| f * z[i]));
                let r = s * (a(i) - fitted);
                energy += r * r;
                let spread = (fit_error.iter().zip(&e_loadings))
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
        let design = &self.prepared.design;
        let dominant = design.dominant;
        let eliminated = self.prepared.term(dominant);
        let (e_n_levels, e_width) = (eliminated.term.n_levels(), eliminated.term.n_columns());
        let e_loadings: Vec<Option<&[f64]>> = (0..e_width).map(|k| eliminated.loading(k)).collect();
        let e_gram = self.prepared.diagonal(dominant);
        let e_rows = self.prepared.level_rows(dominant);
        let sqrt_weights = self.prepared.sqrt_weights();
        // (term, its block start in the kept coefficients, its first level among all kept levels)
        let mut kept: Vec<(PreparedTerm<'_>, usize, usize)> = Vec::new();
        let mut n_kept_levels = 0;
        for t in 0..design.n_factors() {
            if let Some(start) = design.block_start(Unknowns::WithoutDominant, t) {
                let term = self.prepared.term(t);
                let n_levels = term.term.n_levels();
                kept.push((term, start, n_kept_levels));
                n_kept_levels += n_levels;
            }
        }
        let grams: Vec<&[f64]> = (0..design.n_factors())
            .filter(|&t| t != dominant)
            .map(|t| self.prepared.diagonal(t))
            .collect();
        let n_kept = design.n_unknowns(Unknowns::WithoutDominant);
        let kept_loadings: Vec<Vec<Option<&[f64]>>> = (kept.iter())
            .map(|(t, ..)| (0..t.term.n_columns()).map(|c| t.loading(c)).collect())
            .collect();
        let max_levels = kept
            .iter()
            .map(|(t, ..)| t.term.n_levels())
            .max()
            .unwrap_or(0);
        let tasks = &self.tasks;
        // Per kept coefficient `Σ_e cross²/g` over the eliminated levels of one leaf, and per kept
        // level its row count; each level's cross sums its rows in ascending order.
        let leaf = |leaf: Range<usize>| {
            let mut removed = vec![0.0; n_kept];
            let mut counts = vec![0usize; n_kept_levels];
            // `cross` is compact: one `width × e_width` block per kept level the run touches.
            let (mut cross, mut touched, mut rows_of) = (Vec::new(), Vec::new(), Vec::new());
            let mut slot = vec![u32::MAX; max_levels];
            for e in tasks[leaf.start].start..tasks[leaf.end - 1].end {
                let rows = e_rows.positions(e);
                for ((term, start, first), loadings) in kept.iter().zip(&kept_loadings) {
                    let (levels, n_levels) = (term.term.levels(), term.term.n_levels());
                    let width = loadings.len();
                    let block = width * e_width;
                    touched.clear();
                    rows_of.clear();
                    cross.clear();
                    for i in rows.clone() {
                        let l = levels[i] as usize;
                        if slot[l] == u32::MAX {
                            slot[l] = touched.len() as u32;
                            touched.push(l);
                            rows_of.push(0);
                            cross.resize(cross.len() + block, 0.0);
                        }
                        let p = slot[l] as usize;
                        rows_of[p] += 1;
                        let at = p * block;
                        // Weight first: a zero weight kills a huge loading.
                        let w = row_weight(sqrt_weights, i);
                        for (c, loading) in loadings.iter().enumerate() {
                            let wz = loading.map_or(w, |z| w * z[i]);
                            for (k, e_loading) in e_loadings.iter().enumerate() {
                                cross[at + c * e_width + k] += e_loading.map_or(wz, |z| wz * z[i]);
                            }
                        }
                    }
                    for (p, &l) in touched.iter().enumerate() {
                        slot[l] = u32::MAX;
                        counts[first + l] += rows_of[p];
                        for k in 0..e_width {
                            let g = e_gram[k * e_n_levels + e];
                            for c in 0..width {
                                let x = cross[p * block + c * e_width + k];
                                if g > 0.0 {
                                    removed[start + c * n_levels + l] += x / g * x;
                                }
                            }
                        }
                    }
                }
            }
            (removed, counts)
        };
        let merge = |(l, lc): &mut (Vec<f64>, Vec<usize>), (r, rc): &(Vec<f64>, Vec<usize>)| {
            l.iter_mut().zip(r).for_each(|(l, r)| *l += r);
            lc.iter_mut().zip(rc).for_each(|(l, r)| *l += r);
        };
        let leaves = &self.leaves;
        let (removed, counts) = fixed_tree(
            0..leaves.len(),
            1,
            &|k: Range<usize>| leaf(leaves[k.start].clone()),
            &merge,
        );
        let mut diagonal = vec![0.0; n_kept];
        let mut flagged: Vec<(usize, usize, usize)> = Vec::new();
        for (k, ((term, start, first), gram)) in kept.iter().zip(&grams).enumerate() {
            let n_levels = term.term.n_levels();
            for c in 0..term.term.n_columns() {
                for l in 0..n_levels {
                    let (at, norm) = (start + c * n_levels + l, gram[c * n_levels + l]);
                    let entry = norm - removed[at];
                    // The difference cancels below `n·ε` of the squared norm; there it is resummed.
                    let noise = 4.0 * counts[first + l] as f64 * f64::EPSILON;
                    match entry <= noise * norm {
                        true => flagged.push((k, l, c)),
                        false => diagonal[at] = entry,
                    }
                }
            }
        }
        if flagged.is_empty() {
            return diagonal;
        }
        let mut wanted = vec![false; n_kept_levels];
        for &(k, l, _) in &flagged {
            wanted[kept[k].2 + l] = true;
        }
        // The eliminated levels each flagged kept level meets, ascending.
        let pairs: Vec<Vec<(usize, u32)>> = tasks
            .par_iter()
            .map(|task| {
                let (mut pairs, mut met) = (Vec::new(), Vec::new());
                for e in task.clone() {
                    met.clear();
                    for i in e_rows.positions(e) {
                        for (term, _, first) in &kept {
                            let at = first + term.term.levels()[i] as usize;
                            if wanted[at] {
                                met.push(at);
                            }
                        }
                    }
                    met.sort_unstable();
                    met.dedup();
                    pairs.extend(met.iter().map(|&at| (at, e as u32)));
                }
                pairs
            })
            .collect();
        let mut met: Vec<(usize, u32)> = pairs.concat();
        met.par_sort_by_key(|&(at, _)| at);
        let entries: Vec<f64> = flagged
            .par_iter()
            .map(|&(k, l, c)| {
                let at = kept[k].2 + l;
                let lo = met.partition_point(|p| p.0 < at);
                let hi = met.partition_point(|p| p.0 <= at);
                let e_levels = met[lo..hi].iter().map(|&(_, e)| e);
                self.projected_energy(&kept[k].0, l, c, e_levels)
            })
            .collect();
        for (&(k, l, c), entry) in flagged.iter().zip(entries) {
            let (term, start, _) = &kept[k];
            diagonal[start + c * term.term.n_levels() + l] = entry;
        }
        diagonal
    }
}

impl ProjectedColumns<'_> {
    /// `(D_k x)_i`.
    #[inline(always)]
    fn kept_row<const PLAIN: bool>(&self, x: &[f64], i: usize) -> f64 {
        let row = |(columns, offset): &(TermRows<'_>, usize)| columns.row::<PLAIN>(x, *offset, i);
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
        let mut add = |(columns, offset): &(TermRows<'_>, usize)| {
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
            intercept: self.eliminated.intercept,
            slopes: self.eliminated.slopes,
            gram: self.gram,
            n_levels: self.eliminated.n_levels,
            level: l,
        };
        (rows, basis)
    }

    /// `k` moments per eliminated column, a cache line clear of other threads' buffers at each
    /// end: small buffers that shared a line ran up to 2× slower.
    fn moments(&self, k: usize) -> Vec<f64> {
        vec![0.0; 2 * LINE + k * self.eliminated.width()]
    }

    /// `out ← (I − P_e) out` in one pass, `set` first writing each row's value; `w = W^½`.
    fn project<const PLAIN: bool>(
        &self,
        set: impl Fn(usize, &mut f64) + Sync,
        w: impl Fn(usize) -> f64 + Sync,
        out: &mut [f64],
    ) {
        let mut parts = Vec::with_capacity(self.system.tasks.len());
        let mut rest = out;
        for levels in &self.system.tasks {
            let start = self.rows.positions(levels.start).start;
            let end = self.rows.positions(levels.end - 1).end;
            let (head, tail) = std::mem::take(&mut rest).split_at_mut(end - start);
            parts.push((levels.clone(), start, head));
            rest = tail;
        }
        let width = self.eliminated.width();
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
                    solve_rescued(&basis, v, fit);
                    for (p, x) in v.iter_mut().enumerate() {
                        *x = remove_fit(&basis, p, *x, fit);
                    }
                }
            },
        );
    }

    /// `A_rᵀ v`, one accumulator per leaf summed in a fixed tree.
    fn adjoint<const PLAIN: bool>(&self, v: &[f64], w: impl Fn(usize) -> f64 + Sync) -> Vec<f64> {
        let width = self.eliminated.width();
        let ReducedDesign {
            prepared,
            tasks: all,
            leaves,
            ..
        } = self.system;
        let n_kept = prepared.design.n_unknowns(Unknowns::WithoutDominant);
        let leaf = |tasks: Range<usize>| {
            let mut acc = vec![0.0; n_kept];
            let mut moments = self.moments(1);
            let mut plain = [0.0];
            let fit = match PLAIN {
                true => &mut plain[..],
                false => &mut moments[LINE..][..width],
            };
            for l in all[tasks.start].start..all[tasks.end - 1].end {
                let (rows, basis) = self.level::<PLAIN, _>(l, &w);
                let u = &v[rows.clone()];
                fit.fill(0.0);
                for (p, &x) in u.iter().enumerate() {
                    add_moments(&basis, p, x, fit);
                }
                // Unrescued: the check costs a 3FE product 7%; LSMR keeps ‖u‖ within 1e±64.
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
        let leaf = |k: Range<usize>| leaf(leaves[k.start].clone());
        fixed_tree(0..leaves.len(), 1, &leaf, &add)
    }
}

/// Level-major `by_level` (`width` entries per level) read column by column.
fn column_major(by_level: &[f64], width: usize) -> impl Iterator<Item = f64> + '_ {
    (0..width).flat_map(move |c| by_level.iter().skip(c).step_by(width).copied())
}

/// Doubles in a cache line, 128 bytes on Apple silicon and 64 elsewhere.
const LINE: usize = 16;

/// One eliminated level's columns `W^½ D_e,l`, row `p` of them at design row `start + p`.
struct LevelBasis<'r, const PLAIN: bool, W> {
    start: usize,
    w: W,
    intercept: bool,
    slopes: &'r [Vec<f64>],
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
        match c.checked_sub(self.intercept as usize) {
            None => w,
            Some(j) => w * self.slopes[j][i],
        }
    }

    #[inline(always)]
    fn norm2(&self, c: usize) -> f64 {
        self.gram[c * self.n_levels + self.level]
    }
}

impl Operator for ReducedDesign<'_> {
    fn nrows(&self) -> usize {
        self.prepared.design.n_obs
    }

    fn ncols(&self) -> usize {
        self.prepared.design.n_unknowns(Unknowns::WithoutDominant)
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), schwarz_precond::SolveError> {
        debug_assert_eq!(x.len(), self.ncols());
        let columns = self.columns();
        with_shape!(columns, |PLAIN, w| {
            columns.project::<PLAIN>(|i, y| *y = w(i) * columns.kept_row::<PLAIN>(x, i), w, y)
        });
        Ok(())
    }

    fn apply_adjoint(&self, v: &[f64], y: &mut [f64]) -> Result<(), schwarz_precond::SolveError> {
        debug_assert_eq!(y.len(), self.ncols());
        let columns = self.columns();
        with_shape!(columns, |PLAIN, w| y
            .copy_from_slice(&columns.adjoint::<PLAIN>(v, w)));
        Ok(())
    }
}
