//! The reduced system: the kept terms with the eliminated term's span projected out, one sweep
//! per product over the rows grouped by eliminated level.

use std::borrow::Cow;
use std::ops::Range;

use rayon::prelude::*;
use schwarz_precond::{MemBuffer, MemStack, Operator, StackReq, Threads};

use crate::domain::{
    cut, fixed_tree, row_weight, tree_mid, Design, LevelRows, PreparedDesign, TermRows,
    MIN_LEAF_ROWS,
};
use crate::linalg::{
    add_moments, fit_twice, remove_fit, solve, solve_rescued, DiagonalGramBasis, GramBasisWorkspace,
};
use crate::BuildError;

/// Rows per parallel task, cut at an eliminated level's end so no level spans two tasks.
const TASK_ROWS: usize = 16_384;

/// `A_r = (I − P_e) A_k`, where `P_e` projects onto the eliminated term's columns `A_e`, with
/// every column whose projection is rounding off the span of its level's larger ones set to 0.
///
/// The kept coefficients are the full ones without the eliminated term's block.
pub(crate) struct ReducedDesign<'a> {
    prepared: PreparedDesign<'a>,
    /// Consecutive eliminated levels per parallel task.
    tasks: Vec<Range<usize>>,
    /// Consecutive tasks per adjoint accumulator, fixed by the data alone.
    leaves: Vec<Range<usize>>,
    /// `projected_gram`, computed once.
    diagonal: Vec<f64>,
    /// The kept coefficients whose diagonal is 0: the products treat their columns as zero.
    vanished: Vec<usize>,
}

/// `A_e` and `A_k` borrowed for one product.
struct ProjectedColumns<'r> {
    system: &'r ReducedDesign<'r>,
    eliminated: TermRows<'r>,
    /// `diag(A_eᵀA_e)`, the whole Gram block of each level after whitening.
    gram: &'r [f64],
    /// The rows of each eliminated level, contiguous: the design is sorted by it.
    rows: &'r LevelRows,
    kept: Box<Kept<'r>>,
}

/// Runs `$body` with the eliminated term's width `$width` (an intercept and its slopes; 0: read
/// at run time), whether the kept terms are lone intercepts `$kp` when asked, and the row weight
/// `$w` fixed at compile time; `$kept` holds one or two kept terms by value, more behind a slice.
macro_rules! with_shape {
    ($columns:ident, |$width:ident, $kp:ident, $kept:ident, $w:ident| $body:expr) => {
        // A loop over a slice costs a 3FE product a fifth of its time.
        match $columns.kept[..] {
            [a] => with_shape!(@plain $columns, $width, $kp, ($kept = [a]), $w, $body),
            [a, b] => with_shape!(@plain $columns, $width, $kp, ($kept = [a, b]), $w, $body),
            ref kept => with_shape!(@plain $columns, $width, $kp, ($kept = kept), $w, $body),
        }
    };
    (@plain $columns:ident, $width:ident, $kp:ident, ($kept:ident = $kv:expr),
     $w:ident, $body:expr) => {{
        let $kept = $kv;
        match ($columns.kept.iter()).all(|(t, _)| t.intercept && t.slopes.is_empty()) {
            true => with_shape!(@width $columns, $width, ($kp = true), $w, $body),
            false => with_shape!(@width $columns, $width, ($kp = false), $w, $body),
        }
    }};
    ($columns:ident, |$width:ident, $w:ident| $body:expr) => {
        with_shape!(@width $columns, $width, (), $w, $body)
    };
    (@width $columns:ident, $width:ident, ($($kp:ident = $kv:expr)?), $w:ident,
     $body:expr) => {
        match ($columns.eliminated.intercept, $columns.eliminated.slopes.len()) {
            (true, 0) => {
                with_shape!(@arm $columns, ($width = 1 $(, $kp = $kv)?), $w, $body)
            }
            (true, 1) => {
                with_shape!(@arm $columns, ($width = 2 $(, $kp = $kv)?), $w, $body)
            }
            (true, 2) => {
                with_shape!(@arm $columns, ($width = 3 $(, $kp = $kv)?), $w, $body)
            }
            _ => with_shape!(@arm $columns, ($width = 0 $(, $kp = $kv)?), $w, $body),
        }
    };
    (@arm $columns:ident,
     ($width:ident = $wv:expr $(, $kp:ident = $kv:expr)?),
     $w:ident, $body:expr) => {{
        const $width: usize = $wv;
        $(const $kp: bool = $kv;)?
        match $columns.system.prepared.sqrt_weights() {
            None => {
                let $w = |_: usize| 1.0;
                $body
            }
            Some(sw) => {
                let $w = move |i: usize| sw[i];
                $body
            }
        }
    }};
}

impl<'a> ReducedDesign<'a> {
    pub(crate) fn new(design: Design<'a>, weights: Option<&[f64]>) -> Result<Self, BuildError> {
        let dominant = design.dominant;
        let prepared = PreparedDesign::new(design, weights)?;
        let design = &prepared.design;
        let rows = prepared.eliminated_rows();
        let tasks = cut(design.terms[dominant].n_levels(), TASK_ROWS, |l| {
            rows.positions(l).len()
        });
        let task_rows =
            |t: usize| rows.positions(tasks[t].end - 1).end - rows.positions(tasks[t].start).start;
        // Merging costs ∝ the kept coefficients per leaf: 4× as many rows, capped at 8 leaves' worth.
        let n_kept = design.n_unknowns();
        let leaf_rows = (4 * n_kept).min(design.n_obs.div_ceil(8));
        let leaves = cut(tasks.len(), MIN_LEAF_ROWS.max(leaf_rows), task_rows);
        let mut system = Self {
            prepared,
            tasks,
            leaves,
            diagonal: Vec::new(),
            vanished: Vec::new(),
        };
        system.diagonal = system.projected_gram();
        system.vanished = (system.diagonal.iter().enumerate())
            .filter_map(|(j, &d)| (d == 0.0).then_some(j))
            .collect();
        Ok(system)
    }

    /// `diag(A_rᵀ A_r)`.
    pub(crate) fn gram_diagonal(&self) -> Vec<f64> {
        self.diagonal.clone()
    }

    /// `x` with the vanished coefficients at 0, borrowed when there are none.
    fn without_vanished<'x>(&self, x: &'x [f64]) -> Cow<'x, [f64]> {
        if self.vanished.is_empty() {
            return Cow::Borrowed(x);
        }
        let mut x = x.to_vec();
        self.zero_vanished(&mut x);
        Cow::Owned(x)
    }

    /// How much of an `n`-long `x` an apply copies: all of it when some coefficients vanished.
    fn copied(&self, n: usize) -> usize {
        if self.vanished.is_empty() {
            0
        } else {
            n
        }
    }

    fn zero_vanished(&self, v: &mut [f64]) {
        self.vanished.iter().for_each(|&j| v[j] = 0.0);
    }

    pub(crate) fn prepared(&self) -> &PreparedDesign<'a> {
        &self.prepared
    }

    fn columns(&self) -> ProjectedColumns<'_> {
        let design = &self.prepared.design;
        let kept: Box<Kept<'_>> = (0..design.n_factors())
            .filter_map(|t| {
                let start = design.block_start(t)?;
                Some((TermRows::new(&self.prepared.term(t)), start))
            })
            .collect();
        ProjectedColumns {
            system: self,
            eliminated: TermRows::new(&self.prepared.term(design.dominant)),
            gram: self.prepared.diagonal(design.dominant),
            rows: self.prepared.eliminated_rows(),
            kept,
        }
    }

    /// `(I − P_e) b`, with `b` read row by row in internal order.
    pub(crate) fn project_rhs(&self, b: impl Fn(usize) -> f64 + Sync) -> Vec<f64> {
        let columns = self.columns();
        let mut out = vec![0.0; self.prepared.design.n_obs];
        let mut scratch = MemBuffer::new(columns.project_scratch(true));
        let stack = MemStack::new(&mut scratch);
        with_shape!(columns, |W, w| {
            columns.project::<W>(|i, x| *x = b(i), w, &mut out, Threads::All, stack);
            // One pass rounds away entries of heavy rows that light rows' weights make matter.
            columns.project::<W>(|_, _| {}, w, &mut out, Threads::All, stack);
        });
        out
    }

    /// The full coefficients: `x_k` plus the eliminated block fitted to `b − A_k x_k`.
    pub(crate) fn recover(&self, b: impl Fn(usize) -> f64 + Sync, x_k: &[f64]) -> Vec<f64> {
        let x_k = &*self.without_vanished(x_k);
        let columns = self.columns();
        let (n_levels, width) = (columns.eliminated.n_levels, columns.eliminated.width());
        let mut by_level = vec![0.0; n_levels * width];
        // Cold: the general kernels, bitwise the same as the plain ones.
        with_shape!(columns, |W, w| {
            let levels = by_level.par_chunks_mut(width).enumerate().with_min_len(256);
            levels.for_each_init(
                || (vec![0.0; 2 * LINE + 2 * width], Vec::new()),
                |(moments, v), (l, out)| {
                    let (fit, step) = moments[LINE..][..2 * width].split_at_mut(width);
                    let (rows, basis) = columns.level::<W, _>(l, w);
                    v.clear();
                    fit.fill(0.0);
                    for (p, i) in rows.enumerate() {
                        let x = b(i) - w(i) * kept_row::<false>(&columns.kept, x_k, i);
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

    /// `diag(A_rᵀ A_r)`: each kept column's squared norm less its projection onto `A_e`; 0 when
    /// what is left is rounding off the span of its level's larger columns.
    fn projected_gram(&self) -> Vec<f64> {
        let columns = self.columns();
        let design = &self.prepared.design;
        let (eliminated, e_gram) = (&columns.eliminated, columns.gram);
        let (e_n_levels, e_width) = (eliminated.n_levels, eliminated.width());
        let sqrt_weights = self.prepared.sqrt_weights();
        let mut n_kept_levels = 0;
        let kept: Vec<KeptTerm<'_>> = (0..design.n_factors())
            .filter_map(|t| {
                let start = design.block_start(t)?;
                let rows = TermRows::new(&self.prepared.term(t));
                let first = n_kept_levels;
                n_kept_levels += rows.n_levels;
                let gram = self.prepared.diagonal(t);
                Some(KeptTerm {
                    rows,
                    start,
                    first,
                    gram,
                })
            })
            .collect();
        let n_kept = design.n_unknowns();
        let max_levels = kept.iter().map(|t| t.rows.n_levels).max().unwrap_or(0);
        let tasks = &self.tasks;
        let leaf = |leaf: Range<usize>| {
            let mut removed = vec![0.0; n_kept];
            let mut counts = vec![0usize; n_kept_levels];
            // One `width × e_width` block per kept level the run touches, not per kept level.
            let (mut cross, mut touched) = (Vec::new(), Vec::new());
            let mut slot = vec![u32::MAX; max_levels];
            for e in tasks[leaf.start].start..tasks[leaf.end - 1].end {
                let rows = columns.rows.positions(e);
                for term in &kept {
                    let (levels, n_levels) = (term.rows.levels, term.rows.n_levels);
                    let width = term.rows.width();
                    let block = width * e_width;
                    touched.clear();
                    cross.clear();
                    for i in rows.clone() {
                        let l = levels[i] as usize;
                        if slot[l] == u32::MAX {
                            slot[l] = touched.len() as u32;
                            touched.push(l);
                            cross.resize(cross.len() + block, 0.0);
                        }
                        counts[term.first + l] += 1;
                        let at = slot[l] as usize * block;
                        // Weight first: a zero weight kills a huge loading.
                        let w = row_weight(sqrt_weights, i);
                        for c in 0..width {
                            let wz = w * term.rows.column(i, c);
                            for k in 0..e_width {
                                cross[at + c * e_width + k] += wz * eliminated.column(i, k);
                            }
                        }
                    }
                    for (p, &l) in touched.iter().enumerate() {
                        slot[l] = u32::MAX;
                        for k in 0..e_width {
                            let g = e_gram[k * e_n_levels + e];
                            if g <= 0.0 {
                                continue;
                            }
                            for c in 0..width {
                                let x = cross[p * block + c * e_width + k];
                                removed[term.start + c * n_levels + l] += x / g * x;
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
            Threads::All,
            &|k: Range<usize>| leaf(leaves[k.start].clone()),
            &merge,
        );
        let mut diagonal = vec![0.0; n_kept];
        let mut wanted = vec![false; n_kept_levels];
        // (kept term, level)
        let mut flagged: Vec<(usize, usize)> = Vec::new();
        for (t, term) in kept.iter().enumerate() {
            let n_levels = term.rows.n_levels;
            for l in 0..n_levels {
                // The difference cancels below `n·ε` of the squared norm; there it is resummed.
                let noise = 4.0 * counts[term.first + l] as f64 * f64::EPSILON;
                let entries = (0..term.rows.width()).map(|c| {
                    let (at, norm) = (term.start + c * n_levels + l, term.gram[c * n_levels + l]);
                    (at, norm - removed[at], norm)
                });
                if entries
                    .clone()
                    .any(|(_, entry, norm)| entry <= noise * norm)
                {
                    wanted[term.first + l] = true;
                    flagged.push((t, l));
                } else {
                    entries.for_each(|(at, entry, _)| diagonal[at] = entry);
                }
            }
        }
        if flagged.is_empty() {
            return diagonal;
        }
        let searched: Vec<&KeptTerm<'_>> = (kept.iter())
            .filter(|t| wanted[t.first..][..t.rows.n_levels].contains(&true))
            .collect();
        // The eliminated levels each flagged kept level meets, ascending.
        let mut met: Vec<(usize, u32)> = tasks
            .par_iter()
            .flat_map_iter(|task| {
                let (mut pairs, mut seen) = (Vec::new(), Vec::new());
                for e in task.clone() {
                    seen.clear();
                    for i in columns.rows.positions(e) {
                        for term in &searched {
                            let at = term.first + term.rows.levels[i] as usize;
                            if wanted[at] {
                                seen.push(at);
                            }
                        }
                    }
                    seen.sort_unstable();
                    seen.dedup();
                    pairs.extend(seen.iter().map(|&at| (at, e as u32)));
                }
                pairs
            })
            .collect();
        met.par_sort_unstable();
        for levels in flagged.chunk_by(|a, b| a.0 == b.0) {
            let term = &kept[levels[0].0];
            let (width, n_levels) = (term.rows.width(), term.rows.n_levels);
            let mut energies = vec![0.0; levels.len() * width];
            energies.par_chunks_mut(width).zip(levels).for_each_init(
                || EnergyScratch {
                    fit: vec![0.0; width * e_width],
                    fit_error: vec![0.0; width * e_width],
                    e_row: vec![0.0; e_width],
                    r: vec![0.0; width],
                    cross: vec![0.0; width * width],
                    bound: vec![0.0; width],
                    basis: GramBasisWorkspace::new(width),
                },
                |scratch, (out, &(_, l))| {
                    let at = term.first + l;
                    let lo = met.partition_point(|p| p.0 < at);
                    let hi = met.partition_point(|p| p.0 <= at);
                    let e_levels = met[lo..hi].iter().map(|&(_, e)| e);
                    columns.projected_energies(scratch, &term.rows, l, e_levels, out);
                },
            );
            for (&(_, l), energies) in levels.iter().zip(energies.chunks_exact(width)) {
                for (c, &entry) in energies.iter().enumerate() {
                    diagonal[term.start + c * n_levels + l] = entry;
                }
            }
        }
        diagonal
    }
}

/// A kept term as `projected_gram` sweeps it.
struct KeptTerm<'p> {
    rows: TermRows<'p>,
    /// Its block start in the kept coefficients.
    start: usize,
    /// Its first level among all kept terms' levels.
    first: usize,
    /// Its block of `diag(AᵀA)`.
    gram: &'p [f64],
}

/// Buffers `projected_energies` reuses across one kept term's levels.
struct EnergyScratch {
    /// `width × e_width`: each kept column's fit on the eliminated level's columns.
    fit: Vec<f64>,
    fit_error: Vec<f64>,
    /// A row's columns of `A_e`.
    e_row: Vec<f64>,
    /// A row's residual in each kept column.
    r: Vec<f64>,
    /// Upper triangle of the residual cross-Gram, row-major `width × width`.
    cross: Vec<f64>,
    bound: Vec<f64>,
    basis: GramBasisWorkspace,
}

impl ProjectedColumns<'_> {
    /// `‖(I − P_e) a‖²` of each column `a` of `kept`'s `level` into `out`, summed row by row over
    /// the rows of `e_levels`; 0 for a column within the rounding of its own evaluation of the
    /// span of the level's larger columns.
    fn projected_energies(
        &self,
        scratch: &mut EnergyScratch,
        kept: &TermRows<'_>,
        level: usize,
        e_levels: impl Iterator<Item = u32>,
        out: &mut [f64],
    ) {
        let EnergyScratch {
            fit,
            fit_error,
            e_row,
            r,
            cross,
            bound,
            basis,
        } = scratch;
        let (eliminated, gram) = (&self.eliminated, self.gram);
        let sqrt_weights = self.system.prepared.sqrt_weights();
        let (width, e_width) = (kept.width(), eliminated.width());
        // Running error bounds (Higham, ASNA 2nd ed., §3.3); a row with weight 0 rounds nothing.
        cross.fill(0.0);
        bound.fill(0.0);
        let row_error = (e_width + 2) as f64 * f64::EPSILON;
        for e in e_levels.map(|e| e as usize) {
            fit.fill(0.0);
            fit_error.fill(0.0);
            let rows = self.rows.positions(e);
            for i in rows.clone().filter(|&i| kept.levels[i] as usize == level) {
                let w = row_weight(sqrt_weights, i);
                for (k, z) in e_row.iter_mut().enumerate() {
                    *z = eliminated.column(i, k);
                }
                let by_column = fit
                    .chunks_exact_mut(e_width)
                    .zip(fit_error.chunks_exact_mut(e_width));
                for (c, (fit, fit_error)) in by_column.enumerate() {
                    let wa = w * kept.column(i, c);
                    for ((f, error), &z) in fit.iter_mut().zip(fit_error).zip(&*e_row) {
                        let term = z * wa;
                        if term != 0.0 {
                            *f += term;
                            *error += f.abs() + 2.0 * term.abs();
                        }
                    }
                }
            }
            let by_entry = fit_error
                .iter_mut()
                .zip(fit.iter_mut())
                .zip((0..e_width).cycle());
            for ((error, f), k) in by_entry {
                let g = gram[k * eliminated.n_levels + e];
                (*f, *error) = if g > 0.0 {
                    let fit = *f / g;
                    (fit, f64::EPSILON * (*error / g + fit.abs()))
                } else {
                    (0.0, 0.0)
                };
            }
            for i in rows {
                let s = sqrt_weights.map_or(1.0, |s| s[i]);
                let in_level = kept.levels[i] as usize == level;
                for (k, z) in e_row.iter_mut().enumerate() {
                    *z = eliminated.column(i, k);
                }
                let by_column = fit
                    .chunks_exact(e_width)
                    .zip(fit_error.chunks_exact(e_width));
                for (c, (fit, fit_error)) in by_column.enumerate() {
                    let a = if in_level { kept.column(i, c) } else { 0.0 };
                    let fitted = (fit.iter().zip(&*e_row)).fold(-0.0, |acc, (f, z)| acc + f * z);
                    r[c] = s * (a - fitted);
                    let spread =
                        (fit_error.iter().zip(&*e_row)).fold(0.0, |acc, (d, z)| acc + d * z.abs());
                    let error = s * (row_error * (a.abs() + fitted.abs()) + spread);
                    bound[c] += error * error;
                }
                for (c, row) in cross.chunks_exact_mut(width).enumerate() {
                    let r_c = r[c];
                    let upper = row[c..].iter_mut().zip(&r[c..]);
                    upper.for_each(|(x, r_d)| *x += r_c * r_d);
                }
            }
        }
        let fill = |g: &mut [f64]| {
            for (j, g) in g.iter_mut().enumerate() {
                let (c, d) = (j / width, j % width);
                *g = cross[c.min(d) * width + c.max(d)];
            }
        };
        let kept_columns = basis.orthonormalize(fill, |c, _| 4.0 * bound[c]).kept;
        for ((out, &kept), energy) in out
            .iter_mut()
            .zip(kept_columns)
            .zip(cross.iter().step_by(width + 1))
        {
            *out = if kept { *energy } else { 0.0 };
        }
    }

    /// Level `l`'s rows and its columns of `A_e`.
    #[inline(always)]
    fn level<const W: usize, Wt: Fn(usize) -> f64>(
        &self,
        l: usize,
        w: Wt,
    ) -> (Range<usize>, LevelBasis<'_, W, Wt>) {
        let rows = self.rows.positions(l);
        let eliminated = self.eliminated;
        let basis = LevelBasis {
            start: rows.start,
            w,
            eliminated,
            slopes: std::array::from_fn(|k| {
                let slope = eliminated.slopes.get(k);
                slope.and_then(|z| z.get(rows.clone())).unwrap_or_default()
            }),
            gram: self.gram,
            level: l,
        };
        (rows, basis)
    }

    /// The moments of one level, a cache line clear of other threads' memory at each end: small
    /// buffers that shared a line ran up to 2× slower.
    fn moments_len(&self) -> usize {
        2 * LINE + self.eliminated.width()
    }

    fn moments(&self) -> StackReq {
        StackReq::new::<f64>(self.moments_len())
    }

    /// What `project` takes: moments per task when the tasks run at once.
    fn project_scratch(&self, parallel: bool) -> StackReq {
        let tasks = self.system.tasks.len();
        self.moments().array(if parallel { tasks } else { 1 })
    }

    /// `out ← (I − P_e) out` in one pass, `set` first writing each row's value; `w = W^½`.
    fn project<const W: usize>(
        &self,
        set: impl Fn(usize, &mut f64) + Sync + Copy,
        w: impl Fn(usize) -> f64 + Sync + Copy,
        out: &mut [f64],
        threads: Threads,
        stack: &mut MemStack,
    ) {
        let width = self.eliminated.width();
        let moments = self.moments_len();
        let solve_part =
            |levels: Range<usize>, base: usize, part: &mut [f64], stack: &mut MemStack| {
                // Copied: read through the captures, each row reloads them after a store to `part`.
                let (set, w) = (set, w);
                let (mut moments, _) = stack.make_with(moments, |_| 0.0);
                // A fixed width keeps the moments in registers; on the heap they cost 12%.
                let mut fixed = [0.0; W];
                let fit = match W {
                    0 => &mut moments[LINE..][..width],
                    _ => &mut fixed[..],
                };
                for l in levels {
                    let (rows, basis) = self.level::<W, _>(l, w);
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
            };
        let parallel = threads != Threads::ONE && stack.can_hold(self.project_scratch(true));
        let tasks = 0..self.system.tasks.len();
        self.project_tasks(tasks, out, parallel, stack, &solve_part);
    }

    /// `solve_part` on each task's levels, its first row and its rows of `out`, halving at once.
    fn project_tasks(
        &self,
        tasks: Range<usize>,
        out: &mut [f64],
        parallel: bool,
        stack: &mut MemStack,
        solve_part: &(impl Fn(Range<usize>, usize, &mut [f64], &mut MemStack) + Sync),
    ) {
        let all = &self.system.tasks;
        let start = |t: usize| self.rows.positions(all[t].start).start;
        if parallel && tasks.len() > 1 {
            let mid = tasks.start + tasks.len() / 2;
            let (left, right) = out.split_at_mut(start(mid) - start(tasks.start));
            let carved = self.moments().array(mid - tasks.start);
            join_carved(
                stack,
                carved,
                |stack| self.project_tasks(tasks.start..mid, left, true, stack, solve_part),
                |stack| self.project_tasks(mid..tasks.end, right, true, stack, solve_part),
            );
            return;
        }
        let mut rest = out;
        for t in tasks {
            let levels = all[t].clone();
            let end = self.rows.positions(levels.end - 1).end;
            let (head, tail) = std::mem::take(&mut rest).split_at_mut(end - start(t));
            solve_part(levels, start(t), head, stack);
            rest = tail;
        }
    }

    /// What `adjoint` over `leaves` takes: an accumulator per right child, moments per leaf, and
    /// both children's at once when they run at once.
    fn adjoint_scratch(&self, leaves: &Range<usize>, parallel: bool) -> StackReq {
        if leaves.len() <= 1 {
            return self.moments();
        }
        let mid = tree_mid(leaves, 1);
        let left = self.adjoint_scratch(&(leaves.start..mid), parallel);
        let right = self.adjoint_scratch(&(mid..leaves.end), parallel);
        let children = if parallel {
            left.and(right)
        } else {
            left.or(right)
        };
        let n_kept = self.system.prepared.design.n_unknowns();
        StackReq::new::<f64>(n_kept).and(children)
    }

    /// `y = A_rᵀ v`, one accumulator per leaf summed in [`fixed_tree`]'s tree.
    fn adjoint<'k, const W: usize, const KEPT_PLAIN: bool>(
        &self,
        kept: impl AsRef<Kept<'k>> + Sync + Copy,
        v: &[f64],
        w: impl Fn(usize) -> f64 + Sync + Copy,
        y: &mut [f64],
        threads: Threads,
        stack: &mut MemStack,
    ) {
        let width = self.eliminated.width();
        let moments = self.moments_len();
        let all = &self.system.tasks;
        let leaf = |tasks: Range<usize>, acc: &mut [f64], stack: &mut MemStack| {
            // Copied: read through the captures, each row reloads them after a store to `acc`.
            let (kept, w) = (kept, w);
            let (mut moments, _) = stack.make_with(moments, |_| 0.0);
            let mut fixed = [0.0; W];
            let fit = match W {
                0 => &mut moments[LINE..][..width],
                _ => &mut fixed[..],
            };
            for l in all[tasks.start].start..all[tasks.end - 1].end {
                let (rows, basis) = self.level::<W, _>(l, w);
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
                    add_kept_row::<KEPT_PLAIN>(&kept, acc, i, projected);
                }
            }
        };
        let leaves = 0..self.system.leaves.len();
        let parallel =
            threads != Threads::ONE && stack.can_hold(self.adjoint_scratch(&leaves, true));
        y.fill(0.0);
        self.adjoint_tree(leaves, y, parallel, stack, &leaf);
    }

    /// Adds `leaf` over `leaves` into `out`, zero on entry: the right child into an accumulator
    /// of its own, added after the left.
    fn adjoint_tree(
        &self,
        leaves: Range<usize>,
        out: &mut [f64],
        parallel: bool,
        stack: &mut MemStack,
        leaf: &(impl Fn(Range<usize>, &mut [f64], &mut MemStack) + Sync),
    ) {
        if leaves.len() <= 1 {
            return leaf(self.system.leaves[leaves.start].clone(), out, stack);
        }
        let mid = tree_mid(&leaves, 1);
        let (left, right) = (leaves.start..mid, mid..leaves.end);
        let (mut acc, stack) = stack.make_with(out.len(), |_| 0.0);
        if parallel {
            let carved = self.adjoint_scratch(&left, true);
            join_carved(
                stack,
                carved,
                |stack| self.adjoint_tree(left, out, true, stack, leaf),
                |stack| self.adjoint_tree(right, &mut acc, true, stack, leaf),
            );
        } else {
            self.adjoint_tree(left, out, false, stack, leaf);
            self.adjoint_tree(right, &mut acc, false, stack, leaf);
        }
        out.iter_mut().zip(acc.iter()).for_each(|(l, r)| *l += r);
    }
}

/// `left` on a region of `carved` from `stack` and `right` on the rest, at once.
fn join_carved(
    stack: &mut MemStack,
    carved: StackReq,
    left: impl FnOnce(&mut MemStack) + Send,
    right: impl FnOnce(&mut MemStack) + Send,
) {
    let (mine, rest) = stack.make_aligned_uninit::<u8>(carved.size_bytes(), carved.align_bytes());
    rayon::join(|| left(MemStack::new(mine)), || right(rest));
}

/// Per kept term, its columns and its block start in the kept coefficients.
type Kept<'r> = [(TermRows<'r>, usize)];

/// `(D_k x)_i`.
#[inline(always)]
fn kept_row<'r, const PLAIN: bool>(kept: &impl AsRef<Kept<'r>>, x: &[f64], i: usize) -> f64 {
    let row = |(columns, offset): &(TermRows<'_>, usize)| columns.row::<PLAIN>(x, *offset, i);
    (kept.as_ref().iter()).fold(-0.0, |acc, term| acc + row(term))
}

/// `acc += v · D_kᵀ e_i`.
#[inline(always)]
fn add_kept_row<'r, const PLAIN: bool>(
    kept: &impl AsRef<Kept<'r>>,
    acc: &mut [f64],
    i: usize,
    v: f64,
) {
    for (columns, offset) in kept.as_ref() {
        columns.add_row::<PLAIN>(acc, *offset, i, v);
    }
}

/// Level-major `by_level` (`width` entries per level) read column by column.
fn column_major(by_level: &[f64], width: usize) -> impl Iterator<Item = f64> + '_ {
    (0..width).flat_map(move |c| by_level.iter().skip(c).step_by(width).copied())
}

/// Doubles in a cache line, 128 bytes on Apple silicon and 64 elsewhere.
const LINE: usize = 16;

/// One eliminated level's columns `W^½ D_e,l`, row `p` of them at design row `start + p`: an
/// intercept and `W − 1` slopes, or `eliminated`'s columns when `W` is 0.
struct LevelBasis<'r, const W: usize, Wt> {
    start: usize,
    w: Wt,
    eliminated: TermRows<'r>,
    /// Slope `k` (column `k + 1`) on this level's rows; slot `W − 1` is empty.
    slopes: [&'r [f64]; W],
    /// `diag(A_eᵀA_e)`, column-major over the term's `n_levels`.
    gram: &'r [f64],
    level: usize,
}

impl<const W: usize, Wt: Fn(usize) -> f64> DiagonalGramBasis for LevelBasis<'_, W, Wt> {
    #[inline(always)]
    fn entry(&self, p: usize, c: usize) -> f64 {
        let i = self.start + p;
        let w = (self.w)(i);
        match (W, c) {
            (0, _) => w * self.eliminated.column(i, c),
            (_, 0) => w,
            _ => w * self.slopes[c - 1][p],
        }
    }

    #[inline(always)]
    fn norm2(&self, c: usize) -> f64 {
        self.gram[c * self.eliminated.n_levels + self.level]
    }
}

impl Operator for ReducedDesign<'_> {
    fn nrows(&self) -> usize {
        self.prepared.design.n_obs
    }

    fn ncols(&self) -> usize {
        self.prepared.design.n_unknowns()
    }

    /// `apply`'s copy of `x` and its projection, or the adjoint's accumulators.
    fn scratch(&self, _columns: usize, threads: Threads) -> StackReq {
        let columns = self.columns();
        let parallel = threads != Threads::ONE;
        let leaves = 0..self.leaves.len();
        let copy = StackReq::new::<f64>(self.copied(self.ncols()));
        (copy.and(columns.project_scratch(parallel))).or(columns.adjoint_scratch(&leaves, parallel))
    }

    fn apply(
        &self,
        x: &[f64],
        y: &mut [f64],
        threads: Threads,
        stack: &mut MemStack,
    ) -> Result<(), schwarz_precond::SolveError> {
        debug_assert_eq!(x.len(), self.ncols());
        let (mut copy, stack) = stack.make_with(self.copied(x.len()), |j| x[j]);
        let x = match copy.is_empty() {
            true => x,
            false => {
                self.zero_vanished(&mut copy);
                &*copy
            }
        };
        let columns = self.columns();
        with_shape!(columns, |W, KEPT_PLAIN, kept, w| {
            columns.project::<W>(
                move |i, y| *y = w(i) * kept_row::<KEPT_PLAIN>(&kept, x, i),
                w,
                y,
                threads,
                stack,
            )
        });
        Ok(())
    }

    fn apply_adjoint(
        &self,
        v: &[f64],
        y: &mut [f64],
        threads: Threads,
        stack: &mut MemStack,
    ) -> Result<(), schwarz_precond::SolveError> {
        debug_assert_eq!(y.len(), self.ncols());
        let columns = self.columns();
        with_shape!(columns, |W, KEPT_PLAIN, kept, w| {
            columns.adjoint::<W, KEPT_PLAIN>(kept, v, w, y, threads, stack)
        });
        self.zero_vanished(y);
        Ok(())
    }
}
