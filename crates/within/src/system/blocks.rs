//! `D̃`'s rows split around `e`, read a row at a time as `A`'s ([`SplitRows`]): `A_k`'s rows and
//! `P_l` over `e`'s segments, each at fixed widths where [`with_split`] fuses them. The rows are
//! sorted by `e`, so `A_e` is block-diagonal by segment and `G_ee` diagonal: `P_e` acts on each
//! level's segment alone, `P_l = A_{e,l} G_l⁻¹ A_{e,l}ᵀ`.

use std::ops::Range;

use rayon::prelude::*;
use schwarz_precond::{MemBuffer, MemStack, StackReq};

use super::term::{Row, Rows, TermBlock, TermRow};
use super::SplitDesign;
use crate::design::{FittedMap, FittedSum, LevelSegments};

/// `D̃_k`, the kept terms' block, read a row at a time.
pub(super) trait KeptTerms: Sync {
    /// `f(j, d)` for each `d = D̃_k[i, j]` of row `i`, zero or not.
    fn slots(&self, i: usize, f: impl FnMut(usize, f64));
}

/// A kept term's rows placed in `x`: column `c` of level `l` at `start + c · n_levels + l`.
#[derive(Clone, Copy)]
pub(super) struct KeptRow<'p, R> {
    row: R,
    levels: &'p [u32],
    /// Where the term's block starts in `x`.
    start: usize,
    /// The term's level count: the stride between its columns.
    n_levels: usize,
}

impl<'p, R> KeptRow<'p, R> {
    fn new(term: &TermBlock<'p>, row: R, start: usize) -> Self {
        Self {
            row,
            levels: term.levels(),
            start,
            n_levels: term.layout().n_levels(),
        }
    }
}

impl<R: TermRow> KeptTerms for KeptRow<'_, R> {
    #[inline(always)]
    fn slots(&self, i: usize, mut f: impl FnMut(usize, f64)) {
        let at = self.start + self.levels[i] as usize;
        self.row.row(i, |c, d| f(at + c * self.n_levels, d));
    }
}

impl<A: KeptTerms, B: KeptTerms> KeptTerms for (A, B) {
    #[inline(always)]
    fn slots(&self, i: usize, mut f: impl FnMut(usize, f64)) {
        self.0.slots(i, &mut f);
        self.1.slots(i, &mut f);
    }
}

impl<T: KeptTerms + ?Sized> KeptTerms for &T {
    #[inline(always)]
    fn slots(&self, i: usize, f: impl FnMut(usize, f64)) {
        (**self).slots(i, f);
    }
}

impl<K: KeptTerms> KeptTerms for [K] {
    #[inline(always)]
    fn slots(&self, i: usize, mut f: impl FnMut(usize, f64)) {
        for term in self {
            term.slots(i, &mut f);
        }
    }
}

/// Every kept term at any count, built once per solve, not per row: intercept-only terms read
/// without a dispatch, any other mix each at its own shape.
pub(super) enum DynamicKept<'p> {
    Plain(Vec<KeptRow<'p, Row<'p, true, 0>>>),
    Mixed(Vec<KeptRow<'p, Rows<'p>>>),
}

impl<'p> DynamicKept<'p> {
    pub(super) fn new(design: &'p SplitDesign<'_>) -> Self {
        let terms = || (design.unknowns().blocks()).map(|(t, at)| (design.term(t), at.start));
        let plain: Option<Vec<_>> = terms()
            .map(|(term, at)| match term.rows() {
                Rows::I0(row) => Some(KeptRow::new(&term, row, at)),
                _ => None,
            })
            .collect();
        plain.map_or_else(
            || {
                Self::Mixed(
                    terms()
                        .map(|(term, at)| KeptRow::new(&term, term.rows(), at))
                        .collect(),
                )
            },
            Self::Plain,
        )
    }
}

impl KeptTerms for DynamicKept<'_> {
    #[inline(always)]
    fn slots(&self, i: usize, f: impl FnMut(usize, f64)) {
        match self {
            Self::Plain(terms) => terms[..].slots(i, f),
            Self::Mixed(terms) => terms[..].slots(i, f),
        }
    }
}

/// `W^{1/2}`'s diagonal, read a row at a time.
pub(super) trait RowScale: Copy + Sync {
    /// Row `i`'s entry.
    fn at(&self, i: usize) -> f64;
}

/// `W = I`: its unit scale folds away.
#[derive(Clone, Copy)]
pub(super) struct Unweighted;

impl RowScale for Unweighted {
    #[inline(always)]
    fn at(&self, _: usize) -> f64 {
        1.0
    }
}

impl RowScale for &[f64] {
    #[inline(always)]
    fn at(&self, i: usize) -> f64 {
        self[i]
    }
}

impl RowScale for Option<&[f64]> {
    #[inline(always)]
    fn at(&self, i: usize) -> f64 {
        self.map_or(1.0, |s| s[i])
    }
}

/// `A = W^{1/2} D̃`'s rows split around `e`: `A_k`'s rows, and `P_l` on each of `e`'s segments
/// from `A_e`'s rows and `G_ee⁻¹`. A fit holds [`width`](Self::width) values.
#[derive(Clone, Copy)]
pub(super) struct SplitRows<'s, T, R, W = Option<&'s [f64]>> {
    /// `D̃_k`.
    kept: T,
    /// `D̃_e`.
    eliminated: R,
    sqrt: W,
    /// `G_ee⁻¹`, level-major.
    inverse: &'s [f64],
    /// `e`'s segments.
    segments: &'s LevelSegments,
}

impl<T: KeptTerms, R: TermRow, W: RowScale> SplitRows<'_, T, R, W> {
    /// `W^{1/2}`'s entry for row `i`.
    #[inline(always)]
    fn scale(&self, i: usize) -> f64 {
        self.sqrt.at(i)
    }

    /// `f(j, a)` for each `a = A_k[i, j]` of row `i`.
    #[inline(always)]
    fn slots(&self, i: usize, mut f: impl FnMut(usize, f64)) {
        let s = self.scale(i);
        self.kept.slots(i, |j, d| f(j, s * d));
    }

    /// `(A_k x)_i`.
    #[inline(always)]
    fn dot(&self, i: usize, x: &[f64]) -> f64 {
        // Fold from -0.0, the additive identity, so it folds away.
        let mut acc = -0.0;
        self.kept.slots(i, |j, d| acc += x[j] * d);
        self.scale(i) * acc
    }

    /// `acc += value · A_k[i, :]ᵀ`.
    #[inline(always)]
    fn add(&self, i: usize, value: f64, acc: &mut [f64]) {
        let value = self.scale(i) * value;
        self.kept.slots(i, |j, d| acc[j] += value * d);
    }

    /// `e`'s column count.
    #[inline(always)]
    fn width(&self) -> usize {
        self.eliminated.width()
    }

    /// `G_l⁻¹`, level `level`'s diagonal: 0 where `G_l`'s is.
    #[inline(always)]
    fn inverse(&self, level: usize) -> &[f64] {
        let width = self.width();
        &self.inverse[level * width..(level + 1) * width]
    }

    /// `f(c, a)` for each `a = A_e[i, c]` of row `i`.
    #[inline(always)]
    fn row(&self, i: usize, mut f: impl FnMut(usize, f64)) {
        let s = self.scale(i);
        self.eliminated.row(i, |c, r| f(c, s * r));
    }

    /// `m = A_eᵀ (scale · v)` over the rows of `v` from row `start`.
    #[inline(always)]
    fn moments(&self, start: usize, v: &[f64], scale: f64, m: &mut [f64]) {
        let weight = |i: usize| self.scale(i) * scale;
        self.eliminated.moments(start, v, weight, m);
    }

    /// `fit = G_l⁻¹ fit`.
    #[inline(always)]
    fn solve(&self, level: usize, fit: &mut [f64]) {
        for (f, &g) in fit.iter_mut().zip(self.inverse(level)) {
            *f *= g;
        }
    }

    /// `fit = G_l⁻¹ m`, `m = A_{e,l}ᵀ v` in `fit`, refolded from `v`, level `level`'s segment
    /// from row `start`, if a moment overflowed.
    #[inline(always)]
    fn join(&self, level: usize, start: usize, v: &[f64], fit: &mut [f64]) {
        self.solve(level, fit);
        if fit.iter().any(|f| !f.is_finite()) {
            self.refold(level, start, v, fit);
        }
    }

    /// `v_i − A_e[i, :] · fit`.
    #[inline(always)]
    fn remove(&self, i: usize, v_i: f64, fit: &[f64]) -> f64 {
        let mut f = 0.0;
        self.eliminated.row(i, |c, r| f += r * fit[c]);
        v_i - self.scale(i) * f
    }

    /// `segment ← segment − A_e fit` over the rows of `segment` from row `start`.
    #[inline(always)]
    fn remove_fit(&self, start: usize, segment: &mut [f64], fit: &[f64]) {
        for (k, v) in segment.iter_mut().enumerate() {
            *v = self.remove(start + k, *v, fit);
        }
    }

    /// `fit` from `v` scaled to a unit peak, so its moments stay finite.
    #[cold]
    fn refold(&self, level: usize, start: usize, v: &[f64], fit: &mut [f64]) {
        let peak = v.iter().fold(0.0f64, |m, x| m.max(x.abs()));
        // A segment with no finite value has no scale; its fit stays non-finite.
        if peak == 0.0 || !peak.is_finite() {
            return;
        }
        let exponent = (peak.log2().floor() as i32).clamp(-1022, 1022);
        self.moments(start, v, 2f64.powi(-exponent), fit);
        self.solve(level, fit);
        let unscale = 2f64.powi(exponent);
        fit.iter_mut().for_each(|f| *f *= unscale);
    }

    /// What [`map_projected`](Self::map_projected) and [`sum_projected`](Self::sum_projected)
    /// take from a stack for a sum of length `n`.
    pub(super) fn scratch(&self, n: usize) -> StackReq {
        self.segments.scratch(n, self.width())
    }

    /// `y = (I − P_e) A_k x`, rescued.
    pub(super) fn map_projected(&self, y: &mut [f64], stack: &mut MemStack, x: &[f64])
    where
        T: Copy,
    {
        let pass = ProjectMap { rows: self, x };
        self.segments
            .map_fitted(y, self.width(), stack, &pass, None);
    }

    /// `out = A_kᵀ (I − P_e) v`, unrescued.
    pub(super) fn sum_projected(&self, out: &mut [f64], stack: &mut MemStack, v: &[f64])
    where
        T: Copy,
    {
        let pass = ProjectSum { rows: self, v };
        self.segments.sum_fitted(out, self.width(), stack, &pass);
    }

    /// `v ← (I − P_e) v` twice, rescued, the second pass removing what the first left in
    /// rounding. Returns `G_ee⁻¹ A_eᵀ v`, both passes' fits summed, level-major.
    pub(super) fn partial_out(&self, v: &mut [f64]) -> Vec<f64> {
        let width = self.width();
        let mut buffer = MemBuffer::new(self.segments.scratch(0, width));
        let mut fits = [vec![0.0; self.inverse.len()], vec![0.0; self.inverse.len()]];
        for fits in &mut fits {
            let (pass, stack) = (PartialOut { rows: self }, MemStack::new(&mut buffer));
            (self.segments).map_fitted(v, width, stack, &pass, Some(fits));
        }
        let [mut first, second] = fits;
        for (f, s) in first.iter_mut().zip(&second) {
            *f += s;
        }
        first
    }

    /// `W^{1/2} y − A_k x`.
    pub(super) fn residual(&self, y: &[f64], x: &[f64]) -> Vec<f64> {
        (y.par_iter().enumerate())
            .map(|(i, &y_i)| self.scale(i) * y_i - self.dot(i, x))
            .collect()
    }

    /// `out_j = Σ_l (‖a_{j,l}‖² − Σ_c (a_{c,l}ᵀ a_j)² / g_{c,l})` over `e`'s levels `l`: each
    /// level's part of `A_k`'s column `a_j` less its energy in that level's span, counted where
    /// `keep(rows, own, energy)` holds for the level's `rows` and `own = ‖a_{j,l}‖²`.
    pub(super) fn column_energies(
        &self,
        out: &mut [f64],
        keep: &(impl Fn(usize, f64, f64) -> bool + Sync),
    ) {
        let n = out.len();
        let mut buffer = MemBuffer::new(self.segments.scratch(n, 3 * n));
        // Per level, `own[j] = ‖a_{j,l}‖²` and `absorbed[j]` gather over its rows; per column `c`,
        // `a_{c,l}ᵀ a_j` gathers in `cross`. Each pass after the first reads each touched `j` and
        // clears what it read.
        let level = |level: usize, rows: Range<usize>, acc: &mut [f64], scratch: &mut [f64]| {
            let (cross, rest) = scratch.split_at_mut(n);
            let (own, absorbed) = rest.split_at_mut(n);
            for i in rows.clone() {
                self.slots(i, |j, d| own[j] += d * d);
            }
            for (c, &g) in self.inverse(level).iter().enumerate() {
                if g == 0.0 {
                    continue;
                }
                for i in rows.clone() {
                    let mut a = 0.0;
                    self.row(i, |cc, v| {
                        if cc == c {
                            a = v;
                        }
                    });
                    self.slots(i, |j, d| cross[j] += a * d);
                }
                for i in rows.clone() {
                    self.slots(i, |j, _| {
                        // `g` before the second factor, so large weights do not overflow `cross²`.
                        absorbed[j] += cross[j] * g * cross[j];
                        cross[j] = 0.0;
                    });
                }
            }
            for i in rows.clone() {
                self.slots(i, |j, _| {
                    let energy = own[j] - absorbed[j];
                    if keep(rows.len(), own[j], energy) {
                        acc[j] += energy;
                    }
                    (own[j], absorbed[j]) = (0.0, 0.0);
                });
            }
        };
        (self.segments).sum(out, 3 * n, MemStack::new(&mut buffer), &level);
    }
}

/// [`SplitRows::map_projected`]'s pass.
struct ProjectMap<'a, 's, T, R, W> {
    rows: &'a SplitRows<'s, T, R, W>,
    x: &'a [f64],
}

impl<T: KeptTerms + Copy, R: TermRow, W: RowScale> FittedMap for ProjectMap<'_, '_, T, R, W> {
    #[inline]
    fn moments(&self, _: usize, start: usize, segment: &mut [f64], m: &mut [f64]) {
        // Copied: read through `self`, each row reloads `rows` after a store to `segment`.
        let rows = *self.rows;
        for (k, y) in segment.iter_mut().enumerate() {
            *y = rows.dot(start + k, self.x);
        }
        rows.moments(start, segment, 1.0, m);
    }

    #[inline]
    fn fit(&self, level: usize, start: usize, segment: &[f64], m: &mut [f64]) {
        self.rows.join(level, start, segment, m);
    }

    #[inline]
    fn apply(&self, _: usize, start: usize, segment: &mut [f64], fit: &[f64]) {
        self.rows.remove_fit(start, segment, fit);
    }
}

/// [`SplitRows::partial_out`]'s pass: `v ← (I − P_e) v`, rescued.
struct PartialOut<'a, 's, T, R, W> {
    rows: &'a SplitRows<'s, T, R, W>,
}

impl<T: KeptTerms, R: TermRow, W: RowScale> FittedMap for PartialOut<'_, '_, T, R, W> {
    #[inline]
    fn moments(&self, _: usize, start: usize, segment: &mut [f64], m: &mut [f64]) {
        self.rows.moments(start, segment, 1.0, m);
    }

    #[inline]
    fn fit(&self, level: usize, start: usize, segment: &[f64], m: &mut [f64]) {
        self.rows.join(level, start, segment, m);
    }

    #[inline]
    fn apply(&self, _: usize, start: usize, segment: &mut [f64], fit: &[f64]) {
        self.rows.remove_fit(start, segment, fit);
    }
}

/// [`SplitRows::sum_projected`]'s pass.
struct ProjectSum<'a, 's, T, R, W> {
    rows: &'a SplitRows<'s, T, R, W>,
    v: &'a [f64],
}

impl<T: KeptTerms + Copy, R: TermRow, W: RowScale> FittedSum for ProjectSum<'_, '_, T, R, W> {
    #[inline]
    fn moments(&self, _: usize, rows: Range<usize>, m: &mut [f64]) {
        self.rows.moments(rows.start, &self.v[rows], 1.0, m);
    }

    #[inline]
    fn fit(&self, level: usize, m: &mut [f64]) {
        self.rows.solve(level, m);
    }

    #[inline]
    fn apply(&self, _: usize, rows: Range<usize>, fit: &[f64], acc: &mut [f64]) {
        // Copied: read through `self`, each row reloads `split` after a store to `acc`.
        let split = *self.rows;
        for (i, &v_i) in rows.clone().zip(&self.v[rows]) {
            split.add(i, split.remove(i, v_i, fit), acc);
        }
    }
}

impl<'s> SplitDesign<'s> {
    /// `A` split around `e` with `kept` for `A_k`, `eliminated` for `A_e` and `sqrt` for `W^{1/2}`.
    fn split<T, R, W>(&'s self, kept: T, eliminated: R, sqrt: W) -> SplitRows<'s, T, R, W> {
        SplitRows {
            kept,
            eliminated,
            sqrt,
            inverse: &self.eliminated_inverse,
            segments: self.incidence.segments(),
        }
    }

    /// `A` split around `e` at any shape.
    pub(super) fn dynamic(&'s self) -> SplitRows<'s, DynamicKept<'s>, TermBlock<'s>> {
        self.split(
            DynamicKept::new(self),
            self.term(self.eliminated()),
            self.weights.sqrt(),
        )
    }
}

/// A product over `A`'s rows split around `e`.
pub(super) trait ReducedKernel {
    fn run<T: KeptTerms + Copy, R: TermRow, W: RowScale>(self, rows: &SplitRows<'_, T, R, W>);
}

/// Runs `kernel` on `A` split around `e` at fixed widths where fused: `e` at every [`Rows`] shape,
/// `A_k` at `kept`'s, and `W^{1/2}` as [`Unweighted`] where `W = I`.
pub(super) fn with_split(
    design: &SplitDesign<'_>,
    kept: &KeptSide<'_>,
    kernel: impl ReducedKernel,
) {
    match design.term(design.eliminated()).rows() {
        Rows::I0(row) => with_kept(design, kept, kernel, row),
        Rows::I1(row) => with_kept(design, kept, kernel, row),
        Rows::I2(row) => with_kept(design, kept, kernel, row),
        Rows::I3(row) => with_kept(design, kept, kernel, row),
        Rows::S1(row) => with_kept(design, kept, kernel, row),
        Rows::S2(row) => with_kept(design, kept, kernel, row),
        Rows::S3(row) => with_kept(design, kept, kernel, row),
        Rows::Dynamic(block) => with_kept(design, kept, kernel, block),
    }
}

/// `A_k`'s rows at the shape they run at, classified once per solve: one or two kept terms at
/// fixed widths, each an intercept with at most two slopes, else [`DynamicKept`].
pub(super) enum KeptSide<'p> {
    One(FusedKept<'p>),
    Pair(FusedKept<'p>, FusedKept<'p>),
    Dynamic(DynamicKept<'p>),
}

impl<'p> KeptSide<'p> {
    pub(super) fn new(design: &'p SplitDesign<'_>) -> Self {
        let mut kept =
            (design.unknowns().blocks()).map(|(t, at)| FusedKept::of(design.term(t), at.start));
        match (kept.next(), kept.next(), kept.next()) {
            (Some(Some(a)), None, None) => Self::One(a),
            (Some(Some(a)), Some(Some(b)), None) => Self::Pair(a, b),
            _ => Self::Dynamic(DynamicKept::new(design)),
        }
    }
}

/// A kept term at a shape the kept side fuses.
#[derive(Clone, Copy)]
pub(super) enum FusedKept<'p> {
    Plain(KeptRow<'p, Row<'p, true, 0>>),
    OneSlope(KeptRow<'p, Row<'p, true, 1>>),
    TwoSlopes(KeptRow<'p, Row<'p, true, 2>>),
}

impl<'p> FusedKept<'p> {
    /// `term`, its block from `start` in `x`, if an intercept with at most two slopes.
    fn of(term: TermBlock<'p>, start: usize) -> Option<Self> {
        match term.rows() {
            Rows::I0(row) => Some(Self::Plain(KeptRow::new(&term, row, start))),
            Rows::I1(row) => Some(Self::OneSlope(KeptRow::new(&term, row, start))),
            Rows::I2(row) => Some(Self::TwoSlopes(KeptRow::new(&term, row, start))),
            _ => None,
        }
    }
}

/// Runs `kernel` with `eliminated` and `A_k` at `kept`'s shape.
#[inline(always)]
fn with_kept<R: TermRow>(
    design: &SplitDesign<'_>,
    kept: &KeptSide<'_>,
    kernel: impl ReducedKernel,
    eliminated: R,
) {
    use FusedKept::{OneSlope, Plain, TwoSlopes};
    macro_rules! run {
        ($kept:expr) => {
            match design.weights.sqrt() {
                None => kernel.run(&design.split($kept, eliminated, Unweighted)),
                Some(sqrt) => kernel.run(&design.split($kept, eliminated, sqrt)),
            }
        };
    }
    match kept {
        KeptSide::One(a) => match *a {
            Plain(a) => run!(a),
            OneSlope(a) => run!(a),
            TwoSlopes(a) => run!(a),
        },
        KeptSide::Pair(a, b) => match (*a, *b) {
            (Plain(a), Plain(b)) => run!((a, b)),
            (OneSlope(a), OneSlope(b)) => run!((a, b)),
            (TwoSlopes(a), TwoSlopes(b)) => run!((a, b)),
            // Narrower first, whichever comes first in `x`.
            (Plain(a), OneSlope(b)) | (OneSlope(b), Plain(a)) => run!((a, b)),
            (Plain(a), TwoSlopes(b)) | (TwoSlopes(b), Plain(a)) => run!((a, b)),
            (OneSlope(a), TwoSlopes(b)) | (TwoSlopes(b), OneSlope(a)) => run!((a, b)),
        },
        KeptSide::Dynamic(dynamic) => match dynamic {
            DynamicKept::Plain(terms) => run!(&terms[..]),
            DynamicKept::Mixed(terms) => run!(&terms[..]),
        },
    }
}
