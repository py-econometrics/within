//! `D̃`'s products over its term blocks: `y = s ⊙ Σ_t D̃_t x_t`, and `x_t = D̃_tᵀ (s ⊙ v)` summed in
//! the order term `t`'s `LevelReduction` fixes.

use rayon::prelude::*;
use schwarz_precond::parallel::{for_each_piece, Pieces, Split, Workspaces};
use schwarz_precond::MemStack;

use super::OrthogonalDesign;
use crate::design::LevelSlots;
use crate::design::{RowKernel, TermBlock};
use crate::PAR_THRESHOLD;

/// `dst = D̃ src`, row `i` times `scale[i]` when given.
pub(super) fn gather(
    design: &OrthogonalDesign<'_>,
    src: &[f64],
    dst: &mut [f64],
    scale: Option<&[f64]>,
) {
    debug_assert!(scale.is_none_or(|s| s.len() == dst.len()));
    debug_assert_eq!(design.n_obs(), dst.len());
    debug_assert_eq!(src.len(), design.unknowns().len());

    dst.fill(0.0);

    for_each_chunk(dst, |chunk, row_start| {
        for (t, term) in design.terms().enumerate() {
            term.visit_rows(GatherTerm {
                chunk: &mut *chunk,
                row_start,
                block: &src[design.unknowns().block(t)],
                term: &term,
            });
        }
        if let Some(scale) = scale {
            for (s, dst_val) in scale[row_start..].iter().zip(chunk.iter_mut()) {
                *dst_val *= s;
            }
        }
    });
}

/// Sweep `dst` in cache-sized chunks, parallel above [`PAR_THRESHOLD`] rows.
fn for_each_chunk(dst: &mut [f64], kernel: impl Fn(&mut [f64], usize) + Sync) {
    if dst.len() > PAR_THRESHOLD {
        const CHUNK_SIZE: usize = 4096;

        dst.par_chunks_mut(CHUNK_SIZE)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| kernel(chunk, chunk_idx * CHUNK_SIZE));
    } else {
        kernel(dst, 0);
    }
}

/// One term's gather into the rows `row_start..` of `chunk`.
struct GatherTerm<'k, 't> {
    chunk: &'k mut [f64],
    row_start: usize,
    /// The term's block of `x`.
    block: &'k [f64],
    term: &'k TermBlock<'t>,
}

impl RowKernel for GatherTerm<'_, '_> {
    #[inline(always)]
    fn fixed<const C: usize>(self, row: impl Fn(usize) -> [f64; C] + Sync) {
        let (layout, block) = (self.term.layout(), self.block);
        let cols = std::array::from_fn(|c| &block[layout.block_column(c)]);
        gather_term(self.chunk, self.row_start, self.term.levels(), cols, row);
    }

    #[inline(always)]
    fn dynamic(self) {
        // A dynamic column count cannot monomorphize a fixed arity.
        let (t, levels) = (self.term, self.term.levels());
        let block = self.block;
        for (local, dst_val) in self.chunk.iter_mut().enumerate() {
            let i = self.row_start + local;
            let lev = levels[i] as usize;
            let mut acc = 0.0;
            for c in 0..t.layout().n_columns() {
                let coef = block[t.layout().block_column(c)][lev];
                acc += match t.loading(c) {
                    None => coef,
                    Some(z) => coef * z[i],
                };
            }
            *dst_val += acc;
        }
    }
}

/// One term sweep with a compile-time column count.
#[inline(always)]
fn gather_term<const N: usize>(
    chunk: &mut [f64],
    row_start: usize,
    levels: &[u32],
    cols: [&[f64]; N],
    weights: impl Fn(usize) -> [f64; N],
) {
    for (local, dst_val) in chunk.iter_mut().enumerate() {
        let i = row_start + local;
        let lev = levels[i] as usize;
        let row = cols.iter().zip(weights(i)).map(|(col, w)| col[lev] * w);
        // Fold from -0.0, not 0.0: the true additive identity, folds away.
        *dst_val += row.fold(-0.0, |acc, term| acc + term);
    }
}

/// `dst = D̃ᵀ (scale ⊙ src)`. The terms' blocks and fold workspaces are disjoint, so the terms run
/// concurrently. `folds` holds each term's fold workspace, in term order, and `stack` its scratch.
pub(super) fn scatter(
    design: &OrthogonalDesign<'_>,
    folds: &Workspaces,
    src: &[f64],
    dst: &mut [f64],
    scale: Option<&[f64]>,
    stack: &mut MemStack,
) {
    debug_assert!(scale.is_none_or(|s| s.len() == src.len()));
    debug_assert_eq!(design.n_obs(), src.len());
    let parts = (
        Pieces::new(dst, design.unknowns().starts()),
        folds.carve(stack),
    );
    for_each_piece(parts, &|i, (block, workspace)| {
        let (t, block) = (&design.term(i), block.into_items());
        let stack = MemStack::new(workspace.into_items());
        match scale {
            Some(s) => scatter_block(t, block, &|i| s[i] * src[i], stack),
            None => scatter_block(t, block, &|i| src[i], stack),
        }
    });
}

/// One term's `block = D_tᵀ v`, with `base(i)` the row value of `v`.
#[inline(always)]
fn scatter_block(
    t: &TermBlock<'_>,
    block: &mut [f64],
    base: &(impl Fn(usize) -> f64 + Sync),
    stack: &mut MemStack,
) {
    t.visit_rows(ScatterTerm {
        block,
        term: t,
        base,
        stack,
    });
}

struct ScatterTerm<'k, 't, B> {
    block: &'k mut [f64],
    term: &'k TermBlock<'t>,
    base: &'k B,
    stack: &'k mut MemStack,
}

impl<B: Fn(usize) -> f64 + Sync> RowKernel for ScatterTerm<'_, '_, B> {
    #[inline(always)]
    fn fixed<const C: usize>(self, row: impl Fn(usize) -> [f64; C] + Sync) {
        let base = self.base;
        scatter_term::<C>(
            self.block,
            self.term,
            |i| {
                let b = base(i);
                row(i).map(|r| r * b)
            },
            self.stack,
        );
    }

    #[inline(always)]
    fn dynamic(self) {
        let (t, base, stack) = (self.term, self.base, self.stack);
        for c in 0..t.layout().n_columns() {
            let slot = &mut self.block[t.layout().block_column(c)];
            match t.loading(c) {
                None => scatter_term::<1>(slot, t, |i| [base(i)], stack),
                Some(z) => scatter_term::<1>(slot, t, move |i| [z[i] * base(i)], stack),
            }
        }
    }
}

/// Write one term's coefficient block: `block[c·L + l] = Σ_{i : level(i) = l} values(i)[c]`.
fn scatter_term<const C: usize>(
    block: &mut [f64],
    term: &TermBlock<'_>,
    values: impl Fn(usize) -> [f64; C] + Sync,
    stack: &mut MemStack,
) {
    let n_levels = term.layout().n_levels();
    let mut columns = block.chunks_mut(n_levels);
    let columns = Columns(std::array::from_fn(|_| columns.next().expect("C columns")));
    term.incidence().fold_with(
        [0.0; C],
        &|sums: &mut [f64; C], _, i| {
            for (s, v) in sums.iter_mut().zip(values(i)) {
                *s += v;
            }
        },
        &|left: &mut [f64; C], right: &[f64; C]| {
            for (l, r) in left.iter_mut().zip(right) {
                *l += r;
            }
        },
        columns,
        stack,
    );
}

/// A column-major `C × L` block as per-level `[f64; C]` slots.
struct Columns<'a, const C: usize>([&'a mut [f64]; C]);

impl<const C: usize> Split for Columns<'_, C> {
    fn n_items(&self) -> usize {
        self.0[0].len()
    }

    fn split_at(self, mid: usize) -> (Self, Self) {
        let mut left = self.0;
        let right = left.each_mut().map(|column| {
            let (l, r) = std::mem::take(column).split_at_mut(mid);
            *column = l;
            r
        });
        (Self(left), Self(right))
    }
}

impl<const C: usize> LevelSlots<[f64; C]> for Columns<'_, C> {
    fn set(&mut self, level: usize, sums: [f64; C]) {
        for (column, s) in self.0.iter_mut().zip(sums) {
            column[level] = s;
        }
    }
}
