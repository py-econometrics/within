//! Scatter: observation space → coefficient space, `x_t = D_tᵀ (s ⊙ v)`, summed per term in the
//! order its [`LevelReduction`] fixes.

use std::mem::MaybeUninit;

use rayon::prelude::*;
use schwarz_precond::{MemStack, StackReq};

use crate::design::LevelSlots;
use crate::design::{RowKernel, TermBlock};

/// What [`scatter`] takes from its stack: each term's fold workspace, side by side.
pub(crate) fn scatter_scratch(terms: &[TermBlock<'_>]) -> StackReq {
    terms
        .iter()
        .map(term_workspace)
        .fold(StackReq::EMPTY, StackReq::and)
}

/// One term's fold workspace for its widest per-level state, `[f64; C]` over its `C` columns.
fn term_workspace(t: &TermBlock<'_>) -> StackReq {
    t.fold_scratch::<f64>().array(t.layout().n_columns())
}

/// `dst[dofs_t] = D_tᵀ (scale ⊙ src)` for `terms` in layout order; other slots are left as they
/// are. The blocks and their workspaces are disjoint, so the terms run concurrently. `stack` holds
/// at least [`scatter_scratch`].
pub(crate) fn scatter(
    terms: &[TermBlock<'_>],
    src: &[f64],
    dst: &mut [f64],
    scale: Option<&[f64]>,
    stack: &mut MemStack,
) {
    debug_assert!(scale.is_none_or(|s| s.len() == src.len()));
    debug_assert!(terms.iter().all(|t| t.levels().len() == src.len()));
    let mut parts: Vec<(&mut [f64], &mut [MaybeUninit<u8>])> = Vec::with_capacity(terms.len());
    let (mut rest, mut start, mut stack) = (dst, 0, stack);
    for t in terms {
        let dofs = t.dofs();
        let tail = std::mem::take(&mut rest).split_at_mut(dofs.start - start).1;
        let (block, tail) = tail.split_at_mut(dofs.len());
        let req = term_workspace(t);
        let (workspace, below) =
            stack.make_aligned_uninit::<u8>(req.size_bytes(), req.align_bytes());
        parts.push((block, workspace));
        (rest, start, stack) = (tail, dofs.end, below);
    }
    terms
        .par_iter()
        .zip(parts)
        .for_each(|(t, (block, workspace))| {
            let stack = MemStack::new(workspace);
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
    term.fold_with(
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

impl<const C: usize> LevelSlots<[f64; C]> for Columns<'_, C> {
    fn len(&self) -> usize {
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

    fn set(&mut self, level: usize, sums: [f64; C]) {
        for (column, s) in self.0.iter_mut().zip(sums) {
            column[level] = s;
        }
    }
}
