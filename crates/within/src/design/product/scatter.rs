//! Scatter: observation space → coefficient space, `x_t = D_tᵀ (s ⊙ v)`, summed per term in the
//! order its [`LevelReduction`] fixes.

use rayon::prelude::*;

use crate::design::LevelSlots;
use crate::design::{RowKernel, TermBlock};

/// `dst[dofs_t] = D_tᵀ (scale ⊙ src)` for `terms` in layout order; other slots are left as they
/// are. The blocks are disjoint, so the terms run concurrently.
pub(crate) fn scatter(
    terms: &[TermBlock<'_>],
    src: &[f64],
    dst: &mut [f64],
    scale: Option<&[f64]>,
) {
    debug_assert!(scale.is_none_or(|s| s.len() == src.len()));
    debug_assert!(terms.iter().all(|t| t.levels().len() == src.len()));
    let mut blocks = Vec::with_capacity(terms.len());
    let (mut rest, mut start) = (dst, 0);
    for t in terms {
        let dofs = t.layout().dofs();
        let tail = std::mem::take(&mut rest).split_at_mut(dofs.start - start).1;
        let (block, tail) = tail.split_at_mut(dofs.len());
        blocks.push(block);
        (rest, start) = (tail, dofs.end);
    }
    terms
        .par_iter()
        .zip(blocks)
        .for_each(|(t, block)| match scale {
            Some(s) => scatter_block(t, block, &|i| s[i] * src[i]),
            None => scatter_block(t, block, &|i| src[i]),
        });
}

/// One term's `block = D_tᵀ v`, with `base(i)` the row value of `v`.
#[inline(always)]
fn scatter_block(t: &TermBlock<'_>, block: &mut [f64], base: &(impl Fn(usize) -> f64 + Sync)) {
    t.visit_rows(ScatterTerm {
        block,
        term: t,
        base,
    });
}

struct ScatterTerm<'k, 't, B> {
    block: &'k mut [f64],
    term: &'k TermBlock<'t>,
    base: &'k B,
}

impl<B: Fn(usize) -> f64 + Sync> RowKernel for ScatterTerm<'_, '_, B> {
    #[inline(always)]
    fn fixed<const C: usize>(self, row: impl Fn(usize) -> [f64; C] + Sync) {
        let base = self.base;
        scatter_term::<C>(self.block, self.term, |i| {
            let b = base(i);
            row(i).map(|r| r * b)
        });
    }

    #[inline(always)]
    fn dynamic(self) {
        let (t, base) = (self.term, self.base);
        for c in 0..t.layout().n_columns() {
            let slot = &mut self.block[t.layout().block_column(c)];
            match t.loading(c) {
                None => scatter_term::<1>(slot, t, |i| [base(i)]),
                Some(z) => scatter_term::<1>(slot, t, move |i| [z[i] * base(i)]),
            }
        }
    }
}

/// Write one term's coefficient block: `block[c·L + l] = Σ_{i : level(i) = l} values(i)[c]`.
fn scatter_term<const C: usize>(
    block: &mut [f64],
    term: &TermBlock<'_>,
    values: impl Fn(usize) -> [f64; C] + Sync,
) {
    let n_levels = term.layout().n_levels();
    let mut columns = block.chunks_mut(n_levels);
    let columns = Columns(std::array::from_fn(|_| columns.next().expect("C columns")));
    term.fold(
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
