//! Scatter: observation space → coefficient space, `x_t = D_tᵀ (s ⊙ v)`, summed per term in the
//! order its [`LevelReduction`] fixes.

use schwarz_precond::parallel::{for_each_piece, Pieces, Split};
use schwarz_precond::{MemStack, StackReq};

use crate::design::LevelSlots;
use crate::design::{RowKernel, TermBlock};

/// Every term's workspace starts on this boundary.
const WORKSPACE_ALIGN: usize = 64;

/// What [`scatter`] takes from its stack: where each term's block and workspace start, then the
/// workspaces.
pub(crate) fn scatter_scratch(terms: &[TermBlock<'_>]) -> StackReq {
    let bounds = StackReq::new::<usize>(terms.len() + 1);
    let workspaces = workspace_bounds(terms).last().unwrap_or(0);
    bounds
        .and(bounds)
        .and(StackReq::new_aligned::<u8>(workspaces, WORKSPACE_ALIGN))
}

/// Each term's first DOF, then the last term's end.
fn block_bounds<'a>(terms: &'a [TermBlock<'_>]) -> impl Iterator<Item = usize> + 'a {
    let end = terms.last().map_or(0, |t| t.dofs().end);
    terms.iter().map(|t| t.dofs().start).chain([end])
}

/// Each term's workspace offset, then their end: a fold workspace for the per-level state
/// [`TermBlock::visit_rows`] hands its kernel.
fn workspace_bounds<'a>(terms: &'a [TermBlock<'_>]) -> impl Iterator<Item = usize> + 'a {
    std::iter::once(0).chain(terms.iter().scan(0, |end, t| {
        let workspace = t.fold_scratch::<f64>().array(state_width(t));
        debug_assert!(workspace.align_bytes() <= WORKSPACE_ALIGN);
        *end += workspace.size_bytes().next_multiple_of(WORKSPACE_ALIGN);
        Some(*end)
    }))
}

/// Columns per state of `t`'s fold: all of them where fused, else one at a time.
fn state_width(t: &TermBlock<'_>) -> usize {
    struct Width<'a>(&'a mut usize);
    impl RowKernel for Width<'_> {
        fn fixed<const C: usize>(self, _: impl Fn(usize) -> [f64; C] + Sync) {
            *self.0 = C;
        }
        fn dynamic(self) {
            *self.0 = 1;
        }
    }
    let mut width = 0;
    t.visit_rows(Width(&mut width));
    width
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
    let (blocks, stack) = stack.collect(block_bounds(terms));
    let (workspace_bounds, stack) = stack.collect(workspace_bounds(terms));
    let (workspaces, _) =
        stack.make_aligned_uninit::<u8>(workspace_bounds[terms.len()], WORKSPACE_ALIGN);
    let parts = (
        Pieces::new(dst, &blocks),
        Pieces::new(workspaces, &workspace_bounds),
    );
    for_each_piece(parts, &|i, (block, workspace)| {
        let t = &terms[i];
        let block = &mut block.into_items()[..t.dofs().len()];
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
