//! How a term's per-level reduction `E_tᵀ` is cut into fixed parallel pieces, chosen from its rows.

use inflight::parallel::{fold_leaves, Blocks, Chunks, Split, Threads};
use schwarz_precond::{MemStack, StackReq};

use super::{LevelSegments, RowIndex};

/// Rows per level from which splitting by row beats splitting by level.
const FOLD_ROWS_PER_LEVEL: usize = 12;

/// Where a fold writes each level's state, one item per level; splits into disjoint level ranges
/// for parallel tasks.
pub(crate) trait LevelSlots<S>: Split {
    fn set(&mut self, level: usize, state: S);
}

impl<S: Send> LevelSlots<S> for &mut [S] {
    fn set(&mut self, level: usize, state: S) {
        self[level] = state;
    }
}

/// The pieces of `z[l] = Σ_{level(i) = l} vᵢ`; the thread count only schedules them.
#[derive(Clone, Debug)]
pub(crate) enum LevelReduction {
    /// Each level sums its segment of positions, rows themselves when `None`: fixed leaves in
    /// position order, added by a fixed binary tree.
    Segments(LevelSegments, Option<RowIndex>),
    /// Fixed row blocks scatter into partial blocks that a fixed binary tree adds up.
    ByRow,
}

impl LevelReduction {
    /// The reduction of `levels`.
    pub(crate) fn new(levels: &[u32], n_levels: usize) -> Self {
        if let Some(segments) = LevelSegments::of(levels, n_levels) {
            return Self::Segments(segments, None);
        }
        if levels.len() >= FOLD_ROWS_PER_LEVEL * n_levels {
            return Self::ByRow;
        }
        let (offsets, rows) = RowIndex::argsort(levels, n_levels);
        Self::Segments(LevelSegments::new(offsets), Some(rows))
    }
}

impl LevelReduction {
    /// What [`fold_with`](Self::fold_with) takes from its stack for states `S` over `n_rows` rows.
    pub(super) fn scratch<S>(&self, n_rows: usize, n_levels: usize) -> StackReq {
        match self {
            Self::Segments(..) => StackReq::EMPTY,
            Self::ByRow => {
                let leaves = fold_leaves(n_rows, by_row_leaf_len(n_rows, n_levels));
                StackReq::new::<S>(leaves * n_levels)
            }
        }
    }

    /// Each level's fold of its rows: `step(state, level, row)` from `init` over pieces this reduction
    /// fixes, `merge`d left to right in a fixed binary tree; threads only schedule the pieces. `stack`
    /// holds at least [`scratch`](Self::scratch).
    pub(super) fn fold_with<S: Copy + Send + Sync, O: LevelSlots<S>>(
        &self,
        levels: &[u32],
        init: S,
        step: &(impl Fn(&mut S, usize, usize) + Sync),
        merge: &(impl Fn(&mut S, &S) + Sync),
        mut out: O,
        stack: &mut MemStack,
    ) {
        let n_levels = out.n_items();
        match self {
            Self::Segments(segments, rows) => segments.fold(rows.as_ref(), init, step, merge, out),
            Self::ByRow => {
                let leaf_len = by_row_leaf_len(levels.len(), n_levels);
                let leaves = fold_leaves(levels.len(), leaf_len);
                let (mut partials, _) = stack.make_with(leaves * n_levels, |_| init);
                let states = Threads::All.fold_tree(
                    (
                        Blocks::new(levels, leaf_len),
                        Chunks::equal(&mut partials, leaves),
                    ),
                    |((start, levels), states)| scatter_rows(start, levels, states, step),
                    |left, right| {
                        for (l, r) in left.iter_mut().zip(&*right) {
                            merge(l, r);
                        }
                        left
                    },
                );
                for (level, &state) in states.iter().enumerate() {
                    out.set(level, state);
                }
            }
        }
    }
}

/// `step` over each row of `levels`, the first `start`, into its level's state of `states`.
/// Not a closure: `step` as an argument lets LLVM hoist a load of it out of the row loop.
fn scatter_rows<'a, S>(
    start: usize,
    levels: &[u32],
    states: &'a mut [S],
    step: &impl Fn(&mut S, usize, usize),
) -> &'a mut [S] {
    for (i, &level) in (start..).zip(levels) {
        let level = level as usize;
        step(&mut states[level], level, i);
    }
    states
}

/// Rows per leaf of a by-row fold: merging costs ∝ L per state word, a leaf ∝ its rows per word,
/// so 4·L rows bound the merges.
fn by_row_leaf_len(n_rows: usize, n_levels: usize) -> usize {
    (4 * n_levels).min(n_rows.div_ceil(2)).max(65_536)
}
