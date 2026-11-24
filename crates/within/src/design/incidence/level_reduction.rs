//! How a term's per-level reduction `E_tᵀ` is cut into fixed parallel pieces, chosen from its rows.

use std::sync::Arc;

use inflight::parallel::{fold_leaves, Blocks, Chunks, Threads};

use super::{LevelSegments, RowIndex};

/// Rows per level from which splitting by row beats splitting by level.
const FOLD_ROWS_PER_LEVEL: usize = 12;

/// The pieces of `z[l] = Σ_{level(i) = l} vᵢ`; the thread count only schedules them.
#[derive(Clone, Debug)]
pub(crate) enum LevelReduction {
    /// Each level sums its segment of positions, rows themselves when `None`: fixed leaves in
    /// position order, added by a fixed binary tree. Shared, as the sort term's are the row order's.
    Segments(Arc<LevelSegments>, Option<RowIndex>),
    /// Fixed row blocks scatter into partial blocks that a fixed binary tree adds up.
    ByRow,
}

impl LevelReduction {
    /// The reduction of `levels`.
    pub(crate) fn new(levels: &[u32], n_levels: usize) -> Self {
        if let Some(segments) = LevelSegments::of(levels, n_levels) {
            return Self::Segments(Arc::new(segments), None);
        }
        if levels.len() >= FOLD_ROWS_PER_LEVEL * n_levels {
            return Self::ByRow;
        }
        let (offsets, rows) = RowIndex::argsort(levels, n_levels);
        Self::Segments(Arc::new(LevelSegments::new(offsets)), Some(rows))
    }

    /// Each level's fold of its rows: `step(state, level, row)` from `init` over pieces this
    /// reduction fixes, `merge`d left to right in a fixed binary tree, into `out[level]`; threads
    /// only schedule the pieces.
    pub(super) fn fold<S: Copy + Send + Sync>(
        &self,
        levels: &[u32],
        init: S,
        step: &(impl Fn(&mut S, usize, usize) + Sync),
        merge: &(impl Fn(&mut S, &S) + Sync),
        out: &mut [S],
    ) {
        let n_levels = out.len();
        match self {
            Self::Segments(segments, rows) => segments.fold(rows.as_ref(), init, step, merge, out),
            Self::ByRow => {
                let leaf_len = by_row_leaf_len(levels.len(), n_levels);
                let leaves = fold_leaves(levels.len(), leaf_len);
                let mut partials = vec![init; leaves * n_levels];
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
                out.copy_from_slice(states);
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
