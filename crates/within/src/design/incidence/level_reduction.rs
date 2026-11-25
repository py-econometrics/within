//! How a term's per-level reduction `E_tᵀ` is cut into fixed parallel pieces, chosen from its rows.

use std::ops::Range;

use schwarz_precond::parallel::{fold, for_each_piece, Split};
use schwarz_precond::{MemStack, StackReq};

use super::RowIndex;
use crate::PAR_THRESHOLD;

/// Rows per level from which splitting by row beats splitting by level.
const FOLD_ROWS_PER_LEVEL: usize = 12;

/// Levels per parallel task of a per-level pass.
pub(crate) const LEVEL_GROUP: usize = 1024;

/// Rows per leaf of one level's fold; above `PAR_THRESHOLD`, so a sequential fold never joins.
pub(crate) const LEVEL_LEAF: usize = PAR_THRESHOLD.next_power_of_two();

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
    /// Each level sums its own rows: fixed row blocks in row order, added by a fixed binary tree.
    ByLevel(LevelRows),
    /// Fixed row blocks scatter into partial blocks that a fixed binary tree adds up.
    ByRow,
}

impl LevelReduction {
    /// The reduction of `levels`.
    pub(crate) fn new(levels: &[u32], n_levels: usize) -> Self {
        if let Some(reduction) = Self::of_sorted(levels, n_levels) {
            return reduction;
        }
        if levels.len() >= FOLD_ROWS_PER_LEVEL * n_levels {
            return Self::ByRow;
        }
        let (offsets, rows) = RowIndex::argsort(levels, n_levels);
        Self::ByLevel(LevelRows {
            offsets,
            rows: Some(rows),
        })
    }

    /// The reduction of `levels` if they are non-decreasing, else `None`.
    pub(crate) fn of_sorted(levels: &[u32], n_levels: usize) -> Option<Self> {
        let mut offsets = vec![0; n_levels + 1];
        let mut previous = None;
        // One store per run: a per-row increment chains each run through one counter.
        for run in levels.chunk_by(|a, b| a == b) {
            if previous > Some(run[0]) {
                return None;
            }
            previous = Some(run[0]);
            offsets[run[0] as usize + 1] = run.len();
        }
        for level in 0..n_levels {
            offsets[level + 1] += offsets[level];
        }
        Some(Self::sorted(offsets))
    }

    /// The reduction of sorted rows whose level `l` owns rows `offsets[l]..offsets[l + 1]`.
    pub(crate) fn sorted(offsets: Vec<usize>) -> Self {
        Self::ByLevel(LevelRows {
            offsets,
            rows: None,
        })
    }
}

impl LevelReduction {
    /// What [`fold_with`](Self::fold_with) takes from its stack for states `S` over `n_rows` rows.
    pub(super) fn scratch<S>(&self, n_rows: usize, n_levels: usize) -> StackReq {
        match self {
            Self::ByLevel(_) => StackReq::EMPTY,
            Self::ByRow => StackReq::new::<S>(by_row_leaves(n_rows, n_levels) * n_levels),
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
        let parallel = levels.len() > PAR_THRESHOLD;
        let n_levels = out.n_items();
        match self {
            Self::ByLevel(rows) => {
                let level_fold = |level: usize| {
                    let leaf = |positions: Range<usize>| {
                        let mut state = init;
                        rows.for_each_row(positions, |i| step(&mut state, level, i));
                        state
                    };
                    fold(
                        rows.positions(level),
                        LEVEL_LEAF,
                        &leaf,
                        &|mut left, right| {
                            merge(&mut left, &right);
                            left
                        },
                    )
                };
                let fill = |first: usize, mut slots: O| {
                    for k in 0..slots.n_items() {
                        slots.set(k, level_fold(first + k));
                    }
                };
                if parallel {
                    for_each_piece(Groups(out), &|g, Groups(group)| {
                        fill(g * LEVEL_GROUP, group)
                    });
                } else {
                    fill(0, out);
                }
            }
            Self::ByRow => {
                let leaf_len = by_row_leaf_len(levels.len(), n_levels);
                let leaves = by_row_leaves(levels.len(), n_levels);
                let (mut partials, _) = stack.make_with(leaves * n_levels, |_| init);
                let all = RowLeaves {
                    rows: 0..levels.len(),
                    partials: &mut partials,
                    leaf_len,
                    n_levels,
                };
                let states = fold(
                    all,
                    leaf_len,
                    &|leaf| leaf.scatter(levels, step),
                    &|left, right| {
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

/// Level slots cut into [`LEVEL_GROUP`] levels per piece, the last one shorter.
struct Groups<O>(O);

impl<O: Split> Split for Groups<O> {
    fn n_items(&self) -> usize {
        self.0.n_items().div_ceil(LEVEL_GROUP)
    }
    fn split_at(self, mid: usize) -> (Self, Self) {
        let cut = (mid * LEVEL_GROUP).min(self.0.n_items());
        let (left, right) = self.0.split_at(cut);
        (Groups(left), Groups(right))
    }
}

/// Rows per leaf of a by-row fold: merging costs ∝ L per state word, a leaf ∝ its rows per word,
/// so 4·L rows bound the merges.
fn by_row_leaf_len(n_rows: usize, n_levels: usize) -> usize {
    (4 * n_levels).min(n_rows.div_ceil(2)).max(65_536)
}

/// Leaves of a by-row fold over `n_rows` rows, at least one.
fn by_row_leaves(n_rows: usize, n_levels: usize) -> usize {
    n_rows.div_ceil(by_row_leaf_len(n_rows, n_levels)).max(1)
}

/// Rows in leaves of `leaf_len`, each leaf with its own `n_levels` states of `partials`.
struct RowLeaves<'a, S> {
    rows: Range<usize>,
    partials: &'a mut [S],
    leaf_len: usize,
    n_levels: usize,
}

impl<'a, S> RowLeaves<'a, S> {
    /// `step` over every row into this leaf's states, which it returns.
    fn scatter(self, levels: &[u32], step: &impl Fn(&mut S, usize, usize)) -> &'a mut [S] {
        let states = &mut self.partials[..self.n_levels];
        for i in self.rows {
            let level = levels[i] as usize;
            step(&mut states[level], level, i);
        }
        states
    }
}

impl<S: Send> Split for RowLeaves<'_, S> {
    fn n_items(&self) -> usize {
        self.rows.len()
    }
    fn split_at(self, mid: usize) -> (Self, Self) {
        let (rows, rest) = self.rows.split_at(mid);
        let (partials, others) = self
            .partials
            .split_at_mut(mid / self.leaf_len * self.n_levels);
        let left = RowLeaves {
            rows,
            partials,
            ..self
        };
        let right = RowLeaves {
            rows: rest,
            partials: others,
            ..self
        };
        (left, right)
    }
}

/// The rows of each level, ascending: the inverse of the row → level map.
#[derive(Clone, Debug)]
pub(crate) struct LevelRows {
    /// Level `l` owns positions `offsets[l]..offsets[l + 1]`.
    offsets: Vec<usize>,
    /// Row at each position; `None` when the levels are sorted, so positions are rows.
    rows: Option<RowIndex>,
}

impl LevelRows {
    /// Positions of `level`'s rows, for [`for_each_row`](Self::for_each_row).
    fn positions(&self, level: usize) -> Range<usize> {
        self.offsets[level]..self.offsets[level + 1]
    }

    /// Calls `f` on the row at each of `positions`, in order.
    #[inline(always)]
    fn for_each_row(&self, positions: Range<usize>, f: impl FnMut(usize)) {
        match &self.rows {
            None => positions.for_each(f),
            Some(rows) => rows.for_each(positions, f),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{LevelReduction, LevelRows};

    #[test]
    fn of_sorted_counts_runs_and_rejects_a_descent() {
        let reduction = LevelReduction::of_sorted(&[0, 0, 1, 3, 3], 4).unwrap();
        let LevelReduction::ByLevel(LevelRows {
            offsets,
            rows: None,
        }) = reduction
        else {
            panic!("sorted levels split by level without a row index")
        };
        assert_eq!(offsets, [0, 2, 3, 3, 5]);
        assert!(LevelReduction::of_sorted(&[1, 1, 0], 2).is_none());
    }
}
