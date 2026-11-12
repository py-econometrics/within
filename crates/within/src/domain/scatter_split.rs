//! How a term's scatter-add splits into fixed parallel pieces, chosen from its rows alone.

use std::ops::Range;

use rayon::prelude::*;

use super::{Term, PAR_THRESHOLD};

/// Rows per level from which splitting by row beats splitting by level.
const FOLD_ROWS_PER_LEVEL: usize = 12;

/// Levels per parallel task of a per-level pass.
pub(crate) const LEVEL_GROUP: usize = 1024;

/// Rows per leaf of one level's fold; above `PAR_THRESHOLD`, so a sequential fold never joins.
pub(crate) const LEVEL_LEAF: usize = PAR_THRESHOLD.next_power_of_two();

/// Where a fold writes each level's state; splits into disjoint level ranges for parallel tasks.
pub(crate) trait LevelSlots<S>: Send + Sized {
    fn len(&self) -> usize;
    fn split_at(self, mid: usize) -> (Self, Self);
    fn set(&mut self, level: usize, state: S);
}

impl<S: Send> LevelSlots<S> for &mut [S] {
    fn len(&self) -> usize {
        <[S]>::len(self)
    }

    fn split_at(self, mid: usize) -> (Self, Self) {
        self.split_at_mut(mid)
    }

    fn set(&mut self, level: usize, state: S) {
        self[level] = state;
    }
}

/// The split of `z[l] = Σ_{level(i) = l} vᵢ` into pieces; the thread count only schedules them.
pub(crate) enum ScatterSplit {
    /// Each level sums its own rows: fixed row blocks in row order, added by a fixed binary tree.
    ByLevel(LevelRows),
    /// Fixed row blocks scatter into partial blocks that a fixed binary tree adds up.
    ByRow,
}

impl ScatterSplit {
    pub(crate) fn new(term: &Term<'_>) -> Self {
        let (levels, n_levels) = (term.levels(), term.n_levels());
        if !term.sorted() {
            // An unsorted term's row index stores u32 rows.
            if levels.len() >= FOLD_ROWS_PER_LEVEL * n_levels
                || u32::try_from(levels.len()).is_err()
            {
                return Self::ByRow;
            }
            let (offsets, rows) = super::stable_argsort(levels, n_levels);
            let rows = Some(rows.into_boxed_slice());
            return Self::ByLevel(LevelRows { offsets, rows });
        }
        let mut offsets = vec![0; n_levels + 1];
        // One store per run: a per-row increment chains each run through one counter.
        for run in levels.chunk_by(|a, b| a == b) {
            offsets[run[0] as usize + 1] = run.len();
        }
        for level in 0..n_levels {
            offsets[level + 1] += offsets[level];
        }
        Self::ByLevel(LevelRows {
            offsets,
            rows: None,
        })
    }
}

impl ScatterSplit {
    /// Each level's fold of its rows: `step(state, level, row)` from `init` over pieces this split
    /// fixes, `merge`d left to right in a fixed binary tree; threads only schedule the pieces.
    pub(crate) fn fold<S: Copy + Send + Sync, O: LevelSlots<S>>(
        &self,
        levels: &[u32],
        init: S,
        step: &(impl Fn(&mut S, usize, usize) + Sync),
        merge: &(impl Fn(&mut S, &S) + Sync),
        mut out: O,
    ) {
        let parallel = levels.len() > PAR_THRESHOLD;
        let n_levels = out.len();
        match self {
            Self::ByLevel(rows) => {
                let level_fold = |level: usize| {
                    let leaf = |positions: Range<usize>| {
                        let mut state = init;
                        rows.for_each_row(positions, |i| step(&mut state, level, i));
                        state
                    };
                    let positions = rows.positions(level);
                    if positions.len() <= LEVEL_LEAF {
                        return leaf(positions);
                    }
                    fixed_tree(positions, LEVEL_LEAF, &leaf, merge)
                };
                let mut groups = Vec::with_capacity(n_levels.div_ceil(LEVEL_GROUP));
                while out.len() > 0 {
                    let mid = LEVEL_GROUP.min(out.len());
                    let (group, rest) = out.split_at(mid);
                    groups.push(group);
                    out = rest;
                }
                let fill = |(g, mut group): (usize, O)| {
                    for k in 0..LevelSlots::len(&group) {
                        group.set(k, level_fold(g * LEVEL_GROUP + k));
                    }
                };
                if parallel {
                    groups.into_par_iter().enumerate().for_each(fill);
                } else {
                    groups.into_iter().enumerate().for_each(fill);
                }
            }
            Self::ByRow => {
                // Merging costs ∝ L per state word, a leaf ∝ its rows per word: 4·L rows bound merges.
                let leaf_len = (4 * n_levels).min(levels.len().div_ceil(2)).max(65_536);
                let scatter = |rows: Range<usize>| {
                    let mut states = vec![init; n_levels];
                    for i in rows {
                        let level = levels[i] as usize;
                        step(&mut states[level], level, i);
                    }
                    states
                };
                let merge_all = |left: &mut Vec<S>, right: &Vec<S>| {
                    for (l, r) in left.iter_mut().zip(right) {
                        merge(l, r);
                    }
                };
                let states = fixed_tree(0..levels.len(), leaf_len, &scatter, &merge_all);
                for (level, state) in states.into_iter().enumerate() {
                    out.set(level, state);
                }
            }
        }
    }
}

/// `leaf` over `leaf_len` pieces of `items`, merged in a binary tree fixed by both lengths alone.
fn fixed_tree<T: Send>(
    items: Range<usize>,
    leaf_len: usize,
    leaf: &(impl Fn(Range<usize>) -> T + Sync),
    merge: &(impl Fn(&mut T, &T) + Sync),
) -> T {
    if items.len() <= leaf_len {
        return leaf(items);
    }
    let mid = items.start + (items.len() / 2 / leaf_len).max(1) * leaf_len;
    let (mut left, right) = rayon::join(
        || fixed_tree(items.start..mid, leaf_len, leaf, merge),
        || fixed_tree(mid..items.end, leaf_len, leaf, merge),
    );
    merge(&mut left, &right);
    left
}

/// The rows of each level, ascending: the inverse of the row → level map.
pub(crate) struct LevelRows {
    /// Level `l` owns positions `offsets[l]..offsets[l + 1]`.
    offsets: Vec<usize>,
    /// Row at each position; `None` when the levels are sorted, so positions are rows.
    rows: Option<Box<[u32]>>,
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
            Some(rows) => rows[positions].iter().map(|&r| r as usize).for_each(f),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{Design, Effect};

    #[test]
    fn level_rows_list_each_levels_rows_ascending() {
        let ScatterSplit::ByLevel(rows) = split(&[2, 0, 2, 1, 0]) else {
            panic!()
        };
        assert_eq!(
            (rows.offsets, rows.rows.as_deref()),
            (vec![0, 2, 3, 5], Some(&[1, 4, 3, 0, 2][..]))
        );
        let ScatterSplit::ByLevel(sorted) = split(&[0, 0, 1, 1, 1]) else {
            panic!()
        };
        assert_eq!((sorted.offsets, sorted.rows), (vec![0, 2, 5], None));
    }

    fn split(levels: &[u32]) -> ScatterSplit {
        let slope = vec![1.0; levels.len()];
        let effect = Effect::new(levels, true, vec![&slope[..]; 4]).expect("valid effect");
        ScatterSplit::new(&Design::new_unsorted([effect]).expect("valid design").terms[0])
    }

    #[test]
    fn by_row_needs_unsorted_rows_and_enough_rows_per_level() {
        let levels: Vec<u32> = (0..48).map(|i| i % 2).collect();
        assert!(matches!(split(&levels), ScatterSplit::ByRow));
        let sorted: Vec<u32> = (0..48).map(|i| i / 24).collect();
        assert!(matches!(split(&sorted), ScatterSplit::ByLevel(_)));
        assert!(matches!(split(&levels[..23]), ScatterSplit::ByLevel(_)));
        assert!(matches!(split(&levels[..24]), ScatterSplit::ByRow));
    }
}
