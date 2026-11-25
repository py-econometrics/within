//! Each level's positions as one segment, and the segmented passes over them.

use std::ops::Range;

use inflight::parallel::{Blocks, Pieces, Split, Threads};

use super::{LevelSlots, RowIndex};
use crate::PAR_THRESHOLD;

/// Rows of whole segments per task, at least.
const TASK_ROWS: usize = 16_384;
/// Rows per leaf of one level's fold; above `PAR_THRESHOLD`, so a sequential fold never joins.
pub(crate) const LEVEL_LEAF: usize = PAR_THRESHOLD.next_power_of_two();

/// Level `l` owns the positions `offsets[l]..offsets[l + 1]`: rows, for rows sorted by level, or
/// positions of a [`RowIndex`]. Its passes cut the segments into tasks of whole segments by the
/// offsets alone: every pass is independent of the thread count.
#[derive(Clone, Debug)]
pub(crate) struct LevelSegments {
    offsets: Vec<usize>,
    /// Task `k` covers levels `tasks[k]..tasks[k + 1]`.
    tasks: Vec<usize>,
}

impl LevelSegments {
    /// The segments of `levels` if they are non-decreasing, else `None`.
    pub(super) fn of(levels: &[u32], n_levels: usize) -> Option<Self> {
        let mut offsets = vec![0; n_levels + 1];
        let mut previous = None;
        // One store per segment: a per-row increment chains each through one counter.
        for segment in levels.chunk_by(|a, b| a == b) {
            if previous > Some(segment[0]) {
                return None;
            }
            previous = Some(segment[0]);
            offsets[segment[0] as usize + 1] = segment.len();
        }
        for level in 0..n_levels {
            offsets[level + 1] += offsets[level];
        }
        Some(Self::new(offsets))
    }

    /// Segments whose level `l` owns rows `offsets[l]..offsets[l + 1]`, in tasks of at least
    /// [`TASK_ROWS`] rows.
    pub(super) fn new(offsets: Vec<usize>) -> Self {
        let n_levels = offsets.len() - 1;
        let (mut tasks, mut start) = (vec![0], 0);
        for (level, &end) in offsets.iter().enumerate().skip(1) {
            if end - start >= TASK_ROWS || level == n_levels {
                tasks.push(level);
                start = end;
            }
        }
        Self { offsets, tasks }
    }

    pub(crate) fn n_levels(&self) -> usize {
        self.offsets.len() - 1
    }

    /// Level `level`'s segment.
    pub(crate) fn rows(&self, level: usize) -> Range<usize> {
        self.offsets[level]..self.offsets[level + 1]
    }

    /// Each level's fold of its rows: `step(state, level, row)` from `init` over leaves of
    /// [`LEVEL_LEAF`] positions, `merge`d in a fixed binary tree, into `out`'s slot of the level;
    /// `rows` maps positions to rows, `None` for rows sorted by level.
    pub(super) fn fold<S: Copy + Send + Sync>(
        &self,
        rows: Option<&RowIndex>,
        init: S,
        step: &(impl Fn(&mut S, usize, usize) + Sync),
        merge: &(impl Fn(&mut S, &S) + Sync),
        out: impl LevelSlots<S>,
    ) {
        let level_fold = |level: usize| {
            let leaf = |at: Range<usize>| {
                let mut state = init;
                match rows {
                    None => at.for_each(|i| step(&mut state, level, i)),
                    Some(rows) => rows.for_each(at, |i| step(&mut state, level, i)),
                }
                state
            };
            let merge = |mut left: S, right: S| {
                merge(&mut left, &right);
                left
            };
            let at = self.rows(level);
            Threads::All.fold_tree(Blocks::new(at, LEVEL_LEAF), |(_, at)| leaf(at), merge)
        };
        self.for_each_task(Pieces::new(out, &self.tasks), |levels, mut slots| {
            for (k, level) in levels.enumerate() {
                slots.set(k, level_fold(level));
            }
        });
    }

    /// `f(levels, part)` for each task, `part` its item of `parts`, one per task; tasks run
    /// concurrently.
    fn for_each_task<P: Split>(&self, parts: P, f: impl Fn(Range<usize>, P::Item) + Sync) {
        let levels = Pieces::new(0..self.n_levels(), &self.tasks);
        Threads::All.for_each_item((levels, parts), |_, (levels, part)| f(levels, part));
    }
}

#[cfg(test)]
mod tests {
    use super::LevelSegments;

    #[test]
    fn sorted_levels_count_their_segments_and_a_descent_has_none() {
        let segments = LevelSegments::of(&[0, 0, 1, 3, 3], 4).unwrap();
        assert_eq!(segments.offsets, [0, 2, 3, 3, 5]);
        assert!(LevelSegments::of(&[1, 1, 0], 2).is_none());
    }
}
