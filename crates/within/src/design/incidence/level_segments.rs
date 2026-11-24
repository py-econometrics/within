//! Each level's positions as one segment, and the segmented passes over them.

use std::mem;
use std::ops::Range;

use inflight::parallel::{Accumulators, Blocks, Chunks, Pieces, Split, Threads};
use schwarz_precond::{MemStack, StackReq};

use super::RowIndex;
use crate::PAR_THRESHOLD;

/// Rows of whole segments per task, at least.
const TASK_ROWS: usize = 16_384;
/// Rows per leaf of one level's fold; above `PAR_THRESHOLD`, so a sequential fold never joins.
pub(crate) const LEVEL_LEAF: usize = PAR_THRESHOLD.next_power_of_two();
/// Rows per leaf of a segmented sum, at least.
const LEAF_ROWS: usize = 65_536;

/// Level `l` owns the positions `offsets[l]..offsets[l + 1]`: rows, for rows sorted by level, or
/// positions of a [`RowIndex`]. Its passes cut the segments into tasks of whole segments, and a
/// sum's tasks into leaves, by the offsets and the sum's length alone: every pass is independent
/// of the thread count.
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

    fn n_rows(&self) -> usize {
        self.offsets[self.n_levels()]
    }

    fn n_tasks(&self) -> usize {
        self.tasks.len() - 1
    }

    /// Tasks per leaf of a sum of length `n`: leaves of at least [`LEAF_ROWS`] rows, and more for
    /// long sums, so a leaf's accumulator stays small next to its rows.
    fn per_leaf(&self, n: usize) -> usize {
        let leaf_rows = (4 * n).min(self.n_rows().div_ceil(8)).max(LEAF_ROWS);
        leaf_rows.div_ceil(TASK_ROWS)
    }

    /// `f(level, start, segment, scratch)` for each level, `segment` its rows of `v` from row
    /// `start`; tasks run concurrently, each with `len` values of scratch.
    pub(crate) fn map(
        &self,
        v: &mut [f64],
        len: usize,
        stack: &mut MemStack,
        f: &(impl Fn(usize, usize, &mut [f64], &mut [f64]) + Sync),
    ) {
        let (mut scratch, _) = stack.make_with(self.n_tasks() * len, |_| 0.0);
        let rows = Pieces::new(Pieces::new(v, &self.offsets), &self.tasks);
        let parts = (rows, Chunks::equal(&mut scratch, self.n_tasks()));
        self.for_each_task(parts, |levels, (rows, scratch)| {
            let mut rows = rows.into_inner();
            for level in levels {
                let at = self.rows(level);
                let (segment, tail) = mem::take(&mut rows).split_at_mut(at.len());
                f(level, at.start, segment, scratch);
                rows = tail;
            }
        });
    }

    /// `f(state, level, start, segment, out)` for each level, `segment` its rows of `v` from row
    /// `start`, `out` its `width` values of `outs`; tasks run concurrently, each on its own
    /// `state` from `init`.
    pub(crate) fn map_levels<S>(
        &self,
        v: &mut [f64],
        outs: &mut [f64],
        width: usize,
        init: impl Fn() -> S + Sync,
        f: &(impl Fn(&mut S, usize, usize, &mut [f64], &mut [f64]) + Sync),
    ) {
        let rows = Pieces::new(Pieces::new(v, &self.offsets), &self.tasks);
        let outs = Pieces::new(Chunks::equal(outs, self.n_levels()), &self.tasks);
        self.for_each_task((rows, outs), |levels, (rows, outs)| {
            let (mut state, mut rows) = (init(), rows.into_inner());
            for (level, out) in levels.zip(outs.into_values().chunks_exact_mut(width)) {
                let at = self.rows(level);
                let (segment, tail) = mem::take(&mut rows).split_at_mut(at.len());
                f(&mut state, level, at.start, segment, out);
                rows = tail;
            }
        });
    }

    /// `f(level, rows, out)` for each level, `out` its value of `outs`; tasks run concurrently.
    fn for_each_level<T: Send>(
        &self,
        outs: &mut [T],
        f: impl Fn(usize, Range<usize>, &mut T) + Sync,
    ) {
        let outs = Pieces::new(Chunks::equal(outs, self.n_levels()), &self.tasks);
        self.for_each_task(outs, |levels, outs| {
            for (level, out) in levels.zip(outs.into_values()) {
                f(level, self.rows(level), out);
            }
        });
    }

    /// Each level's fold of its rows: `step(state, level, row)` from `init` over leaves of
    /// [`LEVEL_LEAF`] positions, `merge`d in a fixed binary tree, into `out[level]`; `rows` maps
    /// positions to rows, `None` for rows sorted by level.
    pub(super) fn fold<S: Copy + Send + Sync>(
        &self,
        rows: Option<&RowIndex>,
        init: S,
        step: &(impl Fn(&mut S, usize, usize) + Sync),
        merge: &(impl Fn(&mut S, &S) + Sync),
        out: &mut [S],
    ) {
        let level_fold = |level: usize, at: Range<usize>| {
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
            Threads::All.fold_tree(Blocks::new(at, LEVEL_LEAF), |(_, at)| leaf(at), merge)
        };
        self.for_each_level(out, |level, at, slot| *slot = level_fold(level, at));
    }

    /// `f(levels, part)` for each task, `part` its item of `parts`, one per task; tasks run
    /// concurrently.
    fn for_each_task<P: Split>(&self, parts: P, f: impl Fn(Range<usize>, P::Item) + Sync) {
        let levels = Pieces::new(0..self.n_levels(), &self.tasks);
        Threads::All.for_each_item((levels, parts), |_, (levels, part)| f(levels, part));
    }

    /// `out = Σ_l f_l`, `f(level, rows, acc, scratch)` adding level `level`'s part into its leaf's
    /// zeroed `acc`, with `len` values of scratch per leaf. The leaves add up in a tree fixed by
    /// their count; the leftmost leaf's `acc` is `out` itself.
    pub(crate) fn sum(
        &self,
        out: &mut [f64],
        len: usize,
        stack: &mut MemStack,
        f: &(impl Fn(usize, Range<usize>, &mut [f64], &mut [f64]) + Sync),
    ) {
        let (n, per_leaf) = (out.len(), self.per_leaf(out.len()));
        let n_leaves = self.n_tasks().div_ceil(per_leaf);
        let (mut rest, stack) = stack.make_with((n_leaves - 1) * n, |_| 0.0);
        let (mut scratch, _) = stack.make_with(n_leaves * len, |_| 0.0);
        let leaves = (
            Blocks::new(0..self.n_tasks(), per_leaf),
            Accumulators::new(out, &mut rest, n_leaves),
            Chunks::equal(&mut scratch, n_leaves),
        );
        Threads::All.fold_tree(
            leaves,
            |((_, tasks), first, scratch)| {
                first.fill(0.0);
                for level in self.tasks[tasks.start]..self.tasks[tasks.end] {
                    f(level, self.rows(level), first, scratch);
                }
                first
            },
            |left, right| {
                for (a, b) in left.iter_mut().zip(&*right) {
                    *a += b;
                }
                left
            },
        );
    }

    /// What [`map`](Self::map) and [`sum`](Self::sum) take from a stack for a sum of length `n`
    /// and `len` values of scratch.
    pub(crate) fn scratch(&self, n: usize, len: usize) -> StackReq {
        let n_leaves = self.n_tasks().div_ceil(self.per_leaf(n));
        let map = StackReq::new::<f64>(self.n_tasks() * len);
        let sum =
            StackReq::new::<f64>((n_leaves - 1) * n).and(StackReq::new::<f64>(n_leaves * len));
        map.or(sum)
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
