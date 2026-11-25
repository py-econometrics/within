//! Each level's positions as one segment, and the segmented passes over them.

use std::mem;
use std::ops::Range;

use inflight::parallel::{Accumulators, Blocks, Chunks, Pieces, Split, Threads};
use schwarz_precond::{MemStack, StackReq};

use super::RowIndex;
use crate::PAR_THRESHOLD;

/// Rows of whole segments per task, at least; rows per piece of a long level, about.
const TASK_ROWS: usize = 16_384;
/// Rows past which a level is long: a fitted pass runs it in pieces, not as one task.
const LONG_ROWS: usize = 4 * TASK_ROWS;
/// Rows per leaf of one level's fold; above `PAR_THRESHOLD`, so a sequential fold never joins.
pub(crate) const LEVEL_LEAF: usize = PAR_THRESHOLD.next_power_of_two();
/// Rows per leaf of a segmented sum, at least.
const LEAF_ROWS: usize = 65_536;

/// Level `l` owns the positions `offsets[l]..offsets[l + 1]`: rows, for rows sorted by level, or
/// positions of a [`RowIndex`]. Its passes cut the segments into tasks of whole segments, a fitted
/// pass's long levels into pieces, and a sum's tasks into leaves, by the offsets and the sum's
/// length alone: every pass is independent of the thread count.
#[derive(Clone, Debug)]
pub(crate) struct LevelSegments {
    offsets: Vec<usize>,
    /// The tasks: whole levels, a long level alone.
    tasks: Cuts,
    /// A fitted pass's runs: the tasks, with each long level's cut into pieces.
    runs: Cuts,
}

/// A map that fits each level from its segment's moments: [`moments`](Self::moments) over runs of a
/// level's rows add up to the level's, [`fit`](Self::fit) turns them into its fit, and
/// [`apply`](Self::apply) runs on runs of its rows with it.
pub(crate) trait FittedMap: Sync {
    /// `m`, the moments of the rows from `start` of level `level`, `segment` their values.
    fn moments(&self, level: usize, start: usize, segment: &mut [f64], m: &mut [f64]);
    /// Level `level`'s moments `m` into its fit; `segment` is all its rows, from `start`.
    fn fit(&self, level: usize, start: usize, segment: &[f64], m: &mut [f64]);
    /// The rows from `start` of level `level`, `segment` their values, with its `fit`.
    fn apply(&self, level: usize, start: usize, segment: &mut [f64], fit: &[f64]);
}

/// A sum that fits each level from its moments, as [`FittedMap`]: `apply` adds into `acc`.
pub(crate) trait FittedSum: Sync {
    /// `m`, the moments of rows `rows` of level `level`.
    fn moments(&self, level: usize, rows: Range<usize>, m: &mut [f64]);
    /// Level `level`'s moments `m` into its fit.
    fn fit(&self, level: usize, m: &mut [f64]);
    /// Rows `rows` of level `level` into `acc`, with its `fit`.
    fn apply(&self, level: usize, rows: Range<usize>, fit: &[f64], acc: &mut [f64]);
}

/// Runs of positions: run `k` from position `rows[k]`, in level `levels[k]`, to the next run's
/// start.
#[derive(Clone, Debug, Default)]
struct Cuts {
    levels: Vec<usize>,
    rows: Vec<usize>,
}

impl Cuts {
    fn push(&mut self, level: usize, row: usize) {
        self.levels.push(level);
        self.rows.push(row);
    }

    /// Number of runs.
    fn len(&self) -> usize {
        self.levels.len() - 1
    }
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
    /// [`TASK_ROWS`] rows but for those cut before a long level, which is a task alone.
    pub(super) fn new(offsets: Vec<usize>) -> Self {
        let n_levels = offsets.len() - 1;
        let (mut tasks, mut start) = (Cuts::default(), 0);
        tasks.push(0, offsets[0]);
        for level in 0..n_levels {
            let (first, end) = (offsets[level], offsets[level + 1]);
            if end - first > LONG_ROWS && first > start {
                tasks.push(level, first);
                start = first;
            }
            if end - start >= TASK_ROWS || level + 1 == n_levels {
                tasks.push(level + 1, end);
                start = end;
            }
        }
        let mut runs = Cuts::default();
        runs.push(tasks.levels[0], tasks.rows[0]);
        for k in 0..tasks.len() {
            let (level, first, next) = (tasks.levels[k], tasks.rows[k], tasks.rows[k + 1]);
            if offsets[level + 1] - offsets[level] > LONG_ROWS {
                let pieces = (next - first).div_ceil(TASK_ROWS);
                for piece in 1..pieces {
                    runs.push(level, first + piece * (next - first) / pieces);
                }
            }
            runs.push(tasks.levels[k + 1], next);
        }
        Self {
            offsets,
            tasks,
            runs,
        }
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
        self.tasks.len()
    }

    fn n_runs(&self) -> usize {
        self.runs.len()
    }

    /// Whether level `level` runs in pieces in a fitted pass.
    fn long(&self, level: usize) -> bool {
        self.rows(level).len() > LONG_ROWS
    }

    /// Run `run`'s rows.
    fn run_rows(&self, run: usize) -> Range<usize> {
        self.runs.rows[run]..self.runs.rows[run + 1]
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
        let rows = Pieces::new(Pieces::new(v, &self.offsets), &self.tasks.levels);
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

    /// `f(level, rows, out)` for each level, `out` its value of `outs`; tasks run concurrently.
    fn for_each_level<T: Send>(
        &self,
        outs: &mut [T],
        f: impl Fn(usize, Range<usize>, &mut T) + Sync,
    ) {
        let outs = Pieces::new(Chunks::equal(outs, self.n_levels()), &self.tasks.levels);
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
        let levels = Pieces::new(0..self.n_levels(), &self.tasks.levels);
        Threads::All.for_each_item((levels, parts), |_, (levels, part)| f(levels, part));
    }

    /// Each long level's fit, in each of its pieces' `len` values of `fits`, each piece's moments
    /// before: the pieces' moments summed in order, then `fit(level, moments)`.
    fn join_pieces(&self, fits: &mut [f64], len: usize, mut fit: impl FnMut(usize, &mut [f64])) {
        let mut start = 0;
        for group in self.runs.levels[..self.n_runs()].chunk_by(|a, b| a == b) {
            let pieces = start * len..(start + group.len()) * len;
            start += group.len();
            if !self.long(group[0]) {
                continue;
            }
            let (head, rest) = fits[pieces].split_at_mut(len);
            for piece in rest.chunks_exact(len) {
                head.iter_mut().zip(piece).for_each(|(h, p)| *h += p);
            }
            fit(group[0], head);
            for piece in rest.chunks_exact_mut(len) {
                piece.copy_from_slice(head);
            }
        }
    }

    /// `pass` on each level's segment of `v`, its moments `len` values; each level's fit into
    /// `fits`, level-major, if given. A long level's rows run in pieces at once, its moments
    /// before its fit before its applies.
    pub(crate) fn map_fitted(
        &self,
        v: &mut [f64],
        len: usize,
        stack: &mut MemStack,
        pass: &impl FittedMap,
        mut fits: Option<&mut [f64]>,
    ) {
        let (mut scratch, _) = stack.make_with(self.n_runs() * len, |_| 0.0);
        if self.runs.len() > self.tasks.len() {
            let parts = (
                Pieces::new(&mut *v, &self.runs.rows),
                Chunks::equal(&mut scratch, self.n_runs()),
            );
            Threads::All.for_each_item(parts, |run, (segment, m)| {
                let (level, row) = (self.runs.levels[run], self.runs.rows[run]);
                if self.long(level) {
                    pass.moments(level, row, segment, m);
                }
            });
            self.join_pieces(&mut scratch, len, |level, m| {
                pass.fit(level, self.offsets[level], &v[self.rows(level)], m);
                if let Some(fits) = fits.as_deref_mut() {
                    fits[level * len..(level + 1) * len].copy_from_slice(m);
                }
            });
        }
        // A short level fits into its own slot of `fits` if given, else into its run's scratch.
        let outs = fits.unwrap_or_default();
        let parts = (
            Pieces::new(v, &self.runs.rows),
            Chunks::equal(&mut scratch, self.n_runs()),
            Pieces::new(Chunks::equal(outs, self.n_levels()), &self.runs.levels),
        );
        Threads::All.for_each_item(parts, |run, (mut rows, m, outs)| {
            let (level, row) = (self.runs.levels[run], self.runs.rows[run]);
            if self.long(level) {
                return pass.apply(level, row, rows, m);
            }
            let mut outs = outs.into_values().chunks_exact_mut(len);
            for level in level..self.runs.levels[run + 1] {
                let at = self.rows(level);
                let (segment, tail) = mem::take(&mut rows).split_at_mut(at.len());
                let m = outs.next().unwrap_or(&mut *m);
                pass.moments(level, at.start, segment, m);
                pass.fit(level, at.start, segment, m);
                pass.apply(level, at.start, segment, m);
                rows = tail;
            }
        });
    }

    /// `out = Σ_l` `pass` on each level, its applies into their leaf's zeroed `acc`; long levels
    /// as in [`map_fitted`](Self::map_fitted), the leaves added up as in [`sum`](Self::sum).
    pub(crate) fn sum_fitted(
        &self,
        out: &mut [f64],
        len: usize,
        stack: &mut MemStack,
        pass: &impl FittedSum,
    ) {
        let (n, per_leaf) = (out.len(), self.per_leaf(out.len()));
        let n_leaves = self.n_runs().div_ceil(per_leaf);
        let (mut rest, stack) = stack.make_with((n_leaves - 1) * n, |_| 0.0);
        let (mut fits, _) = stack.make_with(self.n_runs() * len, |_| 0.0);
        if self.runs.len() > self.tasks.len() {
            let moments = Chunks::equal(&mut fits, self.n_runs());
            Threads::All.for_each_item(moments, |run, m| {
                let level = self.runs.levels[run];
                if self.long(level) {
                    pass.moments(level, self.run_rows(run), m);
                }
            });
            self.join_pieces(&mut fits, len, |level, m| pass.fit(level, m));
        }
        let leaves = (
            Blocks::new(Chunks::equal(&mut fits, self.n_runs()), per_leaf),
            Accumulators::new(out, &mut rest, n_leaves),
        );
        Threads::All.fold_tree(
            leaves,
            |((start, fits), first)| {
                first.fill(0.0);
                let runs = start..start + fits.n_items();
                for (run, m) in runs.zip(fits.into_values().chunks_exact_mut(len)) {
                    let level = self.runs.levels[run];
                    if self.long(level) {
                        pass.apply(level, self.run_rows(run), m, first);
                        continue;
                    }
                    for level in level..self.runs.levels[run + 1] {
                        let at = self.rows(level);
                        pass.moments(level, at.clone(), m);
                        pass.fit(level, m);
                        pass.apply(level, at, m, first);
                    }
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
                for level in self.tasks.levels[tasks.start]..self.tasks.levels[tasks.end] {
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

    /// What any pass takes from a stack for a sum of length `n` and `len` values of scratch.
    pub(crate) fn scratch(&self, n: usize, len: usize) -> StackReq {
        let per_leaf = self.per_leaf(n);
        let sum = |items: usize, scratch: usize| {
            let rest = (items.div_ceil(per_leaf) - 1) * n;
            StackReq::new::<f64>(rest).and(StackReq::new::<f64>(scratch * len))
        };
        let n_leaves = self.n_tasks().div_ceil(per_leaf);
        // A fitted pass's scratch is per run; `map`'s per task, at most one per run.
        let map = StackReq::new::<f64>(self.n_runs() * len);
        map.or(sum(self.n_tasks(), n_leaves))
            .or(sum(self.n_runs(), self.n_runs()))
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

    /// A long level is a task alone, and a fitted pass's runs cut it into pieces.
    #[test]
    fn a_long_level_is_a_task_alone_in_pieces() {
        let segments = LevelSegments::new(vec![0, 10, 70_010, 70_015]);
        assert_eq!(segments.tasks.rows, [0, 10, 70_010, 70_015]);
        let pieces = [14_010, 28_010, 42_010, 56_010];
        assert_eq!(
            segments.runs.rows,
            [&[0, 10][..], &pieces, &[70_010, 70_015]].concat()
        );
    }
}
