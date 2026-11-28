use std::ops::Range;

use inflight::parallel::{Blocks, Pieces, Threads};
use rayon::prelude::*;

/// Rows per task of a gather, or per chunk of a grouping's count, at least. On the 3-FE panel
/// shuffled at 12 threads, a gather's time is flat from 2^12 to 2^18 (0.165–0.167 s at 40M rows,
/// 0.64–0.70 s at 160M). A grouping's chunks are a thread's share of the rows above 2^14 rows a
/// thread, so there it does not move the cut.
const ROW_GRAIN: usize = 1 << 14;
/// Keys up to which rows are placed by key in one pass. On 40M rows at 12 threads, one pass beats
/// two up to 2^16 keys, uniform or with 90% of rows on 2^14 keys (at 2^16: 0.043 s against 0.063,
/// and 0.026 against 0.063); at 2^17 uniform keys two passes win, 0.046 s against 0.061.
const ONE_PASS_KEYS: usize = 1 << 16;
/// Keys per range of a grouping in two passes: the first places rows by range, the second sorts
/// each range by key, its counts in cache. On the 3-FE panel shuffled, 2^14 groups 160M rows in
/// 0.21 s, against 0.23–0.27 s from 2^12 to 2^16 and 0.39 s at 2^18; at 40M rows, 2^12 to 2^16 take
/// 0.056–0.063 s.
const RANGE_BITS: u32 = 14;

/// Observation indices, 4 bytes each while every row fits `u32`.
#[derive(Clone, Debug)]
pub(crate) enum RowIndex {
    Narrow(Box<[u32]>),
    Wide(Box<[usize]>),
}

impl RowIndex {
    /// Rows grouped by key, ascending within each, and where each key's rows start.
    pub(crate) fn argsort(key: &[u32], n_keys: usize) -> (Vec<usize>, Self) {
        let mut offsets = vec![0; n_keys + 1];
        let rows = if u32::try_from(key.len().saturating_sub(1)).is_ok() {
            Self::Narrow(grouped(key, &mut offsets, |i| i as u32))
        } else {
            Self::Wide(grouped(key, &mut offsets, |i| i))
        };
        (offsets, rows)
    }

    #[inline(always)]
    pub(crate) fn for_each(&self, positions: Range<usize>, f: impl FnMut(usize)) {
        match self {
            Self::Narrow(rows) => rows[positions].iter().map(|&r| r as usize).for_each(f),
            Self::Wide(rows) => rows[positions].iter().copied().for_each(f),
        }
    }

    /// `out[k] = col[self[k]]`.
    pub(crate) fn gather<T: Copy + Send + Sync>(&self, col: &[T]) -> Vec<T> {
        match self {
            Self::Narrow(rows) => gather(col, rows, |k| k as usize),
            Self::Wide(rows) => gather(col, rows, |k| k),
        }
    }

    /// `out[self[k]] = v[k]`, the inverse of [`gather`](Self::gather).
    pub(crate) fn scatter(&self, v: &[f64]) -> Vec<f64> {
        match self {
            Self::Narrow(rows) => scatter(v, rows, |k| k as usize),
            Self::Wide(rows) => scatter(v, rows, |k| k),
        }
    }
}

/// Each row's index grouped by key, stable, and `offsets[k]` where key `k`'s rows start. Up to
/// [`ONE_PASS_KEYS`], rows are placed by key at once; past it, first by range of keys, with their
/// keys, then each range by key on its own.
fn grouped<I: Copy + Default + Send + Sync>(
    key: &[u32],
    offsets: &mut [usize],
    index: impl Fn(usize) -> I + Sync,
) -> Box<[I]> {
    let (n, n_keys) = (key.len(), offsets.len() - 1);
    let mut rows = vec![I::default(); n];
    offsets[n_keys] = n;
    if n_keys <= ONE_PASS_KEYS {
        let by_key = |&k: &u32| k as usize;
        let (offsets, entry) = (&mut offsets[..n_keys], |i, _: &u32| index(i));
        // A chunk of at least as many rows as keys holds its row of counts in 8 B/obs at most.
        place_by_key(
            ROW_GRAIN.max(n_keys),
            key,
            by_key,
            entry,
            &mut rows,
            offsets,
            0,
        );
        return rows.into_boxed_slice();
    }
    let n_ranges = n_keys.div_ceil(1 << RANGE_BITS);
    let mut range_starts = vec![n; n_ranges + 1];
    let mut pairs = vec![(0u32, I::default()); n];
    let by_range = |&k: &u32| (k >> RANGE_BITS) as usize;
    let pair = |i, &k: &u32| (k, index(i));
    let starts = &mut range_starts[..n_ranges];
    place_by_key(ROW_GRAIN, key, by_range, pair, &mut pairs, starts, 0);
    let ranges = (
        Pieces::new(&pairs[..], &range_starts),
        Pieces::new(&mut rows[..], &range_starts),
        Blocks::new(&mut offsets[..n_keys], 1 << RANGE_BITS),
    );
    // A range of more than a thread's share of the rows, as many keys hold, splits across threads.
    let share = n.div_ceil(Threads::All.tasks());
    Threads::All.for_each_item(ranges, |r, (pairs, rows, (first, offsets))| {
        let by_key = |&(k, _): &(u32, I)| k as usize - first;
        let row = |_, &(_, i): &(u32, I)| i;
        place_by_key(share, pairs, by_key, row, rows, offsets, range_starts[r]);
    });
    rows.into_boxed_slice()
}

/// `out` holds `entry(i, item)` of each item, grouped by `key(item)` in input order, and
/// `offsets[k]` is where key `k`'s entries start, after `base`, one offset a key; in chunks of at
/// least `grain` items.
fn place_by_key<T: Sync, O: Send>(
    grain: usize,
    items: &[T],
    key: impl Fn(&T) -> usize + Sync,
    entry: impl Fn(usize, &T) -> O + Sync,
    out: &mut [O],
    offsets: &mut [usize],
    base: usize,
) {
    let places = Threads::All.places(items, grain, offsets.len(), |_, items, counts| {
        items.iter().for_each(|item| counts[key(item)] += 1);
    });
    for (o, p) in offsets.iter_mut().zip(places.offsets()) {
        *o = base + p;
    }
    places.place(items, out, |start, items, slots| {
        for (i, item) in (start..).zip(items) {
            slots.push(key(item), entry(i, item));
        }
    });
}

fn gather<T: Copy + Send + Sync, I: Copy + Sync>(
    col: &[T],
    rows: &[I],
    row: impl Fn(I) -> usize + Sync,
) -> Vec<T> {
    (rows.par_iter().with_min_len(ROW_GRAIN))
        .map(|&k| col[row(k)])
        .collect()
}

fn scatter<I: Copy>(v: &[f64], rows: &[I], row: impl Fn(I) -> usize) -> Vec<f64> {
    let mut out = vec![0.0; v.len()];
    for (&k, &x) in rows.iter().zip(v) {
        out[row(k)] = x;
    }
    out
}

#[cfg(test)]
mod tests {
    use inflight::parallel::testing::in_pool;

    use super::*;

    fn rows(index: &RowIndex, n: usize) -> Vec<usize> {
        let mut rows = Vec::new();
        index.for_each(0..n, |r| rows.push(r));
        rows
    }

    #[test]
    fn wide_rows_read_as_narrow_rows() {
        let key = [2u32, 0, 1, 0, 2, 1, 1];
        let mut offsets = vec![0; 4];
        let narrow = RowIndex::Narrow(grouped(&key, &mut offsets, |i| i as u32));
        let wide = RowIndex::Wide(grouped(&key, &mut offsets, |i| i));
        let col = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];
        let read = |index: &RowIndex| {
            let mut some = Vec::new();
            index.for_each(2..6, |r| some.push(r));
            let gathered = index.gather(&col);
            (
                some,
                rows(index, key.len()),
                index.scatter(&gathered),
                gathered,
            )
        };
        assert_eq!(read(&wide), read(&narrow));
        assert_eq!(read(&narrow).2, col);
    }

    /// A stable sort by key on pools of each size: placed by key at once up to
    /// [`ONE_PASS_KEYS`] and past it by range then key, keys spread or packed into one range, rows
    /// indexed by `u32` or `usize`.
    #[rstest::rstest]
    fn rows_grouped_by_key_are_a_stable_sort_on_any_pool(
        #[values(ONE_PASS_KEYS, ONE_PASS_KEYS + 1)] n_keys: usize,
        #[values(false, true)] packed: bool,
    ) {
        let n = 300_000;
        let key: Vec<u32> = (0..n)
            .map(|i| match packed && i % 10 != 0 {
                true => (i * 7_919) % n_keys.min(1 << RANGE_BITS),
                false => (i * 7_919 + i / 3) % n_keys,
            } as u32)
            .collect();
        let mut sorted: Vec<usize> = (0..n).collect();
        sorted.sort_by_key(|&i| key[i]);
        let mut offsets = vec![0; n_keys + 1];
        let wide = grouped(&key, &mut offsets, |i| i);
        assert_eq!(*wide, *sorted);
        let mut counted = vec![0; n_keys + 1];
        key.iter().for_each(|&k| counted[k as usize + 1] += 1);
        (1..=n_keys).for_each(|k| counted[k] += counted[k - 1]);
        assert_eq!(offsets, counted);
        for pool in [1, 2, 3, 8] {
            let (offsets, index) = in_pool(pool, || RowIndex::argsort(&key, n_keys));
            assert_eq!(rows(&index, n), sorted, "{pool} threads");
            for (k, w) in offsets.windows(2).enumerate() {
                assert!(sorted[w[0]..w[1]].iter().all(|&i| key[i] as usize == k));
            }
            assert_eq!(offsets[n_keys], n);
        }
    }
}
