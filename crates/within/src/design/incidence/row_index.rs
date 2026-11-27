use std::ops::Range;

use inflight::parallel::{Blocks, Pieces, Threads};

/// Keys per chunk of a parallel grouping, at least.
const GROUP_GRAIN: usize = 1 << 16;
/// Keys per range of a grouping's second pass: its counts stay in cache.
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
    pub(crate) fn gather<T: Copy + Default + Send + Sync>(&self, col: &[T]) -> Vec<T> {
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

/// Each row's index grouped by key, stable, and `offsets[k]` where key `k`'s rows start: rows
/// placed first into ranges of keys, then each range sorted by its owner.
fn grouped<I: Copy + Default + Send + Sync>(
    key: &[u32],
    offsets: &mut [usize],
    index: impl Fn(usize) -> I + Sync,
) -> Box<[I]> {
    let n_keys = offsets.len() - 1;
    let n_ranges = n_keys.div_ceil(1 << RANGE_BITS).max(1);
    let mut places = Threads::All.count(key, GROUP_GRAIN, n_ranges, |_, chunk, row| {
        for &k in chunk {
            row[(k >> RANGE_BITS) as usize] += 1;
        }
    });
    let ranges = places.offsets();
    let mut pairs = vec![(0u32, I::default()); key.len()];
    places.place(Threads::All, key, &mut pairs, |start, chunk, slots| {
        for (i, &k) in chunk.iter().enumerate() {
            slots.push((k >> RANGE_BITS) as usize, (k, index(start + i)));
        }
    });
    let mut rows = vec![I::default(); key.len()];
    let split = (
        Pieces::new(&pairs[..], &ranges),
        Pieces::new(&mut rows[..], &ranges),
        Blocks::new(&mut offsets[..n_keys], 1 << RANGE_BITS),
    );
    Threads::All.for_each_item(split, |r, (pairs, rows, (_, offsets))| {
        let first = r << RANGE_BITS;
        let mut places = Threads::ONE.count(pairs, 1, offsets.len(), |_, pairs, row| {
            for &(k, _) in pairs {
                row[k as usize - first] += 1;
            }
        });
        for (o, p) in offsets.iter_mut().zip(places.offsets()) {
            *o = ranges[r] + p;
        }
        places.place(Threads::ONE, pairs, rows, |_, pairs, slots| {
            for &(k, i) in pairs {
                slots.push(k as usize - first, i);
            }
        });
    });
    offsets[n_keys] = key.len();
    rows.into_boxed_slice()
}

#[cfg(test)]
fn stable_argsort<I: Copy + Default>(
    key: &[u32],
    offsets: &[usize],
    index: impl Fn(usize) -> I,
) -> Box<[I]> {
    let mut cursors = offsets[..offsets.len() - 1].to_vec();
    let mut rows = vec![I::default(); key.len()];
    for (i, &k) in key.iter().enumerate() {
        let cursor = &mut cursors[k as usize];
        rows[*cursor] = index(i);
        *cursor += 1;
    }
    rows.into_boxed_slice()
}

// Inlined into `Design::build`, its loop measured +3–8% on slope designs.
#[inline(never)]
fn gather<T: Copy + Default + Send + Sync, I: Copy + Sync>(
    col: &[T],
    rows: &[I],
    row: impl Fn(I) -> usize + Sync,
) -> Vec<T> {
    let mut out = vec![T::default(); rows.len()];
    Threads::All.for_each_chunk(&mut out[..], GROUP_GRAIN, |start, chunk| {
        for (o, &k) in chunk.iter_mut().zip(&rows[start..]) {
            *o = col[row(k)];
        }
    });
    out
}

fn scatter<I: Copy>(v: &[f64], rows: &[I], row: impl Fn(I) -> usize) -> Vec<f64> {
    let mut out = vec![0.0; v.len()];
    for (&k, &x) in rows.iter().zip(v) {
        out[row(k)] = x;
    }
    out
}

#[cfg(test)]
/// Group `k` of the counted `keys` owns `offsets[k]..offsets[k + 1]`.
fn offsets(n_keys: usize, keys: impl IntoIterator<Item = usize>) -> Vec<usize> {
    let mut offsets = vec![0usize; n_keys + 1];
    for k in keys {
        offsets[k + 1] += 1;
    }
    for i in 1..offsets.len() {
        offsets[i] += offsets[i - 1];
    }
    offsets
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(index: &RowIndex, n: usize) -> Vec<usize> {
        let mut rows = Vec::new();
        index.for_each(0..n, |r| rows.push(r));
        rows
    }

    #[test]
    fn wide_rows_read_as_narrow_rows() {
        let key = [2u32, 0, 1, 0, 2, 1, 1];
        let offsets = offsets(3, key.iter().map(|&k| k as usize));
        let narrow = RowIndex::Narrow(stable_argsort(&key, &offsets, |i| i as u32));
        let wide = RowIndex::Wide(stable_argsort(&key, &offsets, |i| i));
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
}
