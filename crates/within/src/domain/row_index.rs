use std::ops::Range;

/// Observation indices, 4 bytes each while every row fits `u32`.
#[derive(Clone, Debug)]
pub(crate) enum RowIndex {
    Narrow(Box<[u32]>),
    Wide(Box<[usize]>),
}

impl RowIndex {
    /// Rows grouped by key, ascending within each, and where each key's rows start.
    pub(crate) fn argsort(key: &[u32], n_keys: usize) -> (Vec<usize>, Self) {
        let offsets = super::offsets(n_keys, key.iter().map(|&k| k as usize));
        let rows = if u32::try_from(key.len().saturating_sub(1)).is_ok() {
            Self::Narrow(stable_argsort(key, &offsets, |i| i as u32))
        } else {
            Self::Wide(stable_argsort(key, &offsets, |i| i))
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
    pub(crate) fn gather<T: Copy>(&self, col: &[T]) -> Vec<T> {
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
fn gather<T: Copy, I: Copy>(col: &[T], rows: &[I], row: impl Fn(I) -> usize) -> Vec<T> {
    rows.iter().map(|&k| col[row(k)]).collect()
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
    use super::*;
    use crate::domain::offsets;

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
