//! Parallel sums whose summation order is fixed by the input length, never by the thread count.

use std::ops::Range;

/// An ordered input that [`fold`] can cut at an index into a prefix and a suffix.
pub(crate) trait Split: Sized + Send {
    fn n_items(&self) -> usize;
    fn split_at(self, mid: usize) -> (Self, Self);
}

impl<T: Sync> Split for &[T] {
    fn n_items(&self) -> usize {
        <[T]>::len(self)
    }
    fn split_at(self, mid: usize) -> (Self, Self) {
        <[T]>::split_at(self, mid)
    }
}

impl<T: Send> Split for &mut [T] {
    fn n_items(&self) -> usize {
        <[T]>::len(self)
    }
    fn split_at(self, mid: usize) -> (Self, Self) {
        <[T]>::split_at_mut(self, mid)
    }
}

impl Split for Range<usize> {
    fn n_items(&self) -> usize {
        self.end - self.start
    }
    fn split_at(self, mid: usize) -> (Self, Self) {
        let cut = self.start + mid;
        (self.start..cut, cut..self.end)
    }
}

impl<A: Split, B: Split> Split for (A, B) {
    fn n_items(&self) -> usize {
        debug_assert_eq!(self.0.n_items(), self.1.n_items());
        self.0.n_items()
    }
    fn split_at(self, mid: usize) -> (Self, Self) {
        let (a0, a1) = self.0.split_at(mid);
        let (b0, b1) = self.1.split_at(mid);
        ((a0, b0), (a1, b1))
    }
}

/// `leaf` over `block_len` pieces of `input`, combined in a tree fixed by the length alone.
pub(crate) fn fold<P: Split, T: Send>(
    input: P,
    block_len: usize,
    leaf: &(impl Fn(P) -> T + Sync),
    combine: &(impl Fn(T, T) -> T + Sync),
) -> T {
    let n = input.n_items();
    let pieces = n.div_ceil(block_len);
    if pieces <= 1 {
        return leaf(input);
    }
    let (left, right) = input.split_at(pieces / 2 * block_len);
    let (l, r) = if n > SEQUENTIAL_ITEMS {
        rayon::join(
            || fold(left, block_len, leaf, combine),
            || fold(right, block_len, leaf, combine),
        )
    } else {
        (
            fold(left, block_len, leaf, combine),
            fold(right, block_len, leaf, combine),
        )
    };
    combine(l, r)
}

/// Subtrees this small run on the calling thread: a `join` per piece costs more than it spreads.
const SEQUENTIAL_ITEMS: usize = 1 << 16;

/// Asserts `f` returns the same value in pools of 1, 2, 3 and 8 threads.
#[cfg(test)]
pub(crate) fn assert_pool_independent<R: PartialEq + Send>(f: impl Fn() -> R + Sync) {
    let in_pool = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("pool")
            .install(&f)
    };
    let reference = in_pool(1);
    for threads in [2, 3, 8] {
        // Not `assert_eq!`: a printed solution vector would bury the thread count.
        assert!(in_pool(threads) == reference, "{threads} threads");
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case(0)]
    #[case(1)]
    #[case(7)]
    #[case(8)]
    #[case(9)]
    #[case(43)]
    #[case(1_000)]
    fn leaves_cover_the_input_once_in_order(#[case] n: usize) {
        let leaves = fold(0..n, 8, &|r: Range<usize>| vec![r], &|mut a, b| {
            a.extend(b);
            a
        });
        assert!(leaves
            .iter()
            .all(|r| r.len() <= 8 && (n == 0 || !r.is_empty())));
        let covered: Vec<usize> = leaves.into_iter().flatten().collect();
        assert_eq!(covered, (0..n).collect::<Vec<_>>());
    }

    #[test]
    fn the_sum_does_not_depend_on_the_pool() {
        // Magnitudes over 30 decades, so any change of summation order changes the bits.
        let x: Vec<f64> = (0..100_003u64)
            .map(|i| ((i * 2_654_435_761) % 997) as f64 * 10f64.powi((i % 31) as i32 - 15))
            .collect();
        assert_pool_independent(|| {
            fold(
                x.as_slice(),
                64,
                &|c: &[f64]| c.iter().sum::<f64>(),
                &|a, b| a + b,
            )
            .to_bits()
        });
    }
}
