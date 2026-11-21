//! Parallel sums whose summation order is fixed by the input length, never by the thread count, and
//! parallel loops over pieces that each own their part of a buffer.

use std::convert::Infallible;
use std::mem::MaybeUninit;
use std::ops::Range;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use dyn_stack::{MemStack, StackReq};
use rayon::iter::ParallelIterator;

/// An ordered input that [`fold`] can cut at an index into a prefix and a suffix.
pub trait Split: Sized + Send {
    /// Number of items.
    fn n_items(&self) -> usize;
    /// The first `mid` items and the rest.
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
pub fn fold<P: Split, T: Send>(
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

/// A buffer cut into consecutive pieces: piece `i` owns `items[offsets[i]..offsets[i + 1]]`.
pub struct Pieces<'a, T> {
    items: &'a mut [T],
    offsets: &'a [usize],
}

impl<'a, T> Pieces<'a, T> {
    /// `items` cut at the non-decreasing, non-empty `offsets`; items outside their span are left out.
    pub fn new(items: &'a mut [T], offsets: &'a [usize]) -> Self {
        let span = offsets[0]..offsets[offsets.len() - 1];
        Self {
            items: &mut items[span],
            offsets,
        }
    }

    /// The items of every piece, in order.
    pub fn into_items(self) -> &'a mut [T] {
        self.items
    }
}

impl<T: Send> Split for Pieces<'_, T> {
    fn n_items(&self) -> usize {
        self.offsets.len() - 1
    }
    fn split_at(self, mid: usize) -> (Self, Self) {
        let (left, right) = self.items.split_at_mut(self.offsets[mid] - self.offsets[0]);
        (
            Self {
                items: left,
                offsets: &self.offsets[..=mid],
            },
            Self {
                items: right,
                offsets: &self.offsets[mid..],
            },
        )
    }
}

/// One workspace per piece, side by side on a stack, each on one alignment: a cache line, or the
/// widest a piece asks.
#[derive(Clone)]
pub struct Workspaces {
    /// Piece `i` owns bytes `offsets[i]..offsets[i + 1]`.
    offsets: Arc<[usize]>,
    align: usize,
}

impl Workspaces {
    /// A workspace for each of `each`, in order; panics if one overflows.
    pub fn new(each: impl IntoIterator<Item = StackReq>) -> Self {
        let each: Vec<StackReq> = each.into_iter().collect();
        assert!(
            each.iter().all(|w| w.align_bytes() > 0),
            "a workspace overflows"
        );
        let align = each.iter().map(StackReq::align_bytes).fold(64, usize::max);
        let offsets = std::iter::once(0)
            .chain(each.iter().scan(0usize, |end, w| {
                *end = w
                    .size_bytes()
                    .checked_next_multiple_of(align)
                    .and_then(|size| end.checked_add(size))
                    .expect("a workspace overflows");
                Some(*end)
            }))
            .collect();
        Self { offsets, align }
    }

    /// What [`carve`](Self::carve) takes from a stack.
    pub fn scratch(&self) -> StackReq {
        StackReq::new_aligned::<u8>(self.len(), self.align)
    }

    /// Every piece's workspace, from `stack`.
    pub fn carve<'s>(&'s self, stack: &'s mut MemStack) -> Pieces<'s, MaybeUninit<u8>> {
        let (bytes, _) = stack.make_aligned_uninit::<u8>(self.len(), self.align);
        Pieces::new(bytes, &self.offsets)
    }

    fn len(&self) -> usize {
        self.offsets[self.offsets.len() - 1]
    }
}

/// `piece(i, part)` for each item `i` of `input`, `part` holding item `i` alone; the pool decides
/// which run at once. Once a piece fails no new piece starts, but running ones finish, and which of
/// several errors returns is unspecified.
pub fn try_for_each_piece<P: Split, E: Send>(
    input: P,
    piece: &(impl Fn(usize, P) -> Result<(), E> + Sync),
) -> Result<(), E> {
    let failed = AtomicBool::new(false);
    rayon::iter::split((0, input), |(start, input): (usize, P)| {
        let n = input.n_items();
        if n < 2 {
            return ((start, input), None);
        }
        let (left, right) = input.split_at(n / 2);
        ((start, left), Some((start + n / 2, right)))
    })
    .try_for_each(|(start, mut rest)| {
        for i in start..start + rest.n_items() {
            if failed.load(Ordering::Acquire) {
                break;
            }
            let (part, tail) = rest.split_at(1);
            piece(i, part).inspect_err(|_| failed.store(true, Ordering::Release))?;
            rest = tail;
        }
        Ok(())
    })
}

/// [`try_for_each_piece`] for a `piece` that cannot fail.
pub fn for_each_piece<P: Split>(input: P, piece: &(impl Fn(usize, P) + Sync)) {
    let Ok(()) = try_for_each_piece(input, &|i, part| {
        piece(i, part);
        Ok::<(), Infallible>(())
    });
}

/// Runs `f` in a fresh pool of `threads`.
#[cfg(any(test, feature = "testing"))]
pub fn in_pool<R: Send>(threads: usize, f: impl FnOnce() -> R + Send) -> R {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("pool")
        .install(f)
}

/// Asserts `f` returns the same value in pools of 1, 2, 3 and 8 threads.
#[cfg(any(test, feature = "testing"))]
pub fn assert_pool_independent<R: PartialEq + Send>(f: impl Fn() -> R + Sync) {
    let reference = in_pool(1, &f);
    for threads in [2, 3, 8] {
        // Not `assert_eq!`: a printed solution vector would bury the thread count.
        assert!(in_pool(threads, &f) == reference, "{threads} threads");
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
    fn each_piece_gets_its_own_items_and_the_rest_stay() {
        let offsets = [2, 2, 5, 6, 10];
        let mut items = [usize::MAX; 12];
        in_pool(3, || {
            for_each_piece(Pieces::new(&mut items, &offsets), &|i, part| {
                part.into_items().fill(i);
            });
        });
        let max = usize::MAX;
        assert_eq!(items, [max, max, 1, 1, 1, 2, 3, 3, 3, 3, max, max]);
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

    #[test]
    #[should_panic(expected = "a workspace overflows")]
    fn workspaces_past_the_address_space_panic() {
        Workspaces::new([StackReq::new::<u8>(1 << 62); 4]);
    }
}
