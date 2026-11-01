//! Budgets: [`join`] divides `Par` between side-by-side regions, [`fork`] lends all of it.

use std::num::NonZeroUsize;
use std::ops::Range;

use crate::fold::for_each_piece;

use faer::dyn_stack::{MemStack, StackReq};
pub(crate) use faer::Par;

/// `par` split into `⌈n/2⌉` and `⌊n/2⌋`; a part of one or none is `Par::Seq`.
pub(crate) fn halves(par: Par) -> (Par, Par) {
    let n = par.degree();
    let part = |k: usize| match NonZeroUsize::new(k) {
        Some(k) if k.get() > 1 => Par::Rayon(k),
        _ => Par::Seq,
    };
    (part(n.div_ceil(2)), part(n / 2))
}

/// `a` and `b` on rayon, each with its half of `par`; both inline under a degree of one.
pub fn join<A: Send, B: Send>(
    par: Par,
    a: impl FnOnce(Par) -> A + Send,
    b: impl FnOnce(Par) -> B + Send,
) -> (A, B) {
    if par.degree() <= 1 {
        return (a(Par::Seq), b(Par::Seq));
    }
    let (l, r) = halves(par);
    rayon::join(|| a(l), || b(r))
}

/// [`join`] with `stack` split: `a` gets what `need` asks for under its half, `b` the rest.
pub(crate) fn join_on_stack<A: Send, B: Send>(
    par: Par,
    need: impl FnOnce(Par) -> StackReq,
    stack: &mut MemStack,
    a: impl FnOnce(Par, &mut MemStack) -> A + Send,
    b: impl FnOnce(Par, &mut MemStack) -> B + Send,
) -> (A, B) {
    let (left, right) = split_stack(stack, need(halves(par).0));
    join(par, |p| a(p, left), |p| b(p, right))
}

/// [`fork`] with `stack` split: `a` gets what `need` asks for under all of `par`, `b` the rest.
pub fn fork_on_stack<A: Send, B: Send>(
    par: Par,
    need: impl FnOnce(Par) -> StackReq,
    stack: &mut MemStack,
    a: impl FnOnce(Par, &mut MemStack) -> A + Send,
    b: impl FnOnce(Par, &mut MemStack) -> B + Send,
) -> (A, B) {
    let (left, right) = split_stack(stack, need(par));
    fork(par, |p| a(p, left), |p| b(p, right))
}

fn split_stack(stack: &mut MemStack, left: StackReq) -> (&mut MemStack, &mut MemStack) {
    let (bytes, right) = stack.make_aligned_uninit::<u8>(left.size_bytes(), left.align_bytes());
    (MemStack::new(bytes), right)
}

/// The halves of one data-parallel kernel, each with all of `par`; inline under a degree of one.
pub fn fork<A: Send, B: Send>(
    par: Par,
    a: impl FnOnce(Par) -> A + Send,
    b: impl FnOnce(Par) -> B + Send,
) -> (A, B) {
    if par.degree() <= 1 {
        return (a(Par::Seq), b(Par::Seq));
    }
    rayon::join(|| a(par), || b(par))
}

/// `f(start, piece)` over `chunk_len` pieces of `data`, forked down to single pieces.
pub fn for_each_chunk<T: Send>(
    par: Par,
    data: &mut [T],
    chunk_len: usize,
    f: &(impl Fn(usize, &mut [T]) + Sync),
) {
    if data.is_empty() {
        return;
    }
    let rows = 0..data.len();
    for_each_piece(par, (rows, data), chunk_len.max(1), &|(r, d): (
        Range<usize>,
        &mut [T],
    )| {
        f(r.start, d)
    });
}

/// `par`, or `Par::Seq` below `threshold` items, where a rayon wake costs more than it spreads.
pub fn at_least(par: Par, items: usize, threshold: usize) -> Par {
    if items >= threshold {
        par
    } else {
        Par::Seq
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn halves_cover_the_budget() {
        assert_eq!(halves(Par::Seq), (Par::Seq, Par::Seq));
        assert_eq!(halves(Par::rayon(5)), (Par::rayon(3), Par::rayon(2)));
        assert_eq!(halves(Par::rayon(3)), (Par::rayon(2), Par::Seq));
    }

    #[test]
    fn chunks_visit_every_piece_once_with_its_start() {
        for par in [Par::Seq, Par::rayon(3), Par::rayon(8)] {
            let mut data: Vec<usize> = vec![0; 1003];
            for_each_chunk(par, &mut data, 64, &|start, piece: &mut [usize]| {
                for (k, x) in piece.iter_mut().enumerate() {
                    *x += start + k + 1;
                }
            });
            assert!(data.iter().enumerate().all(|(i, &x)| x == i + 1), "{par:?}");
        }
    }
}
