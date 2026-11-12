//! Scatter kernel `Dᵀ x`, summed per term in the order its [`ScatterSplit`] fixes.

use crate::domain::{LevelSlots, PreparedDesign, PreparedTerm};

/// Adjoint scatter over all terms; `base(i)` is the row value each column scales by its loading.
pub(super) fn scatter_apply(
    prepared: &PreparedDesign<'_>,
    dst: &mut [f64],
    base: &(impl Fn(usize) -> f64 + Sync),
) {
    let design = &prepared.design;
    debug_assert_eq!(dst.len(), design.n_dofs);

    for t in prepared.terms() {
        let term = t.term;
        let block = &mut dst[term.dofs()];
        match (term.intercept, t.slopes) {
            (true, []) => scatter_term::<1>(block, &t, |i| [base(i)]),
            (true, [z0]) => {
                let z0 = &z0[..];
                scatter_term::<2>(block, &t, |i| {
                    let b = base(i);
                    [b, z0[i] * b]
                })
            }
            (true, [z0, z1]) => {
                let (z0, z1) = (&z0[..], &z1[..]);
                scatter_term::<3>(block, &t, |i| {
                    let b = base(i);
                    [b, z0[i] * b, z1[i] * b]
                })
            }
            (true, [z0, z1, z2]) => {
                let (z0, z1, z2) = (&z0[..], &z1[..], &z2[..]);
                scatter_term::<4>(block, &t, |i| {
                    let b = base(i);
                    [b, z0[i] * b, z1[i] * b, z2[i] * b]
                })
            }
            (false, [z0, z1]) => {
                let (z0, z1) = (&z0[..], &z1[..]);
                scatter_term::<2>(block, &t, |i| {
                    let b = base(i);
                    [z0[i] * b, z1[i] * b]
                })
            }
            (false, [z0, z1, z2]) => {
                let (z0, z1, z2) = (&z0[..], &z1[..], &z2[..]);
                scatter_term::<3>(block, &t, |i| {
                    let b = base(i);
                    [z0[i] * b, z1[i] * b, z2[i] * b]
                })
            }
            _ => {
                for c in 0..term.n_columns() {
                    let start = c * term.n_levels();
                    let slot = &mut block[start..start + term.n_levels()];
                    match t.loading(c) {
                        None => scatter_term::<1>(slot, &t, |i| [base(i)]),
                        Some(z) => scatter_term::<1>(slot, &t, move |i| [z[i] * base(i)]),
                    }
                }
            }
        }
    }
}

/// Write one term's coefficient block: `block[c·L + l] = Σ_{i : level(i) = l} values(i)[c]`.
fn scatter_term<const C: usize>(
    block: &mut [f64],
    term: &PreparedTerm<'_>,
    values: impl Fn(usize) -> [f64; C] + Sync,
) {
    let n_levels = term.term.n_levels();
    debug_assert_eq!(block.len(), C * n_levels);
    let mut columns = block.chunks_mut(n_levels);
    let columns = Columns(std::array::from_fn(|_| columns.next().expect("C columns")));
    term.split.fold(
        term.term.levels(),
        [0.0; C],
        &|sums: &mut [f64; C], _, i| {
            for (s, v) in sums.iter_mut().zip(values(i)) {
                *s += v;
            }
        },
        &|left: &mut [f64; C], right: &[f64; C]| {
            for (l, r) in left.iter_mut().zip(right) {
                *l += r;
            }
        },
        columns,
    );
}

/// A column-major `C × L` block as per-level `[f64; C]` slots.
struct Columns<'a, const C: usize>([&'a mut [f64]; C]);

impl<const C: usize> LevelSlots<[f64; C]> for Columns<'_, C> {
    fn len(&self) -> usize {
        self.0[0].len()
    }

    fn split_at(self, mid: usize) -> (Self, Self) {
        let mut left = self.0;
        let right = left.each_mut().map(|column| {
            let (l, r) = std::mem::take(column).split_at_mut(mid);
            *column = l;
            r
        });
        (Self(left), Self(right))
    }

    fn set(&mut self, level: usize, sums: [f64; C]) {
        for (column, s) in self.0.iter_mut().zip(sums) {
            column[level] = s;
        }
    }
}
