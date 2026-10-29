//! Scatter kernel `Dᵀ x`, summed per term in the order its [`ScatterSplit`] fixes.

use std::ops::Range;

use rayon::prelude::*;

use super::PAR_THRESHOLD;
use crate::domain::{LevelRows, PreparedDesign, PreparedTerm, ScatterSplit};

/// Adjoint scatter over all terms; `base(i)` is the row value each column scales by its loading.
pub(super) fn scatter_apply(
    prepared: &PreparedDesign<'_>,
    dst: &mut [f64],
    base: &(impl Fn(usize) -> f64 + Sync),
) {
    let design = &prepared.design;
    debug_assert_eq!(dst.len(), design.n_dofs);
    let parallel = design.n_obs > PAR_THRESHOLD;

    for t in prepared.terms() {
        let term = t.term;
        let block = &mut dst[term.dofs()];
        match (term.intercept, t.slopes) {
            (true, []) => scatter_term::<1>(block, &t, parallel, |i| [base(i)]),
            (true, [z0]) => {
                let z0 = &z0[..];
                scatter_term::<2>(block, &t, parallel, |i| {
                    let b = base(i);
                    [b, z0[i] * b]
                })
            }
            (true, [z0, z1]) => {
                let (z0, z1) = (&z0[..], &z1[..]);
                scatter_term::<3>(block, &t, parallel, |i| {
                    let b = base(i);
                    [b, z0[i] * b, z1[i] * b]
                })
            }
            (true, [z0, z1, z2]) => {
                let (z0, z1, z2) = (&z0[..], &z1[..], &z2[..]);
                scatter_term::<4>(block, &t, parallel, |i| {
                    let b = base(i);
                    [b, z0[i] * b, z1[i] * b, z2[i] * b]
                })
            }
            (false, [z0, z1]) => {
                let (z0, z1) = (&z0[..], &z1[..]);
                scatter_term::<2>(block, &t, parallel, |i| {
                    let b = base(i);
                    [z0[i] * b, z1[i] * b]
                })
            }
            (false, [z0, z1, z2]) => {
                let (z0, z1, z2) = (&z0[..], &z1[..], &z2[..]);
                scatter_term::<3>(block, &t, parallel, |i| {
                    let b = base(i);
                    [z0[i] * b, z1[i] * b, z2[i] * b]
                })
            }
            _ => {
                for c in 0..term.n_columns() {
                    let start = c * term.n_levels();
                    let slot = &mut block[start..start + term.n_levels()];
                    match t.loading(c) {
                        None => scatter_term::<1>(slot, &t, parallel, |i| [base(i)]),
                        Some(z) => scatter_term::<1>(slot, &t, parallel, move |i| [z[i] * base(i)]),
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
    parallel: bool,
    values: impl Fn(usize) -> [f64; C] + Sync,
) {
    let (n_levels, levels) = (term.term.n_levels(), term.term.levels());
    debug_assert_eq!(block.len(), C * n_levels);
    match term.split {
        ScatterSplit::ByLevel(rows) => level_sums::<C>(block, rows, parallel, &values),
        ScatterSplit::ByRow => {
            // Adding two leaves costs C·L ≤ C·n/12 adds; leaves of 4·C·L rows bound how many add.
            let leaf_len = (4 * C * n_levels).min(levels.len().div_ceil(2)).max(65_536);
            let scatter = |rows: Range<usize>| {
                let mut sums = vec![0.0; C * n_levels];
                for i in rows {
                    let level = levels[i] as usize;
                    for (c, v) in values(i).into_iter().enumerate() {
                        sums[c * n_levels + level] += v;
                    }
                }
                sums
            };
            block.copy_from_slice(&fixed_tree(0..levels.len(), leaf_len, &scatter));
        }
    }
}

/// Levels per parallel task in [`level_sums`].
const LEVEL_GROUP: usize = 1024;

/// Rows per leaf of one level's sum; above `PAR_THRESHOLD`, so a sequential scatter never joins.
const LEVEL_LEAF: usize = PAR_THRESHOLD.next_power_of_two();

/// Each level's `C` column sums over its own rows, written to `block[c·L + l]`.
fn level_sums<const C: usize>(
    block: &mut [f64],
    rows: &LevelRows,
    parallel: bool,
    values: &(impl Fn(usize) -> [f64; C] + Sync),
) {
    let level_sum = |level: usize| {
        let leaf = |positions: Range<usize>| {
            let mut sums = [0.0; C];
            rows.for_each_row(positions, |i| {
                for (s, v) in sums.iter_mut().zip(values(i)) {
                    *s += v;
                }
            });
            sums
        };
        let positions = rows.positions(level);
        if positions.len() <= LEVEL_LEAF {
            return leaf(positions);
        }
        fixed_tree(positions, LEVEL_LEAF, &leaf)
    };
    let n_levels = block.len() / C;
    let mut columns = block
        .chunks_mut(n_levels)
        .map(|c| c.chunks_mut(LEVEL_GROUP));
    let mut columns: [_; C] = std::array::from_fn(|_| columns.next().expect("C columns"));
    let groups = (0..n_levels.div_ceil(LEVEL_GROUP)).map(|_| {
        columns
            .each_mut()
            .map(|c| c.next().expect("a group per column"))
    });
    let fill = |(g, mut group): (usize, [&mut [f64]; C])| {
        for k in 0..group[0].len() {
            let sums = level_sum(g * LEVEL_GROUP + k);
            for (column, s) in group.iter_mut().zip(sums) {
                column[k] = s;
            }
        }
    };
    if parallel {
        groups
            .collect::<Vec<_>>()
            .into_par_iter()
            .enumerate()
            .for_each(fill);
    } else {
        groups.enumerate().for_each(fill);
    }
}

/// `leaf` over `leaf_len` pieces of `items`, added in a binary tree fixed by both lengths alone.
fn fixed_tree<T: AsMut<[f64]> + Send>(
    items: Range<usize>,
    leaf_len: usize,
    leaf: &(impl Fn(Range<usize>) -> T + Sync),
) -> T {
    if items.len() <= leaf_len {
        return leaf(items);
    }
    let mid = items.start + (items.len() / 2 / leaf_len).max(1) * leaf_len;
    let (mut left, mut right) = rayon::join(
        || fixed_tree(items.start..mid, leaf_len, leaf),
        || fixed_tree(mid..items.end, leaf_len, leaf),
    );
    for (x, y) in left.as_mut().iter_mut().zip(right.as_mut()) {
        *x += *y;
    }
    left
}
