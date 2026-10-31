//! Scatter kernel `Dᵀ x`, summed per term in the order its [`ScatterSplit`] fixes.

use std::ops::Range;

use schwarz_precond::par::{at_least, fork, fork_on_stack};
use schwarz_precond::Par;
use schwarz_precond::{MemStack, StackReq};

use super::PAR_THRESHOLD;
use crate::domain::{LevelRows, PreparedDesign, PreparedTerm, ScatterSplit};

/// The by-row terms' partial sums, the only scratch a scatter under `par` takes.
pub(super) fn scatter_scratch(prepared: &PreparedDesign<'_>, par: Par) -> StackReq {
    let par = scatter_par(prepared, par);
    let reqs: Vec<StackReq> = prepared
        .terms()
        .filter(|t| matches!(t.split, ScatterSplit::ByRow))
        .flat_map(|t| {
            let (rows, l) = (t.term.levels().len(), t.term.n_levels());
            // A fused arity sums all columns at once; any other arity sums one column at a time.
            [t.term.n_columns() * l, l]
                .map(|width| partials_scratch(par, 0..rows, by_row_leaf_len(width, rows), width))
        })
        .collect();
    StackReq::any_of(&reqs)
}

fn scatter_par(prepared: &PreparedDesign<'_>, par: Par) -> Par {
    at_least(par, prepared.design.n_obs, PAR_THRESHOLD + 1)
}

/// Adjoint scatter over all terms; `base(i)` is the row value each column scales by its loading.
pub(super) fn scatter_apply(
    prepared: &PreparedDesign<'_>,
    dst: &mut [f64],
    base: &(impl Fn(usize) -> f64 + Sync),
    par: Par,
    stack: &mut MemStack,
) {
    let design = &prepared.design;
    debug_assert_eq!(dst.len(), design.n_dofs);
    let par = scatter_par(prepared, par);

    for t in prepared.terms() {
        let term = t.term;
        let block = &mut dst[term.dofs()];
        match (term.intercept, t.slopes) {
            (true, []) => scatter_term::<1>(block, &t, par, stack, |i| [base(i)]),
            (true, [z0]) => {
                let z0 = &z0[..];
                scatter_term::<2>(block, &t, par, stack, |i| {
                    let b = base(i);
                    [b, z0[i] * b]
                })
            }
            (true, [z0, z1]) => {
                let (z0, z1) = (&z0[..], &z1[..]);
                scatter_term::<3>(block, &t, par, stack, |i| {
                    let b = base(i);
                    [b, z0[i] * b, z1[i] * b]
                })
            }
            (true, [z0, z1, z2]) => {
                let (z0, z1, z2) = (&z0[..], &z1[..], &z2[..]);
                scatter_term::<4>(block, &t, par, stack, |i| {
                    let b = base(i);
                    [b, z0[i] * b, z1[i] * b, z2[i] * b]
                })
            }
            (false, [z0, z1]) => {
                let (z0, z1) = (&z0[..], &z1[..]);
                scatter_term::<2>(block, &t, par, stack, |i| {
                    let b = base(i);
                    [z0[i] * b, z1[i] * b]
                })
            }
            (false, [z0, z1, z2]) => {
                let (z0, z1, z2) = (&z0[..], &z1[..], &z2[..]);
                scatter_term::<3>(block, &t, par, stack, |i| {
                    let b = base(i);
                    [z0[i] * b, z1[i] * b, z2[i] * b]
                })
            }
            _ => {
                for c in 0..term.n_columns() {
                    let start = c * term.n_levels();
                    let slot = &mut block[start..start + term.n_levels()];
                    match t.loading(c) {
                        None => scatter_term::<1>(slot, &t, par, stack, |i| [base(i)]),
                        Some(z) => {
                            scatter_term::<1>(slot, &t, par, stack, move |i| [z[i] * base(i)])
                        }
                    }
                }
            }
        }
    }
}

/// Rows per by-row leaf; leaves of 4·C·L rows bound how many C·L-long partials get added.
fn by_row_leaf_len(width: usize, rows: usize) -> usize {
    (4 * width).min(rows.div_ceil(2)).max(65_536)
}

/// Write one term's coefficient block: `block[c·L + l] = Σ_{i : level(i) = l} values(i)[c]`.
fn scatter_term<const C: usize>(
    block: &mut [f64],
    term: &PreparedTerm<'_>,
    par: Par,
    stack: &mut MemStack,
    values: impl Fn(usize) -> [f64; C] + Sync,
) {
    let (n_levels, levels) = (term.term.n_levels(), term.term.levels());
    debug_assert_eq!(block.len(), C * n_levels);
    match term.split {
        ScatterSplit::ByLevel(rows) => level_sums::<C>(block, rows, par, &values),
        ScatterSplit::ByRow => {
            let leaf_len = by_row_leaf_len(C * n_levels, levels.len());
            let scatter = |rows: Range<usize>, sums: &mut [f64]| {
                sums.fill(0.0);
                for i in rows {
                    let level = levels[i] as usize;
                    for (c, v) in values(i).into_iter().enumerate() {
                        sums[c * n_levels + level] += v;
                    }
                }
            };
            partial_sums(par, 0..levels.len(), leaf_len, &scatter, block, stack);
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
    par: Par,
    values: &(impl Fn(usize) -> [f64; C] + Sync),
) {
    let level_sum = |level: usize, par: Par| {
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
        fixed_tree(par, positions, LEVEL_LEAF, &leaf)
    };
    let n_levels = block.len() / C;
    let mut columns = block.chunks_mut(n_levels);
    let columns: [&mut [f64]; C] = std::array::from_fn(|_| columns.next().expect("C columns"));
    let fill = |first: usize, group: [&mut [f64]; C], par: Par| {
        let mut group = group;
        for k in 0..group[0].len() {
            let sums = level_sum(first + k, par);
            for (column, s) in group.iter_mut().zip(sums) {
                column[k] = s;
            }
        }
    };
    level_groups(par, 0, columns, &fill);
}

/// `fill(first level, columns, par)` over [`LEVEL_GROUP`]-aligned pieces of the `C` columns.
fn level_groups<const C: usize>(
    par: Par,
    first: usize,
    columns: [&mut [f64]; C],
    fill: &(impl Fn(usize, [&mut [f64]; C], Par) + Sync),
) {
    let groups = columns[0].len().div_ceil(LEVEL_GROUP);
    if par.degree() <= 1 || groups <= 1 {
        fill(first, columns, par);
        return;
    }
    let mid = groups / 2 * LEVEL_GROUP;
    let mut halves_of = columns.map(|c| {
        let (l, r) = c.split_at_mut(mid);
        (Some(l), Some(r))
    });
    let left: [&mut [f64]; C] = std::array::from_fn(|c| halves_of[c].0.take().expect("left"));
    let right: [&mut [f64]; C] = std::array::from_fn(|c| halves_of[c].1.take().expect("right"));
    fork(
        par,
        |p| level_groups(p, first, left, fill),
        |p| level_groups(p, first + mid, right, fill),
    );
}

/// `leaf` over `leaf_len` pieces of `items`, added in a binary tree fixed by both lengths alone.
fn fixed_tree<const C: usize>(
    par: Par,
    items: Range<usize>,
    leaf_len: usize,
    leaf: &(impl Fn(Range<usize>) -> [f64; C] + Sync),
) -> [f64; C] {
    if items.len() <= leaf_len {
        return leaf(items);
    }
    let mid = tree_mid(&items, leaf_len);
    let (mut left, right) = fork(
        par,
        |p| fixed_tree(p, items.start..mid, leaf_len, leaf),
        |p| fixed_tree(p, mid..items.end, leaf_len, leaf),
    );
    for (x, y) in left.iter_mut().zip(right) {
        *x += y;
    }
    left
}

/// Where [`fixed_tree`] and [`partial_sums`] cut `items`: a leaf multiple near the middle.
fn tree_mid(items: &Range<usize>, leaf_len: usize) -> usize {
    items.start + (items.len() / 2 / leaf_len).max(1) * leaf_len
}

/// [`fixed_tree`] into `out`, each right subtree's partial off `stack`.
fn partial_sums(
    par: Par,
    items: Range<usize>,
    leaf_len: usize,
    leaf: &(impl Fn(Range<usize>, &mut [f64]) + Sync),
    out: &mut [f64],
    stack: &mut MemStack,
) {
    if items.len() <= leaf_len {
        leaf(items, out);
        return;
    }
    let mid = tree_mid(&items, leaf_len);
    let width = out.len();
    let (mut right_out, stack) = stack.make_with(width, |_| 0.0);
    let (left, right) = (items.start..mid, mid..items.end);
    if par.degree() <= 1 {
        // Depth-first: the right subtree reuses what the left one freed.
        partial_sums(par, left, leaf_len, leaf, out, stack);
        partial_sums(par, right, leaf_len, leaf, &mut right_out, stack);
    } else {
        let left_items = left.clone();
        let need = move |p| partials_scratch(p, left_items, leaf_len, width);
        fork_on_stack(
            par,
            need,
            stack,
            |p, s| partial_sums(p, left, leaf_len, leaf, out, s),
            |p, s| partial_sums(p, right, leaf_len, leaf, &mut right_out, s),
        );
    }
    for (x, y) in out.iter_mut().zip(right_out.iter()) {
        *x += *y;
    }
}

/// The stack [`partial_sums`] takes for `items` under `par`.
fn partials_scratch(par: Par, items: Range<usize>, leaf_len: usize, width: usize) -> StackReq {
    if items.len() <= leaf_len {
        return StackReq::EMPTY;
    }
    let mid = tree_mid(&items, leaf_len);
    let left = partials_scratch(par, items.start..mid, leaf_len, width);
    let right = partials_scratch(par, mid..items.end, leaf_len, width);
    let children = if par.degree() <= 1 {
        left.or(right)
    } else {
        left.and(right)
    };
    StackReq::new::<f64>(width).and(children)
}
