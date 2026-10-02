//! Observation space → coefficient space with disjoint per-level destinations.

use crate::domain::{LevelMembership, PreparedDesign};
use rayon::prelude::*;
use std::ops::Range;

/// Every level sums stable row blocks, then combines them with a fixed binary
/// tree. Scheduling can change, but neither row order nor arithmetic can.
fn level_sum(
    membership: &LevelMembership,
    range: Range<usize>,
    value: &(impl Fn(usize) -> f64 + Sync),
) -> f64 {
    const CHUNK: usize = 4096;
    if range.len() <= CHUNK {
        return range
            .map(|position| value(membership.row(position)))
            .fold(0.0, |sum, value| sum + value);
    }
    let mid = range.start + (range.len() / 2 / CHUNK).max(1) * CHUNK;
    let (left, right) = rayon::join(
        || level_sum(membership, range.start..mid, value),
        || level_sum(membership, mid..range.end, value),
    );
    left + right
}

pub(super) fn scatter_apply(
    prepared: &PreparedDesign<'_>,
    dst: &mut [f64],
    base: &(impl Fn(usize) -> f64 + Sync),
) {
    let design = &prepared.design;
    debug_assert_eq!(dst.len(), design.n_dofs);
    for (q, term) in design.terms.iter().enumerate() {
        let membership = &design.membership[q];
        let prepared_term = prepared.term(q);
        for column in 0..term.n_columns() {
            let slot = &mut dst[term.column_dofs(column)];
            let loading = prepared_term.loading(column);
            let value = |row| loading.map_or_else(|| base(row), |z| z[row] * base(row));
            let apply = |(level, output): (usize, &mut f64)| {
                *output = level_sum(membership, membership.level_range(level), &value);
            };
            if design.n_obs > super::PAR_THRESHOLD {
                slot.par_iter_mut().enumerate().for_each(apply);
            } else {
                slot.iter_mut().enumerate().for_each(apply);
            }
        }
    }
}
