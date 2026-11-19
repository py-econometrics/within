//! `E`: per term the level of each row, the coefficient layout and the internal row order.

mod encoding;
mod layout;
mod row_index;
mod scatter_split;

pub(crate) use encoding::FactorEncoding;
pub use layout::CoefficientLayout;
pub(crate) use layout::{Column, TermLayout};
pub(crate) use row_index::RowIndex;
#[cfg(test)]
pub(crate) use scatter_split::LEVEL_LEAF;
pub(crate) use scatter_split::{LevelSlots, ScatterSplit, LEVEL_GROUP};

use std::borrow::Cow;

/// One term's level of each observation, in the design's row order.
#[derive(Debug, Clone)]
pub(crate) struct TermRows<'a> {
    /// Internal level position of every observation.
    pub(super) levels: Cow<'a, [u32]>,
    /// The rows of each level, as the scatter-add splits them.
    pub(super) split: ScatterSplit,
}

impl TermRows<'_> {
    pub(crate) fn levels(&self) -> &[u32] {
        &self.levels
    }

    /// Each level's fold of its rows, in the pieces the row split fixes: see [`ScatterSplit::fold`].
    pub(crate) fn fold<S: Copy + Send + Sync>(
        &self,
        init: S,
        step: &(impl Fn(&mut S, usize, usize) + Sync),
        merge: &(impl Fn(&mut S, &S) + Sync),
        out: impl LevelSlots<S>,
    ) {
        self.split.fold(&self.levels, init, step, merge, out);
    }

    /// Per level, its rows as one contiguous run, when the rows are sorted by level.
    pub(crate) fn sorted_runs(&self) -> Option<&[usize]> {
        self.split.sorted_offsets()
    }

    pub(super) fn into_owned(self) -> TermRows<'static> {
        TermRows {
            levels: Cow::Owned(self.levels.into_owned()),
            split: self.split,
        }
    }
}

/// Group `k` of the counted `keys` owns `offsets[k]..offsets[k + 1]`.
pub(crate) fn offsets(n_keys: usize, keys: impl IntoIterator<Item = usize>) -> Vec<usize> {
    let mut offsets = vec![0usize; n_keys + 1];
    for k in keys {
        offsets[k + 1] += 1;
    }
    for i in 1..offsets.len() {
        offsets[i] += offsets[i - 1];
    }
    offsets
}

/// `E`: per term the level of each row, with the coefficient layout and the row order. Weight-free,
/// so a design and its orthogonalization share it.
#[derive(Clone, Debug)]
pub(crate) struct Incidence<'a> {
    pub(super) layout: CoefficientLayout,
    /// Rows in internal order (caller's, or an owned locality-sorted copy).
    pub(super) terms: Vec<TermRows<'a>>,
    /// `obs_perm[k]` = caller's original index of the observation at internal position `k`.
    pub(super) obs_perm: Option<RowIndex>,
}

impl Incidence<'_> {
    pub(crate) fn layout(&self) -> &CoefficientLayout {
        &self.layout
    }

    pub(crate) fn n_obs(&self) -> usize {
        self.terms[0].levels.len()
    }

    /// Caller order → internal order: `out[k] = v[obs_perm[k]]`; borrows when unpermuted.
    pub(crate) fn permute_obs_in<'v>(&self, v: &'v [f64]) -> Cow<'v, [f64]> {
        match &self.obs_perm {
            None => Cow::Borrowed(v),
            Some(perm) => Cow::Owned(perm.gather(v)),
        }
    }

    /// Internal order → caller order: `out[obs_perm[k]] = v[k]`.
    pub(crate) fn permute_obs_out(&self, v: Vec<f64>) -> Vec<f64> {
        match &self.obs_perm {
            None => v,
            Some(perm) => perm.scatter(&v),
        }
    }
}
