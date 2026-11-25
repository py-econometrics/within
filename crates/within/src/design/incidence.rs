//! `E`: per term the level of each row, the coefficient layout and the internal row order.

mod encoding;
mod layout;
mod level_reduction;
mod level_segments;
mod row_index;

pub(crate) use encoding::FactorEncoding;
pub use layout::CoefficientLayout;
pub(crate) use layout::{Column, TermLayout, Unknowns};
use level_reduction::LevelReduction;
pub(crate) use level_segments::LevelSegments;
#[cfg(test)]
pub(crate) use level_segments::LEVEL_LEAF;
use row_index::RowIndex;

use std::borrow::Cow;
use std::sync::Arc;

use rayon::prelude::*;

/// One term's level of each observation, in the design's row order.
#[derive(Debug, Clone)]
struct TermRows<'a> {
    /// Internal level position of every observation.
    levels: Cow<'a, [u32]>,
    /// The rows of each level, as its per-level folds split them.
    reduction: LevelReduction,
}

impl TermRows<'_> {
    fn into_owned(self) -> TermRows<'static> {
        TermRows {
            levels: Cow::Owned(self.levels.into_owned()),
            reduction: self.reduction,
        }
    }
}

/// `E_t`: term `t`'s level of each row, and where its coefficients sit.
#[derive(Clone, Copy)]
pub(crate) struct TermIncidence<'p> {
    layout: &'p TermLayout,
    levels: &'p [u32],
    reduction: &'p LevelReduction,
}

impl<'p> TermIncidence<'p> {
    /// Where the term's coefficients sit.
    pub(crate) fn layout(&self) -> &'p TermLayout {
        self.layout
    }

    /// The level of every row.
    pub(crate) fn levels(&self) -> &'p [u32] {
        self.levels
    }

    /// Each level's fold of its rows, in the pieces its reduction fixes: see
    /// [`LevelReduction::fold`].
    pub(crate) fn fold<S: Copy + Send + Sync>(
        &self,
        init: S,
        step: &(impl Fn(&mut S, usize, usize) + Sync),
        merge: &(impl Fn(&mut S, &S) + Sync),
        out: &mut [S],
    ) {
        (self.reduction).fold(self.levels, init, step, merge, out);
    }

    /// Whether the term's fold scatters by row.
    #[cfg(test)]
    pub(crate) fn by_row(&self) -> bool {
        matches!(self.reduction, LevelReduction::ByRow)
    }
}

/// The internal row order: the caller's rows sorted by the sort term's levels.
#[derive(Clone, Debug)]
pub(crate) struct RowOrder {
    /// `perm[k]` = caller's index of the row at internal position `k`; `None` when the caller's
    /// rows were already sorted.
    perm: Option<RowIndex>,
}

impl RowOrder {
    /// Caller order → internal order: `out[k] = v[perm[k]]`; passes `v` through when unpermuted.
    pub(crate) fn permute_in<'v>(&self, v: impl Into<Cow<'v, [f64]>>) -> Cow<'v, [f64]> {
        let v = v.into();
        match &self.perm {
            None => v,
            Some(perm) => Cow::Owned(perm.gather(&v)),
        }
    }

    /// Internal order → caller order: `out[perm[k]] = v[k]`.
    pub(crate) fn permute_out(&self, v: Vec<f64>) -> Vec<f64> {
        match &self.perm {
            None => v,
            Some(perm) => perm.scatter(&v),
        }
    }
}

/// `E`: per term the level of each row, with the coefficient layout and the row order. Weight-free,
/// so a design and its orthogonalization share it.
#[derive(Clone, Debug)]
pub(crate) struct Incidence<'a> {
    layout: CoefficientLayout,
    order: RowOrder,
    /// Every term, in term order, its rows in the internal order.
    terms: Vec<TermRows<'a>>,
    /// The sort term: the one the rows are sorted by.
    sort_term: usize,
    /// The sort term's segments: the rows of each of its levels, shared with its reduction.
    segments: Arc<LevelSegments>,
}

impl<'a> Incidence<'a> {
    /// `E` for each term's internal level positions, its rows sorted by the term with the most
    /// coefficients, ties to the last, so each of that term's levels is one segment of rows.
    pub(super) fn new(mut levels: Vec<Cow<'a, [u32]>>, layout: CoefficientLayout) -> Self {
        let sort_term = (0..layout.n_terms())
            .max_by_key(|&t| layout.term(t).n_dofs())
            .expect("a design has a term");
        let n_levels = |t: usize| layout.term(t).n_levels();
        let (segments, perm) = match LevelSegments::of(&levels[sort_term], n_levels(sort_term)) {
            Some(segments) => (Arc::new(segments), None),
            None => {
                let (offsets, perm) = RowIndex::argsort(&levels[sort_term], n_levels(sort_term));
                for levels in levels.iter_mut() {
                    *levels = Cow::Owned(perm.gather(levels));
                }
                (Arc::new(LevelSegments::new(offsets)), Some(perm))
            }
        };
        // Factors nested in the sorted one come out sorted and need no row index.
        let terms = levels
            .into_par_iter()
            .enumerate()
            .map(|(t, levels)| TermRows {
                reduction: if t == sort_term {
                    LevelReduction::Segments(Arc::clone(&segments), None)
                } else {
                    LevelReduction::new(&levels, n_levels(t))
                },
                levels,
            })
            .collect();
        Self {
            layout,
            order: RowOrder { perm },
            terms,
            sort_term,
            segments,
        }
    }

    /// Convert every level column to owned, dropping ties to caller buffers.
    pub(super) fn into_owned(self) -> Incidence<'static> {
        Incidence {
            layout: self.layout,
            order: self.order,
            terms: self.terms.into_iter().map(TermRows::into_owned).collect(),
            sort_term: self.sort_term,
            segments: self.segments,
        }
    }

    pub(crate) fn layout(&self) -> &CoefficientLayout {
        &self.layout
    }

    /// The internal row order.
    pub(crate) fn order(&self) -> &RowOrder {
        &self.order
    }

    /// The sort term: the one the rows are sorted by.
    pub(crate) fn sort_term(&self) -> usize {
        self.sort_term
    }

    /// The sort term's segments: the rows of each of its levels.
    pub(crate) fn segments(&self) -> &LevelSegments {
        &self.segments
    }

    /// Term `term`'s `E_t`.
    pub(crate) fn term(&self, term: usize) -> TermIncidence<'_> {
        let rows = &self.terms[term];
        TermIncidence {
            layout: self.layout.term(term),
            levels: &rows.levels,
            reduction: &rows.reduction,
        }
    }

    pub(crate) fn n_obs(&self) -> usize {
        self.terms[self.sort_term].levels.len()
    }
}
