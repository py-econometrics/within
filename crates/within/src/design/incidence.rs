//! `E`: per term the level of each row, the coefficient layout and the internal row order.

mod encoding;
mod layout;
mod level_reduction;
mod level_segments;
mod row_index;

pub(crate) use encoding::FactorEncoding;
pub use layout::CoefficientLayout;
pub(crate) use layout::{Column, TermLayout, Unknowns};
pub(crate) use level_reduction::{LevelReduction, LevelSlots};
use level_segments::LevelSegments;
#[cfg(test)]
pub(crate) use level_segments::LEVEL_LEAF;
use row_index::RowIndex;

use std::borrow::Cow;

use rayon::prelude::*;
use schwarz_precond::{MemBuffer, MemStack, StackReq};

/// One term's level of each observation, in the design's row order.
#[derive(Debug, Clone)]
struct TermRows<'a> {
    /// Internal level position of every observation.
    levels: Cow<'a, [u32]>,
    /// The rows of each level, as the scatter-add splits them.
    reduction: LevelReduction,
}

/// `E_t`: term `t`'s level of each row, and where its coefficients sit.
#[derive(Clone, Copy)]
pub(crate) struct TermIncidence<'p> {
    layout: &'p TermLayout,
    rows: &'p TermRows<'p>,
}

impl<'p> TermIncidence<'p> {
    /// Where the term's coefficients sit.
    pub(crate) fn layout(&self) -> &'p TermLayout {
        self.layout
    }

    /// The level of every row.
    pub(crate) fn levels(&self) -> &'p [u32] {
        &self.rows.levels
    }

    /// Each level's fold of its rows, in the pieces its reduction fixes: see [`LevelReduction::fold`].
    pub(crate) fn fold<S: Copy + Send + Sync>(
        &self,
        init: S,
        step: &(impl Fn(&mut S, usize, usize) + Sync),
        merge: &(impl Fn(&mut S, &S) + Sync),
        out: impl LevelSlots<S>,
    ) {
        let mut buffer = MemBuffer::new(self.fold_scratch::<S>());
        self.fold_with(init, step, merge, out, MemStack::new(&mut buffer));
    }

    /// What [`fold_with`](Self::fold_with) takes from its stack for states `S`.
    pub(crate) fn fold_scratch<S>(&self) -> StackReq {
        let n_levels = self.layout.n_levels();
        self.rows
            .reduction
            .scratch::<S>(self.rows.levels.len(), n_levels)
    }

    /// [`fold`](Self::fold) on `stack`, which holds at least [`fold_scratch`](Self::fold_scratch).
    pub(crate) fn fold_with<S: Copy + Send + Sync>(
        &self,
        init: S,
        step: &(impl Fn(&mut S, usize, usize) + Sync),
        merge: &(impl Fn(&mut S, &S) + Sync),
        out: impl LevelSlots<S>,
        stack: &mut MemStack,
    ) {
        self.rows
            .reduction
            .fold_with(&self.rows.levels, init, step, merge, out, stack);
    }

    #[cfg(test)]
    pub(crate) fn reduction(&self) -> &'p LevelReduction {
        &self.rows.reduction
    }
}

impl TermRows<'_> {
    fn into_owned(self) -> TermRows<'static> {
        TermRows {
            levels: Cow::Owned(self.levels.into_owned()),
            reduction: self.reduction,
        }
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
            Some(segments) => (segments, None),
            None => {
                let (offsets, perm) = RowIndex::argsort(&levels[sort_term], n_levels(sort_term));
                for levels in levels.iter_mut() {
                    *levels = Cow::Owned(perm.gather(levels));
                }
                (LevelSegments::new(offsets), Some(perm))
            }
        };
        let sorted = TermRows {
            levels: levels.remove(sort_term),
            reduction: LevelReduction::Segments(segments, None),
        };
        let others: Vec<usize> = (0..layout.n_terms()).filter(|&t| t != sort_term).collect();
        // Factors nested in the sorted one come out sorted and need no row index.
        let mut terms: Vec<TermRows> = levels
            .into_par_iter()
            .zip(others)
            .map(|(levels, t)| TermRows {
                reduction: LevelReduction::new(&levels, n_levels(t)),
                levels,
            })
            .collect();
        terms.insert(sort_term, sorted);
        Self {
            layout,
            order: RowOrder { perm },
            terms,
        }
    }

    /// Convert every level column to owned, dropping ties to caller buffers.
    pub(super) fn into_owned(self) -> Incidence<'static> {
        Incidence {
            layout: self.layout,
            order: self.order,
            terms: self.terms.into_iter().map(TermRows::into_owned).collect(),
        }
    }

    pub(crate) fn layout(&self) -> &CoefficientLayout {
        &self.layout
    }

    /// The internal row order.
    pub(crate) fn order(&self) -> &RowOrder {
        &self.order
    }

    /// Term `term`'s `E_t`.
    pub(crate) fn term(&self, term: usize) -> TermIncidence<'_> {
        TermIncidence {
            layout: self.layout.term(term),
            rows: &self.terms[term],
        }
    }

    pub(crate) fn n_obs(&self) -> usize {
        self.terms[0].levels.len()
    }
}
