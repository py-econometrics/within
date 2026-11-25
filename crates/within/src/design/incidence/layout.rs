//! Flat coefficient index ⇄ caller-visible [`CoefficientAddress`] translation.

use std::ops::Range;
use std::sync::Arc;

use super::FactorEncoding;
use crate::design::{Channel, CoefficientAddress, CoefficientPosition};

/// What one coefficient column of a term multiplies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Column {
    Intercept,
    /// The term's slope `j`.
    Slope(usize),
}

/// Translates a [`CoefficientAddress`] to its flat index in [`SolveResult::x`](crate::SolveResult::x)
/// and back, including the translation between caller-visible labels and
/// internal compact level positions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoefficientLayout {
    /// Term blocks in ascending, contiguous order.
    pub(in crate::design) terms: Arc<[TermLayout]>,
}

/// One term's block of coefficients: column `c` of `level` sits at `c · n_levels + level` in it.
/// Where the block sits is the index set's: [`CoefficientLayout`] for `β`, [`Unknowns`] for `x`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TermLayout {
    pub(in crate::design) encoding: FactorEncoding,
    /// Column 0 is an intercept; the slopes follow it.
    pub(in crate::design) intercept: bool,
    pub(in crate::design) n_slopes: usize,
}

impl TermLayout {
    pub(crate) fn n_levels(&self) -> usize {
        self.encoding.n_levels()
    }

    pub(crate) fn intercept(&self) -> bool {
        self.intercept
    }

    pub(crate) fn n_slopes(&self) -> usize {
        self.n_slopes
    }

    pub(crate) fn n_columns(&self) -> usize {
        self.intercept as usize + self.n_slopes
    }

    pub(crate) fn n_dofs(&self) -> usize {
        self.n_columns() * self.n_levels()
    }

    /// Column `column`'s `n_levels` slots within the block.
    pub(crate) fn block_column(&self, column: usize) -> Range<usize> {
        let start = column * self.n_levels();
        start..start + self.n_levels()
    }

    /// What column `index` multiplies.
    pub(crate) fn column(&self, index: usize) -> Column {
        match index.checked_sub(self.intercept as usize) {
            None => Column::Intercept,
            Some(j) => Column::Slope(j),
        }
    }

    pub(crate) fn intercept_column(&self) -> Option<usize> {
        self.intercept.then_some(0)
    }

    pub(crate) fn slope_column(&self, slope: usize) -> usize {
        self.intercept as usize + slope
    }
}

impl CoefficientLayout {
    /// Total number of coefficients (the length of [`SolveResult::x`](crate::SolveResult::x)).
    pub fn n_dofs(&self) -> usize {
        self.terms.iter().map(TermLayout::n_dofs).sum()
    }

    /// Number of terms in the design.
    pub fn n_terms(&self) -> usize {
        self.terms.len()
    }

    /// Level count of `term`, or `None` if `term` is out of range.
    pub fn n_levels(&self, term: usize) -> Option<usize> {
        self.terms.get(term).map(TermLayout::n_levels)
    }

    /// Coefficient-column count of `term` (`intercept? + slopes`, ordered
    /// `[intercept?, slopes…]`), or `None` if `term` is out of range.
    pub fn n_columns(&self, term: usize) -> Option<usize> {
        self.terms.get(term).map(TermLayout::n_columns)
    }

    /// Flat [`SolveResult::x`](crate::SolveResult::x) index of `at`, or `None` if its term,
    /// column, or caller-visible level label is out of range.
    pub fn index(&self, at: CoefficientAddress) -> Option<usize> {
        let term = self.terms.get(at.channel.term)?;
        if at.channel.column >= term.n_columns() {
            return None;
        }
        let position = term.encoding.position(at.level)?;
        Some(self.dofs(at.channel).start + position)
    }

    /// The address of flat index `i`, or `None` if `i >= n_dofs`.
    pub fn address(&self, i: usize) -> Option<CoefficientAddress> {
        let mut start = 0;
        for (term, layout) in self.terms.iter().enumerate() {
            if i < start + layout.n_dofs() {
                let (within, n_levels) = (i - start, layout.n_levels());
                return Some(self.caller_address(CoefficientPosition {
                    channel: Channel {
                        term,
                        column: within / n_levels,
                    },
                    level: within % n_levels,
                }));
            }
            start += layout.n_dofs();
        }
        None
    }

    pub(crate) fn term(&self, term: usize) -> &TermLayout {
        &self.terms[term]
    }

    pub(crate) fn terms(&self) -> &[TermLayout] {
        &self.terms
    }

    /// Term `term`'s block in `β`: the terms in order, back to back.
    pub(crate) fn term_dofs(&self, term: usize) -> Range<usize> {
        let start: usize = self.terms[..term].iter().map(TermLayout::n_dofs).sum();
        start..start + self.terms[term].n_dofs()
    }

    /// `channel`'s `n_levels` slots in `β`.
    pub(crate) fn dofs(&self, channel: Channel) -> Range<usize> {
        let (block, local) = (
            self.term_dofs(channel.term),
            self.terms[channel.term].block_column(channel.column),
        );
        block.start + local.start..block.start + local.end
    }

    /// The term's coefficient columns in layout order.
    pub(crate) fn channels(&self, term: usize) -> impl Iterator<Item = Channel> + '_ {
        (0..self.terms[term].n_columns()).map(move |column| Channel { term, column })
    }

    /// What `channel` multiplies.
    pub(crate) fn column(&self, channel: Channel) -> Column {
        self.terms[channel.term].column(channel.column)
    }

    pub(crate) fn caller_address(&self, position: CoefficientPosition) -> CoefficientAddress {
        let level = self.terms[position.channel.term]
            .encoding
            .label(position.level)
            .expect("coefficient position belongs to its term");
        CoefficientAddress {
            channel: position.channel,
            level,
        }
    }
}

/// The coefficients a system solves for, `x`: every term's block, back to back in term order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Unknowns {
    /// Term `t`'s block in `x` is `starts[t]..starts[t + 1]`.
    starts: Vec<usize>,
}

impl Unknowns {
    pub(crate) fn new(layout: &CoefficientLayout) -> Self {
        let starts = std::iter::once(0)
            .chain(layout.terms.iter().scan(0, |end, term| {
                *end += term.n_dofs();
                Some(*end)
            }))
            .collect();
        Self { starts }
    }

    /// The number of unknowns.
    pub(crate) fn len(&self) -> usize {
        self.starts[self.starts.len() - 1]
    }

    /// Term `term`'s block in `x`.
    pub(crate) fn block(&self, term: usize) -> Range<usize> {
        self.starts[term]..self.starts[term + 1]
    }

    /// Where each term's block starts, then where the last ends.
    pub(crate) fn starts(&self) -> &[usize] {
        &self.starts
    }

    /// `channel`'s `n_levels` slots in `x`.
    pub(crate) fn dofs(&self, layout: &CoefficientLayout, channel: Channel) -> Range<usize> {
        let (block, local) = (
            self.block(channel.term),
            layout.terms[channel.term].block_column(channel.column),
        );
        block.start + local.start..block.start + local.end
    }
}
