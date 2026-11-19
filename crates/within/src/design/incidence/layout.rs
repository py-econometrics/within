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
/// Each start is stored once; ends follow from the term's own counts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TermLayout {
    pub(in crate::design) encoding: FactorEncoding,
    /// Start of the block in the flat coefficient vector.
    pub(in crate::design) offset: usize,
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

    /// The block in the flat coefficient vector.
    pub(crate) fn dofs(&self) -> Range<usize> {
        self.offset..self.offset + self.n_dofs()
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
        self.terms.last().map_or(0, |t| t.dofs().end)
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
        if i >= self.n_dofs() {
            return None;
        }
        // Term blocks ascend by offset, so the owner is the last one not exceeding `i`.
        let term = self.terms.partition_point(|t| t.offset <= i) - 1;
        let within = i - self.terms[term].offset;
        let n_levels = self.terms[term].n_levels();
        Some(self.caller_address(CoefficientPosition {
            channel: Channel {
                term,
                column: within / n_levels,
            },
            level: within % n_levels,
        }))
    }

    pub(crate) fn term(&self, term: usize) -> &TermLayout {
        &self.terms[term]
    }

    pub(crate) fn terms(&self) -> &[TermLayout] {
        &self.terms
    }

    /// The term with the most coefficients, ties to the last: the sort key and the eliminated term.
    pub(crate) fn dominant(&self) -> usize {
        (0..self.terms.len())
            .max_by_key(|&t| self.terms[t].n_dofs())
            .expect("a design has a term")
    }

    /// `channel`'s `n_levels` slots in the flat coefficient vector.
    pub(crate) fn dofs(&self, channel: Channel) -> Range<usize> {
        let term = &self.terms[channel.term];
        let local = term.block_column(channel.column);
        term.offset + local.start..term.offset + local.end
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
