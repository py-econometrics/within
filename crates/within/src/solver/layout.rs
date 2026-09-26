//! Flat coefficient index ⇄ caller-visible [`CoefficientAddress`] translation.

use crate::channel::{Channel, Coefficient, CoefficientAddress};
use crate::domain::{Design, FactorEncoding};

impl Coefficient<usize> {
    pub(super) fn to_caller_address(self, design: &Design) -> CoefficientAddress {
        let level = design.terms[self.channel.term]
            .encoding
            .label(self.level)
            .expect("coefficient position belongs to its term");

        CoefficientAddress {
            channel: self.channel,
            level,
        }
    }
}

/// Translates a [`CoefficientAddress`] to its flat index in [`SolveResult::x`](crate::SolveResult::x)
/// and back, including the translation between caller-visible labels and
/// internal compact level positions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoefficientLayout {
    terms: Vec<TermLayout>,
    n_dofs: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TermLayout {
    offset: usize,
    encoding: FactorEncoding,
    n_columns: usize,
}

impl CoefficientLayout {
    pub(crate) fn from_design(design: &Design) -> Self {
        let terms = design
            .terms
            .iter()
            .map(|t| TermLayout {
                offset: t.offset,
                encoding: t.encoding.clone(),
                n_columns: t.n_columns(),
            })
            .collect();
        Self {
            terms,
            n_dofs: design.n_dofs,
        }
    }

    /// Total number of coefficients (the length of [`SolveResult::x`](crate::SolveResult::x)).
    pub fn n_dofs(&self) -> usize {
        self.n_dofs
    }

    /// Number of terms in the design.
    pub fn n_terms(&self) -> usize {
        self.terms.len()
    }

    /// Level count of `term`, or `None` if `term` is out of range.
    pub fn n_levels(&self, term: usize) -> Option<usize> {
        self.terms.get(term).map(|t| t.encoding.n_levels())
    }

    /// Coefficient-column count of `term` (`intercept? + slopes`, ordered
    /// `[intercept?, slopes…]`), or `None` if `term` is out of range.
    pub fn n_columns(&self, term: usize) -> Option<usize> {
        self.terms.get(term).map(|t| t.n_columns)
    }

    /// Flat [`SolveResult::x`](crate::SolveResult::x) index of `at`, or `None` if its term,
    /// column, or caller-visible level label is out of range.
    pub fn index(&self, at: CoefficientAddress) -> Option<usize> {
        let term = self.terms.get(at.channel.term)?;
        if at.channel.column >= term.n_columns {
            return None;
        }
        let position = term.encoding.position(at.level)?;
        Some(term.offset + at.channel.column * term.encoding.n_levels() + position)
    }

    /// The address of flat index `i`, or `None` if `i >= n_dofs`.
    pub fn address(&self, i: usize) -> Option<CoefficientAddress> {
        if i >= self.n_dofs {
            return None;
        }
        // Term blocks ascend by offset, so the owner is the last one not exceeding `i`.
        let term = self.terms.partition_point(|t| t.offset <= i) - 1;
        let t = &self.terms[term];
        let within = i - t.offset;
        let n_levels = t.encoding.n_levels();
        let level = t
            .encoding
            .label(within % n_levels)
            .expect("coefficient position belongs to the term encoding");
        Some(CoefficientAddress {
            channel: Channel {
                term,
                column: within / n_levels,
            },
            level,
        })
    }
}
