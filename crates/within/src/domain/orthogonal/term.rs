//! One term's column block `D̃_t` of an [`OrthogonalDesign`], and the operations every kernel
//! reads it through.

use super::OrthogonalDesign;
use crate::domain::{Column, LevelSlots, TermLayout, TermRows};

/// Term `t`'s block `D̃_t`: row `i` holds `r(i) = [1?, u_1(i), …, u_v(i)]` in the columns of
/// level `level(i)`. Hides how the design stores rows and slopes.
#[derive(Clone, Copy)]
pub(crate) struct TermBlock<'p> {
    /// Where the block's coefficients sit.
    pub(crate) layout: &'p TermLayout,
    rows: &'p TermRows<'p>,
    /// Orthogonalized slopes in column order; empty for a slope-free term.
    slopes: &'p [Vec<f64>],
}

impl OrthogonalDesign<'_> {
    /// Term `term`'s block.
    pub(crate) fn term(&self, term: usize) -> TermBlock<'_> {
        TermBlock {
            layout: self.incidence.layout.term(term),
            rows: &self.incidence.terms[term],
            slopes: self.slopes[term].columns(),
        }
    }

    pub(crate) fn terms(&self) -> impl ExactSizeIterator<Item = TermBlock<'_>> {
        (0..self.incidence.layout.n_terms()).map(|term| self.term(term))
    }
}

/// A kernel over a block's rows `r(i)`.
pub(crate) trait RowKernel {
    /// Runs with `r` as a `C`-array, for the column counts that are fused.
    fn fixed<const C: usize>(self, row: impl Fn(usize) -> [f64; C] + Sync);
    /// Runs for wider terms, which read their columns through [`TermBlock::loading`].
    fn dynamic(self);
}

impl<'p> TermBlock<'p> {
    /// The level of every row.
    pub(crate) fn levels(&self) -> &'p [u32] {
        self.rows.levels()
    }

    /// Column `column`'s loading; `None` is the intercept.
    pub(crate) fn loading(&self, column: usize) -> Option<&'p [f64]> {
        match self.layout.column(column) {
            Column::Intercept => None,
            Column::Slope(j) => Some(&self.slopes[j][..]),
        }
    }

    /// The slope columns, in order.
    #[cfg(test)]
    pub(crate) fn slopes(&self) -> impl ExactSizeIterator<Item = &'p [f64]> {
        self.slopes.iter().map(|z| &z[..])
    }

    /// Hands `kernel` the row `r(i)`, at a fixed column count where one is fused.
    #[inline(always)]
    pub(crate) fn visit_rows(&self, kernel: impl RowKernel) {
        let z = |j: usize| &self.slopes[j][..];
        match (self.layout.intercept(), self.slopes.len()) {
            (true, 0) => kernel.fixed(|_| [1.0]),
            (true, 1) => {
                let z0 = z(0);
                kernel.fixed(move |i| [1.0, z0[i]])
            }
            (true, 2) => {
                let (z0, z1) = (z(0), z(1));
                kernel.fixed(move |i| [1.0, z0[i], z1[i]])
            }
            (true, 3) => {
                let (z0, z1, z2) = (z(0), z(1), z(2));
                kernel.fixed(move |i| [1.0, z0[i], z1[i], z2[i]])
            }
            (false, 1) => {
                let z0 = z(0);
                kernel.fixed(move |i| [z0[i]])
            }
            (false, 2) => {
                let (z0, z1) = (z(0), z(1));
                kernel.fixed(move |i| [z0[i], z1[i]])
            }
            (false, 3) => {
                let (z0, z1, z2) = (z(0), z(1), z(2));
                kernel.fixed(move |i| [z0[i], z1[i], z2[i]])
            }
            _ => kernel.dynamic(),
        }
    }

    /// Each level's fold of its rows, in the pieces the row split fixes: see [`ScatterSplit::fold`].
    pub(crate) fn fold<S: Copy + Send + Sync>(
        &self,
        init: S,
        step: &(impl Fn(&mut S, usize, usize) + Sync),
        merge: &(impl Fn(&mut S, &S) + Sync),
        out: impl LevelSlots<S>,
    ) {
        self.rows.fold(init, step, merge, out);
    }

    #[cfg(test)]
    pub(crate) fn split(&self) -> &'p crate::domain::ScatterSplit {
        &self.rows.split
    }
}
