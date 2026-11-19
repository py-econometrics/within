//! One term's column block `D_t = E_t • X_t` of a design, and the operations every kernel reads
//! it through.

use std::ops::Range;

use super::{Column, LevelSlots, TermIncidence, TermLayout};

/// Term `t`'s block `D_t`: row `i` holds `r(i) = [1?, z_1(i), …, z_v(i)]` in the columns of
/// level `level(i)`.
#[derive(Clone, Copy)]
pub(crate) struct TermBlock<'p> {
    incidence: TermIncidence<'p>,
    /// The layout's `n_slopes` columns, in order.
    slopes: &'p [Vec<f64>],
    /// Its first coefficient in the vector the products read and write.
    at: usize,
}

/// A kernel over a block's rows `r(i)`.
pub(crate) trait RowKernel {
    /// Runs with `r` as a `C`-array, for the column counts that are fused.
    fn fixed<const C: usize>(self, row: impl Fn(usize) -> [f64; C] + Sync);
    /// Runs for wider terms, which read their columns through [`TermBlock::loading`].
    fn dynamic(self);
}

impl<'p> TermBlock<'p> {
    pub(crate) fn new(incidence: TermIncidence<'p>, slopes: &'p [Vec<f64>], at: usize) -> Self {
        Self {
            incidence,
            slopes,
            at,
        }
    }

    /// The block's coefficients in the vector the products read and write.
    pub(crate) fn dofs(&self) -> Range<usize> {
        self.at..self.at + self.layout().n_dofs()
    }

    /// The block's shape: levels and columns.
    pub(crate) fn layout(&self) -> &'p TermLayout {
        self.incidence.layout()
    }

    /// The level of every row.
    pub(crate) fn levels(&self) -> &'p [u32] {
        self.incidence.levels()
    }

    /// Column `column`'s loading; `None` is the intercept.
    pub(crate) fn loading(&self, column: usize) -> Option<&'p [f64]> {
        match self.layout().column(column) {
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
        match (self.layout().intercept(), self.layout().n_slopes()) {
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

    /// Each level's fold of its rows, in the pieces its reduction fixes: see [`LevelReduction::fold`].
    pub(crate) fn fold<S: Copy + Send + Sync>(
        &self,
        init: S,
        step: &(impl Fn(&mut S, usize, usize) + Sync),
        merge: &(impl Fn(&mut S, &S) + Sync),
        out: impl LevelSlots<S>,
    ) {
        self.incidence.fold(init, step, merge, out);
    }

    #[cfg(test)]
    pub(crate) fn reduction(&self) -> &'p crate::design::LevelReduction {
        self.incidence.reduction()
    }
}
