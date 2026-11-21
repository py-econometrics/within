//! One term's column block `D_t = E_t • X_t` of a design, and the operations every kernel reads
//! it through.

use schwarz_precond::StackReq;

use super::{Column, TermIncidence, TermLayout};

/// Term `t`'s block `D_t`: row `i` holds `r(i) = [1?, z_1(i), …, z_v(i)]` in the columns of
/// level `level(i)`.
#[derive(Clone, Copy)]
pub(crate) struct TermBlock<'p> {
    incidence: TermIncidence<'p>,
    /// The layout's `n_slopes` columns, in order.
    slopes: &'p [Vec<f64>],
}

/// A kernel over a block's rows `r(i)`.
pub(crate) trait RowKernel {
    /// Runs with `r` as a `C`-array, for the column counts that are fused.
    fn fixed<const C: usize>(self, row: impl Fn(usize) -> [f64; C] + Sync);
    /// Runs for wider terms, which read their columns through [`TermBlock::loading`].
    fn dynamic(self);
}

impl<'p> TermBlock<'p> {
    pub(crate) fn new(incidence: TermIncidence<'p>, slopes: &'p [Vec<f64>]) -> Self {
        Self { incidence, slopes }
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

    /// What `E_t`'s fold takes from its stack for the rows [`visit_rows`](Self::visit_rows) hands a
    /// kernel.
    pub(crate) fn fold_scratch(&self) -> StackReq {
        struct Width<'a>(&'a mut usize);
        impl RowKernel for Width<'_> {
            fn fixed<const C: usize>(self, _: impl Fn(usize) -> [f64; C] + Sync) {
                *self.0 = C;
            }
            fn dynamic(self) {
                *self.0 = 1;
            }
        }
        let mut width = 0;
        self.visit_rows(Width(&mut width));
        self.incidence.fold_scratch::<f64>().array(width)
    }

    /// `E_t`, which folds the block's rows per level.
    pub(crate) fn incidence(&self) -> TermIncidence<'p> {
        self.incidence
    }
}
