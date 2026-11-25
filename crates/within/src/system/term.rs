//! One term's column block `D̃_t = E_t • X̃_t` of the orthogonal design, and the operations every
//! kernel reads it through.

use crate::design::{Column, TermIncidence, TermLayout};

/// Term `t`'s block `D̃_t`: row `i` holds `r(i) = [1?, u_1(i), …, u_v(i)]` in the columns of
/// level `level(i)`.
#[derive(Clone, Copy)]
pub(crate) struct TermBlock<'p> {
    incidence: TermIncidence<'p>,
    /// The layout's `n_slopes` columns, in order.
    slopes: &'p [Vec<f64>],
}

/// A row `r(i)` of a term's block: its value in each of the term's columns.
pub(crate) trait TermRow: Sync + Copy {
    /// The term's column count.
    fn width(&self) -> usize;

    /// `f(c, r_c(i))` for each column `c`, in order.
    fn row(&self, i: usize, f: impl FnMut(usize, f64));

    /// `fit = Σ_k scale(i) · v_k · r(i)`, `i = start + k`.
    #[inline(always)]
    fn moments(&self, start: usize, v: &[f64], scale: impl Fn(usize) -> f64, fit: &mut [f64]) {
        fit.fill(0.0);
        for (k, &vk) in v.iter().enumerate() {
            let i = start + k;
            let sv = scale(i) * vk;
            self.row(i, |c, r| fit[c] += r * sv);
        }
    }
}

/// The widest fused row.
const FUSED_WIDTH: usize = 4;

/// A row at a fused shape: `[1, z_1(i), …, z_S(i)]` with the intercept `I`, `[z_1(i), …]` without.
#[derive(Clone, Copy)]
pub(crate) struct Row<'p, const I: bool, const S: usize> {
    slopes: [&'p [f64]; S],
}

impl<const I: bool, const S: usize> TermRow for Row<'_, I, S> {
    #[inline(always)]
    fn width(&self) -> usize {
        I as usize + S
    }

    #[inline(always)]
    fn row(&self, i: usize, mut f: impl FnMut(usize, f64)) {
        if I {
            f(0, 1.0);
        }
        for (j, z) in self.slopes.iter().enumerate() {
            f(I as usize + j, z[i]);
        }
    }

    /// The moments in registers: on the heap they cost 12%.
    #[inline(always)]
    fn moments(&self, start: usize, v: &[f64], scale: impl Fn(usize) -> f64, fit: &mut [f64]) {
        let mut sums = [0.0; FUSED_WIDTH];
        for (k, &vk) in v.iter().enumerate() {
            let i = start + k;
            let sv = scale(i) * vk;
            self.row(i, |c, r| sums[c] += r * sv);
        }
        fit.copy_from_slice(&sums[..self.width()]);
    }
}

/// Any shape, each column read through its loading.
impl TermRow for TermBlock<'_> {
    fn width(&self) -> usize {
        self.layout().n_columns()
    }

    fn row(&self, i: usize, mut f: impl FnMut(usize, f64)) {
        for c in 0..self.width() {
            f(c, self.loading(c).map_or(1.0, |z| z[i]));
        }
    }
}

/// A block's rows as the type that fits its shape: `I` with the intercept, `S` slopes only, each
/// with its slope count; `Dynamic` at any other shape.
#[derive(Clone, Copy)]
pub(crate) enum Rows<'p> {
    I0(Row<'p, true, 0>),
    I1(Row<'p, true, 1>),
    I2(Row<'p, true, 2>),
    I3(Row<'p, true, 3>),
    S1(Row<'p, false, 1>),
    S2(Row<'p, false, 2>),
    S3(Row<'p, false, 3>),
    Dynamic(TermBlock<'p>),
}

/// Each call dispatched on the shape: how a kept side of more terms than fuse reads each one.
impl TermRow for Rows<'_> {
    #[inline(always)]
    fn width(&self) -> usize {
        on_shape!(self, row => TermRow::width(row))
    }

    #[inline(always)]
    fn row(&self, i: usize, f: impl FnMut(usize, f64)) {
        on_shape!(self, row => TermRow::row(row, i, f))
    }
}

/// `$body` with `$row` bound to the row of whichever shape `$rows` holds.
macro_rules! on_shape {
    ($rows:expr, $row:ident => $body:expr) => {
        match $rows {
            Rows::I0($row) => $body,
            Rows::I1($row) => $body,
            Rows::I2($row) => $body,
            Rows::I3($row) => $body,
            Rows::S1($row) => $body,
            Rows::S2($row) => $body,
            Rows::S3($row) => $body,
            Rows::Dynamic($row) => $body,
        }
    };
}
use on_shape;

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

    /// The block's row as a `Row<I, S>`, the block's own shape.
    fn row<const I: bool, const S: usize>(&self) -> Row<'p, I, S> {
        debug_assert!(I as usize + S <= FUSED_WIDTH);
        Row {
            slopes: std::array::from_fn(|j| &self.slopes[j][..]),
        }
    }

    /// The block's rows as the type that fits its shape: a [`Row`] where fused, else the block.
    pub(crate) fn rows(&self) -> Rows<'p> {
        match (self.layout().intercept(), self.layout().n_slopes()) {
            (true, 0) => Rows::I0(self.row()),
            (true, 1) => Rows::I1(self.row()),
            (true, 2) => Rows::I2(self.row()),
            (true, 3) => Rows::I3(self.row()),
            (false, 1) => Rows::S1(self.row()),
            (false, 2) => Rows::S2(self.row()),
            (false, 3) => Rows::S3(self.row()),
            _ => Rows::Dynamic(*self),
        }
    }
}
