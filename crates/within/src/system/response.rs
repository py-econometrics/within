//! The change of variables to and from the caller's problem: `b = (I − P_e) W^{1/2} P y` in;
//! `x_e`, `β = R [x; x_e]` and `y − D β` out. Each pass runs once per response over `e`'s
//! segments: `b` and `x_e` read `A`'s rows [split](super::blocks), `y − D β` reads `D̃`'s.

use std::borrow::Cow;

use schwarz_precond::{MemBuffer, MemStack};

use super::blocks::{DynamicKept, KeptTerms};
use super::term::TermRow;
use super::SplitDesign;
use crate::design::Channel;
use crate::WithinError;

/// One response in the internal observation order of `design`, with its reduced image `b`.
pub(crate) struct Response<'r> {
    design: &'r SplitDesign<'r>,
    y: Cow<'r, [f64]>,
    rhs: Vec<f64>,
}

impl<'r> Response<'r> {
    /// `b = (I − P_e) W^{1/2} P y`: `y` in the internal observation order, weighted, and with `e`
    /// partialled out twice, the second pass removing what the first left in rounding.
    pub(super) fn new(design: &'r SplitDesign<'r>, y: &'r [f64]) -> Result<Self, WithinError> {
        // `W^{1/2} y` zips y with sqrt-weights, silently truncating when `y.len() > n_rows`.
        if y.len() != design.n_obs() {
            return Err(WithinError::ResponseLengthMismatch {
                expected: design.n_obs(),
                got: y.len(),
            });
        }
        if let Some((index, &value)) = y.iter().enumerate().find(|&(_, &v)| !v.is_finite()) {
            return Err(WithinError::InvalidResponse { index, value });
        }
        let y = design.incidence.order().permute_in(y);
        let mut rhs = design.weights.scaled(&y).unwrap_or_else(|| y.to_vec());
        design.dynamic().partial_out(&mut rhs);
        Ok(Response { design, y, rhs })
    }

    /// `b = (I − P_e) W^{1/2} P y`.
    pub(crate) fn rhs(&self) -> &[f64] {
        &self.rhs
    }

    /// `x_e`, `β = R [x; x_e]` and `y − D β`, read off the measured `b − A_r x` unless a zero
    /// weight erased a row.
    pub(crate) fn fit(self, x: Vec<f64>, measured: Option<Vec<f64>>) -> Fit {
        let design = self.design;
        let x_e = recover(design, &self.y, &x);
        let demeaned = match measured {
            Some(mut residual) if !design.weights.erases_rows() => {
                design.weights.unscale(&mut residual);
                residual
            }
            _ => demean(design, &self.y, &x, &x_e),
        };
        let mut beta = design.place(&x, &x_e);
        design.back_transform(&mut beta);
        Fit {
            x: beta,
            demeaned: design.incidence.order().permute_out(demeaned),
        }
    }
}

/// Coefficients in the caller's parametrization; `y − D x` in the caller's observation order.
pub(crate) struct Fit {
    pub(crate) x: Vec<f64>,
    pub(crate) demeaned: Vec<f64>,
}

impl SplitDesign<'_> {
    /// `[x; x_e]` in the layout's coefficient order: each kept channel's slots, then `e`'s from
    /// the level-major `x_e`.
    fn place(&self, x: &[f64], x_e: &[f64]) -> Vec<f64> {
        let layout = self.layout();
        let mut beta = vec![0.0; layout.n_dofs()];
        for (channel, at) in self.unknowns().channels(layout) {
            beta[layout.dofs(channel)].copy_from_slice(&x[at]);
        }
        let e = self.eliminated();
        let width = layout.term(e).n_columns();
        for column in 0..width {
            let slots = &mut beta[layout.dofs(Channel { term: e, column })];
            for (level, b) in slots.iter_mut().enumerate() {
                *b = x_e[level * width + column];
            }
        }
        beta
    }

    /// Map solve-basis coefficients back to the caller's parametrization: `x ← R·x`.
    pub(super) fn back_transform(&self, x: &mut [f64]) {
        let layout = self.layout();
        for (term, slopes) in self.slopes.iter().enumerate() {
            slopes.back_transform(layout.term(term), &mut x[layout.term_dofs(term)]);
        }
    }
}

/// `x_e = G_ee⁻¹ A_eᵀ (W^{1/2} y − A_k x)` per level, level-major, fitted twice: the second fit
/// recovers what the first left in rounding.
fn recover(design: &SplitDesign<'_>, y: &[f64], x: &[f64]) -> Vec<f64> {
    let rows = design.dynamic();
    rows.partial_out(&mut rows.residual(y, x))
}

/// `y − D̃_k x − D̃_e x_e` over `D̃`'s own rows: every row, erased by a zero weight or not.
fn demean(design: &SplitDesign<'_>, y: &[f64], x: &[f64], x_e: &[f64]) -> Vec<f64> {
    let (kept, e) = (DynamicKept::new(design), design.term(design.eliminated()));
    let (width, segments) = (e.width(), design.incidence.segments());
    let mut buffer = MemBuffer::new(segments.scratch(0, 0));
    let mut demeaned = vec![0.0; y.len()];
    let level = |level: usize, start: usize, segment: &mut [f64], _: &mut [f64]| {
        let fit = &x_e[level * width..(level + 1) * width];
        for (k, out) in segment.iter_mut().enumerate() {
            let i = start + k;
            // Fold from -0.0, the additive identity, so it folds away.
            let (mut kept_i, mut e_i) = (-0.0, 0.0);
            kept.slots(i, |j, d| kept_i += x[j] * d);
            e.row(i, |c, d| e_i += d * fit[c]);
            *out = y[i] - kept_i - e_i;
        }
    };
    segments.map(&mut demeaned, 0, MemStack::new(&mut buffer), &level);
    demeaned
}
