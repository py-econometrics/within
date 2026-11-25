//! The change of variables to and from the caller's problem: `b = W^{1/2} P y` in; `β = R x̂`
//! and `y − D β` out.

use std::borrow::Cow;

use super::{product, OrthogonalDesign};
use crate::WithinError;

/// One response in the internal observation order of `design`, with its `W^{1/2}`-weighted image.
pub(crate) struct Response<'r> {
    design: &'r OrthogonalDesign<'r>,
    y: Cow<'r, [f64]>,
    /// `None` when unweighted, where the weighted image is `y` itself.
    weighted: Option<Vec<f64>>,
}

impl<'r> Response<'r> {
    /// `b = W^{1/2} P y`: `y` in the internal observation order and weighted.
    pub(super) fn new(design: &'r OrthogonalDesign<'r>, y: &'r [f64]) -> Result<Self, WithinError> {
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
        let weighted = design.weights.scaled(&y);
        Ok(Response {
            design,
            y,
            weighted,
        })
    }

    /// `b = W^{1/2} P y`.
    pub(crate) fn rhs(&self) -> &[f64] {
        self.weighted.as_deref().unwrap_or(&self.y)
    }

    /// `β = R x̂` and `y − D β`, read off the measured `b − A x̂` unless a zero weight erased a row.
    pub(crate) fn fit(self, mut x: Vec<f64>, measured: Option<Vec<f64>>) -> Fit {
        let design = self.design;
        let demeaned = match measured {
            Some(mut residual) if !design.weights.erases_rows() => {
                design.weights.unscale(&mut residual);
                residual
            }
            measured => {
                let mut demeaned = measured.unwrap_or_else(|| vec![0.0; self.y.len()]);
                product::gather(design, &x, &mut demeaned, None);
                for (d, &yi) in demeaned.iter_mut().zip(self.y.iter()) {
                    *d = yi - *d;
                }
                demeaned
            }
        };
        design.back_transform(&mut x);
        Fit {
            x,
            demeaned: design.incidence.order().permute_out(demeaned),
        }
    }
}

/// Coefficients in the caller's parametrization; `y − D x` in the caller's observation order.
pub(crate) struct Fit {
    pub(crate) x: Vec<f64>,
    pub(crate) demeaned: Vec<f64>,
}

impl OrthogonalDesign<'_> {
    /// Map solve-basis coefficients back to the caller's parametrization: `x ← R·x`.
    pub(super) fn back_transform(&self, x: &mut [f64]) {
        let layout = self.layout();
        for (term, slopes) in self.slopes.iter().enumerate() {
            slopes.back_transform(layout.term(term), &mut x[layout.term_dofs(term)]);
        }
    }
}
