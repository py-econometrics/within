//! A solve's two ends: `b = W^{1/2} P y` in, the caller's `β = R x̂` and `y − D β` out.

use std::borrow::Cow;

use super::OrthogonalDesign;
use crate::SolveError;

/// One response in the internal observation order, with its `W^{1/2}`-weighted image.
pub(crate) struct Response<'s> {
    y: Cow<'s, [f64]>,
    /// `None` when unweighted, where the weighted image is `y` itself.
    weighted: Option<Vec<f64>>,
}

impl<'s> Response<'s> {
    pub(crate) fn new(design: &OrthogonalDesign<'_>, y: &'s [f64]) -> Result<Self, SolveError> {
        // `W^{1/2} y` zips y with sqrt-weights, silently truncating when `y.len() > n_rows`.
        if y.len() != design.n_obs() {
            return Err(SolveError::InvalidInput {
                context: "Solver::solve",
                message: format!(
                    "response vector length ({}) does not match number of observations ({})",
                    y.len(),
                    design.n_obs()
                ),
            });
        }
        if let Some((index, &value)) = y.iter().enumerate().find(|&(_, &v)| !v.is_finite()) {
            return Err(SolveError::InvalidInput {
                context: "Solver::solve",
                message: format!("response at index {index} must be finite, got {value}"),
            });
        }
        let y = design.incidence.permute_obs_in(y);
        let weighted = design.weights.scaled(&y);
        Ok(Self { y, weighted })
    }

    pub(crate) fn weighted(&self) -> &[f64] {
        self.weighted.as_deref().unwrap_or(&self.y)
    }
}

/// Coefficients in the caller's parametrization; `y − D x` in the caller's observation order.
pub(crate) struct Fit {
    pub(crate) x: Vec<f64>,
    pub(crate) demeaned: Vec<f64>,
}

impl Fit {
    /// `β = R x̂` and `y − D β`, read off the measured `b − A x̂` unless a zero weight erased a row.
    pub(crate) fn new(
        design: &OrthogonalDesign<'_>,
        rhs: Response<'_>,
        mut x: Vec<f64>,
        measured: Option<Vec<f64>>,
    ) -> Self {
        let demeaned = match measured {
            Some(mut residual) if !design.weights.erases_rows() => {
                design.weights.unscale(&mut residual);
                residual
            }
            measured => {
                let mut demeaned = measured.unwrap_or_else(|| vec![0.0; rhs.y.len()]);
                super::product::gather(design, &x, &mut demeaned, None);
                for (d, &yi) in demeaned.iter_mut().zip(rhs.y.iter()) {
                    *d = yi - *d;
                }
                demeaned
            }
        };
        design.back_transform(&mut x);
        Fit {
            x,
            demeaned: design.incidence.permute_obs_out(demeaned),
        }
    }
}
