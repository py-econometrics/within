//! A solve's two ends: `b = W^{1/2} y` in internal order, and the caller's fit out.

use std::borrow::Cow;

use crate::domain::PreparedDesign;
use crate::SolveError;

/// One response in the internal observation order, with its `W^{1/2}`-weighted image.
pub(crate) struct Response<'s> {
    y: Cow<'s, [f64]>,
    /// `None` when unweighted, where the weighted image is `y` itself.
    weighted: Option<Vec<f64>>,
}

impl<'s> Response<'s> {
    pub(crate) fn new(prepared: &'s PreparedDesign<'_>, y: &'s [f64]) -> Result<Self, SolveError> {
        // `W^{1/2} y` zips y with sqrt-weights, silently truncating when `y.len() > n_rows`.
        if y.len() != prepared.design.n_obs() {
            return Err(SolveError::InvalidInput {
                context: "Solver::solve",
                message: format!(
                    "response vector length ({}) does not match number of observations ({})",
                    y.len(),
                    prepared.design.n_obs()
                ),
            });
        }
        if let Some((index, &value)) = y.iter().enumerate().find(|&(_, &v)| !v.is_finite()) {
            return Err(SolveError::InvalidInput {
                context: "Solver::solve",
                message: format!("response at index {index} must be finite, got {value}"),
            });
        }
        let y = prepared.design.permute_obs_in(y);
        let weighted = prepared
            .weights
            .sqrt()
            .map(|sw| y.iter().zip(sw).map(|(&yi, &swi)| swi * yi).collect());
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

impl PreparedDesign<'_> {
    /// Caller coefficients and `y − D x` from the iterate and, if measured, its `b − A x`.
    pub(crate) fn finish(
        &self,
        rhs: Response<'_>,
        mut x: Vec<f64>,
        measured: Option<Vec<f64>>,
    ) -> Result<Fit, SolveError> {
        let demeaned = self.demeaned(&x, &rhs.y, measured);
        self.back_transform(&mut x);
        Ok(Fit {
            x,
            demeaned: self.design.permute_obs_out(demeaned),
        })
    }
}
