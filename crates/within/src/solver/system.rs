//! What a least-squares system owes the solve driver, and the full system's own answer.

use std::borrow::Cow;

use schwarz_precond::Operator;

use crate::config::LocalSolverConfig;
use crate::domain::PreparedDesign;
use crate::operator::schwarz::{build_schwarz, Preconditioner};
use crate::{BuildError, BuildWarning, SolveError};

/// `min ‖b − A x‖` over the coefficients LSMR iterates on, and the way back to the full fit.
pub(crate) trait LeastSquaresSystem: Sync {
    type Matrix: Operator;
    type Rhs<'s>: Send
    where
        Self: 's;

    fn matrix(&self) -> &Self::Matrix;
    /// Validate `y`, then form this system's right-hand side.
    fn prepare<'s>(&'s self, y: &'s [f64]) -> Result<Self::Rhs<'s>, SolveError>;
    /// What LSMR iterates on; tolerances are relative to its norm.
    fn b<'r>(&self, rhs: &'r Self::Rhs<'_>) -> &'r [f64];
    /// Caller coefficients and `y − D x` from the iterate and, if measured, its `b − A x`.
    fn finish(
        &self,
        rhs: Self::Rhs<'_>,
        x: Vec<f64>,
        measured: Option<Vec<f64>>,
    ) -> Result<Fit, SolveError>;
    /// The rung a stalled ladder escalates to; `None` without a factor-pair subdomain.
    fn build_schwarz(
        &self,
        local_solver: &LocalSolverConfig,
    ) -> Result<(Option<Preconditioner>, Vec<BuildWarning>), BuildError>;
}

/// One response in the internal observation order, with its `W^{1/2}`-weighted image.
pub(crate) struct Response<'s> {
    y: Cow<'s, [f64]>,
    /// `None` when unweighted, where the weighted image is `y` itself.
    weighted: Option<Vec<f64>>,
}

impl<'s> Response<'s> {
    pub(crate) fn new(prepared: &'s PreparedDesign<'_>, y: &'s [f64]) -> Result<Self, SolveError> {
        // `W^{1/2} y` zips y with sqrt-weights, silently truncating when `y.len() > n_rows`.
        if y.len() != prepared.design.n_obs {
            return Err(SolveError::InvalidInput {
                context: "Solver::solve",
                message: format!(
                    "response vector length ({}) does not match number of observations ({})",
                    y.len(),
                    prepared.design.n_obs
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
            .sqrt_weights()
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

impl LeastSquaresSystem for PreparedDesign<'_> {
    type Matrix = Self;
    type Rhs<'s>
        = Response<'s>
    where
        Self: 's;

    fn matrix(&self) -> &Self {
        self
    }

    fn prepare<'s>(&'s self, y: &'s [f64]) -> Result<Response<'s>, SolveError> {
        Response::new(self, y)
    }

    fn b<'r>(&self, rhs: &'r Response<'_>) -> &'r [f64] {
        rhs.weighted()
    }

    fn finish(
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

    fn build_schwarz(
        &self,
        local_solver: &LocalSolverConfig,
    ) -> Result<(Option<Preconditioner>, Vec<BuildWarning>), BuildError> {
        build_schwarz(self, local_solver)
    }
}
