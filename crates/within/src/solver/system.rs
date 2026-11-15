//! What a least-squares system owes the solve driver, and the full system's own answer.

use std::borrow::Cow;

use schwarz_precond::Operator;

use crate::config::RungSettings;
use crate::domain::{PreparedDesign, Unknowns, UnknownsLayout};
use crate::operator::design::ReducedDesign;
use crate::operator::schwarz::{build_schwarz, diagonal_map, RungBuild, RungMap};
use crate::{BuildError, SolveError};

pub(crate) type Build<'s> = Box<dyn Fn() -> Result<RungBuild, BuildError> + Send + Sync + 's>;

/// `rung` over `unknowns` of `prepared`, whose normal matrix has the diagonal `gram_diagonal`.
fn offer<'s>(
    prepared: &'s PreparedDesign<'_>,
    unknowns: Unknowns,
    gram_diagonal: impl Fn() -> Vec<f64> + Send + Sync + 's,
    rung: RungSettings<'s>,
) -> Option<Build<'s>> {
    match rung {
        RungSettings::Diagonal { .. } => Some(Box::new(move || {
            Ok(RungBuild {
                map: Some(RungMap::Diagonal(diagonal_map(gram_diagonal())?)),
                warnings: Vec::new(),
            })
        })),
        // Schwarz subdomains are factor pairs; one factor's normal matrix is diagonal.
        RungSettings::Schwarz(_) if prepared.design.n_factors() < 2 => None,
        RungSettings::Schwarz(schwarz) => Some(Box::new(move || {
            build_schwarz(prepared, &schwarz.local_solver, unknowns)
        })),
        RungSettings::TwoLevel => None,
    }
}

/// `min ‖b − A x‖` over the coefficients LSMR iterates on, and the way back to the full fit.
pub(crate) trait LeastSquaresSystem: Sync {
    type Matrix: Operator;
    type Rhs<'s>: Send
    where
        Self: 's;

    fn matrix(&self) -> &Self::Matrix;
    fn prepared(&self) -> &PreparedDesign<'_>;
    /// The coefficients LSMR iterates on.
    fn unknowns(&self) -> Unknowns;
    /// The coefficients every map built on this system acts on.
    fn layout(&self) -> UnknownsLayout {
        self.prepared().design.unknowns_layout(self.unknowns())
    }
    /// The build of `rung` for this system's normal matrix; `None` for a rung it does not offer.
    fn offer<'s>(&'s self, rung: RungSettings<'s>) -> Option<Build<'s>>;
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

impl PreparedDesign<'_> {
    /// The caller's fit from solve-basis coefficients `x` and, if measured, `b − A x`.
    pub(crate) fn fit(
        &self,
        rhs: Response<'_>,
        mut x: Vec<f64>,
        measured: Option<Vec<f64>>,
    ) -> Fit {
        let demeaned = self.demeaned(&x, &rhs.y, measured);
        self.back_transform(&mut x);
        Fit {
            x,
            demeaned: self.design.permute_obs_out(demeaned),
        }
    }
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

    fn prepared(&self) -> &PreparedDesign<'_> {
        self
    }

    fn unknowns(&self) -> Unknowns {
        Unknowns::All
    }

    fn offer<'s>(&'s self, rung: RungSettings<'s>) -> Option<Build<'s>> {
        offer(self, self.unknowns(), || self.gram_diagonal(), rung)
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
        x: Vec<f64>,
        measured: Option<Vec<f64>>,
    ) -> Result<Fit, SolveError> {
        Ok(self.fit(rhs, x, measured))
    }
}

/// One response and the reduced system's right-hand side `(I − P_e) W^{1/2} y`.
pub(crate) struct ReducedResponse<'s> {
    response: Response<'s>,
    projected: Vec<f64>,
}

impl LeastSquaresSystem for ReducedDesign<'_> {
    type Matrix = Self;
    type Rhs<'s>
        = ReducedResponse<'s>
    where
        Self: 's;

    fn matrix(&self) -> &Self {
        self
    }

    fn prepared(&self) -> &PreparedDesign<'_> {
        ReducedDesign::prepared(self)
    }

    fn unknowns(&self) -> Unknowns {
        Unknowns::WithoutDominant
    }

    fn offer<'s>(&'s self, rung: RungSettings<'s>) -> Option<Build<'s>> {
        offer(
            ReducedDesign::prepared(self),
            self.unknowns(),
            || self.gram_diagonal(),
            rung,
        )
    }

    fn prepare<'s>(&'s self, y: &'s [f64]) -> Result<ReducedResponse<'s>, SolveError> {
        let response = Response::new(ReducedDesign::prepared(self), y)?;
        let projected = self.project_rhs(response.weighted());
        Ok(ReducedResponse {
            response,
            projected,
        })
    }

    fn b<'r>(&self, rhs: &'r ReducedResponse<'_>) -> &'r [f64] {
        &rhs.projected
    }

    fn finish(
        &self,
        rhs: ReducedResponse<'_>,
        x: Vec<f64>,
        measured: Option<Vec<f64>>,
    ) -> Result<Fit, SolveError> {
        drop(rhs.projected);
        let x = self.recover(rhs.response.weighted(), &x);
        // `(I − P_e)(b − A_k x_k)` is `b − A x` once the eliminated block is recovered.
        Ok(ReducedDesign::prepared(self).fit(rhs.response, x, measured))
    }
}
