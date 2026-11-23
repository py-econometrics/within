//! The reduced system as the solve driver sees it: the rungs it offers, its right-hand side and
//! the way back to the caller's fit.

use crate::config::RungSettings;
use crate::domain::PreparedDesign;
use crate::operator::design::ReducedDesign;
use crate::operator::schwarz::{build_schwarz, diagonal_map, RungBuild, RungMap};
use crate::operator::two_level::build_two_level;
use crate::{BuildError, BuildWarning, SolveError};

pub(crate) type Build<'s> = Box<dyn Fn() -> Result<RungBuild, BuildError> + Send + Sync + 's>;

/// The build of `rung` for `system`'s normal matrix over `below`, the map of the rung under it;
/// `None` for a rung it does not offer.
pub(crate) fn offer<'s>(
    system: &'s ReducedDesign<'_>,
    rung: RungSettings<'s>,
    below: Option<&'s RungMap>,
) -> Option<Build<'s>> {
    let prepared = system.prepared();
    match rung {
        RungSettings::Diagonal { .. } => Some(Box::new(move || {
            Ok(RungBuild {
                map: Some(RungMap::Diagonal(diagonal_map(system.gram_diagonal())?)),
                warnings: Vec::new(),
            })
        })),
        // Schwarz subdomains are factor pairs; one factor's normal matrix is diagonal.
        RungSettings::Schwarz(_) if prepared.design.n_factors() < 2 => None,
        RungSettings::Schwarz(schwarz) => Some(Box::new(move || {
            build_schwarz(
                prepared,
                &schwarz.local_solver,
                std::env::var_os("ZZ_TB").is_some(),
            )
        })),
        // ZZ prototype: with two terms the term blocks are Schwarz itself.
        RungSettings::TermBlocks { .. } if prepared.design.n_factors() < 3 => None,
        RungSettings::TermBlocks { schwarz, .. } => Some(Box::new(move || {
            build_schwarz(prepared, &schwarz.local_solver, true)
        })),
        RungSettings::TwoLevel(coarse) => {
            let Some(RungMap::Schwarz(one_level)) = below else {
                return None;
            };
            Some(Box::new(move || {
                Ok(match build_two_level(prepared, one_level, coarse) {
                    Ok(map) => RungBuild {
                        map: Some(RungMap::TwoLevel(map)),
                        warnings: Vec::new(),
                    },
                    Err(decline) => RungBuild {
                        map: None,
                        warnings: vec![BuildWarning::CoarseDeclined(decline)],
                    },
                })
            }))
        }
    }
}

/// A response validated against the design, in the caller's observation order.
#[derive(Clone, Copy)]
pub(crate) struct Response<'s> {
    y: &'s [f64],
}

impl<'s> Response<'s> {
    pub(crate) fn new(prepared: &'s PreparedDesign<'_>, y: &'s [f64]) -> Result<Self, SolveError> {
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
        Ok(Self { y })
    }

    /// Row `i` of `W^{1/2} y` in internal order.
    fn weighted_row<'p>(self, prepared: &'p PreparedDesign<'_>) -> impl Fn(usize) -> f64 + Sync + 'p
    where
        's: 'p,
    {
        let (perm, sqrt_weights) = (prepared.design.obs_perm.as_deref(), prepared.sqrt_weights());
        move |i| {
            let yi = self.y[perm.map_or(i, |p| p.get(i))];
            sqrt_weights.map_or(yi, |sw| sw[i] * yi)
        }
    }
}

/// Coefficients in the caller's parametrization; `y − D x` in the caller's observation order.
pub(crate) struct Fit {
    pub(crate) x: Vec<f64>,
    pub(crate) demeaned: Vec<f64>,
}

/// A validated response and the reduced system's right-hand side `(I − P_e) W^{1/2} y`.
pub(crate) struct ReducedResponse<'s> {
    response: Response<'s>,
    /// What LSMR iterates on; tolerances are relative to its norm.
    pub(crate) projected: Vec<f64>,
}

impl ReducedDesign<'_> {
    /// Validate `y`, then form the right-hand side LSMR iterates on.
    pub(crate) fn prepare<'s>(&'s self, y: &'s [f64]) -> Result<ReducedResponse<'s>, SolveError> {
        let prepared = self.prepared();
        let response = Response::new(prepared, y)?;
        let projected = self.project_rhs(response.weighted_row(prepared));
        Ok(ReducedResponse {
            response,
            projected,
        })
    }

    /// Caller coefficients and `y − D x` from the iterate and, if measured, its `b − A x`.
    pub(crate) fn finish(
        &self,
        rhs: ReducedResponse<'_>,
        x: Vec<f64>,
        measured: Option<Vec<f64>>,
    ) -> Fit {
        drop(rhs.projected);
        let prepared = self.prepared();
        let mut x = self.recover(rhs.response.weighted_row(prepared), &x);
        // `(I − P_e)(b − A_k x_k)` is `b − A x` once the eliminated block is recovered.
        let demeaned = prepared.demeaned(&x, rhs.response.y, measured);
        prepared.back_transform(&mut x);
        Fit {
            x,
            demeaned: prepared.design.permute_obs_out(demeaned),
        }
    }
}
