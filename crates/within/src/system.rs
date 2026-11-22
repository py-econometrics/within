//! The least-squares system in solve coordinates for one weighting `W`: `D̃ = D·R`, `A = W^{1/2} D̃`,
//! `G = AᵀA`, and the change of variables to and from the caller's problem: `b = W^{1/2} P y` in,
//! `β = R x̂` and `y − D β` out.

mod basis;
mod gramian;
mod operator;
mod product;
#[cfg(test)]
mod tests;
mod weights;

pub(crate) use gramian::Gramian;
pub(crate) use operator::DesignOperator;
use weights::RowWeights;

use std::borrow::Cow;
use std::sync::{Arc, OnceLock};

use rayon::prelude::*;

use crate::design::TermBlock;
use crate::design::{Channel, CoefficientPosition};
use crate::design::{CoefficientLayout, Design, Incidence, Unknowns};
use crate::{BuildError, WithinError};
use basis::OrthogonalSlopes;

/// `D̃ = E • X̃ = D·R` for one weighting `W`: `R` orthogonalizes each term's columns within each
/// level, so `D̃_tᵀ W D̃_t` is diagonal.
pub(crate) struct OrthogonalDesign<'a> {
    /// Shared with the design it came from.
    incidence: Arc<Incidence<'a>>,
    /// `X̃ = X·R` with `R`, per term.
    slopes: Vec<OrthogonalSlopes>,
    weights: RowWeights,
    /// Per term, each level's weight `w_l`, summed once when first read.
    level_weights: Vec<OnceLock<Vec<f64>>>,
    unknowns: Unknowns,
}

impl<'a> Design<'a> {
    /// `D̃ = D·R` for caller-order `weights` (`None` unweighted): reuses `E` and orthogonalizes
    /// each term's slopes in place.
    pub(crate) fn orthogonalize(
        self,
        weights: Option<&[f64]>,
    ) -> Result<OrthogonalDesign<'a>, BuildError> {
        let Design { incidence, slopes } = self;
        let weights = RowWeights::new(&incidence, weights)?;
        let slopes = slopes
            .into_par_iter()
            .enumerate()
            .map(|(term, slopes)| slopes.orthogonalize(incidence.term(term), &weights))
            .collect();
        let unknowns = Unknowns::new(incidence.layout());
        let level_weights = (0..incidence.layout().n_terms())
            .map(|_| OnceLock::new())
            .collect();
        Ok(OrthogonalDesign {
            incidence,
            slopes,
            weights,
            level_weights,
            unknowns,
        })
    }
}

impl<'a> OrthogonalDesign<'a> {
    #[cfg(test)]
    pub(crate) fn incidence(&self) -> &Incidence<'a> {
        &self.incidence
    }

    /// Term `term`'s block `D̃_t`.
    fn term(&self, term: usize) -> TermBlock<'_> {
        TermBlock::new(self.incidence.term(term), self.slopes[term].columns())
    }

    pub(crate) fn terms(&self) -> impl ExactSizeIterator<Item = TermBlock<'_>> {
        (0..self.layout().n_terms()).map(|term| self.term(term))
    }

    pub(crate) fn layout(&self) -> &CoefficientLayout {
        self.incidence.layout()
    }

    pub(crate) fn is_weighted(&self) -> bool {
        self.weights.sqrt().is_some()
    }

    pub(crate) fn n_obs(&self) -> usize {
        self.incidence.n_obs()
    }

    /// Each level's weight `w_l` in term `term`: the intercept's entry of `diag(G)`; `0` where `W`
    /// erases the level.
    fn level_weights(&self, term: usize) -> &[f64] {
        self.level_weights[term]
            .get_or_init(|| self.weights.level_weights(self.incidence.term(term)))
    }

    /// The coefficients the system solves for.
    pub(crate) fn unknowns(&self) -> &Unknowns {
        &self.unknowns
    }

    /// Map solve-basis coefficients back to the caller's parametrization: `x ← R·x`.
    fn back_transform(&self, x: &mut [f64]) {
        let layout = self.layout();
        for (term, slopes) in self.slopes.iter().enumerate() {
            slopes.back_transform(layout.term(term), &mut x[layout.term_dofs(term)]);
        }
    }

    /// `b = W^{1/2} P y`: `y` in the internal observation order and weighted.
    pub(crate) fn response<'r>(&'r self, y: &'r [f64]) -> Result<Response<'r>, WithinError> {
        // `W^{1/2} y` zips y with sqrt-weights, silently truncating when `y.len() > n_rows`.
        if y.len() != self.n_obs() {
            return Err(WithinError::ResponseLengthMismatch {
                expected: self.n_obs(),
                got: y.len(),
            });
        }
        if let Some((index, &value)) = y.iter().enumerate().find(|&(_, &v)| !v.is_finite()) {
            return Err(WithinError::InvalidResponse { index, value });
        }
        let y = self.incidence.permute_obs_in(y);
        let weighted = self.weights.scaled(&y);
        Ok(Response {
            design: self,
            y,
            weighted,
        })
    }

    /// Directions the data cannot identify, ascending in `(term, level, column)`: the intercepts of
    /// levels `W` erases and dropped slopes.
    pub(crate) fn unidentified(&self) -> impl Iterator<Item = CoefficientPosition> + '_ {
        let terms = self.layout().terms().iter().zip(&self.slopes);
        terms.enumerate().flat_map(move |(term, (layout, slopes))| {
            // Only a zero-weight row can erase a level.
            let weights = (self.weights.erases_rows() && layout.intercept())
                .then(|| self.level_weights(term));
            (0..layout.n_levels()).flat_map(move |level| {
                let intercept = layout
                    .intercept_column()
                    .filter(|_| weights.is_some_and(|w| w[level] == 0.0));
                let dropped = slopes.dropped(level).map(|j| layout.slope_column(j));
                intercept
                    .into_iter()
                    .chain(dropped)
                    .map(move |column| CoefficientPosition {
                        channel: Channel { term, column },
                        level,
                    })
            })
        })
    }
}

/// One response in the internal observation order of `design`, with its `W^{1/2}`-weighted image.
pub(crate) struct Response<'r> {
    design: &'r OrthogonalDesign<'r>,
    y: Cow<'r, [f64]>,
    /// `None` when unweighted, where the weighted image is `y` itself.
    weighted: Option<Vec<f64>>,
}

impl Response<'_> {
    pub(crate) fn weighted(&self) -> &[f64] {
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
            demeaned: design.incidence.permute_obs_out(demeaned),
        }
    }
}

/// Coefficients in the caller's parametrization; `y − D x` in the caller's observation order.
pub(crate) struct Fit {
    pub(crate) x: Vec<f64>,
    pub(crate) demeaned: Vec<f64>,
}
