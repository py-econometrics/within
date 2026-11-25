//! The least-squares system in solve coordinates for one weighting `W`: `D̃ = D·R`, `A = W^{1/2} D̃`,
//! `G = AᵀA`, and the change of variables to and from the caller's problem: `b = W^{1/2} P y` in,
//! `β = R x̂` and `y − D β` out.

mod basis;
mod gramian;
mod operator;
mod product;
mod response;
#[cfg(test)]
mod tests;
mod weights;

pub(crate) use gramian::Gramian;
pub(crate) use operator::SystemMatrix;
pub(crate) use response::{Fit, Response};
use weights::RowWeights;

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

    /// `b = W^{1/2} P y`: `y` in the internal observation order and weighted.
    pub(crate) fn response<'r>(&'r self, y: &'r [f64]) -> Result<Response<'r>, WithinError> {
        Response::new(self, y)
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
