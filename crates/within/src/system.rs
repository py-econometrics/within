//! The least-squares system in solve coordinates for one weighting `W`: `D̃ = D·R`, `A = W^{1/2} D̃`,
//! `G = AᵀA`, and the change of variables to and from the caller's problem.

mod basis;
mod gramian;
mod operator;
mod response;
#[cfg(test)]
mod tests;
mod weights;

pub(crate) use gramian::Gramian;
pub(crate) use operator::DesignOperator;
pub(crate) use response::{Fit, Response};
use weights::RowWeights;

use std::sync::Arc;

use rayon::prelude::*;

use crate::design::TermBlock;
use crate::design::{Channel, CoefficientPosition};
use crate::design::{CoefficientLayout, Design, Incidence, Unknowns};
use crate::BuildError;
use basis::OrthogonalSlopes;

/// `D̃ = E • X̃ = D·R` for one weighting `W`: `R` orthogonalizes each term's columns within each
/// level, so `D̃_tᵀ W D̃_t` is diagonal.
pub(crate) struct OrthogonalDesign<'a> {
    /// Shared with the design it came from.
    incidence: Arc<Incidence<'a>>,
    /// `X̃ = X·R` with `R`, per term.
    slopes: Vec<OrthogonalSlopes>,
    weights: RowWeights,
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
        Ok(OrthogonalDesign {
            incidence,
            slopes,
            weights,
            unknowns,
        })
    }
}

impl<'a> OrthogonalDesign<'a> {
    #[cfg(test)]
    pub(crate) fn incidence(&self) -> &Incidence<'a> {
        &self.incidence
    }

    /// Term `term`'s block `D̃_t`, at its slots in `x`.
    fn term(&self, term: usize) -> TermBlock<'_> {
        let at = self.unknowns.block(term).start;
        TermBlock::new(self.incidence.term(term), self.slopes[term].columns(), at)
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

    /// Directions the data cannot identify, ascending in `(term, level, column)`: a slope term's
    /// weightless levels and dropped slopes.
    pub(crate) fn unidentified(&self) -> impl Iterator<Item = CoefficientPosition> + '_ {
        let terms = self.layout().terms().iter().zip(&self.slopes);
        terms.enumerate().flat_map(|(term, (layout, slopes))| {
            slopes.kept().enumerate().flat_map(move |(level, kept)| {
                let intercept = layout
                    .intercept_column()
                    .filter(|_| slopes.weightless(level));
                let dropped = (0..kept.len())
                    .filter(|&j| !kept[j])
                    .map(|j| layout.slope_column(j));
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
