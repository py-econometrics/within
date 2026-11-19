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
use crate::design::{CoefficientLayout, Design, Incidence};
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
        Ok(OrthogonalDesign {
            incidence,
            slopes,
            weights,
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

    pub(crate) fn n_dofs(&self) -> usize {
        self.layout().n_dofs()
    }

    /// Map solve-basis coefficients back to the caller's parametrization: `x ← R·x`.
    fn back_transform(&self, x: &mut [f64]) {
        for (layout, slopes) in self.layout().terms().iter().zip(&self.slopes) {
            slopes.back_transform(layout, &mut x[layout.dofs()]);
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

    /// Whether this design reads `design`'s `E`, not a copy.
    #[cfg(test)]
    pub(crate) fn shares_rows_with(&self, design: &Design<'_>) -> bool {
        std::sync::Arc::ptr_eq(&self.incidence, &design.incidence)
    }
}
