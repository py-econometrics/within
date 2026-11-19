//! `D̃ = D·R`: the design with each term's slopes orthogonalized, for one weighting.

mod gramian;
mod level_moments;
mod slopes;
mod term;
#[cfg(test)]
mod tests;
mod weights;

pub(crate) use gramian::Gramian;
pub(crate) use term::{RowKernel, TermBlock};
pub(crate) use weights::RowWeights;

use std::sync::Arc;

use rayon::prelude::*;

use super::{CoefficientLayout, Design, Incidence};
use crate::channel::{Channel, CoefficientPosition};
use crate::BuildError;
use slopes::OrthogonalSlopes;

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
            .map(|(term, slopes)| {
                let (rows, layout) = (&incidence.terms[term], incidence.layout.term(term));
                slopes.orthogonalize(rows, layout, &weights)
            })
            .collect();
        Ok(OrthogonalDesign {
            incidence,
            slopes,
            weights,
        })
    }
}

impl<'a> OrthogonalDesign<'a> {
    pub(crate) fn incidence(&self) -> &Incidence<'a> {
        &self.incidence
    }

    pub(crate) fn layout(&self) -> &CoefficientLayout {
        self.incidence.layout()
    }

    pub(crate) fn weights(&self) -> &RowWeights {
        &self.weights
    }

    pub(crate) fn n_obs(&self) -> usize {
        self.incidence.n_obs()
    }

    pub(crate) fn n_dofs(&self) -> usize {
        self.incidence.layout.n_dofs()
    }

    /// Map solve-basis coefficients back to the caller's parametrization: `x ← R·x`.
    pub(crate) fn back_transform(&self, x: &mut [f64]) {
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
