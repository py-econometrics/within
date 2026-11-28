//! The least-squares system in solve coordinates for one weighting `W`: `D̃ = D·R` with the term the
//! rows are sorted by, `e`, eliminated: `A_r = (I − P_e) W^{1/2} D̃_k` and `S = A_rᵀA_r`, and the
//! change of variables to and from the caller's problem: `b = (I − P_e) W^{1/2} P y` in; `x_e`,
//! `β = R x̂` and `y − D β` out.

mod basis;
mod blocks;
mod coarse;
mod gramian;
mod operator;
mod response;
mod term;
#[cfg(test)]
mod tests;
mod weights;

pub(crate) use coarse::CoarseSolve;
use gramian::SchurDiagonal;
pub(crate) use gramian::{Gramian, PairBlock, RawMap, RawSlopes};
use operator::Absorbed;
pub(crate) use operator::SystemMatrix;
pub(crate) use response::{Fit, Response};
use weights::RowWeights;

use std::sync::Arc;

use rayon::prelude::*;

use crate::design::{Channel, CoefficientPosition};
use crate::design::{CoefficientLayout, Design, Incidence, Unknowns};
use crate::{BuildError, WithinError};
use basis::OrthogonalSlopes;
use term::TermBlock;

/// `D̃ = E • X̃ = D·R` for one weighting `W`, split around `e`: the kept unknowns and `G_ee⁻¹`.
/// `R` orthogonalizes each term's columns within each level, so `D̃_tᵀ W D̃_t` is diagonal. Both
/// `A_r` and `S` are read from it.
struct SplitDesign<'a> {
    /// Shared with the design it came from.
    incidence: Arc<Incidence<'a>>,
    /// `X̃ = X·R` with `R`, per term.
    slopes: Vec<OrthogonalSlopes>,
    weights: RowWeights,
    /// Every term's coefficients but `e`'s: what the system solves for.
    unknowns: Unknowns,
    /// `G_ee⁻¹`, level-major: `1/w_l` on the intercept, 1 on an identified slope, 0 where `W`
    /// erases the level or the slope is dropped.
    eliminated_inverse: Vec<f64>,
}

/// The split `D̃` with what `A_r`'s columns yield: `diag(S)`, and the kept columns `e` absorbs.
pub(crate) struct ReducedDesign<'a> {
    split: SplitDesign<'a>,
    diag_s: SchurDiagonal,
    absorbed: Absorbed,
}

impl<'a> Design<'a> {
    /// `D̃ = D·R` for caller-order `weights` (`None` unweighted) with `e` eliminated: reuses `E`
    /// and orthogonalizes each term's slopes in place.
    pub(crate) fn reduce(self, weights: Option<&[f64]>) -> Result<ReducedDesign<'a>, BuildError> {
        let Design { incidence, slopes } = self;
        let weights = RowWeights::new(&incidence, weights)?;
        let slopes: Vec<OrthogonalSlopes> = slopes
            .into_par_iter()
            .enumerate()
            .map(|(term, slopes)| slopes.orthogonalize(incidence.term(term), &weights))
            .collect();
        let e = incidence.sort_term();
        let unknowns = Unknowns::new(incidence.layout(), e);
        let eliminated_inverse = eliminated_inverse(&incidence, &slopes[e], &weights);
        let split = SplitDesign {
            incidence,
            slopes,
            weights,
            unknowns,
            eliminated_inverse,
        };
        let diag_s = SchurDiagonal::of(&split);
        Ok(ReducedDesign {
            absorbed: Absorbed::of(diag_s.values()),
            diag_s,
            split,
        })
    }
}

/// `G_ee⁻¹`, level-major. `D̃_eᵀ W D̃_e` is diagonal per level: `w_l` on the intercept, 1 on each
/// identified slope; the inverse is 0 where `W` erases the level or the slope is dropped.
fn eliminated_inverse(
    incidence: &Incidence<'_>,
    slopes: &OrthogonalSlopes,
    weights: &RowWeights,
) -> Vec<f64> {
    let e = incidence.term(incidence.sort_term());
    let (layout, width) = (e.layout(), e.layout().n_columns());
    let mut inverse = vec![0.0; layout.n_levels() * width];
    if let Some(column) = layout.intercept_column() {
        for (level, w) in weights.level_weights(e).into_iter().enumerate() {
            inverse[level * width + column] = if w > 0.0 { 1.0 / w } else { 0.0 };
        }
    }
    for level in 0..layout.n_levels() {
        for j in 0..slopes.rank(level) {
            inverse[level * width + layout.slope_column(j)] = 1.0;
        }
    }
    inverse
}

impl<'a> ReducedDesign<'a> {
    #[cfg(test)]
    pub(crate) fn incidence(&self) -> &Incidence<'a> {
        &self.split.incidence
    }

    pub(crate) fn layout(&self) -> &CoefficientLayout {
        self.split.layout()
    }

    pub(crate) fn is_weighted(&self) -> bool {
        self.split.weights.sqrt().is_some()
    }

    pub(crate) fn n_obs(&self) -> usize {
        self.split.n_obs()
    }

    /// The coefficients the system solves for.
    pub(crate) fn unknowns(&self) -> &Unknowns {
        self.split.unknowns()
    }

    /// `b = (I − P_e) W^{1/2} P y`: `y` in the internal observation order, weighted, and with `e`
    /// partialled out twice, the second pass removing what the first left in rounding.
    pub(crate) fn response<'r>(&'r self, y: &'r [f64]) -> Result<Response<'r>, WithinError> {
        Response::new(&self.split, y)
    }

    /// Directions the data cannot identify, ascending in `(term, level, column)`: the intercepts of
    /// levels `W` erases and dropped slopes.
    pub(crate) fn unidentified(&self) -> impl Iterator<Item = CoefficientPosition> + '_ {
        let design = &self.split;
        let terms = design.layout().terms().iter().zip(&design.slopes);
        terms.enumerate().flat_map(move |(term, (layout, slopes))| {
            (0..layout.n_levels()).flat_map(move |level| {
                let intercept = layout
                    .intercept_column()
                    .filter(|_| design.weights.erases_level(term, level));
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

impl SplitDesign<'_> {
    /// Term `term`'s block `D̃_t`.
    fn term(&self, term: usize) -> TermBlock<'_> {
        TermBlock::new(self.incidence.term(term), self.slopes[term].columns())
    }

    fn layout(&self) -> &CoefficientLayout {
        self.incidence.layout()
    }

    fn n_obs(&self) -> usize {
        self.incidence.n_obs()
    }

    /// The coefficients the system solves for.
    fn unknowns(&self) -> &Unknowns {
        &self.unknowns
    }

    /// The term the system eliminates: the one the rows are sorted by.
    fn eliminated(&self) -> usize {
        self.incidence.sort_term()
    }
}
