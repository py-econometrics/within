//! `D = E • X`: the caller's design.

mod effect;

pub use effect::Effect;

use crate::BuildError;
use ndarray::{ArrayView2, Axis};
use std::borrow::Cow;
use std::sync::Arc;

use super::{CoefficientLayout, FactorEncoding, Incidence, TermLayout};

/// Fixed-effects design `D = E • X`: per term, row `i` is `X`'s row `[1?, z(i)]` in level `i`'s
/// columns. Clones share `E`.
#[derive(Clone, Debug)]
pub struct Design<'a> {
    pub(super) incidence: Arc<Incidence<'a>>,
    /// `X`, per term.
    pub(super) slopes: Vec<Slopes<'a>>,
}

/// `X_t`: one term's slope columns in internal row order, borrowed or owned per column.
#[derive(Clone, Debug)]
pub(crate) struct Slopes<'a> {
    pub(in crate::domain) columns: Vec<Cow<'a, [f64]>>,
}

impl<'a> Design<'a> {
    /// Lower effect terms into a design, laid out term-major (`offset[t] + c · L_t + level`).
    pub fn new(effects: impl IntoIterator<Item = Effect<'a>>) -> Result<Self, BuildError> {
        Self::build(effects.into_iter().collect())
    }

    /// Build an intercept-only design from an observation-major categories matrix.
    pub fn from_categories(categories: ArrayView2<'a, u32>) -> Result<Self, BuildError> {
        // Gather strided (C-order) columns once so every downstream read is contiguous.
        let effects = (0..categories.ncols())
            .map(|factor| {
                let column = categories.index_axis_move(Axis(1), factor);
                let levels = match column.to_slice() {
                    Some(values) => Cow::Borrowed(values),
                    None => Cow::Owned(column.to_vec()),
                };
                Effect {
                    levels,
                    intercept: true,
                    slopes: Vec::new(),
                }
            })
            .collect();
        Self::build(effects)
    }

    fn build(effects: Vec<Effect<'a>>) -> Result<Self, BuildError> {
        let n_obs = effects.first().map_or(0, |e| e.levels.len());
        for (column, e) in effects.iter().enumerate() {
            if e.levels.len() != n_obs {
                return Err(BuildError::ObservationCountMismatch {
                    column,
                    expected: n_obs,
                    got: e.levels.len(),
                });
            }
        }
        if n_obs == 0 {
            return Err(BuildError::EmptyObservations);
        }

        let mut layout = Vec::with_capacity(effects.len());
        let mut levels = Vec::with_capacity(effects.len());
        let mut slopes = Vec::with_capacity(effects.len());
        let mut offset = 0;
        for e in effects {
            let f = FactorEncoding::encode_labels(e.levels);
            let term = TermLayout {
                encoding: f.encoding,
                offset,
                intercept: e.intercept,
                n_slopes: e.slopes.len(),
            };
            offset = term.dofs().end;
            layout.push(term);
            levels.push(f.levels);
            slopes.push(e.slopes);
        }
        let layout = CoefficientLayout {
            terms: layout.into(),
        };
        let n_dofs = layout.n_dofs();
        // Rejected here rather than left to panic in `to_u32`.
        if u32::try_from(n_dofs).is_err() {
            return Err(BuildError::DofSpaceExceedsU32 { n_dofs });
        }

        let incidence = Incidence::new(levels, layout);
        // Level columns before any slope: interleaving them measured +2–4% here.
        let slopes = slopes
            .into_iter()
            .map(|columns| Slopes {
                columns: columns
                    .into_iter()
                    .map(|z| incidence.permute_obs_in(z))
                    .collect(),
            })
            .collect();
        Ok(Design {
            incidence: Arc::new(incidence),
            slopes,
        })
    }

    /// Convert every column to owned, dropping ties to caller buffers.
    pub fn into_owned(self) -> Design<'static> {
        Design {
            incidence: Arc::new(Arc::unwrap_or_clone(self.incidence).into_owned()),
            slopes: self
                .slopes
                .into_iter()
                .map(|term| Slopes {
                    columns: term
                        .columns
                        .into_iter()
                        .map(|slope| Cow::Owned(slope.into_owned()))
                        .collect(),
                })
                .collect(),
        }
    }

    /// Number of categorical factors in the design.
    #[inline]
    pub fn n_factors(&self) -> usize {
        self.incidence.layout().n_terms()
    }

    /// Number of observations (rows of D).
    #[inline]
    pub fn n_obs(&self) -> usize {
        self.incidence.n_obs()
    }

    /// Total degrees of freedom (columns of D).
    #[inline]
    pub fn n_dofs(&self) -> usize {
        self.incidence.layout().n_dofs()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    impl Design<'static> {
        pub(crate) fn from_levels_for_test(columns: Vec<Vec<u32>>) -> Self {
            let effects = columns
                .iter()
                .map(|c| Effect::new(c, true, []).expect("intercept effect"));
            Design::new(effects).expect("valid design").into_owned()
        }
    }

    #[test]
    fn build_compacts_large_integer_label() {
        let design = Design::from_levels_for_test(vec![vec![u32::MAX]]);

        assert_eq!(design.n_dofs(), 1);
        assert_eq!(design.incidence.term(0).levels(), &[0]);
        assert_eq!(
            design.incidence.layout().term(0).encoding.label(0),
            Some(u32::MAX)
        );
    }
}
