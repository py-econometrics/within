//! `D = E • X`: the caller's design.

mod effect;

pub use effect::Effect;

use crate::BuildError;
use ndarray::{ArrayView2, Axis};
use rayon::prelude::*;
use std::borrow::Cow;
use std::sync::Arc;

use super::{
    CoefficientLayout, FactorEncoding, Incidence, RowIndex, ScatterSplit, TermLayout, TermRows,
};

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
            slopes.push(Slopes { columns: e.slopes });
        }
        let layout = CoefficientLayout {
            terms: layout.into(),
        };
        let n_dofs = layout.n_dofs();
        // Rejected here rather than left to panic in `to_u32`.
        if u32::try_from(n_dofs).is_err() {
            return Err(BuildError::DofSpaceExceedsU32 { n_dofs });
        }

        // Sort by the dominant term so its gather/scatter runs sequentially.
        let dominant = layout.dominant();
        let n_levels = |t: usize| layout.term(t).n_levels();
        let (dominant_split, obs_perm) =
            match ScatterSplit::of_sorted(&levels[dominant], n_levels(dominant)) {
                Some(split) => (split, None),
                None => {
                    let (offsets, perm) = RowIndex::argsort(&levels[dominant], n_levels(dominant));
                    for levels in levels.iter_mut() {
                        *levels = Cow::Owned(perm.gather(levels));
                    }
                    // Level columns before any slope: interleaving them measured +2–4% here.
                    for slope in slopes.iter_mut().flat_map(|s| &mut s.columns) {
                        *slope = Cow::Owned(perm.gather(slope));
                    }
                    (ScatterSplit::sorted(offsets), Some(perm))
                }
            };
        let dominant_rows = TermRows {
            levels: levels.remove(dominant),
            split: dominant_split,
        };
        let others: Vec<usize> = (0..layout.n_terms()).filter(|&t| t != dominant).collect();
        // Factors nested in the dominant one come out sorted and need no row index.
        let mut terms: Vec<TermRows> = levels
            .into_par_iter()
            .zip(others)
            .map(|(levels, t)| TermRows {
                split: ScatterSplit::new(&levels, n_levels(t)),
                levels,
            })
            .collect();
        terms.insert(dominant, dominant_rows);

        Ok(Design {
            incidence: Arc::new(Incidence {
                layout,
                terms,
                obs_perm,
            }),
            slopes,
        })
    }

    /// Convert every column to owned, dropping ties to caller buffers.
    pub fn into_owned(self) -> Design<'static> {
        let Incidence {
            layout,
            terms,
            obs_perm,
        } = Arc::unwrap_or_clone(self.incidence);
        Design {
            incidence: Arc::new(Incidence {
                layout,
                terms: terms.into_iter().map(TermRows::into_owned).collect(),
                obs_perm,
            }),
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
        self.incidence.layout.n_terms()
    }

    /// Number of observations (rows of D).
    #[inline]
    pub fn n_obs(&self) -> usize {
        self.incidence.n_obs()
    }

    /// Total degrees of freedom (columns of D).
    #[inline]
    pub fn n_dofs(&self) -> usize {
        self.incidence.layout.n_dofs()
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
        assert_eq!(design.incidence.terms[0].levels(), &[0]);
        assert_eq!(
            design.incidence.layout.term(0).encoding.label(0),
            Some(u32::MAX)
        );
    }
}
