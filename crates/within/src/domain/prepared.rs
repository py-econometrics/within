use std::sync::OnceLock;

use rayon::prelude::*;

use super::{row_weight, Column, Design, TermLayout, TermReparam, TermRows};
use crate::channel::CoefficientPosition;
use crate::linalg::add_compensated;
use crate::BuildError;

/// A [`Design`] plus the solve state it and one weight vector determine.
pub(crate) struct PreparedDesign<'a> {
    pub(crate) design: Design<'a>,
    /// `W^{1/2}` in the design's internal observation order; `None` is unweighted.
    sqrt_weights: Option<Vec<f64>>,
    /// Solve-basis slopes in the layout's slope numbering.
    slopes: Vec<Vec<f64>>,
    /// Per design term, its change of basis; `None` is the identity.
    reparams: Vec<Option<TermReparam>>,
    /// Per design term, its block of [`Self::gram_diagonal`], filled on first read.
    diagonals: Vec<OnceLock<Box<[f64]>>>,
}

/// A term in the solve basis: the design's layout and rows plus this preparation's whitened slopes.
pub(crate) struct PreparedTerm<'p> {
    pub(crate) layout: &'p TermLayout,
    pub(crate) rows: &'p TermRows<'p>,
    /// Solve-basis slopes in coefficient-column order; empty for a slope-free term.
    pub(crate) slopes: &'p [Vec<f64>],
}

impl<'p> PreparedTerm<'p> {
    /// Column `column`'s solve-basis loading; `None` is the intercept.
    pub(crate) fn loading(&self, column: usize) -> Option<&'p [f64]> {
        match self.layout.column(column) {
            Column::Intercept => None,
            Column::Slope(j) => Some(&self.slopes[j]),
        }
    }
}

impl<'a> PreparedDesign<'a> {
    pub(crate) fn new(design: Design<'a>, weights: Option<&[f64]>) -> Result<Self, BuildError> {
        let sqrt_weights = weights
            .map(|weights| prepare_sqrt_weights(&design, weights))
            .transpose()?;
        let built: Vec<_> = (0..design.n_factors())
            .into_par_iter()
            .map(|t| TermReparam::build(&design, t, sqrt_weights.as_deref()))
            .collect();
        let mut slopes = Vec::with_capacity(design.slopes.len());
        let reparams = built
            .into_iter()
            .map(|built| {
                built.map(|(reparam, whitened)| {
                    slopes.extend(whitened);
                    reparam
                })
            })
            .collect();
        let diagonals = (0..design.n_factors()).map(|_| OnceLock::new()).collect();
        Ok(Self {
            design,
            sqrt_weights,
            slopes,
            reparams,
            diagonals,
        })
    }

    /// `W^{1/2}` in internal observation order; `None` is unweighted.
    #[inline]
    pub(crate) fn sqrt_weights(&self) -> Option<&[f64]> {
        self.sqrt_weights.as_deref()
    }

    pub(crate) fn term(&self, term: usize) -> PreparedTerm<'_> {
        let layout = self.design.layout.term(term);
        PreparedTerm {
            layout,
            rows: &self.design.terms[term],
            slopes: &self.slopes[layout.slopes()],
        }
    }

    pub(crate) fn terms(&self) -> impl ExactSizeIterator<Item = PreparedTerm<'_>> {
        (0..self.design.n_factors()).map(|term| self.term(term))
    }

    /// Term `term`'s block of `diag(AᵀA)`, laid out like its `dofs()`, summed in observation order.
    pub(crate) fn diagonal(&self, term: usize) -> &[f64] {
        // Serial fill: a rayon job stolen inside it could re-enter this cell and deadlock.
        self.diagonals[term].get_or_init(|| {
            let prepared = self.term(term);
            let (layout, sqrt_weights) = (prepared.layout, self.sqrt_weights());
            let levels = prepared.rows.levels();
            let mut diag = vec![0.0; layout.n_dofs()];
            let mut corrections = vec![0.0; layout.n_dofs()];
            if !prepared.slopes.is_empty() {
                let columns: Vec<(usize, Option<&[f64]>)> = (0..layout.n_columns())
                    .map(|c| (layout.block_column(c).start, prepared.loading(c)))
                    .collect();
                for (obs, &level) in levels.iter().enumerate() {
                    let w = row_weight(sqrt_weights, obs);
                    for &(base, z) in &columns {
                        // Keep `w * z * z` left-to-right: a zero weight kills a huge `z` first.
                        let slot = base + level as usize;
                        add_compensated(
                            &mut diag[slot],
                            &mut corrections[slot],
                            z.map_or(w, |z| w * z[obs] * z[obs]),
                        );
                    }
                }
            } else if let Some(offsets) = prepared.rows.split.sorted_offsets() {
                // A sorted level is one run: a register sum keeps its order with no store per row.
                for (level, run) in offsets.windows(2).enumerate() {
                    // A sum of ones is exact, so the unweighted run sum is its length.
                    diag[level] += match sqrt_weights {
                        None => (run[1] - run[0]) as f64,
                        Some(s) => {
                            let (mut sum, mut correction) = (0.0, 0.0);
                            for &si in &s[run[0]..run[1]] {
                                add_compensated(&mut sum, &mut correction, si * si);
                            }
                            sum + correction
                        }
                    };
                }
            } else {
                for (obs, &level) in levels.iter().enumerate() {
                    let level = level as usize;
                    add_compensated(
                        &mut diag[level],
                        &mut corrections[level],
                        row_weight(sqrt_weights, obs),
                    );
                }
            }
            for (entry, correction) in diag.iter_mut().zip(corrections) {
                *entry += correction;
            }
            diag.into_boxed_slice()
        })
    }

    /// `diag(AᵀA)` in solve coordinates.
    pub(crate) fn gram_diagonal(&self) -> Vec<f64> {
        let blocks: Vec<&[f64]> = (0..self.design.n_factors())
            .into_par_iter()
            .map(|term| self.diagonal(term))
            .collect();
        blocks.concat()
    }

    /// Map solve-basis coefficients back to the user's parametrization.
    pub(crate) fn back_transform(&self, x: &mut [f64]) {
        for (layout, reparam) in self.design.layout.terms().iter().zip(&self.reparams) {
            if let Some(reparam) = reparam {
                reparam.back_transform(layout, &mut x[layout.dofs()]);
            }
        }
    }

    /// Directions the data cannot identify, ascending in `(term, level, column)`.
    pub(crate) fn unidentified(&self) -> impl Iterator<Item = CoefficientPosition> + '_ {
        self.reparams
            .iter()
            .flatten()
            .flat_map(|r| r.unidentified.iter().copied())
    }
}

/// Validate caller-order weights, then return `√w` in the design's internal order.
fn prepare_sqrt_weights(design: &Design<'_>, weights: &[f64]) -> Result<Vec<f64>, BuildError> {
    if weights.len() != design.n_obs() {
        return Err(BuildError::WeightCountMismatch {
            expected: design.n_obs(),
            got: weights.len(),
        });
    }
    // `wi >= 0.0` already rejects NaN; `is_finite` additionally rejects `+∞`.
    if let Some((index, &value)) = weights
        .iter()
        .enumerate()
        .find(|&(_, &wi)| !(wi >= 0.0 && wi.is_finite()))
    {
        return Err(BuildError::InvalidWeight { index, value });
    }
    // The fit ignores a common weight scale; a power of four keeps every level's Σw finite.
    let max = weights.iter().copied().fold(0.0, f64::max);
    let scale = if max > 2f64.powi(960) {
        2f64.powi(-128)
    } else {
        1.0
    };
    Ok(design
        .permute_obs_in(weights)
        .iter()
        .map(|weight| (scale * weight).sqrt())
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    impl<'a> PreparedDesign<'a> {
        /// Unweighted, whitened like a solver would.
        pub(crate) fn unweighted_for_test(design: Design<'a>) -> Self {
            Self::new(design, None).expect("unweighted preparation cannot fail")
        }
    }

    impl PreparedDesign<'static> {
        pub(crate) fn from_levels_for_test(columns: Vec<Vec<u32>>) -> Self {
            Self::unweighted_for_test(Design::from_levels_for_test(columns))
        }
    }

    #[test]
    fn owns_sqrt_weights_in_internal_observation_order() {
        // Dominant factor [2,0,1,0] argsorts to caller positions [1,3,2,0].
        let design = Design::from_levels_for_test(vec![vec![2, 0, 1, 0]]);
        let prepared = PreparedDesign::new(design, Some(&[1.0, 4.0, 9.0, 16.0])).unwrap();

        assert_eq!(prepared.sqrt_weights(), Some(&[2.0, 4.0, 3.0, 1.0][..]));
        assert_eq!(
            (0..4)
                .map(|obs| row_weight(prepared.sqrt_weights(), obs))
                .collect::<Vec<_>>(),
            [4.0, 16.0, 9.0, 1.0]
        );
    }

    /// Independent floating-point expansion: retain each exact two-sum remainder
    /// before rounding the short expansion, rather than reusing Neumaier's kernel.
    fn expansion_sum(values: Vec<f64>) -> f64 {
        let mut partials: Vec<f64> = Vec::new();
        for mut x in values {
            let mut kept = 0;
            for i in 0..partials.len() {
                let mut y = partials[i];
                if x.abs() < y.abs() {
                    std::mem::swap(&mut x, &mut y);
                }
                let hi = x + y;
                let lo = y - (hi - x);
                if lo != 0.0 {
                    partials[kept] = lo;
                    kept += 1;
                }
                x = hi;
            }
            partials.truncate(kept);
            partials.push(x);
        }
        let mut hi = partials.pop().unwrap_or(0.0);
        let mut lo = 0.0;
        while let Some(y) = partials.pop() {
            let x = hi;
            hi = x + y;
            lo = y - (hi - x);
            if lo != 0.0 {
                break;
            }
        }
        // A same-sign tail resolves a halfway rounding at the leading pair.
        if partials.last().is_some_and(|tail| lo * tail > 0.0) {
            let twice = 2.0 * lo;
            let rounded = hi + twice;
            if rounded - hi == twice {
                hi = rounded;
            }
        }
        hi
    }

    /// Explicit per-column, per-level contributions, independently summed.
    fn observation_sums(prepared: &PreparedDesign<'_>) -> Vec<f64> {
        let mut entries = vec![Vec::new(); prepared.design.n_dofs()];
        for term in prepared.terms() {
            for column in 0..term.layout.n_columns() {
                let block = &mut entries[term.layout.column_dofs(column)];
                let z = term.loading(column);
                for (obs, &level) in term.rows.levels().iter().enumerate() {
                    let w = row_weight(prepared.sqrt_weights(), obs);
                    block[level as usize].push(z.map_or(w, |z| w * z[obs] * z[obs]));
                }
            }
        }
        entries.into_iter().map(expansion_sum).collect()
    }

    #[test]
    fn gram_diagonal_matches_independent_expansion_sums() {
        // Level 1 of `f` has z1 = 2·z0 (a rank drop).
        let f = [0u32, 1, 0, 1, 0, 1, 2, 2];
        let z0 = [1.0, 2.0, 3.0, 4.0, 5.5, 6.0, 0.3, 0.7];
        let z1 = [9.0, 4.0, 1.0, 8.0, 2.0, 12.0, 5.0, 1.1];
        let g = [2u32, 0, 1, 1, 0, 2, 0, 1];
        let h = [1u32, 0, 0, 1, 1, 0, 1, 0];
        let zh = [0.5, 1.5, -2.0, 3.0, 0.25, -1.0, 7.0, 2.0];
        // Nested in `f`, so it comes out sorted and is summed by runs.
        let p = [0u32, 1, 0, 1, 0, 1, 1, 1];
        let weights = [2.0, 0.0, 3.0, 0.1, 1.0, 7.0, 0.3, 5.0];
        for weights in [None, Some(&weights[..])] {
            let effects = vec![
                crate::Effect::new(&f, true, [&z0[..], &z1[..]]).unwrap(),
                crate::Effect::new(&g, true, []).unwrap(),
                crate::Effect::new(&h, false, [&zh[..]]).unwrap(),
                crate::Effect::new(&p, true, []).unwrap(),
            ];
            let prepared = PreparedDesign::new(Design::new(effects).unwrap(), weights).unwrap();
            let terms = &prepared.design.terms;
            assert!(
                terms[3].split.sorted_offsets().is_some()
                    && terms[1].split.sorted_offsets().is_none()
            );
            assert!(prepared.unidentified().next().is_some());
            assert_eq!(
                prepared.gram_diagonal(),
                observation_sums(&prepared),
                "weights {weights:?}"
            );
        }
    }

    #[test]
    fn gram_diagonal_matches_hand_computed_values() {
        // `z` is constant in level 1 of `f`, so that level's slope is dropped.
        let f = [0u32, 0, 0, 1, 1, 2, 2];
        let z = [1.0, 2.0, 4.0, 3.0, 3.0, 5.0, 7.0];
        let g = [0u32, 1, 0, 1, 0, 1, 1];
        // Perfect squares, so `(√w)²` is exact.
        let weights = [1.0, 4.0, 9.0, 0.25, 4.0, 1.0, 16.0];
        for (weights, f_sums, g_sums) in [
            (None, [3.0, 2.0, 2.0], [3.0, 4.0]),
            (Some(&weights[..]), [14.0, 4.25, 17.0], [14.0, 21.25]),
        ] {
            let effects = vec![
                crate::Effect::new(&f, true, [&z[..]]).unwrap(),
                crate::Effect::new(&g, true, []).unwrap(),
            ];
            let prepared = PreparedDesign::new(Design::new(effects).unwrap(), weights).unwrap();
            let diag = prepared.gram_diagonal();
            let block =
                |term: usize, column| &diag[prepared.design.layout.term(term).column_dofs(column)];
            assert_eq!(block(0, 0), f_sums);
            assert_eq!(block(1, 0), g_sums);
            // A kept whitened slope has unit weighted norm; a dropped one is an exact-zero column.
            let slope = block(0, 1);
            assert_eq!(slope[1], 0.0, "weights {weights:?}");
            for level in [0, 2] {
                assert!(
                    (slope[level] - 1.0).abs() < 1e-12,
                    "{slope:?}, weights {weights:?}"
                );
            }
        }
    }

    #[test]
    fn rejects_invalid_weights_before_permuting_them() {
        let design = Design::from_levels_for_test(vec![vec![2, 0, 1, 0]]);

        assert!(PreparedDesign::new(design.clone(), None).is_ok());
        assert!(PreparedDesign::new(design.clone(), Some(&[1.0, 2.0, 3.0, 4.0])).is_ok());
        // Zero weights are valid (an excluded observation).
        assert!(PreparedDesign::new(design.clone(), Some(&[0.0, 1.0, 2.0, 3.0])).is_ok());
        assert!(matches!(
            PreparedDesign::new(design.clone(), Some(&[1.0, 2.0])),
            Err(BuildError::WeightCountMismatch {
                expected: 4,
                got: 2
            })
        ));
        assert!(matches!(
            PreparedDesign::new(design.clone(), Some(&[1.0, -2.0, 3.0, 4.0])),
            Err(BuildError::InvalidWeight { index: 1, .. })
        ));
        assert!(matches!(
            PreparedDesign::new(design.clone(), Some(&[1.0, 2.0, f64::NAN, 4.0])),
            Err(BuildError::InvalidWeight { index: 2, .. })
        ));
        assert!(matches!(
            PreparedDesign::new(design, Some(&[1.0, 2.0, 3.0, f64::INFINITY])),
            Err(BuildError::InvalidWeight { index: 3, .. })
        ));
    }
}
