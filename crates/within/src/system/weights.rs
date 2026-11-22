//! `W^{1/2}`: a fit's row weights in the design's internal row order.

use rayon::prelude::*;

use crate::design::{Incidence, TermIncidence};
use crate::linalg::add_compensated;
use crate::BuildError;
use crate::PAR_THRESHOLD;

/// `W^{1/2}` in a design's internal row order; the identity when unweighted.
#[derive(Debug, Clone)]
pub(super) struct RowWeights {
    sqrt: Option<Vec<f64>>,
    /// Some row weighs zero, so `W^{1/2}` cannot be undone.
    erases_rows: bool,
}

impl RowWeights {
    /// Validate caller-order weights, then hold `√w` in `E`'s internal row order.
    pub(super) fn new(
        incidence: &Incidence<'_>,
        weights: Option<&[f64]>,
    ) -> Result<Self, BuildError> {
        let Some(weights) = weights else {
            return Ok(Self {
                sqrt: None,
                erases_rows: false,
            });
        };
        if weights.len() != incidence.n_obs() {
            return Err(BuildError::WeightCountMismatch {
                expected: incidence.n_obs(),
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
        let sqrt: Vec<f64> = incidence
            .permute_obs_in(weights)
            .iter()
            .map(|weight| (scale * weight).sqrt())
            .collect();
        Ok(Self {
            erases_rows: sqrt.contains(&0.0),
            sqrt: Some(sqrt),
        })
    }

    /// `√w` per row, for a kernel that picks its loop once; `None` is unweighted.
    pub(super) fn sqrt(&self) -> Option<&[f64]> {
        self.sqrt.as_deref()
    }

    /// The Gram weight `w` of row `obs`: the operator applies `√w`, so its normal matrix carries `w`.
    pub(super) fn weight(&self, obs: usize) -> f64 {
        self.sqrt.as_deref().map_or(1.0, |s| s[obs] * s[obs])
    }

    /// `E_tᵀ w`: each level's weight `w_l`, summed in the pieces the term's reduction fixes.
    pub(super) fn level_weights(&self, term: TermIncidence<'_>) -> Vec<f64> {
        let mut sums = vec![0.0; term.layout().n_levels()];
        match self.sqrt() {
            // A sum of ones is exact in any order.
            None => term.fold(
                0.0,
                &|s: &mut f64, _, _| *s += 1.0,
                &|l: &mut f64, r: &f64| *l += r,
                &mut sums[..],
            ),
            Some(_) => {
                let mut compensated = vec![(0.0, 0.0); sums.len()];
                term.fold(
                    (0.0, 0.0),
                    &|(s, c): &mut (f64, f64), _, row| add_compensated(s, c, self.weight(row)),
                    &|(s, c): &mut (f64, f64), &(rs, rc): &(f64, f64)| {
                        add_compensated(s, c, rs);
                        *c += rc;
                    },
                    &mut compensated[..],
                );
                for (d, (s, c)) in sums.iter_mut().zip(compensated) {
                    *d = s + c;
                }
            }
        }
        sums
    }

    /// Some row weighs zero, so [`Self::unscale`] cannot recover it.
    pub(super) fn erases_rows(&self) -> bool {
        self.erases_rows
    }

    /// `W^{1/2} v`; `None` when unweighted, where `v` is its own image.
    pub(super) fn scaled(&self, v: &[f64]) -> Option<Vec<f64>> {
        let s = self.sqrt.as_deref()?;
        Some(v.iter().zip(s).map(|(&vi, &si)| si * vi).collect())
    }

    /// Undo `W^{1/2}` on a weighted row vector.
    pub(super) fn unscale(&self, v: &mut [f64]) {
        if let Some(s) = &self.sqrt {
            debug_assert_eq!(v.len(), s.len());
            v.par_iter_mut()
                .zip(s)
                .with_min_len(PAR_THRESHOLD)
                .for_each(|(d, &si)| *d /= si);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_invalid_weights_before_permuting_them() {
        let design = crate::design::Design::from_levels_for_test(vec![vec![2, 0, 1, 0]]);

        assert!(RowWeights::new(&design.incidence, None).is_ok());
        assert!(RowWeights::new(&design.incidence, Some(&[1.0, 2.0, 3.0, 4.0])).is_ok());
        // Zero weights are valid (an excluded observation).
        assert!(RowWeights::new(&design.incidence, Some(&[0.0, 1.0, 2.0, 3.0])).is_ok());
        assert!(matches!(
            RowWeights::new(&design.incidence, Some(&[1.0, 2.0])),
            Err(BuildError::WeightCountMismatch {
                expected: 4,
                got: 2
            })
        ));
        assert!(matches!(
            RowWeights::new(&design.incidence, Some(&[1.0, -2.0, 3.0, 4.0])),
            Err(BuildError::InvalidWeight { index: 1, .. })
        ));
        assert!(matches!(
            RowWeights::new(&design.incidence, Some(&[1.0, 2.0, f64::NAN, 4.0])),
            Err(BuildError::InvalidWeight { index: 2, .. })
        ));
        assert!(matches!(
            RowWeights::new(&design.incidence, Some(&[1.0, 2.0, 3.0, f64::INFINITY])),
            Err(BuildError::InvalidWeight { index: 3, .. })
        ));
    }
}
