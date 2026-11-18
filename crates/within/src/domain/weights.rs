//! `W^{1/2}`: a fit's row weights in the design's internal row order.

use std::ops::Range;

use rayon::prelude::*;

use super::{Design, PAR_THRESHOLD};
use crate::linalg::add_compensated;
use crate::BuildError;

/// `W^{1/2}` in a design's internal row order; the identity when unweighted.
#[derive(Debug, Clone)]
pub(crate) struct RowWeights {
    sqrt: Option<Vec<f64>>,
    /// Some row weighs zero, so `W^{1/2}` cannot be undone.
    erases_rows: bool,
}

impl RowWeights {
    /// Validate caller-order weights, then hold `√w` in `design`'s internal order.
    pub(crate) fn new(design: &Design<'_>, weights: Option<&[f64]>) -> Result<Self, BuildError> {
        let Some(weights) = weights else {
            return Ok(Self {
                sqrt: None,
                erases_rows: false,
            });
        };
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
        let sqrt: Vec<f64> = design
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
    pub(crate) fn sqrt(&self) -> Option<&[f64]> {
        self.sqrt.as_deref()
    }

    /// The Gram weight `w` of row `obs`: the operator applies `√w`, so its normal matrix carries `w`.
    pub(crate) fn weight(&self, obs: usize) -> f64 {
        self.sqrt.as_deref().map_or(1.0, |s| s[obs] * s[obs])
    }

    /// `Σ w` over `rows`, summed in row order; a sum of ones is exact, so unweighted is the length.
    pub(crate) fn sum(&self, rows: Range<usize>) -> f64 {
        match &self.sqrt {
            None => rows.len() as f64,
            Some(s) => {
                let (mut sum, mut correction) = (0.0, 0.0);
                for &si in &s[rows] {
                    add_compensated(&mut sum, &mut correction, si * si);
                }
                sum + correction
            }
        }
    }

    /// Some row weighs zero, so [`Self::unscale`] cannot recover it.
    pub(crate) fn erases_rows(&self) -> bool {
        self.erases_rows
    }

    /// Undo `W^{1/2}` on a weighted row vector.
    pub(crate) fn unscale(&self, v: &mut [f64]) {
        if let Some(s) = &self.sqrt {
            debug_assert_eq!(v.len(), s.len());
            v.par_iter_mut()
                .zip(s)
                .with_min_len(PAR_THRESHOLD)
                .for_each(|(d, &si)| *d /= si);
        }
    }
}
