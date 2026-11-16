use crate::domain::{PreparedDesign, PAR_THRESHOLD};

mod gather;
mod reduced;

#[cfg(test)]
mod tests;

use gather::{gather_apply, unscale};
pub(crate) use reduced::ReducedDesign;

impl PreparedDesign<'_> {
    /// `y − D x`, read off this operator's measured `b − A x` unless a zero weight erased a row;
    /// `y` in the caller's order.
    pub(crate) fn demeaned(&self, x: &[f64], y: &[f64], measured: Option<Vec<f64>>) -> Vec<f64> {
        match (measured, self.sqrt_weights()) {
            (Some(residual), None) => residual,
            (Some(mut residual), Some(sw)) if !sw.contains(&0.0) => {
                unscale(&mut residual, sw);
                residual
            }
            (measured, _) => {
                let mut demeaned = measured.unwrap_or_else(|| vec![0.0; y.len()]);
                gather_apply(self, x, &mut demeaned, None);
                for (d, &yi) in demeaned
                    .iter_mut()
                    .zip(self.design.permute_obs_in(y).iter())
                {
                    *d = yi - *d;
                }
                demeaned
            }
        }
    }
}
