//! Columnar observation storage: categorical + continuous columns, row-aligned.

use std::borrow::Cow;

use crate::error::BuildError;

/// Row-aligned observation columns: per-factor level codes and per-slope loadings.
#[derive(Clone, Debug)]
pub struct ObservationFrame<'a> {
    categorical: Vec<Cow<'a, [u32]>>,
    continuous: Vec<Cow<'a, [f64]>>,
    n_obs: usize,
}

fn gather<T: Copy>(col: &[T], perm: &[u32]) -> Vec<T> {
    perm.iter().map(|&k| col[k as usize]).collect()
}

impl<'a> ObservationFrame<'a> {
    /// Build a frame, validating that all columns share one length.
    pub fn new(
        categorical: Vec<Cow<'a, [u32]>>,
        continuous: Vec<Cow<'a, [f64]>>,
    ) -> Result<Self, BuildError> {
        let n_obs = categorical
            .first()
            .map(|c| c.len())
            .or_else(|| continuous.first().map(|c| c.len()))
            .unwrap_or(0);
        let lens = categorical
            .iter()
            .map(|c| c.len())
            .chain(continuous.iter().map(|c| c.len()));
        for (column, len) in lens.enumerate() {
            if len != n_obs {
                return Err(BuildError::ObservationCountMismatch {
                    column,
                    expected: n_obs,
                    got: len,
                });
            }
        }
        Ok(ObservationFrame {
            categorical,
            continuous,
            n_obs,
        })
    }

    /// Number of observations (rows).
    #[inline]
    pub fn n_obs(&self) -> usize {
        self.n_obs
    }

    /// Number of categorical columns.
    #[inline]
    pub fn n_factors(&self) -> usize {
        self.categorical.len()
    }

    /// Level codes of factor `factor`.
    pub fn level_column(&self, factor: usize) -> &[u32] {
        &self.categorical[factor]
    }

    /// Loadings of continuous column `k`.
    pub fn loading_column(&self, k: usize) -> &[f64] {
        &self.continuous[k]
    }

    /// Replace loading column `i` with an owned column of matching row count.
    pub(crate) fn set_loading_column(&mut self, i: usize, column: Vec<f64>) {
        debug_assert_eq!(column.len(), self.n_obs);
        self.continuous[i] = Cow::Owned(column);
    }

    /// Convert every column to owned, dropping ties to caller buffers.
    pub fn into_owned(self) -> ObservationFrame<'static> {
        ObservationFrame {
            categorical: self
                .categorical
                .into_iter()
                .map(|c| Cow::Owned(c.into_owned()))
                .collect(),
            continuous: self
                .continuous
                .into_iter()
                .map(|c| Cow::Owned(c.into_owned()))
                .collect(),
            n_obs: self.n_obs,
        }
    }

    /// Owned copy with row `i` holding observation `perm[i]` (matches `Design::obs_perm`).
    pub fn permuted(&self, perm: &[u32]) -> ObservationFrame<'static> {
        ObservationFrame {
            categorical: self
                .categorical
                .iter()
                .map(|col| gather(col, perm).into())
                .collect(),
            continuous: self
                .continuous
                .iter()
                .map(|col| gather(col, perm).into())
                .collect(),
            n_obs: perm.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn columns_stay_row_aligned_under_permutation() {
        let frame = ObservationFrame::new(
            vec![vec![2u32, 0, 1, 0].into()],
            vec![vec![10.0f64, 20.0, 30.0, 40.0].into()],
        )
        .unwrap();

        let sorted = frame.permuted(&[1, 3, 2, 0]);

        assert_eq!(sorted.level_column(0), &[0, 0, 1, 2]);
        assert_eq!(sorted.loading_column(0), &[20.0, 40.0, 30.0, 10.0]);
    }

    #[test]
    fn mismatched_column_lengths_rejected() {
        let result = ObservationFrame::new(
            vec![vec![0u32, 1, 0].into()],
            vec![vec![1.0f64, 2.0].into()],
        );
        assert!(matches!(
            result,
            Err(BuildError::ObservationCountMismatch { .. })
        ));
    }
}
