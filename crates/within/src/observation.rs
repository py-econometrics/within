//! Columnar observation storage: row-aligned categorical columns.

use std::borrow::Cow;

use crate::error::BuildError;

/// Row-aligned columns, each borrowed from the caller or owned.
pub(crate) type Columns<'a> = Vec<Cow<'a, [u32]>>;

/// Row-aligned per-factor level codes; slopes enter a design through `Effect`.
#[derive(Clone, Debug)]
pub struct ObservationFrame<'a> {
    categorical: Columns<'a>,
    n_obs: usize,
}

// Inlined into `Design::build`, its loop measured +3–8% on slope designs.
#[inline(never)]
pub(crate) fn gather<T: Copy>(col: &[T], perm: &[u32]) -> Vec<T> {
    perm.iter().map(|&k| col[k as usize]).collect()
}

impl<'a> ObservationFrame<'a> {
    /// Build a frame, validating that all columns share one length.
    pub fn new(categorical: Vec<Cow<'a, [u32]>>) -> Result<Self, BuildError> {
        let n_obs = categorical.first().map_or(0, |c| c.len());
        for (column, c) in categorical.iter().enumerate() {
            if c.len() != n_obs {
                return Err(BuildError::ObservationCountMismatch {
                    column,
                    expected: n_obs,
                    got: c.len(),
                });
            }
        }
        Ok(ObservationFrame { categorical, n_obs })
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

    /// Convert every column to owned, dropping ties to caller buffers.
    pub fn into_owned(self) -> ObservationFrame<'static> {
        ObservationFrame {
            categorical: self
                .categorical
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
            n_obs: perm.len(),
        }
    }

    /// The level columns, moved out without copying.
    pub(crate) fn into_columns(self) -> Columns<'a> {
        self.categorical
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn columns_stay_row_aligned_under_permutation() {
        let frame =
            ObservationFrame::new(vec![vec![2u32, 0, 1, 0].into(), vec![5u32, 6, 7, 8].into()])
                .unwrap();

        let sorted = frame.permuted(&[1, 3, 2, 0]);

        assert_eq!(sorted.level_column(0), &[0, 0, 1, 2]);
        assert_eq!(sorted.level_column(1), &[6, 8, 7, 5]);
    }

    #[test]
    fn mismatched_column_lengths_rejected() {
        let result = ObservationFrame::new(vec![vec![0u32, 1, 0].into(), vec![0u32, 1].into()]);
        assert!(matches!(
            result,
            Err(BuildError::ObservationCountMismatch { .. })
        ));
    }
}
