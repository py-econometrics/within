//! Observation storage: [`Store`] trait + [`FactorMajorStore`] / [`ArrayStore`] backends.

use ndarray::ArrayView2;

use crate::error::BuildError;

// ---------------------------------------------------------------------------
// Store trait
// ---------------------------------------------------------------------------

/// Core abstraction: how observation data is stored and accessed.
///
/// Each backend optimizes for different data characteristics.
/// All implementors must be `Send + Sync` for Rayon parallelism.
pub trait Store: Send + Sync {
    /// Number of observations.
    fn n_obs(&self) -> usize;

    /// Number of factors.
    fn n_factors(&self) -> usize;

    /// Level index for observation `obs` in factor `factor`.
    fn level(&self, obs: usize, factor: usize) -> u32;

    /// Optional fast-path access to a factor-major column of levels.
    ///
    /// Stores that naturally keep `level(obs, factor)` as contiguous
    /// `levels[factor][obs]` should return `Some(&levels[factor])`.
    /// Others should return `None` (default).
    fn factor_column(&self, _factor: usize) -> Option<&[u32]> {
        None
    }
}

/// Resolve the level for row `i` in factor `q`.
///
/// `levels` is the optional fast-path column (a contiguous `&[u32]` view of the
/// factor's levels); when `None`, fall back to the store's virtual lookup.
/// Hoisted out of inner loops so the compiler keeps the row body branch-free.
#[inline]
pub(crate) fn level_at<S: Store>(store: &S, levels: Option<&[u32]>, i: usize, q: usize) -> usize {
    match levels {
        Some(col) => col[i] as usize,
        None => store.level(i, q) as usize,
    }
}

/// Pre-compute the factor-column fast-path slices for every factor of `store`.
pub(crate) fn factor_columns<S: Store>(store: &S) -> Vec<Option<&[u32]>> {
    (0..store.n_factors())
        .map(|q| store.factor_column(q))
        .collect()
}

// ---------------------------------------------------------------------------
// FactorMajorStore
// ---------------------------------------------------------------------------

/// Factor-major observation storage: `factor_levels[q][i]` is the level
/// for observation `i` in factor `q`.
///
/// Construction is nearly free — just convert i64 to usize from Python input.
/// Factor-column access is sequential, making it optimal for Gramian build
/// and domain decomposition (which iterate per-factor).
#[derive(Debug, Clone)]
pub struct FactorMajorStore {
    factor_levels: Vec<Vec<u32>>,
    n_obs: usize,
}

impl FactorMajorStore {
    /// Create a new factor-major store, validating that all columns have length `n_obs`.
    pub fn new(factor_levels: Vec<Vec<u32>>, n_obs: usize) -> Result<Self, BuildError> {
        for (factor, col) in factor_levels.iter().enumerate() {
            if col.len() != n_obs {
                return Err(BuildError::ObservationCountMismatch {
                    factor,
                    expected: n_obs,
                    got: col.len(),
                });
            }
        }
        Ok(Self {
            factor_levels,
            n_obs,
        })
    }
}

impl Store for FactorMajorStore {
    #[inline]
    fn n_obs(&self) -> usize {
        self.n_obs
    }

    #[inline]
    fn n_factors(&self) -> usize {
        self.factor_levels.len()
    }

    #[inline]
    fn level(&self, obs: usize, factor: usize) -> u32 {
        self.factor_levels[factor][obs]
    }

    #[inline]
    fn factor_column(&self, factor: usize) -> Option<&[u32]> {
        Some(&self.factor_levels[factor])
    }
}

// ---------------------------------------------------------------------------
// ArrayStore — zero-copy observation-major backend
// ---------------------------------------------------------------------------

/// Zero-copy store backed by a borrowed `ArrayView2<u32>`.
///
/// `categories[[obs, factor]]` is the level for observation `obs` in factor
/// `factor`. No data is copied — the view points directly into the caller's
/// buffer (e.g. a numpy array from Python).
///
/// For F-contiguous (column-major) arrays, `factor_column()` returns
/// contiguous slices — matching `FactorMajorStore` performance.
/// For C-contiguous arrays, columns are strided and the hot loops fall
/// back to per-element `level()` indexing.
#[derive(Debug)]
pub struct ArrayStore<'a> {
    categories: ArrayView2<'a, u32>,
}

impl<'a> ArrayStore<'a> {
    /// Create a zero-copy store from a borrowed 2-D category array.
    pub fn new(categories: ArrayView2<'a, u32>) -> Result<Self, BuildError> {
        Ok(Self { categories })
    }
}

impl Store for ArrayStore<'_> {
    #[inline]
    fn n_obs(&self) -> usize {
        self.categories.nrows()
    }

    #[inline]
    fn n_factors(&self) -> usize {
        self.categories.ncols()
    }

    #[inline]
    fn level(&self, obs: usize, factor: usize) -> u32 {
        self.categories[[obs, factor]]
    }

    fn factor_column(&self, factor: usize) -> Option<&[u32]> {
        let strides = self.categories.strides();
        // Columns are contiguous only when the row stride is 1 (F-order). The
        // column stride must additionally be positive: a column-reversed view
        // (e.g. `cats[:, ::-1]` of an F-order array) keeps `strides[0] == 1`
        // but has `strides[1] < 1`, which would wrap to a huge `usize` below
        // and produce an out-of-bounds `from_raw_parts`. Fall back to the safe
        // per-element `level()` path in that case.
        if strides[0] != 1 || strides[1] < 1 {
            return None;
        }
        let n_obs = self.categories.nrows();
        let col_stride = strides[1] as usize;
        let ptr = self.categories.as_ptr();
        // Safety: F-contiguous layout guarantees n_obs elements at stride-1
        // starting at ptr + factor * col_stride.
        Some(unsafe { std::slice::from_raw_parts(ptr.add(factor * col_stride), n_obs) })
    }
}

// ---------------------------------------------------------------------------
// Weight validation helpers
// ---------------------------------------------------------------------------

/// Validate that an optional weight slice matches `n_obs` observations.
///
/// `None` is always valid (interpreted as unit weights). `Some(w)` requires
/// `w.len() == n_obs`.
pub(crate) fn validate_weights(weights: Option<&[f64]>, n_obs: usize) -> Result<(), BuildError> {
    if let Some(w) = weights {
        if w.len() != n_obs {
            return Err(BuildError::WeightCountMismatch {
                expected: n_obs,
                got: w.len(),
            });
        }
        // `W^{1/2}` is applied to the design, so each weight must be finite and
        // non-negative; otherwise `sqrt(w)` is NaN and the solution is silently
        // corrupted. `wi >= 0.0` already rejects NaN (comparisons with NaN are
        // false); `is_finite` additionally rejects `+∞`.
        if let Some((index, &value)) = w
            .iter()
            .enumerate()
            .find(|&(_, &wi)| !(wi >= 0.0 && wi.is_finite()))
        {
            return Err(BuildError::InvalidWeight { index, value });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_factor_major_store_basic() {
        let store = FactorMajorStore::new(vec![vec![0, 1, 2, 0], vec![0, 1, 0, 1]], 4)
            .expect("valid factor-major store");
        assert_eq!(store.n_obs(), 4);
        assert_eq!(store.n_factors(), 2);
        assert_eq!(store.level(0, 0), 0);
        assert_eq!(store.level(1, 0), 1);
        assert_eq!(store.level(2, 1), 0);
    }

    #[test]
    fn test_factor_column() {
        let store = FactorMajorStore::new(vec![vec![0u32, 1, 2, 0], vec![3, 2, 1, 0]], 4)
            .expect("valid factor-major store");
        assert_eq!(store.factor_column(0).unwrap(), &[0u32, 1, 2, 0]);
        assert_eq!(store.factor_column(1).unwrap(), &[3u32, 2, 1, 0]);
    }

    #[test]
    fn test_validate_weights() {
        assert!(validate_weights(None, 5).is_ok());
        assert!(validate_weights(Some(&[1.0, 2.0, 3.0, 4.0, 5.0]), 5).is_ok());
        // Zero weights are valid (an excluded observation).
        assert!(validate_weights(Some(&[0.0, 1.0, 2.0, 3.0, 4.0]), 5).is_ok());
        // Length mismatch.
        assert!(validate_weights(Some(&[1.0, 2.0]), 5).is_err());
        // Negative / non-finite weights are rejected with the offending index.
        assert!(matches!(
            validate_weights(Some(&[1.0, -2.0, 3.0, 4.0, 5.0]), 5),
            Err(BuildError::InvalidWeight { index: 1, .. })
        ));
        assert!(matches!(
            validate_weights(Some(&[1.0, 2.0, f64::NAN, 4.0, 5.0]), 5),
            Err(BuildError::InvalidWeight { index: 2, .. })
        ));
        assert!(matches!(
            validate_weights(Some(&[1.0, 2.0, 3.0, f64::INFINITY, 5.0]), 5),
            Err(BuildError::InvalidWeight { index: 3, .. })
        ));
    }
}
