//! Observation storage layer: traits, backends, and metadata.
//!
//! This is the lowest layer of the `within` crate. It defines *how*
//! per-observation data (factor levels) is stored and accessed,
//! without knowing anything about design matrices, operators, or solvers.
//! Observation weights live *outside* the store, carried by the operators
//! that actually need them.
//!
//! # Why pluggable backends?
//!
//! Different callers supply data in different layouts:
//!
//! - **Python (via PyO3)** passes a borrowed numpy array — copying it into a
//!   Rust-owned layout would double memory and add latency.
//! - **Rust tests and benchmarks** build data programmatically as `Vec<Vec<u32>>`,
//!   which is naturally factor-major.
//!
//! The [`Store`] trait abstracts over these layouts so that all
//! upstream code (design matrix operations, domain decomposition, Gramian
//! assembly) is generic and layout-agnostic.
//!
//! # Backends
//!
//! | Backend | Layout | Owns data? | Best for |
//! |---|---|---|---|
//! | [`FactorMajorStore`] | `factor_levels[q][i]` — grouped by factor | Yes | Rust-native construction; sequential factor-column access for Gramian build and domain decomposition |
//! | [`ArrayStore`] | `categories[[i, q]]` — borrowed `ArrayView2` | No (borrows) | Zero-copy from numpy; F-contiguous arrays get contiguous column access matching `FactorMajorStore` performance |
//!
//! Both backends implement the optional [`Store::factor_column`]
//! fast-path, which returns a contiguous `&[u32]` slice for a factor's levels
//! when the memory layout permits it. The design-matrix scatter/gather loops
//! exploit this to avoid per-element virtual dispatch.
//!
//! # Key types
//!
//! - [`FactorMeta`] — per-factor metadata (level count and global DOF offset),
//!   separated from observation data so it can live in the [`Design`](crate::domain::Design).
//! - [`Store`] — the core trait. All implementors must be
//!   `Send + Sync` to support Rayon parallelism in the layers above.

use ndarray::ArrayView2;

use crate::error::{WithinError, WithinResult};

// ---------------------------------------------------------------------------
// FactorMeta — per-factor metadata (no observation data)
// ---------------------------------------------------------------------------

/// Per-factor metadata: level count and global DOF offset.
///
/// Separated from observation data — the factor no longer "owns" categories.
#[derive(Debug, Clone, Copy)]
pub struct FactorMeta {
    /// Number of levels (groups) in this factor.
    pub n_levels: usize,
    /// Starting index in coefficient space for this factor.
    pub offset: usize,
}

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
    pub fn new(factor_levels: Vec<Vec<u32>>, n_obs: usize) -> WithinResult<Self> {
        for (factor, col) in factor_levels.iter().enumerate() {
            if col.len() != n_obs {
                return Err(WithinError::ObservationCountMismatch {
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

    /// Direct access to the level column for a factor (contiguous slice).
    #[inline]
    pub fn factor_column(&self, factor: usize) -> &[u32] {
        &self.factor_levels[factor]
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
        Some(self.factor_column(factor))
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
    pub fn new(categories: ArrayView2<'a, u32>) -> Self {
        Self { categories }
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
        // Columns are contiguous only when the row stride is 1 (F-order).
        if strides[0] != 1 {
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

/// Validate that a weight vector is compatible with `n_obs` observations.
pub fn validate_weights(weights: &[f64], n_obs: usize) -> WithinResult<()> {
    if weights.len() != n_obs {
        return Err(WithinError::WeightCountMismatch {
            expected: n_obs,
            got: weights.len(),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

#[cfg(test)]
fn sample_factor_levels() -> Vec<Vec<u32>> {
    vec![vec![0, 1, 2, 0], vec![0, 1, 0, 1]]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_factor_major_store_basic() {
        let store =
            FactorMajorStore::new(sample_factor_levels(), 4).expect("valid factor-major store");
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
        assert_eq!(store.factor_column(0), &[0u32, 1, 2, 0]);
        assert_eq!(store.factor_column(1), &[3u32, 2, 1, 0]);
    }
}
