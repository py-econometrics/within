//! Residual update strategies for multiplicative Schwarz.
//!
//! In multiplicative Schwarz, after each subdomain solve produces a correction
//! `delta`, the global residual `r` must be updated before the next subdomain
//! can use it.
//!
//! The [`SparseGramianUpdater`] computes `r <- r - G delta` using the pre-built
//! explicit Gramian CSR, restricted to touched rows. Cost is O(nnz_touched) with
//! cache-friendly contiguous CSR reads. This requires O(nnz(G)) memory for
//! storing the Gramian.

use std::sync::Arc;

use schwarz_precond::{ResidualUpdater, SparseMatrix};

/// Sparse Gramian residual updater for multiplicative Schwarz.
///
/// Uses the explicit Gramian CSR to perform residual updates via row scatter:
/// `r -= G * delta` restricted to the touched rows.
///
/// Cost: O(nnz_touched) with contiguous CSR reads. No buffers, no bookkeeping.
/// Trades O(nnz) memory (the Gramian) for faster per-iteration updates compared
/// to an observation-space path.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct SparseGramianUpdater {
    gramian: Arc<SparseMatrix>,
}

impl SparseGramianUpdater {
    pub fn new(gramian: Arc<SparseMatrix>) -> Self {
        Self { gramian }
    }
}

impl ResidualUpdater for SparseGramianUpdater {
    fn update(&mut self, global_indices: &[u32], weighted_correction: &[f64], r_work: &mut [f64]) {
        let indptr = self.gramian.indptr();
        let indices = self.gramian.indices();
        let data = self.gramian.data();

        for (k, &gi) in global_indices.iter().enumerate() {
            let c = weighted_correction[k];
            if c == 0.0 {
                continue;
            }
            let row = gi as usize;
            let start = indptr[row] as usize;
            let end = indptr[row + 1] as usize;
            for idx in start..end {
                r_work[indices[idx] as usize] -= c * data[idx];
            }
        }
    }

    fn reset(&mut self, _r_original: &[f64]) {
        // No-op: each update is a pure incremental r -= G * delta.
    }
}
