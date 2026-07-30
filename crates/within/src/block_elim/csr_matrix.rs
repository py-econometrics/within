//! Validated square `f64`/`u32` matrix in Compressed Sparse Row (CSR) format.
//!
//! Holds the reduced Schur-complement / Laplacian system that
//! [`schur`](super::schur) assembles and [`factor`](super::factor) hands to the
//! sparse factor backend — a `block_elim`-internal representation. It is
//! distinct from the crate-level [`CsrBlock`](crate::csr_block::CsrBlock), which
//! carries the rectangular cross-factor coupling block produced in `domain` and
//! consumed here.

/// Block elimination produces the CSR invariants, so [`CsrMatrix::new`] only debug-checks them.
#[derive(Clone)]
pub(crate) struct CsrMatrix {
    indptr: Vec<u32>,
    indices: Vec<u32>,
    data: Vec<f64>,
    n: usize,
}

impl CsrMatrix {
    /// Trusted in release, `assert!`-checked in debug.
    pub(crate) fn new(indptr: Vec<u32>, indices: Vec<u32>, data: Vec<f64>, n: usize) -> Self {
        #[cfg(debug_assertions)]
        {
            assert_eq!(indptr.len(), n + 1, "indptr length must equal n + 1");
            assert_eq!(indptr[0], 0, "indptr[0] must be 0");
            assert_eq!(
                indices.len(),
                data.len(),
                "indices and data must have equal length"
            );
            assert_eq!(
                indices.len(),
                indptr[n] as usize,
                "indices/data length must equal indptr[n]"
            );
            for row in 0..n {
                let start = indptr[row] as usize;
                let end = indptr[row + 1] as usize;
                assert!(end >= start, "indptr must be non-decreasing");
                let cols = &indices[start..end];
                assert!(
                    cols.iter().all(|&col| (col as usize) < n),
                    "column index out of bounds"
                );
                assert!(
                    cols.windows(2).all(|w| w[0] < w[1]),
                    "row column indices must be strictly ascending"
                );
            }
        }
        Self {
            indptr,
            indices,
            data,
            n,
        }
    }

    /// Row pointer array (length `n + 1`).
    #[inline]
    pub(crate) fn indptr(&self) -> &[u32] {
        &self.indptr
    }

    /// Column indices (length `nnz`).
    #[inline]
    pub(crate) fn indices(&self) -> &[u32] {
        &self.indices
    }

    /// Non-zero values (length `nnz`), parallel to `indices`.
    #[inline]
    pub(crate) fn data(&self) -> &[f64] {
        &self.data
    }

    /// Matrix dimension (the matrix is always square).
    #[inline]
    pub(crate) fn n(&self) -> usize {
        self.n
    }
}
