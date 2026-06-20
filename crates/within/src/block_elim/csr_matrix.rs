//! Validated square `f64`/`u32` matrix in Compressed Sparse Row (CSR) format.
//!
//! Holds the reduced Schur-complement / Laplacian system that
//! [`schur`](super::schur) assembles and [`factor`](super::factor) hands to the
//! sparse factor backend — a `block_elim`-internal representation. It is
//! distinct from the crate-level [`CsrBlock`](crate::csr_block::CsrBlock), which
//! carries the rectangular cross-factor coupling block produced in `domain` and
//! consumed here.

/// Square sparse matrix in Compressed Sparse Row (CSR) format.
///
/// # CSR Invariants
///
/// A well-formed `CsrMatrix` satisfies:
/// - `indptr.len() == n + 1`
/// - `indptr[0] == 0` and `indptr` is non-decreasing
/// - `indices.len() == data.len() == indptr[n] as usize` (the number of non-zeros)
/// - All column indices in `indices` are in `0..n`
/// - Within each row, column indices are sorted ascending with no duplicates
///
/// Block elimination produces these by construction, so [`CsrMatrix::new`] is
/// infallible and the invariants are guarded only in debug builds.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct CsrMatrix {
    indptr: Vec<u32>,
    indices: Vec<u32>,
    data: Vec<f64>,
    n: usize,
}

impl CsrMatrix {
    /// Create a `CsrMatrix` from raw CSR components.
    ///
    /// The components must satisfy the CSR invariants documented on
    /// [`CsrMatrix`]. Block elimination produces them by construction, so this
    /// constructor is infallible; in debug builds the invariants are checked
    /// with `assert!`, and in release builds they are trusted.
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
