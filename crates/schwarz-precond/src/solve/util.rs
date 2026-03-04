//! Shared BLAS-like primitives for iterative solvers.

/// Inner product of two vectors.
#[inline]
pub(crate) fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Euclidean norm of a vector.
#[inline]
pub fn vec_norm(v: &[f64]) -> f64 {
    let mut s = 0.0f64;
    for &x in v {
        s += x * x;
    }
    s.sqrt()
}

/// Column-major flat storage for basis vectors.
///
/// Stores up to `capacity` vectors of dimension `n` in a single contiguous
/// allocation. Avoids per-restart/per-iteration heap allocation.
pub(crate) struct ColumnBasis {
    data: Vec<f64>,
    /// Dimension of each vector.
    pub(crate) n: usize,
    len: usize,
}

impl ColumnBasis {
    /// Create with room for `capacity` vectors of length `n`.
    pub(crate) fn new(capacity: usize, n: usize) -> Self {
        Self {
            data: vec![0.0; capacity * n],
            n,
            len: 0,
        }
    }

    /// Number of vectors currently stored.
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// Return column `j` as a shared slice.
    #[inline]
    pub(crate) fn col(&self, j: usize) -> &[f64] {
        debug_assert!(j < self.len);
        let start = j * self.n;
        &self.data[start..start + self.n]
    }

    /// Return column `j` as a mutable slice.
    ///
    /// # Safety
    /// Caller must ensure no aliasing mutable references to the same column.
    #[inline]
    #[allow(clippy::mut_from_ref)]
    pub(crate) fn col_mut(&self, j: usize) -> &mut [f64] {
        debug_assert!(j < self.len);
        let start = j * self.n;
        unsafe {
            let ptr = self.data.as_ptr().add(start) as *mut f64;
            std::slice::from_raw_parts_mut(ptr, self.n)
        }
    }

    /// Append `src` as the next column, incrementing `len`.
    pub(crate) fn push_from(&mut self, src: &[f64]) {
        debug_assert_eq!(src.len(), self.n);
        let start = self.len * self.n;
        self.data[start..start + self.n].copy_from_slice(src);
        self.len += 1;
    }

    /// Append `src * scale` as the next column, incrementing `len`.
    pub(crate) fn push_scaled(&mut self, src: &[f64], scale: f64) {
        debug_assert_eq!(src.len(), self.n);
        let start = self.len * self.n;
        for (dst, &s) in self.data[start..start + self.n].iter_mut().zip(src) {
            *dst = s * scale;
        }
        self.len += 1;
    }

    /// Append a zero-filled column, incrementing `len`.
    pub(crate) fn push_zeroed(&mut self) {
        let start = self.len * self.n;
        self.data[start..start + self.n].fill(0.0);
        self.len += 1;
    }

    /// Reset the number of stored vectors to zero (does not deallocate).
    pub(crate) fn clear(&mut self) {
        self.len = 0;
    }
}
