use rayon::prelude::*;

/// Minimum number of rows to trigger parallel SpMV.
pub(crate) const PAR_SPMV_THRESHOLD: usize = 10_000;
/// Target number of non-zeros per parallel chunk.
const TARGET_NNZ_PER_CHUNK: usize = 32_768;

/// Rectangular CSR matrix used as the off-diagonal block in bipartite Gramians.
///
/// Stores C (n_q × n_r) or C^T (n_r × n_q). All column indices within each
/// row are sorted in ascending order.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct CsrBlock {
    pub(crate) indptr: Vec<u32>,
    pub(crate) indices: Vec<u32>,
    pub(crate) data: Vec<f64>,
    pub(crate) nrows: usize,
    pub(crate) ncols: usize,
}

impl CsrBlock {
    /// Number of stored non-zeros.
    pub(crate) fn nnz(&self) -> usize {
        self.data.len()
    }

    /// Transpose this CSR block: (nrows × ncols) → (ncols × nrows).
    ///
    /// O(nnz). Rows of the output are automatically sorted because we process
    /// source rows in ascending order.
    pub(crate) fn transpose(&self) -> CsrBlock {
        let nnz = self.nnz();
        let mut row_counts = vec![0u32; self.ncols];
        for &col in &self.indices {
            row_counts[col as usize] += 1;
        }
        let mut indptr = vec![0u32; self.ncols + 1];
        for i in 0..self.ncols {
            indptr[i + 1] = indptr[i] + row_counts[i];
        }
        let mut cursor = indptr[..self.ncols].to_vec();
        let mut indices = vec![0u32; nnz];
        let mut data = vec![0.0f64; nnz];
        for src_row in 0..self.nrows {
            let start = self.indptr[src_row] as usize;
            let end = self.indptr[src_row + 1] as usize;
            for idx in start..end {
                let dst_row = self.indices[idx] as usize;
                let pos = cursor[dst_row] as usize;
                indices[pos] = src_row as u32;
                data[pos] = self.data[idx];
                cursor[dst_row] += 1;
            }
        }
        CsrBlock {
            indptr,
            indices,
            data,
            nrows: self.ncols,
            ncols: self.nrows,
        }
    }

    /// Build a CSR block from a row-major dense table, skipping zeros.
    ///
    /// `table` has layout `table[i * ncols + j]` for row i, column j.
    pub(crate) fn from_dense_table(table: &[f64], nrows: usize, ncols: usize) -> Self {
        debug_assert_eq!(table.len(), nrows * ncols);
        let mut indptr = vec![0u32; nrows + 1];
        for i in 0..nrows {
            let row_start = i * ncols;
            let mut count = 0u32;
            for j in 0..ncols {
                if table[row_start + j] != 0.0 {
                    count += 1;
                }
            }
            indptr[i + 1] = indptr[i] + count;
        }
        let nnz = indptr[nrows] as usize;
        let mut indices = Vec::with_capacity(nnz);
        let mut data = Vec::with_capacity(nnz);
        for i in 0..nrows {
            let row_start = i * ncols;
            for j in 0..ncols {
                let v = table[row_start + j];
                if v != 0.0 {
                    indices.push(j as u32);
                    data.push(v);
                }
            }
        }
        CsrBlock {
            indptr,
            indices,
            data,
            nrows,
            ncols,
        }
    }

    /// y += A * diag(d) * x (sparse triple product, additive).
    ///
    /// Equivalent to `y += A * (d .* x)` but without allocating the
    /// element-wise product. Automatically parallelizes for large matrices.
    #[cfg(test)]
    pub(crate) fn spmv_diag_add(
        &self,
        d: &[f64],
        x: &[f64],
        y: &mut [f64],
        allow_inner_parallelism: bool,
    ) {
        debug_assert!(d.len() >= self.ncols);
        debug_assert!(x.len() >= self.ncols);
        debug_assert!(y.len() >= self.nrows);
        if self.nrows > PAR_SPMV_THRESHOLD && allow_inner_parallelism {
            self.par_spmv_diag_add(d, x, y);
        } else {
            self.seq_spmv_diag_add(d, x, y);
        }
    }

    /// y += A * x (sparse matrix-vector multiply, additive).
    ///
    /// Automatically parallelizes for large matrices using the same chunking
    /// policy as [`Self::spmv_diag_add`].
    #[cfg(test)]
    pub(crate) fn spmv_add(&self, x: &[f64], y: &mut [f64], allow_inner_parallelism: bool) {
        debug_assert!(x.len() >= self.ncols);
        debug_assert!(y.len() >= self.nrows);
        if self.nrows > PAR_SPMV_THRESHOLD && allow_inner_parallelism {
            self.par_spmv_add(x, y);
        } else {
            self.seq_spmv_add(x, y);
        }
    }

    /// y = base + A * x (sparse matrix-vector multiply with explicit base).
    ///
    /// This fuses a `copy_from_slice(base)` with `spmv_add(x, y)` to avoid an
    /// extra pass over the output buffer in block-elimination solves.
    pub(crate) fn spmv_assign_add(
        &self,
        x: &[f64],
        base: &[f64],
        y: &mut [f64],
        allow_inner_parallelism: bool,
    ) {
        debug_assert!(x.len() >= self.ncols);
        debug_assert!(base.len() >= self.nrows);
        debug_assert!(y.len() >= self.nrows);
        if self.nrows > PAR_SPMV_THRESHOLD && allow_inner_parallelism {
            self.par_spmv_assign_add(x, base, y);
        } else {
            self.seq_spmv_assign_add(x, base, y);
        }
    }

    #[cfg(test)]
    fn seq_spmv_diag_add(&self, d: &[f64], x: &[f64], y: &mut [f64]) {
        for (i, yi) in y[..self.nrows].iter_mut().enumerate() {
            let start = self.indptr[i] as usize;
            let end = self.indptr[i + 1] as usize;
            let row_data = &self.data[start..end];
            let row_idx = &self.indices[start..end];
            let mut acc = 0.0;
            for (&val, &col) in row_data.iter().zip(row_idx) {
                let j = col as usize;
                acc += val * d[j] * x[j];
            }
            *yi += acc;
        }
    }

    #[cfg(test)]
    fn seq_spmv_add(&self, x: &[f64], y: &mut [f64]) {
        for (i, yi) in y[..self.nrows].iter_mut().enumerate() {
            let start = self.indptr[i] as usize;
            let end = self.indptr[i + 1] as usize;
            let row_data = &self.data[start..end];
            let row_idx = &self.indices[start..end];
            let mut acc = 0.0;
            for (&val, &col) in row_data.iter().zip(row_idx) {
                acc += val * x[col as usize];
            }
            *yi += acc;
        }
    }

    fn seq_spmv_assign_add(&self, x: &[f64], base: &[f64], y: &mut [f64]) {
        for (i, yi) in y[..self.nrows].iter_mut().enumerate() {
            let start = self.indptr[i] as usize;
            let end = self.indptr[i + 1] as usize;
            let row_data = &self.data[start..end];
            let row_idx = &self.indices[start..end];
            let mut acc = base[i];
            for (&val, &col) in row_data.iter().zip(row_idx) {
                acc += val * x[col as usize];
            }
            *yi = acc;
        }
    }

    #[cfg(test)]
    fn par_spmv_diag_add(&self, d: &[f64], x: &[f64], y: &mut [f64]) {
        let indptr = &self.indptr;
        let indices = &self.indices;
        let data = &self.data;
        let nnz = self.nnz();
        let nrows = self.nrows;
        let avg_nnz_per_row = nnz / nrows.max(1);
        let chunk = (TARGET_NNZ_PER_CHUNK / avg_nnz_per_row.max(1)).clamp(256, 8192);

        y[..self.nrows]
            .par_chunks_mut(chunk)
            .enumerate()
            .for_each(|(chunk_idx, y_chunk)| {
                let row_start = chunk_idx * chunk;
                for (local_i, yi) in y_chunk.iter_mut().enumerate() {
                    let i = row_start + local_i;
                    let start = indptr[i] as usize;
                    let end = indptr[i + 1] as usize;
                    let row_data = &data[start..end];
                    let row_idx = &indices[start..end];
                    let mut acc = 0.0;
                    for (&val, &col) in row_data.iter().zip(row_idx) {
                        let j = col as usize;
                        acc += val * d[j] * x[j];
                    }
                    *yi += acc;
                }
            });
    }

    #[cfg(test)]
    fn par_spmv_add(&self, x: &[f64], y: &mut [f64]) {
        let indptr = &self.indptr;
        let indices = &self.indices;
        let data = &self.data;
        let nnz = self.nnz();
        let nrows = self.nrows;
        let avg_nnz_per_row = nnz / nrows.max(1);
        let chunk = (TARGET_NNZ_PER_CHUNK / avg_nnz_per_row.max(1)).clamp(256, 8192);

        y[..self.nrows]
            .par_chunks_mut(chunk)
            .enumerate()
            .for_each(|(chunk_idx, y_chunk)| {
                let row_start = chunk_idx * chunk;
                for (local_i, yi) in y_chunk.iter_mut().enumerate() {
                    let i = row_start + local_i;
                    let start = indptr[i] as usize;
                    let end = indptr[i + 1] as usize;
                    let row_data = &data[start..end];
                    let row_idx = &indices[start..end];
                    let mut acc = 0.0;
                    for (&val, &col) in row_data.iter().zip(row_idx) {
                        acc += val * x[col as usize];
                    }
                    *yi += acc;
                }
            });
    }

    fn par_spmv_assign_add(&self, x: &[f64], base: &[f64], y: &mut [f64]) {
        let indptr = &self.indptr;
        let indices = &self.indices;
        let data = &self.data;
        let nnz = self.nnz();
        let nrows = self.nrows;
        let avg_nnz_per_row = nnz / nrows.max(1);
        let chunk = (TARGET_NNZ_PER_CHUNK / avg_nnz_per_row.max(1)).clamp(256, 8192);

        y[..self.nrows]
            .par_chunks_mut(chunk)
            .enumerate()
            .for_each(|(chunk_idx, y_chunk)| {
                let row_start = chunk_idx * chunk;
                for (local_i, yi) in y_chunk.iter_mut().enumerate() {
                    let i = row_start + local_i;
                    let start = indptr[i] as usize;
                    let end = indptr[i + 1] as usize;
                    let row_data = &data[start..end];
                    let row_idx = &indices[start..end];
                    let mut acc = base[i];
                    for (&val, &col) in row_data.iter().zip(row_idx) {
                        acc += val * x[col as usize];
                    }
                    *yi = acc;
                }
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_3x4_block() -> CsrBlock {
        // 3x4 matrix:
        // [1.0  0.0  2.0  0.0]
        // [0.0  3.0  0.0  4.0]
        // [5.0  0.0  6.0  0.0]
        CsrBlock::from_dense_table(
            &[1.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 4.0, 5.0, 0.0, 6.0, 0.0],
            3,
            4,
        )
    }

    #[test]
    fn test_from_dense_table_structure() {
        let b = make_3x4_block();
        assert_eq!(b.nrows, 3);
        assert_eq!(b.ncols, 4);
        assert_eq!(b.nnz(), 6);

        // Row 0: cols 0, 2
        assert_eq!(b.indptr[0], 0);
        assert_eq!(b.indptr[1], 2);
        assert_eq!(b.indices[0], 0);
        assert_eq!(b.indices[1], 2);
        assert_eq!(b.data[0], 1.0);
        assert_eq!(b.data[1], 2.0);

        // Row 1: cols 1, 3
        assert_eq!(b.indptr[2], 4);
        assert_eq!(b.indices[2], 1);
        assert_eq!(b.indices[3], 3);
        assert_eq!(b.data[2], 3.0);
        assert_eq!(b.data[3], 4.0);

        // Row 2: cols 0, 2
        assert_eq!(b.indptr[3], 6);
        assert_eq!(b.indices[4], 0);
        assert_eq!(b.indices[5], 2);
        assert_eq!(b.data[4], 5.0);
        assert_eq!(b.data[5], 6.0);
    }

    #[test]
    fn test_from_dense_table_all_zeros() {
        let b = CsrBlock::from_dense_table(&[0.0; 6], 2, 3);
        assert_eq!(b.nrows, 2);
        assert_eq!(b.ncols, 3);
        assert_eq!(b.nnz(), 0);
        assert_eq!(b.indptr, vec![0, 0, 0]);
    }

    #[test]
    fn test_transpose_basic() {
        let b = make_3x4_block();
        let bt = b.transpose();

        assert_eq!(bt.nrows, 4);
        assert_eq!(bt.ncols, 3);
        assert_eq!(bt.nnz(), 6);

        // Verify A^T[j, i] == A[i, j] by checking specific values
        // Original row 0: (0,0)=1, (0,2)=2
        // Transposed: row 0 should have (0,0)=1, (0,2)=5
        // row 2 should have (2,0)=2, (2,2)=6

        // Row 0 of transpose: columns 0 and 2 (from original rows 0 and 2 having col 0)
        let r0_start = bt.indptr[0] as usize;
        let r0_end = bt.indptr[1] as usize;
        let r0_cols: Vec<u32> = bt.indices[r0_start..r0_end].to_vec();
        let r0_vals: Vec<f64> = bt.data[r0_start..r0_end].to_vec();
        assert_eq!(r0_cols, vec![0, 2]); // rows 0, 2 of original
        assert_eq!(r0_vals, vec![1.0, 5.0]); // values at (0,0) and (2,0) of original
    }

    #[test]
    fn test_transpose_transpose_roundtrip() {
        let b = make_3x4_block();
        let btt = b.transpose().transpose();

        assert_eq!(btt.nrows, b.nrows);
        assert_eq!(btt.ncols, b.ncols);
        assert_eq!(btt.nnz(), b.nnz());
        assert_eq!(btt.indptr, b.indptr);
        assert_eq!(btt.indices, b.indices);
        // Values should match
        for (a, e) in btt.data.iter().zip(b.data.iter()) {
            assert!((a - e).abs() < 1e-14);
        }
    }

    #[test]
    fn test_nnz() {
        let b = make_3x4_block();
        assert_eq!(b.nnz(), 6);

        let empty = CsrBlock::from_dense_table(&[0.0; 4], 2, 2);
        assert_eq!(empty.nnz(), 0);
    }

    #[test]
    fn test_spmv_diag_add_basic() {
        // A = [[1, 0, 2],
        //      [0, 3, 0]]
        // d = [2, 1, 3]
        // x = [1, 2, 1]
        // y = [10, 20]
        // A * diag(d) * x = A * [2, 2, 3] = [[1*2 + 2*3], [3*2]] = [8, 6]
        // y += [8, 6] = [18, 26]
        let b = CsrBlock::from_dense_table(&[1.0, 0.0, 2.0, 0.0, 3.0, 0.0], 2, 3);
        let d = vec![2.0, 1.0, 3.0];
        let x = vec![1.0, 2.0, 1.0];
        let mut y = vec![10.0, 20.0];

        b.spmv_diag_add(&d, &x, &mut y, true);

        assert!((y[0] - 18.0).abs() < 1e-14, "y[0] = {}", y[0]);
        assert!((y[1] - 26.0).abs() < 1e-14, "y[1] = {}", y[1]);
    }

    #[test]
    fn test_spmv_diag_add_identity_diag() {
        // With d = [1, 1, ...], spmv_diag_add degenerates to y += A * x
        let b = make_3x4_block();
        let d = vec![1.0; 4];
        let x = vec![1.0, 1.0, 1.0, 1.0];
        let mut y = vec![0.0; 3];

        b.spmv_diag_add(&d, &x, &mut y, true);

        // Row 0: 1*1 + 2*1 = 3
        // Row 1: 3*1 + 4*1 = 7
        // Row 2: 5*1 + 6*1 = 11
        assert!((y[0] - 3.0).abs() < 1e-14);
        assert!((y[1] - 7.0).abs() < 1e-14);
        assert!((y[2] - 11.0).abs() < 1e-14);
    }

    #[test]
    fn test_spmv_add_basic() {
        let b = make_3x4_block();
        let x = vec![1.0, 1.0, 1.0, 1.0];
        let mut y = vec![0.0; 3];

        b.spmv_add(&x, &mut y, true);

        assert!((y[0] - 3.0).abs() < 1e-14);
        assert!((y[1] - 7.0).abs() < 1e-14);
        assert!((y[2] - 11.0).abs() < 1e-14);
    }

    #[test]
    fn test_spmv_assign_add_basic() {
        let b = make_3x4_block();
        let x = vec![1.0, 1.0, 1.0, 1.0];
        let base = vec![10.0, 20.0, 30.0];
        let mut y = vec![0.0; 3];

        b.spmv_assign_add(&x, &base, &mut y, true);

        assert!((y[0] - 13.0).abs() < 1e-14);
        assert!((y[1] - 27.0).abs() < 1e-14);
        assert!((y[2] - 41.0).abs() < 1e-14);
    }

    #[test]
    fn test_spmv_diag_add_zero_x() {
        let b = make_3x4_block();
        let d = vec![1.0; 4];
        let x = vec![0.0; 4];
        let mut y = vec![5.0; 3];

        b.spmv_diag_add(&d, &x, &mut y, true);

        // y should remain unchanged
        assert_eq!(y, vec![5.0; 3]);
    }

    #[test]
    fn test_from_dense_table_single_element() {
        let b = CsrBlock::from_dense_table(&[42.0], 1, 1);
        assert_eq!(b.nrows, 1);
        assert_eq!(b.ncols, 1);
        assert_eq!(b.nnz(), 1);
        assert_eq!(b.data[0], 42.0);
        assert_eq!(b.indices[0], 0);
    }

    #[test]
    fn test_transpose_empty() {
        let b = CsrBlock::from_dense_table(&[0.0; 6], 2, 3);
        let bt = b.transpose();
        assert_eq!(bt.nrows, 3);
        assert_eq!(bt.ncols, 2);
        assert_eq!(bt.nnz(), 0);
    }
}
