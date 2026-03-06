use rayon::prelude::*;

/// Minimum number of rows to trigger parallel SpMV.
const PAR_SPMV_THRESHOLD: usize = 10_000;
const PAR_SPMV_CHUNK: usize = 4096;

/// Rectangular CSR matrix used as the off-diagonal block in bipartite Gramians.
///
/// Stores C (n_q × n_r) or C^T (n_r × n_q). All column indices within each
/// row are sorted in ascending order.
#[derive(Clone)]
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
    pub(crate) fn spmv_diag_add(&self, d: &[f64], x: &[f64], y: &mut [f64]) {
        debug_assert!(d.len() >= self.ncols);
        debug_assert!(x.len() >= self.ncols);
        debug_assert!(y.len() >= self.nrows);
        if self.nrows > PAR_SPMV_THRESHOLD {
            self.par_spmv_diag_add(d, x, y);
        } else {
            self.seq_spmv_diag_add(d, x, y);
        }
    }

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

    fn par_spmv_diag_add(&self, d: &[f64], x: &[f64], y: &mut [f64]) {
        let indptr = &self.indptr;
        let indices = &self.indices;
        let data = &self.data;

        y[..self.nrows]
            .par_chunks_mut(PAR_SPMV_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, y_chunk)| {
                let row_start = chunk_idx * PAR_SPMV_CHUNK;
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
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_block() -> CsrBlock {
        // 3x4 matrix:
        //  [1 0 2 0]
        //  [0 3 0 4]
        //  [5 0 0 6]
        CsrBlock {
            indptr: vec![0, 2, 4, 6],
            indices: vec![0, 2, 1, 3, 0, 3],
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            nrows: 3,
            ncols: 4,
        }
    }

    #[test]
    fn test_nnz() {
        assert_eq!(sample_block().nnz(), 6);
    }

    #[test]
    fn test_transpose_dimensions() {
        let a = sample_block();
        let at = a.transpose();
        assert_eq!(at.nrows, 4);
        assert_eq!(at.ncols, 3);
        assert_eq!(at.nnz(), a.nnz());
    }

    #[test]
    fn test_transpose_roundtrip() {
        let a = sample_block();
        let att = a.transpose().transpose();
        assert_eq!(att.nrows, a.nrows);
        assert_eq!(att.ncols, a.ncols);
        assert_eq!(att.indptr, a.indptr);
        assert_eq!(att.indices, a.indices);
        assert_eq!(att.data, a.data);
    }

    #[test]
    fn test_transpose_values() {
        let a = sample_block();
        let at = a.transpose();
        // A^T should be 4x3:
        //  [1 0 5]
        //  [0 3 0]
        //  [2 0 0]
        //  [0 4 6]
        assert_eq!(at.indptr, vec![0, 2, 3, 4, 6]);
        assert_eq!(at.indices, vec![0, 2, 1, 0, 1, 2]);
        assert_eq!(at.data, vec![1.0, 5.0, 3.0, 2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_from_dense_table() {
        let table = vec![1.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 4.0, 5.0, 0.0, 0.0, 6.0];
        let a = CsrBlock::from_dense_table(&table, 3, 4);
        let expected = sample_block();
        assert_eq!(a.indptr, expected.indptr);
        assert_eq!(a.indices, expected.indices);
        assert_eq!(a.data, expected.data);
        assert_eq!(a.nrows, expected.nrows);
        assert_eq!(a.ncols, expected.ncols);
    }

    #[test]
    fn test_spmv_diag_add() {
        let a = sample_block();
        let d = vec![2.0, 3.0, 1.0, 0.5];
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let mut y = vec![0.0; 3];
        a.spmv_diag_add(&d, &x, &mut y);
        // row 0: 1*(2*1) + 2*(1*3) = 2+6 = 8
        // row 1: 3*(3*2) + 4*(0.5*4) = 18+8 = 26
        // row 2: 5*(2*1) + 6*(0.5*4) = 10+12 = 22
        assert_eq!(y, vec![8.0, 26.0, 22.0]);
    }

    #[test]
    fn test_empty_block() {
        let a = CsrBlock {
            indptr: vec![0, 0, 0],
            indices: vec![],
            data: vec![],
            nrows: 2,
            ncols: 3,
        };
        assert_eq!(a.nnz(), 0);
        let at = a.transpose();
        assert_eq!(at.nrows, 3);
        assert_eq!(at.ncols, 2);
        assert_eq!(at.nnz(), 0);
    }

    #[test]
    fn test_from_dense_table_all_zeros() {
        let table = vec![0.0; 6];
        let a = CsrBlock::from_dense_table(&table, 2, 3);
        assert_eq!(a.nnz(), 0);
        assert_eq!(a.indptr, vec![0, 0, 0]);
    }
}
