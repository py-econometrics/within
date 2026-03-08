use rayon::prelude::*;

/// Minimum number of rows to trigger parallel SpMV.
const PAR_SPMV_THRESHOLD: usize = 10_000;
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
}
