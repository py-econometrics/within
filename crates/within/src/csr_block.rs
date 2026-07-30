use rayon::prelude::*;

/// Minimum number of rows to trigger parallel SpMV.
pub(crate) const PAR_SPMV_THRESHOLD: usize = 10_000;
/// Target number of non-zeros per parallel chunk.
const TARGET_NNZ_PER_CHUNK: usize = 32_768;

/// Checked `usize -> u32`: a silent `as` truncation would corrupt the structure undiagnosed.
#[inline]
pub(crate) fn to_u32(x: usize) -> u32 {
    u32::try_from(x).expect("CSR index exceeds u32::MAX")
}

/// Rectangular CSR block storing `C` or `Cᵀ`, column indices ascending within each row.
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

    /// `nrows`/`ncols` sit beside the arrays, so untrusted bytes can make them disagree.
    pub(crate) fn is_structurally_valid(&self) -> bool {
        if self.nrows.checked_add(1) != Some(self.indptr.len()) {
            return false;
        }
        let nnz = self.indices.len();
        if self.data.len() != nnz {
            return false;
        }
        if self.indptr[0] != 0 || *self.indptr.last().unwrap() as usize != nnz {
            return false;
        }
        if self.indptr.windows(2).any(|w| w[0] > w[1]) {
            return false;
        }
        self.indices.iter().all(|&j| (j as usize) < self.ncols)
    }

    /// Row `i` as `(column, weight)` pairs.
    pub(crate) fn row(&self, i: usize) -> impl Iterator<Item = (usize, f64)> + '_ {
        let start = self.indptr[i] as usize;
        let end = self.indptr[i + 1] as usize;
        self.indices[start..end]
            .iter()
            .zip(&self.data[start..end])
            .map(|(&j, &w)| (j as usize, w))
    }

    /// Transpose in O(nnz); output rows come out sorted because source rows go ascending.
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

    /// Build a CSR block from a row-major dense table (`table[i * ncols + j]`), skipping zeros.
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

    /// `y = base + A x`, fusing the base copy with accumulation to avoid an extra pass.
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

        // A^T[j, i] == A[i, j] on the values, not just the structure.

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
    fn test_spmv_assign_add_zero_x_preserves_base() {
        let b = make_3x4_block();
        let x = vec![0.0; 4];
        let base = vec![5.0; 3];
        let mut y = vec![0.0; 3];

        b.spmv_assign_add(&x, &base, &mut y, true);

        assert_eq!(y, base);
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

    // Chunk *sizing* cannot change the result, so it is deliberately not asserted.
    #[test]
    fn par_spmv_matches_seq_above_threshold() {
        let nrows = PAR_SPMV_THRESHOLD + 2_000;
        let ncols = 64;
        let half = ncols / 2;

        // Values vary with the row, so a misindexed row is observable.
        let mut indptr = vec![0u32; nrows + 1];
        let mut indices = Vec::with_capacity(2 * nrows);
        let mut data = Vec::with_capacity(2 * nrows);
        for i in 0..nrows {
            indices.push((i % half) as u32);
            data.push(i as f64 * 0.5 + 1.0);
            indices.push((half + i % half) as u32);
            data.push(i as f64 * -0.25 - 2.0);
            indptr[i + 1] = indptr[i] + 2;
        }
        let block = CsrBlock {
            indptr,
            indices,
            data,
            nrows,
            ncols,
        };

        let x: Vec<f64> = (0..ncols).map(|j| (j as f64).sin()).collect();
        let base: Vec<f64> = (0..nrows).map(|i| i as f64 * 1e-3).collect();

        let mut y_par = vec![0.0; nrows];
        block.par_spmv_assign_add(&x, &base, &mut y_par);

        let mut y_seq = vec![0.0; nrows];
        block.seq_spmv_assign_add(&x, &base, &mut y_seq);

        assert_eq!(y_par, y_seq);
    }
}
