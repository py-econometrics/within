use rayon::prelude::*;

use crate::operator::csr_block::CsrBlock;

/// Minimum number of rows to trigger parallel back-substitution.
const PAR_BACKSUB_THRESHOLD: usize = 10_000;
const PAR_BACKSUB_CHUNK: usize = 4096;

/// Negate elements in `slice[from..]`.
#[inline]
pub(super) fn negate_block(slice: &mut [f64], from: usize) {
    for val in slice[from..].iter_mut() {
        *val = -*val;
    }
}

/// Subtract the mean of `slice[..n]` from those `n` elements.
#[inline]
pub(super) fn subtract_mean(slice: &mut [f64], n: usize) {
    let mean: f64 = slice[..n].iter().sum::<f64>() / n as f64;
    for val in slice[..n].iter_mut() {
        *val -= mean;
    }
}

/// Back-substitute for the eliminated block.
pub(super) fn backsub_block(
    sol_output: &mut [f64],
    rhs_slice: &[f64],
    cross_matrix: &CsrBlock,
    inv_diag: &[f64],
    sol_source: &[f64],
) {
    let n = sol_output.len();
    if n > PAR_BACKSUB_THRESHOLD {
        sol_output
            .par_chunks_mut(PAR_BACKSUB_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let row_start = chunk_idx * PAR_BACKSUB_CHUNK;
                for (local_i, si) in chunk.iter_mut().enumerate() {
                    let i = row_start + local_i;
                    let start = cross_matrix.indptr[i] as usize;
                    let end = cross_matrix.indptr[i + 1] as usize;
                    let mut sum = 0.0;
                    for idx in start..end {
                        let j = cross_matrix.indices[idx] as usize;
                        sum += cross_matrix.data[idx] * sol_source[j];
                    }
                    *si = inv_diag[i] * (rhs_slice[i] + sum);
                }
            });
    } else {
        for i in 0..n {
            let start = cross_matrix.indptr[i] as usize;
            let end = cross_matrix.indptr[i + 1] as usize;
            let mut sum = 0.0;
            for idx in start..end {
                let j = cross_matrix.indices[idx] as usize;
                sum += cross_matrix.data[idx] * sol_source[j];
            }
            sol_output[i] = inv_diag[i] * (rhs_slice[i] + sum);
        }
    }
}
