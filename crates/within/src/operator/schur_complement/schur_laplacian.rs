//! Laplacian assembly from fill edges or direct row-workspace accumulation.

use rayon::prelude::*;
use schwarz_precond::SparseMatrix;

use super::elimination::Elimination;
use super::Edge;

/// Assembled Schur complement Laplacian matrix.
pub(super) struct SchurLaplacian {
    pub(super) matrix: SparseMatrix,
}

impl SchurLaplacian {
    /// Build a symmetric CSR Laplacian from fill edges (sort-merge pipeline).
    ///
    /// Used by the approximate path: edges are par-sorted, duplicates merged,
    /// negligible entries dropped, then assembled into CSR with row-sum diagonal.
    pub(super) fn from_edges(mut edges: Vec<Edge>, n_keep: usize) -> Self {
        // Sort by (lo, hi), merge duplicates.
        edges.par_sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

        let mut merged: Vec<Edge> = Vec::with_capacity(edges.len());
        for &(lo, hi, w) in &edges {
            if let Some(last) = merged.last_mut() {
                if last.0 == lo && last.1 == hi {
                    last.2 += w;
                    continue;
                }
            }
            merged.push((lo, hi, w));
        }
        merged.retain(|&(_, _, w)| w > f64::EPSILON);

        Self {
            matrix: Self::edges_to_laplacian_csr(&merged, n_keep),
        }
    }

    /// Build the Schur complement via row-workspace accumulation (exact path).
    ///
    /// Computes `S = D_keep − keep_to_elim · diag(inv_diag_elim) · elim_to_keep`
    /// directly, without materializing intermediate edges. Each keep-block row
    /// scatters into a dense workspace, then extracts non-zeros.
    pub(super) fn from_elimination(elim: &Elimination) -> Self {
        let n_keep = elim.n_keep;
        let inv_diag_elim = &elim.inv_diag_elim;
        let diag_keep = elim.diag_keep;
        let keep_to_elim = elim.keep_to_elim;
        let elim_to_keep = elim.elim_to_keep;

        let rows: Vec<(Vec<u32>, Vec<f64>)> = (0..n_keep)
            .into_par_iter()
            .map_init(
                || (vec![0.0f64; n_keep], Vec::new()),
                |(work, touched), i| {
                    work[i] = diag_keep[i];
                    touched.push(i);

                    let fwd_start = keep_to_elim.indptr[i] as usize;
                    let fwd_end = keep_to_elim.indptr[i + 1] as usize;
                    for fwd_idx in fwd_start..fwd_end {
                        let k = keep_to_elim.indices[fwd_idx] as usize;
                        let scale = keep_to_elim.data[fwd_idx] * inv_diag_elim[k];
                        let bwd_start = elim_to_keep.indptr[k] as usize;
                        let bwd_end = elim_to_keep.indptr[k + 1] as usize;
                        for bwd_idx in bwd_start..bwd_end {
                            let j = elim_to_keep.indices[bwd_idx] as usize;
                            if work[j] == 0.0 && j != i {
                                touched.push(j);
                            }
                            work[j] -= scale * elim_to_keep.data[bwd_idx];
                        }
                    }

                    touched.sort_unstable();
                    let mut row_indices = Vec::new();
                    let mut row_data = Vec::new();
                    for &j in touched.iter() {
                        let v = work[j];
                        if v != 0.0 || j == i {
                            row_indices.push(j as u32);
                            row_data.push(v);
                        }
                        work[j] = 0.0;
                    }
                    touched.clear();

                    (row_indices, row_data)
                },
            )
            .collect();

        let mut s_indptr = vec![0u32; n_keep + 1];
        let mut s_indices = Vec::new();
        let mut s_data = Vec::new();
        for (i, (ri, rd)) in rows.into_iter().enumerate() {
            s_indices.extend_from_slice(&ri);
            s_data.extend_from_slice(&rd);
            s_indptr[i + 1] = s_indices.len() as u32;
        }

        Self {
            matrix: SparseMatrix::new(s_indptr, s_indices, s_data, n_keep),
        }
    }

    /// Build a dense row-major Schur matrix from elimination data.
    ///
    /// Intended for tiny reduced systems where dense factorization is cheaper
    /// than sparse setup.
    #[cfg(test)]
    pub(super) fn dense_from_elimination(elim: &Elimination) -> Vec<f64> {
        let n_keep = elim.n_keep;
        let mut dense = vec![0.0; n_keep * n_keep];

        // Start with the keep-block diagonal.
        for i in 0..n_keep {
            dense[i * n_keep + i] = elim.diag_keep[i];
        }

        let inv_diag_elim = &elim.inv_diag_elim;
        let keep_to_elim = elim.keep_to_elim;
        let elim_to_keep = elim.elim_to_keep;

        // S = D_keep - keep_to_elim * diag(inv_diag_elim) * elim_to_keep
        for i in 0..n_keep {
            let fwd_start = keep_to_elim.indptr[i] as usize;
            let fwd_end = keep_to_elim.indptr[i + 1] as usize;
            for fwd_idx in fwd_start..fwd_end {
                let k = keep_to_elim.indices[fwd_idx] as usize;
                let scale = keep_to_elim.data[fwd_idx] * inv_diag_elim[k];
                let bwd_start = elim_to_keep.indptr[k] as usize;
                let bwd_end = elim_to_keep.indptr[k + 1] as usize;
                for bwd_idx in bwd_start..bwd_end {
                    let j = elim_to_keep.indices[bwd_idx] as usize;
                    dense[i * n_keep + j] -= scale * elim_to_keep.data[bwd_idx];
                }
            }
        }

        dense
    }

    /// Build the anchored top-left Schur minor `(n_keep-1) x (n_keep-1)` in row-major.
    ///
    /// This is the matrix actually factored by dense anchored Cholesky, so building
    /// it directly avoids allocating a full `n_keep x n_keep` dense Schur matrix.
    pub(super) fn anchored_minor_from_elimination(elim: &Elimination) -> Vec<f64> {
        let n_keep = elim.n_keep;
        if n_keep <= 1 {
            return Vec::new();
        }

        let m = n_keep - 1;
        let mut dense_minor = vec![0.0; m * m];

        // Start with the kept diagonal block on anchored rows/cols.
        for i in 0..m {
            dense_minor[i * m + i] = elim.diag_keep[i];
        }

        let inv_diag_elim = &elim.inv_diag_elim;
        let keep_to_elim = elim.keep_to_elim;
        let elim_to_keep = elim.elim_to_keep;

        // S_minor = D_keep_minor - keep_to_elim_minor * inv(D_elim) * elim_to_keep_minor
        for i in 0..m {
            let fwd_start = keep_to_elim.indptr[i] as usize;
            let fwd_end = keep_to_elim.indptr[i + 1] as usize;
            for fwd_idx in fwd_start..fwd_end {
                let k = keep_to_elim.indices[fwd_idx] as usize;
                let scale = keep_to_elim.data[fwd_idx] * inv_diag_elim[k];
                let bwd_start = elim_to_keep.indptr[k] as usize;
                let bwd_end = elim_to_keep.indptr[k + 1] as usize;
                for bwd_idx in bwd_start..bwd_end {
                    let j = elim_to_keep.indices[bwd_idx] as usize;
                    if j < m {
                        dense_minor[i * m + j] -= scale * elim_to_keep.data[bwd_idx];
                    }
                }
            }
        }

        dense_minor
    }

    /// Convert merged upper-triangular edge list to symmetric CSR Laplacian.
    ///
    /// Uses a single flat buffer instead of per-row `Vec`s. Each row gets its
    /// diagonal at slot 0, off-diagonal entries via cursors, then a
    /// sort-and-rotate pass places the diagonal in its sorted position.
    fn edges_to_laplacian_csr(edges: &[Edge], n_keep: usize) -> SparseMatrix {
        // Count off-diagonal entries per row, accumulate diagonal weights.
        let mut offdiag_count = vec![0u32; n_keep];
        let mut diag = vec![0.0f64; n_keep];
        for &(lo, hi, w) in edges {
            offdiag_count[lo as usize] += 1;
            offdiag_count[hi as usize] += 1;
            diag[lo as usize] += w;
            diag[hi as usize] += w;
        }

        // Build row offsets (each row = 1 diagonal + off-diag entries).
        let mut offsets = vec![0u32; n_keep + 1];
        for i in 0..n_keep {
            offsets[i + 1] = offsets[i] + offdiag_count[i] + 1;
        }
        let total_nnz = offsets[n_keep] as usize;
        let mut buf: Vec<(u32, f64)> = vec![(0, 0.0); total_nnz];

        // Place diagonal at slot 0 of each row, fill off-diagonal via cursors.
        let mut cursors: Vec<u32> = offsets[..n_keep].to_vec();
        for (i, cur) in cursors.iter_mut().enumerate() {
            buf[*cur as usize] = (i as u32, diag[i]);
            *cur += 1;
        }
        for &(lo, hi, w) in edges {
            let lo = lo as usize;
            let hi = hi as usize;
            buf[cursors[lo] as usize] = (hi as u32, -w);
            cursors[lo] += 1;
            buf[cursors[hi] as usize] = (lo as u32, -w);
            cursors[hi] += 1;
        }

        // Sort each row's off-diagonal portion, merge duplicates, place diagonal.
        for i in 0..n_keep {
            let start = offsets[i] as usize;
            let end = offsets[i + 1] as usize;
            if end - start <= 1 {
                continue;
            }
            buf[start + 1..end].sort_unstable_by(|a, b| a.0.cmp(&b.0));
            let diag_col = i as u32;
            let mut write = start + 1;
            let mut read = start + 1;
            while read < end {
                let col = buf[read].0;
                let mut w = buf[read].1;
                read += 1;
                while read < end && buf[read].0 == col {
                    w += buf[read].1;
                    read += 1;
                }
                if w.abs() > f64::EPSILON {
                    buf[write] = (col, w);
                    write += 1;
                }
            }
            let offdiag_end = write;
            let diag_pos =
                buf[start + 1..offdiag_end].partition_point(|e| e.0 < diag_col) + start + 1;
            buf[start..offdiag_end].rotate_left(1);
            let target = diag_pos - 1;
            if target < offdiag_end - 1 {
                buf[target..offdiag_end].rotate_right(1);
            }
            offsets[i + 1] = offdiag_end as u32;
        }
        for i in 1..=n_keep {
            offsets[i] = offsets[i].max(offsets[i - 1]);
        }

        let final_nnz = offsets[n_keep] as usize;
        let mut indices = Vec::with_capacity(final_nnz);
        let mut data = Vec::with_capacity(final_nnz);
        for &(col, val) in &buf[..final_nnz] {
            indices.push(col);
            data.push(val);
        }

        SparseMatrix::new(offsets, indices, data, n_keep)
    }
}
