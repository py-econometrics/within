//! CSR assembly helpers and compact index mapping for Gramian construction.

use schwarz_precond::SparseMatrix;

use super::super::csr_block::CsrBlock;
use crate::domain::WeightedDesign;
use crate::observation::ObservationStore;

// ---------------------------------------------------------------------------
// CSR assembly helper
// ---------------------------------------------------------------------------

/// Build a symmetric CSR matrix from (row, col, value) entries.
///
/// `emit_entries` is called twice with a callback:
///   1. Counting pass — the callback tallies row counts
///   2. Fill pass — the callback writes (col, value) into CSR arrays
///
/// After filling, each row is sorted by column index (insertion sort).
pub(crate) fn build_symmetric_csr(
    n: usize,
    mut emit_entries: impl FnMut(&mut dyn FnMut(usize, usize, f64)),
) -> SparseMatrix {
    // Pass 1: count NNZ per row
    let mut row_counts = vec![0u32; n];
    emit_entries(&mut |row, _col, _val| {
        row_counts[row] += 1;
    });

    // Build indptr via prefix sum
    let mut indptr = vec![0u32; n + 1];
    for i in 0..n {
        indptr[i + 1] = indptr[i] + row_counts[i];
    }
    let nnz = indptr[n] as usize;

    // Pass 2: fill indices + data using indptr copy as write cursors
    let mut cursor = indptr[..n].to_vec();
    let mut indices = vec![0u32; nnz];
    let mut data = vec![0.0f64; nnz];
    emit_entries(&mut |row, col, val| {
        let pos = cursor[row] as usize;
        indices[pos] = col as u32;
        data[pos] = val;
        cursor[row] += 1;
    });

    // Sort indices within each row by column index.
    // Use indirect permutation sort for large rows to avoid O(n^2) insertion sort.
    for row in 0..n {
        let start = indptr[row] as usize;
        let end = indptr[row + 1] as usize;
        let len = end - start;
        if len <= 1 {
            continue;
        }
        let row_idx = &mut indices[start..end];
        let row_data = &mut data[start..end];
        if len <= 64 {
            // Insertion sort for small rows (cache-friendly, low overhead)
            for i in 1..len {
                let key_col = row_idx[i];
                let key_val = row_data[i];
                let mut j = i;
                while j > 0 && row_idx[j - 1] > key_col {
                    row_idx[j] = row_idx[j - 1];
                    row_data[j] = row_data[j - 1];
                    j -= 1;
                }
                row_idx[j] = key_col;
                row_data[j] = key_val;
            }
        } else {
            // Build permutation, sort it, then apply in-place
            let mut perm: Vec<u32> = (0..len as u32).collect();
            perm.sort_unstable_by_key(|&i| row_idx[i as usize]);
            // Apply permutation to both arrays via cycle-following
            let mut visited = vec![false; len];
            for i in 0..len {
                if visited[i] || perm[i] as usize == i {
                    continue;
                }
                let saved_idx = row_idx[i];
                let saved_data = row_data[i];
                let mut j = i;
                loop {
                    visited[j] = true;
                    let src = perm[j] as usize;
                    if src == i {
                        row_idx[j] = saved_idx;
                        row_data[j] = saved_data;
                        break;
                    }
                    row_idx[j] = row_idx[src];
                    row_data[j] = row_data[src];
                    j = src;
                }
            }
        }
    }

    SparseMatrix::new(indptr, indices, data, n)
}

// ---------------------------------------------------------------------------
// CompactIndexMaps — maps full-size level indices to compact 0-based indices
// ---------------------------------------------------------------------------

/// Maps full-size factor level indices to compact 0-based indices for a
/// connected component. Levels not present in the component map to `u32::MAX`.
pub(crate) struct CompactIndexMaps {
    /// Full-size -> compact index for factor q. `u32::MAX` = inactive.
    pub(crate) q_compact: Vec<u32>,
    /// Full-size -> compact index for factor r. `u32::MAX` = inactive.
    pub(crate) r_compact: Vec<u32>,
    pub(crate) n_active_q: usize,
    pub(crate) n_active_r: usize,
}

impl CompactIndexMaps {
    pub(crate) fn build(
        factors: &[crate::observation::FactorMeta],
        q: usize,
        r: usize,
        global_indices: &[u32],
    ) -> Self {
        let fq = &factors[q];
        let fr = &factors[r];
        let mut q_compact = vec![u32::MAX; fq.n_levels];
        let mut r_compact = vec![u32::MAX; fr.n_levels];

        let mut n_active_q = 0usize;
        for &gi in global_indices {
            let gi = gi as usize;
            if gi >= fq.offset && gi < fq.offset + fq.n_levels {
                let j = gi - fq.offset;
                q_compact[j] = n_active_q as u32;
                n_active_q += 1;
            }
        }
        let mut n_active_r = 0usize;
        for &gi in global_indices {
            let gi = gi as usize;
            if gi >= fr.offset && gi < fr.offset + fr.n_levels {
                let k = gi - fr.offset;
                r_compact[k] = n_active_r as u32;
                n_active_r += 1;
            }
        }

        CompactIndexMaps {
            q_compact,
            r_compact,
            n_active_q,
            n_active_r,
        }
    }

    pub(crate) fn n_local(&self) -> usize {
        self.n_active_q + self.n_active_r
    }
}

// ---------------------------------------------------------------------------
// accumulate_cross_block — shared observation accumulation for CrossTab
// ---------------------------------------------------------------------------

/// Max entries in a flat dense table (~40 MB at 8 bytes each).
pub(super) const DENSE_TABLE_MAX_ENTRIES: usize = 5_000_000;

/// Accumulate observation weights into a cross-tabulation block C plus diagonals.
///
/// Shared by `CrossTab::build` (component-scoped) and `CrossTab::build_for_pair`
/// (full pair). Observations whose compact index is `u32::MAX` are skipped.
///
/// - Dense path (n_q * n_r <= 5M): flat table with O(1) accumulation per observation.
/// - Sparse path: two-pass bucket + workspace-based dedup per row.
pub(crate) fn accumulate_cross_block<S: ObservationStore>(
    design: &WeightedDesign<S>,
    q: usize,
    r: usize,
    q_compact: &[u32],
    r_compact: &[u32],
    n_q: usize,
    n_r: usize,
) -> (CsrBlock, Vec<f64>, Vec<f64>) {
    let n_unique = design.store.n_unique();
    let mut diag_q = vec![0.0f64; n_q];
    let mut diag_r = vec![0.0f64; n_r];
    let table_size = n_q * n_r;

    let c = if table_size <= DENSE_TABLE_MAX_ENTRIES {
        // Dense path: flat table with O(1) accumulation per observation.
        let mut table = vec![0.0f64; table_size];

        for uid in 0..n_unique {
            let j = design.store.unique_level(uid, q) as usize;
            let k = design.store.unique_level(uid, r) as usize;
            let cj = q_compact[j];
            let ck = r_compact[k];
            if cj == u32::MAX || ck == u32::MAX {
                continue;
            }
            let w = design.uid_weight(uid);
            diag_q[cj as usize] += w;
            diag_r[ck as usize] += w;
            table[cj as usize * n_r + ck as usize] += w;
        }

        CsrBlock::from_dense_table(&table, n_q, n_r)
    } else {
        // Sparse path: two-pass bucket + workspace-dedup.
        //
        // Bucket observations by row in two passes (count + fill), then use
        // a dense workspace of size n_r to accumulate and deduplicate each
        // row. The workspace sort is on unique columns only (n_r_active << len).

        // Pass 1: accumulate diags + count entries per row
        let mut row_counts = vec![0u32; n_q];
        for uid in 0..n_unique {
            let j = design.store.unique_level(uid, q) as usize;
            let k = design.store.unique_level(uid, r) as usize;
            let cj = q_compact[j];
            let ck = r_compact[k];
            if cj == u32::MAX || ck == u32::MAX {
                continue;
            }
            let w = design.uid_weight(uid);
            diag_q[cj as usize] += w;
            diag_r[ck as usize] += w;
            row_counts[cj as usize] += 1;
        }

        // Build row-pointer array for the unsorted bucket CSR
        let mut bucket_indptr = vec![0u32; n_q + 1];
        for i in 0..n_q {
            bucket_indptr[i + 1] = bucket_indptr[i] + row_counts[i];
        }
        let total_entries = bucket_indptr[n_q] as usize;

        // Pass 2: fill per-row buckets (col + weight only, no row index)
        let mut bucket_cols = vec![0u32; total_entries];
        let mut bucket_vals = vec![0.0f64; total_entries];
        let mut cursor = bucket_indptr[..n_q].to_vec();
        for uid in 0..n_unique {
            let j = design.store.unique_level(uid, q) as usize;
            let k = design.store.unique_level(uid, r) as usize;
            let cj = q_compact[j];
            let ck = r_compact[k];
            if cj == u32::MAX || ck == u32::MAX {
                continue;
            }
            let w = design.uid_weight(uid);
            let pos = cursor[cj as usize] as usize;
            bucket_cols[pos] = ck;
            bucket_vals[pos] = w;
            cursor[cj as usize] += 1;
        }

        // Pass 3: workspace-based dedup per row.
        // Accumulate into work[col], track touched columns, sort only the
        // unique set, then emit into final CSR.
        let mut work = vec![0.0f64; n_r];
        let mut touched: Vec<u32> = Vec::new();
        let mut c_indptr = vec![0u32; n_q + 1];
        let mut c_indices = Vec::new();
        let mut c_data = Vec::new();

        for row in 0..n_q {
            let start = bucket_indptr[row] as usize;
            let end = bucket_indptr[row + 1] as usize;
            for idx in start..end {
                let col = bucket_cols[idx] as usize;
                if work[col] == 0.0 {
                    touched.push(col as u32);
                }
                work[col] += bucket_vals[idx];
            }
            touched.sort_unstable();
            for &col in &touched {
                let v = work[col as usize];
                if v != 0.0 {
                    c_indices.push(col);
                    c_data.push(v);
                }
                work[col as usize] = 0.0;
            }
            c_indptr[row + 1] = c_indices.len() as u32;
            touched.clear();
        }

        CsrBlock {
            indptr: c_indptr,
            indices: c_indices,
            data: c_data,
            nrows: n_q,
            ncols: n_r,
        }
    };

    (c, diag_q, diag_r)
}
