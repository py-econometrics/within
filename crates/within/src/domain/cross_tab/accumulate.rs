//! Observation accumulation kernels for [`CrossTab`](super::CrossTab) construction.
//!
//! Both the dense and sparse paths scan observations once, decoding each into
//! its compact `(cj, ck, weight)` via [`decode_obs`], and accumulate the
//! cross-tabulation block `C` plus the two diagonals.

use crate::csr_block::CsrBlock;
use crate::domain::Design;
use crate::observation::{factor_columns, level_at, Store};

use super::{to_u32, ActiveLevels};

/// Max entries in a flat dense cross-tab accumulator (~40 MB at 8 bytes each).
/// Absolute hard cap on the dense path: tables larger than this always go
/// sparse, regardless of the cost comparison in `accumulate_cross_block`.
const DENSE_TABLE_MAX_ENTRIES: usize = 5_000_000;

/// Decode observation `uid` into its compact `(cj, ck, weight)`, or `None` if
/// either factor level is inactive (compact index `u32::MAX`) and the
/// observation should be skipped.
#[inline]
fn decode_obs<S: Store>(
    design: &Design<S>,
    weights: Option<&[f64]>,
    cols: &[Option<&[u32]>],
    q: usize,
    r: usize,
    active: &ActiveLevels,
    uid: usize,
) -> Option<(u32, u32, f64)> {
    let j = level_at(&design.store, cols[q], uid, q);
    let k = level_at(&design.store, cols[r], uid, r);
    let cj = active.q_map[j];
    let ck = active.r_map[k];
    if cj == u32::MAX || ck == u32::MAX {
        return None;
    }
    let w = weights.map_or(1.0, |w| w[uid]);
    Some((cj, ck, w))
}

/// Accumulate observation weights into a cross-tabulation block C plus diagonals.
///
/// Used by `CrossTab::build_for_pair_with_active`. Observations whose compact
/// index is `u32::MAX` are skipped.
///
/// Dispatches to a dense or sparse path by comparing their estimated peak
/// transient memory, with a hard dense-table ceiling.
pub(super) fn accumulate_cross_block<S: Store>(
    design: &Design<S>,
    weights: Option<&[f64]>,
    q: usize,
    r: usize,
    active: &ActiveLevels,
) -> (CsrBlock, Vec<f64>, Vec<f64>) {
    // Cost-based dispatch. Both paths produce a bit-identical CSR `C`, `diag_q`,
    // and `diag_r`; only their peak transient allocation differs:
    //   - dense path: a flat `n_q * n_r` f64 table -> ~8 * n_q * n_r bytes;
    //   - sparse path: per-observation buckets of (u32 col, f64 weight) sized
    //     by valid observations (approximated by n_obs) -> ~12 * n_obs bytes.
    // Picking sparse purely on cell count makes the sparse path use MORE memory
    // than the dense table it replaces whenever n_obs >> cells. So use the dense
    // table unconditionally up to `DENSE_TABLE_MAX_ENTRIES` (for small tables
    // memory is a non-issue and the flat table is the faster build); only past
    // that cap do we consider sparse, and even then only when its bucket cost is
    // actually below the dense table cost -- otherwise the large-n_obs blowup
    // this guards against would make sparse the *more* expensive choice, so we
    // keep dense. Saturating math keeps the comparison well-defined even for
    // enormous level counts.
    let table_size = active.n_q.saturating_mul(active.n_r);
    let dense_cost = table_size.saturating_mul(8);
    let sparse_cost = design.store.n_obs().saturating_mul(12);
    let go_sparse = table_size > DENSE_TABLE_MAX_ENTRIES && sparse_cost < dense_cost;
    if go_sparse {
        accumulate_sparse_cross_block(design, weights, q, r, active)
    } else {
        accumulate_dense_cross_block(design, weights, q, r, active)
    }
}

/// Dense path: flat `n_q * n_r` table with O(1) accumulation per observation.
fn accumulate_dense_cross_block<S: Store>(
    design: &Design<S>,
    weights: Option<&[f64]>,
    q: usize,
    r: usize,
    active: &ActiveLevels,
) -> (CsrBlock, Vec<f64>, Vec<f64>) {
    let n_obs = design.store.n_obs();
    let n_q = active.n_q;
    let n_r = active.n_r;
    let cols = factor_columns(&design.store);
    let mut diag_q = vec![0.0f64; n_q];
    let mut diag_r = vec![0.0f64; n_r];
    let mut table = vec![0.0f64; n_q * n_r];

    for uid in 0..n_obs {
        let Some((cj, ck, w)) = decode_obs(design, weights, &cols, q, r, active, uid) else {
            continue;
        };
        debug_assert!((cj as usize) < n_q && (ck as usize) < n_r);
        diag_q[cj as usize] += w;
        diag_r[ck as usize] += w;
        table[cj as usize * n_r + ck as usize] += w;
    }

    let c = CsrBlock::from_dense_table(&table, n_q, n_r);
    (c, diag_q, diag_r)
}

/// Sparse path: two-pass bucket + workspace-based dedup per row.
///
/// Bucket observations by row in two passes (count + fill), then use
/// a dense workspace of size n_r to accumulate and deduplicate each
/// row. The workspace sort is on unique columns only (n_r_active << len).
fn accumulate_sparse_cross_block<S: Store>(
    design: &Design<S>,
    weights: Option<&[f64]>,
    q: usize,
    r: usize,
    active: &ActiveLevels,
) -> (CsrBlock, Vec<f64>, Vec<f64>) {
    let n_obs = design.store.n_obs();
    let n_q = active.n_q;
    let n_r = active.n_r;
    let cols = factor_columns(&design.store);
    let mut diag_q = vec![0.0f64; n_q];
    let mut diag_r = vec![0.0f64; n_r];

    // Pass 1: accumulate diags + count entries per row
    let mut row_counts = vec![0u32; n_q];
    for uid in 0..n_obs {
        let Some((cj, ck, w)) = decode_obs(design, weights, &cols, q, r, active, uid) else {
            continue;
        };
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
    for uid in 0..n_obs {
        let Some((cj, ck, w)) = decode_obs(design, weights, &cols, q, r, active, uid) else {
            continue;
        };
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
                touched.push(to_u32(col));
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
        c_indptr[row + 1] = to_u32(c_indices.len());
        touched.clear();
    }

    let c = CsrBlock {
        indptr: c_indptr,
        indices: c_indices,
        data: c_data,
        nrows: n_q,
        ncols: n_r,
    };
    (c, diag_q, diag_r)
}
