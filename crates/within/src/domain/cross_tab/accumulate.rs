//! Observation accumulation kernels for [`CrossTab`](super::CrossTab) construction.
//!
//! Both the dense and sparse paths scan observations once, decoding each into
//! its compact [`Contribution`] via [`PairColumns::decode`]: `w·lq·lr` to its
//! cell and `w·lq²` / `w·lr²` to the diagonals, where `l` is the channel's
//! loading, so slope channels yield signed cells. Paths are generic over
//! [`Loading`] and monomorphized per pair: intercept channels pass [`Unit`],
//! whose `l ≡ 1` folds the loading math away, so plain pairs keep the
//! pre-slope codegen.

use crate::csr_block::CsrBlock;
use crate::domain::{ChannelPair, Design};

use super::{to_u32, ActiveLevels};

/// Max entries in a flat dense cross-tab accumulator (~40 MB at 8 bytes each).
/// Absolute hard cap on the dense path: tables larger than this always go
/// sparse, regardless of the cost comparison in `accumulate_cross_block`.
const DENSE_TABLE_MAX_ENTRIES: usize = 5_000_000;

/// A channel's per-observation loading.
pub(super) trait Loading: Copy {
    fn at(self, uid: usize) -> f64;
}

/// Intercept loading `l ≡ 1`; LLVM folds the resulting `x · 1.0` away.
#[derive(Clone, Copy)]
pub(super) struct Unit;

impl Loading for Unit {
    #[inline]
    fn at(self, _uid: usize) -> f64 {
        1.0
    }
}

impl Loading for &[f64] {
    #[inline]
    fn at(self, uid: usize) -> f64 {
        self[uid]
    }
}

/// One observation's contribution to the pair's Gram: the signed cell
/// `w·lq·lr` plus the diagonals `w·lq²` / `w·lr²`, `l` the channel's loading.
struct Contribution {
    cj: u32,
    ck: u32,
    cell: f64,
    diag_q: f64,
    diag_r: f64,
}

/// The per-observation input columns backing one channel pair: level codes,
/// loadings, and observation weights.
#[derive(Clone, Copy)]
pub(super) struct PairColumns<'a, Lq: Loading, Lr: Loading> {
    pub(super) levels_q: &'a [u32],
    pub(super) levels_r: &'a [u32],
    pub(super) load_q: Lq,
    pub(super) load_r: Lr,
    pub(super) weights: Option<&'a [f64]>,
}

impl<Lq: Loading, Lr: Loading> PairColumns<'_, Lq, Lr> {
    /// Decode observation `uid` into its compact [`Contribution`], or `None`
    /// if either factor level is inactive (compact index `u32::MAX`) and the
    /// observation should be skipped.
    #[inline]
    fn decode(&self, active: &ActiveLevels, uid: usize) -> Option<Contribution> {
        let cj = active.q_map[self.levels_q[uid] as usize];
        let ck = active.r_map[self.levels_r[uid] as usize];
        if cj == u32::MAX || ck == u32::MAX {
            return None;
        }
        let w = self.weights.map_or(1.0, |w| w[uid]);
        let lq = self.load_q.at(uid);
        let lr = self.load_r.at(uid);
        Some(Contribution {
            cj,
            ck,
            cell: w * lq * lr,
            diag_q: w * lq * lq,
            diag_r: w * lr * lr,
        })
    }
}

/// Accumulate observation weights into a cross-tabulation block C plus diagonals.
///
/// Used by `CrossTab::build_for_pair_with_active`. Observations whose compact
/// index is `u32::MAX` are skipped.
///
/// Dispatches to a dense or sparse path by comparing their estimated peak
/// transient memory, with a hard dense-table ceiling.
pub(super) fn accumulate_cross_block(
    design: &Design<'_>,
    weights: Option<&[f64]>,
    pair: ChannelPair,
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
    let sparse_cost = design.n_obs.saturating_mul(12);
    let go_sparse = table_size > DENSE_TABLE_MAX_ENTRIES && sparse_cost < dense_cost;

    let levels_q = design.frame.level_column(pair.q.term);
    let levels_r = design.frame.level_column(pair.r.term);
    let load = |col: Option<usize>| col.map(|c| design.frame.loading_column(c));
    // One arm per monomorphized loading combination; a generic constructor
    // can't express this (closures aren't generic), so the literals repeat.
    match (load(pair.q.loading), load(pair.r.loading)) {
        (None, None) => accumulate(
            PairColumns {
                levels_q,
                levels_r,
                load_q: Unit,
                load_r: Unit,
                weights,
            },
            active,
            go_sparse,
        ),
        (Some(zq), None) => accumulate(
            PairColumns {
                levels_q,
                levels_r,
                load_q: zq,
                load_r: Unit,
                weights,
            },
            active,
            go_sparse,
        ),
        (None, Some(zr)) => accumulate(
            PairColumns {
                levels_q,
                levels_r,
                load_q: Unit,
                load_r: zr,
                weights,
            },
            active,
            go_sparse,
        ),
        (Some(zq), Some(zr)) => accumulate(
            PairColumns {
                levels_q,
                levels_r,
                load_q: zq,
                load_r: zr,
                weights,
            },
            active,
            go_sparse,
        ),
    }
}

/// Size-dispatched accumulation for one monomorphized loading combination.
fn accumulate<Lq: Loading, Lr: Loading>(
    cols: PairColumns<'_, Lq, Lr>,
    active: &ActiveLevels,
    go_sparse: bool,
) -> (CsrBlock, Vec<f64>, Vec<f64>) {
    if go_sparse {
        accumulate_sparse_cross_block(cols, active)
    } else {
        accumulate_dense_cross_block(cols, active)
    }
}

/// Dense path: flat `n_q * n_r` table with O(1) accumulation per observation.
pub(super) fn accumulate_dense_cross_block<Lq: Loading, Lr: Loading>(
    cols: PairColumns<'_, Lq, Lr>,
    active: &ActiveLevels,
) -> (CsrBlock, Vec<f64>, Vec<f64>) {
    let n_obs = cols.levels_q.len();
    let n_q = active.n_q;
    let n_r = active.n_r;
    let mut diag_q = vec![0.0f64; n_q];
    let mut diag_r = vec![0.0f64; n_r];
    let mut table = vec![0.0f64; n_q * n_r];

    for uid in 0..n_obs {
        let Some(o) = cols.decode(active, uid) else {
            continue;
        };
        debug_assert!((o.cj as usize) < n_q && (o.ck as usize) < n_r);
        diag_q[o.cj as usize] += o.diag_q;
        diag_r[o.ck as usize] += o.diag_r;
        table[o.cj as usize * n_r + o.ck as usize] += o.cell;
    }

    let c = CsrBlock::from_dense_table(&table, n_q, n_r);
    (c, diag_q, diag_r)
}

/// Sparse path: two-pass bucket + workspace-based dedup per row.
///
/// Bucket observations by row in two passes (count + fill), then use
/// a dense workspace of size n_r to accumulate and deduplicate each
/// row. The workspace sort is on unique columns only (n_r_active << len).
pub(super) fn accumulate_sparse_cross_block<Lq: Loading, Lr: Loading>(
    cols: PairColumns<'_, Lq, Lr>,
    active: &ActiveLevels,
) -> (CsrBlock, Vec<f64>, Vec<f64>) {
    let n_obs = cols.levels_q.len();
    let n_q = active.n_q;
    let n_r = active.n_r;
    let mut diag_q = vec![0.0f64; n_q];
    let mut diag_r = vec![0.0f64; n_r];

    // Pass 1: accumulate diags + count entries per row
    let mut row_counts = vec![0u32; n_q];
    for uid in 0..n_obs {
        let Some(o) = cols.decode(active, uid) else {
            continue;
        };
        diag_q[o.cj as usize] += o.diag_q;
        diag_r[o.ck as usize] += o.diag_r;
        row_counts[o.cj as usize] += 1;
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
        let Some(o) = cols.decode(active, uid) else {
            continue;
        };
        let pos = cursor[o.cj as usize] as usize;
        bucket_cols[pos] = o.ck;
        bucket_vals[pos] = o.cell;
        cursor[o.cj as usize] += 1;
    }

    // Pass 3: workspace-based dedup per row.
    // Accumulate into work[col], track touched columns, sort the touched
    // set, then emit into final CSR. A signed cell cancelling to exactly 0.0
    // mid-row re-pushes its column; the duplicate is harmless (first emit
    // resets work[col], the second skips the 0.0) and exact-0 cells drop,
    // matching the dense path.
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
