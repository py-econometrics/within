//! Observation accumulation kernels for [`CrossTab`](super::CrossTab) construction.
//!
//! Both the dense and sparse paths scan observations once, decoding each into
//! its compact [`Contribution`] via [`PairColumns::decode`]: `w·l_row·l_col` to its
//! cell and `w·l_row²` / `w·l_col²` to the diagonals, where `l` is the channel's
//! loading, so slope channels yield signed cells. Paths are generic over
//! [`Loading`] and monomorphized per pair: intercept channels pass [`Unit`],
//! whose `l ≡ 1` folds the loading math away, so plain pairs keep the
//! pre-slope codegen.

use crate::channel::ChannelPair;
use crate::csr_block::CsrBlock;
use crate::domain::Design;

use super::{to_u32, ActiveLevels};
use crate::domain::Loading as ColumnLoading;

/// Hard cap on the dense accumulator (~40 MB); larger tables always go sparse.
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

/// One observation's Gram contribution: signed cell `w·l_row·l_col` plus its diagonals.
struct Contribution {
    cj: usize,
    ck: usize,
    cell: f64,
    row_diag: f64,
    col_diag: f64,
}

/// Per-observation input columns backing one channel pair: level codes, loadings, and weights.
#[derive(Clone, Copy)]
pub(super) struct PairColumns<'a, Lq: Loading, Lr: Loading> {
    pub(super) row_levels: &'a [u32],
    pub(super) col_levels: &'a [u32],
    pub(super) row_load: Lq,
    pub(super) col_load: Lr,
    pub(super) weights: Option<&'a [f64]>,
}

impl<Lq: Loading, Lr: Loading> PairColumns<'_, Lq, Lr> {
    /// `None` when either level is inactive, so the observation is skipped.
    #[inline]
    fn decode(&self, active: &ActiveLevels, uid: usize) -> Option<Contribution> {
        let cj = active.row_map[self.row_levels[uid] as usize];
        let ck = active.col_map[self.col_levels[uid] as usize];
        if cj == u32::MAX || ck == u32::MAX {
            return None;
        }
        let w = self.weights.map_or(1.0, |w| w[uid]);
        let l_row = self.row_load.at(uid);
        let l_col = self.col_load.at(uid);
        Some(Contribution {
            cj: cj as usize,
            ck: ck as usize,
            cell: w * l_row * l_col,
            row_diag: w * l_row * l_row,
            col_diag: w * l_col * l_col,
        })
    }
}

/// Accumulate into `C` plus diagonals, dispatching dense or sparse by peak transient memory.
pub(super) fn accumulate_cross_block(
    design: &Design<'_>,
    weights: Option<&[f64]>,
    pair: ChannelPair,
    active: &ActiveLevels,
) -> (CsrBlock, Vec<f64>, Vec<f64>) {
    // Dispatching on cell count alone would pick sparse where it uses MORE memory.
    let table_size = active.n_rows.saturating_mul(active.n_cols);
    let dense_cost = table_size.saturating_mul(8);
    let sparse_cost = design.n_obs.saturating_mul(12);
    let go_sparse = table_size > DENSE_TABLE_MAX_ENTRIES && sparse_cost < dense_cost;

    let row_levels = design.frame.level_column(pair.rows.term);
    let col_levels = design.frame.level_column(pair.cols.term);
    let load = |col: ColumnLoading<u32>| {
        col.covariate()
            .map(|&c| design.frame.loading_column(c as usize))
    };
    // One arm per loading combination; closures aren't generic, so the literals repeat.
    match (
        load(design.loading(pair.rows)),
        load(design.loading(pair.cols)),
    ) {
        (None, None) => accumulate(
            PairColumns {
                row_levels,
                col_levels,
                row_load: Unit,
                col_load: Unit,
                weights,
            },
            active,
            go_sparse,
        ),
        (Some(zq), None) => accumulate(
            PairColumns {
                row_levels,
                col_levels,
                row_load: zq,
                col_load: Unit,
                weights,
            },
            active,
            go_sparse,
        ),
        (None, Some(zr)) => accumulate(
            PairColumns {
                row_levels,
                col_levels,
                row_load: Unit,
                col_load: zr,
                weights,
            },
            active,
            go_sparse,
        ),
        (Some(zq), Some(zr)) => accumulate(
            PairColumns {
                row_levels,
                col_levels,
                row_load: zq,
                col_load: zr,
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

/// Dense path: flat `n_rows * n_cols` table with O(1) accumulation per observation.
pub(super) fn accumulate_dense_cross_block<Lq: Loading, Lr: Loading>(
    cols: PairColumns<'_, Lq, Lr>,
    active: &ActiveLevels,
) -> (CsrBlock, Vec<f64>, Vec<f64>) {
    let n_obs = cols.row_levels.len();
    let n_rows = active.n_rows;
    let n_cols = active.n_cols;
    let mut row_diag = vec![0.0f64; n_rows];
    let mut col_diag = vec![0.0f64; n_cols];
    let mut table = vec![0.0f64; n_rows * n_cols];

    for uid in 0..n_obs {
        let Some(o) = cols.decode(active, uid) else {
            continue;
        };
        debug_assert!(o.cj < n_rows && o.ck < n_cols);
        row_diag[o.cj] += o.row_diag;
        col_diag[o.ck] += o.col_diag;
        table[o.cj * n_cols + o.ck] += o.cell;
    }

    let c = CsrBlock::from_dense_table(&table, n_rows, n_cols);
    (c, row_diag, col_diag)
}

/// Sparse path: bucket by row, then dedup each row through a dense `n_cols` workspace.
pub(super) fn accumulate_sparse_cross_block<Lq: Loading, Lr: Loading>(
    cols: PairColumns<'_, Lq, Lr>,
    active: &ActiveLevels,
) -> (CsrBlock, Vec<f64>, Vec<f64>) {
    let n_obs = cols.row_levels.len();
    let n_rows = active.n_rows;
    let n_cols = active.n_cols;
    let mut row_diag = vec![0.0f64; n_rows];
    let mut col_diag = vec![0.0f64; n_cols];

    let mut row_counts = vec![0u32; n_rows];
    for uid in 0..n_obs {
        let Some(o) = cols.decode(active, uid) else {
            continue;
        };
        row_diag[o.cj] += o.row_diag;
        col_diag[o.ck] += o.col_diag;
        row_counts[o.cj] += 1;
    }

    let mut bucket_indptr = vec![0u32; n_rows + 1];
    for i in 0..n_rows {
        bucket_indptr[i + 1] = bucket_indptr[i] + row_counts[i];
    }
    let total_entries = bucket_indptr[n_rows] as usize;

    let mut bucket_cols = vec![0u32; total_entries];
    let mut bucket_vals = vec![0.0f64; total_entries];
    let mut cursor = bucket_indptr[..n_rows].to_vec();
    for uid in 0..n_obs {
        let Some(o) = cols.decode(active, uid) else {
            continue;
        };
        let pos = cursor[o.cj] as usize;
        bucket_cols[pos] = to_u32(o.ck);
        bucket_vals[pos] = o.cell;
        cursor[o.cj] += 1;
    }

    // A signed cell cancelling to 0.0 mid-row re-pushes its column; the duplicate is harmless.
    let mut work = vec![0.0f64; n_cols];
    let mut touched: Vec<u32> = Vec::new();
    let mut c_indptr = vec![0u32; n_rows + 1];
    let mut c_indices = Vec::new();
    let mut c_data = Vec::new();

    for row in 0..n_rows {
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
        nrows: n_rows,
        ncols: n_cols,
    };
    (c, row_diag, col_diag)
}
