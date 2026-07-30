//! Cross-tabulation of a channel pair: the bipartite local Gramian.
//!
//! [`CrossTab`] holds `C` as a [`CsrBlock`] plus its precomputed transpose and
//! the two diagonals (rather than assembling the symmetric block matrix), and
//! supports bipartite connected-components splitting and per-component extraction.
//! Levels are stored compactly with a `local_to_global` map for active levels only.

use crate::channel::ChannelPair;
use crate::csr_block::{to_u32, CsrBlock};
use crate::domain::Design;

mod accumulate;
use accumulate::accumulate_cross_block;

/// Compact mapping of active levels for a factor pair, plus its local-to-global vector.
struct ActiveLevels {
    row_map: Vec<u32>,
    n_rows: usize,
    col_map: Vec<u32>,
    n_cols: usize,
    local_to_global: Vec<u32>,
}

/// Scan observations once, marking `active[f][level]` for every level any observation uses.
pub(crate) fn find_all_active_levels(design: &Design<'_>) -> Vec<Vec<bool>> {
    let mut active: Vec<Vec<bool>> = design
        .terms
        .iter()
        .map(|f| vec![false; f.n_levels])
        .collect();
    // Factor-outer/obs-inner: all writes for a factor land in one `active[f]` buffer.
    for (f, col) in active.iter_mut().enumerate() {
        for &v in design.frame.level_column(f) {
            col[v as usize] = true;
        }
    }
    active
}

/// Compact-index each active level, returning the global-to-compact map and active count.
fn compact_map(active: &[bool]) -> (Vec<u32>, usize) {
    let mut map = vec![u32::MAX; active.len()];
    let mut n = 0u32;
    for (j, &a) in active.iter().enumerate() {
        if a {
            map[j] = n;
            n += 1;
        }
    }
    (map, n as usize)
}

/// `base_rows`/`base_cols` are the channels' global DOF offsets.
fn build_compact_mapping(
    active_rows: &[bool],
    active_cols: &[bool],
    base_rows: usize,
    base_cols: usize,
) -> Option<ActiveLevels> {
    let (row_map, n_rows) = compact_map(active_rows);
    let (col_map, n_cols) = compact_map(active_cols);

    if n_rows == 0 || n_cols == 0 {
        return None;
    }

    let mut local_to_global = Vec::with_capacity(n_rows + n_cols);
    for (j, &a) in active_rows.iter().enumerate() {
        if a {
            local_to_global.push(to_u32(base_rows + j));
        }
    }
    for (k, &a) in active_cols.iter().enumerate() {
        if a {
            local_to_global.push(to_u32(base_cols + k));
        }
    }

    Some(ActiveLevels {
        row_map,
        n_rows,
        col_map,
        n_cols,
        local_to_global,
    })
}

/// A connected component in a bipartite factor-pair graph, in compact 0-based parent indices.
pub(crate) struct BipartiteComponent {
    pub(crate) rows: Vec<usize>,
    pub(crate) cols: Vec<usize>,
}

/// Stores `C` and `Cᵀ` only; the solve path never reads the diagonals of `G`.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct CrossTab {
    /// CSR(C): row-block rows (n_rows) x col-block cols (n_cols).
    pub(crate) c: CsrBlock,
    /// CSR(Cᵀ): `n_cols` x `n_rows`, precomputed via `c.transpose()`.
    pub(crate) ct: CsrBlock,
}

/// Folded into the reduced factor during assembly and never read again, so not serialized.
#[derive(Clone)]
pub(crate) struct BlockDiagonals {
    /// Diagonal block for the row factor (length n_rows).
    pub(crate) rows: Vec<f64>,
    /// Diagonal block for the col factor (length n_cols).
    pub(crate) cols: Vec<f64>,
}

impl BlockDiagonals {
    /// Gather a component's diagonal into the flat `[rows | cols]` order `neighbors` indexes.
    pub(crate) fn extract_component(&self, comp: &BipartiteComponent) -> Vec<f64> {
        comp.rows
            .iter()
            .map(|&i| self.rows[i])
            .chain(comp.cols.iter().map(|&i| self.cols[i]))
            .collect()
    }
}

impl CrossTab {
    /// Number of rows in the row block.
    pub(crate) fn n_rows(&self) -> usize {
        self.c.nrows
    }

    /// Number of rows in the col block.
    pub(crate) fn n_cols(&self) -> usize {
        self.c.ncols
    }

    /// Total number of DOFs (n_rows + n_cols).
    pub(crate) fn n_local(&self) -> usize {
        self.c.nrows + self.c.ncols
    }

    /// Reuses pre-computed active flags instead of rescanning; diagonals come back separately.
    pub(crate) fn build_for_pair_with_active(
        design: &Design<'_>,
        weights: Option<&[f64]>,
        pair: ChannelPair,
        all_active: &[Vec<bool>],
    ) -> Option<(Self, BlockDiagonals, Vec<u32>)> {
        let active = build_compact_mapping(
            &all_active[pair.rows.term],
            &all_active[pair.cols.term],
            design.terms[pair.rows.term].column_base(pair.rows.column),
            design.terms[pair.cols.term].column_base(pair.cols.column),
        )?;

        let (c, row_diag, col_diag) = accumulate_cross_block(design, weights, pair, &active);
        let ct = c.transpose();
        let cross_tab = CrossTab { c, ct };
        let diagonals = BlockDiagonals {
            rows: row_diag,
            cols: col_diag,
        };
        Some((cross_tab, diagonals, active.local_to_global))
    }

    /// Symmetric adjacency over local `[q | r]` indexing: q-nodes walk `C`, r-nodes walk `Cᵀ`.
    pub(crate) fn neighbors(&self, i: usize) -> impl Iterator<Item = (usize, f64)> + '_ {
        let n_rows = self.n_rows();
        let (block, row, off) = if i < n_rows {
            (&self.c, i, n_rows)
        } else {
            (&self.ct, i - n_rows, 0)
        };
        block.row(row).map(move |(j, v)| (off + j, v))
    }

    /// Connected components by DFS over [`Self::neighbors`]; O(n_rows + n_cols + nnz).
    pub(crate) fn bipartite_connected_components(&self) -> Vec<BipartiteComponent> {
        let n_rows = self.n_rows();
        let mut visited = vec![false; self.n_local()];
        let mut components = Vec::new();
        let mut stack = Vec::new();

        for start in 0..self.n_local() {
            if visited[start] {
                continue;
            }
            visited[start] = true;
            stack.push(start);
            let mut rows = Vec::new();
            let mut cols = Vec::new();

            while let Some(node) = stack.pop() {
                if node < n_rows {
                    rows.push(node);
                } else {
                    cols.push(node - n_rows);
                }
                for (j, _) in self.neighbors(node) {
                    if !visited[j] {
                        visited[j] = true;
                        stack.push(j);
                    }
                }
            }

            rows.sort_unstable();
            cols.sort_unstable();
            components.push(BipartiteComponent { rows, cols });
        }

        components
    }

    /// `row_remap`/`col_remap` must arrive all-`u32::MAX` and are reset on exit.
    pub(crate) fn extract_component(
        &self,
        comp: &BipartiteComponent,
        row_remap: &mut [u32],
        col_remap: &mut [u32],
    ) -> Self {
        let n_rows = comp.rows.len();
        let n_cols = comp.cols.len();
        debug_assert_eq!(row_remap.len(), self.n_rows());
        debug_assert_eq!(col_remap.len(), self.n_cols());

        for (new_idx, &old_idx) in comp.rows.iter().enumerate() {
            row_remap[old_idx] = to_u32(new_idx);
        }
        for (new_idx, &old_idx) in comp.cols.iter().enumerate() {
            col_remap[old_idx] = to_u32(new_idx);
        }

        let mut c_indptr = vec![0u32; n_rows + 1];
        let mut c_indices = Vec::new();
        let mut c_data = Vec::new();
        for (new_row, &old_row) in comp.rows.iter().enumerate() {
            let start = self.c.indptr[old_row] as usize;
            let end = self.c.indptr[old_row + 1] as usize;
            for idx in start..end {
                let old_rj = self.c.indices[idx] as usize;
                let new_rj = col_remap[old_rj];
                if new_rj != u32::MAX {
                    c_indices.push(new_rj);
                    c_data.push(self.c.data[idx]);
                }
            }
            c_indptr[new_row + 1] = to_u32(c_indices.len());
        }

        let c = CsrBlock {
            indptr: c_indptr,
            indices: c_indices,
            data: c_data,
            nrows: n_rows,
            ncols: n_cols,
        };
        let ct = c.transpose();

        for &old_idx in &comp.rows {
            row_remap[old_idx] = u32::MAX;
        }
        for &old_idx in &comp.cols {
            col_remap[old_idx] = u32::MAX;
        }

        CrossTab { c, ct }
    }
}

#[cfg(test)]
mod tests;
