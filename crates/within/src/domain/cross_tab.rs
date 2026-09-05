//! Cross-tabulation of a channel pair: the bipartite local Gramian.
//!
//! [`CrossTab`] holds `C` as a [`CsrBlock`] plus its precomputed transpose and
//! the two diagonals (rather than assembling the symmetric block matrix), and
//! supports bipartite connected-components splitting and per-component extraction.
//! Levels use the design's compact positions, with a `local_to_global` map into
//! the full coefficient space.

use crate::channel::ChannelPair;
use crate::csr_block::{to_u32, CsrBlock};
use crate::domain::SolverDesign;

mod accumulate;
use accumulate::accumulate_cross_block;

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

    /// Build one channel pair; diagonals come back separately.
    pub(crate) fn build_for_pair(
        solver_design: &SolverDesign<'_>,
        weights: Option<&[f64]>,
        pair: ChannelPair,
    ) -> (Self, BlockDiagonals, Vec<u32>) {
        let design = solver_design.design();
        let n_rows = design.terms[pair.rows.term].n_levels();
        let n_cols = design.terms[pair.cols.term].n_levels();

        let (c, row_diag, col_diag) =
            accumulate_cross_block(solver_design, weights, pair, n_rows, n_cols);
        let ct = c.transpose();
        let cross_tab = CrossTab { c, ct };
        let diagonals = BlockDiagonals {
            rows: row_diag,
            cols: col_diag,
        };
        let row_base = design.terms[pair.rows.term].column_base(pair.rows.column);
        let col_base = design.terms[pair.cols.term].column_base(pair.cols.column);
        let local_to_global = (0..n_rows)
            .map(|level| to_u32(row_base + level))
            .chain((0..n_cols).map(|level| to_u32(col_base + level)))
            .collect();

        (cross_tab, diagonals, local_to_global)
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
