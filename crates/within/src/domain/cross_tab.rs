//! Cross-tabulation of a channel pair: the bipartite local Gramian.
//!
//! [`CrossTab`] holds `C` as a [`CsrBlock`] plus its precomputed transpose and
//! the two diagonals (rather than assembling the symmetric block matrix), and
//! supports bipartite connected-components splitting and per-component extraction.
//! Levels are stored compactly with a `local_to_global` map for active levels only.

use crate::csr_block::{to_u32, CsrBlock};
use crate::domain::{ChannelPair, Design};

mod accumulate;
use accumulate::accumulate_cross_block;

// ---------------------------------------------------------------------------
// BipartiteComponent / SchurData — supporting types for CrossTab
// ---------------------------------------------------------------------------

/// Compact mapping of active levels for a factor pair.
///
/// Maps global level indices to local (compact) indices for the row and col
/// channels,
/// and provides the local-to-global index vector for the combined domain.
struct ActiveLevels {
    row_map: Vec<u32>,
    n_rows: usize,
    col_map: Vec<u32>,
    n_cols: usize,
    local_to_global: Vec<u32>,
}

/// Scan all observations once and mark which levels are active for each factor.
///
/// Returns `active[f][level]` = true if any observation uses that level of factor f.
pub(crate) fn find_all_active_levels(design: &Design<'_>) -> Vec<Vec<bool>> {
    let mut active: Vec<Vec<bool>> = design
        .terms
        .iter()
        .map(|f| vec![false; f.n_levels])
        .collect();
    // Factor-outer / obs-inner: all writes for a factor land in one `active[f]`
    // buffer before moving on, instead of hopping between `n_factors` buffers
    // on every observation.
    for (f, col) in active.iter_mut().enumerate() {
        for &v in design.frame.level_column(f) {
            col[v as usize] = true;
        }
    }
    active
}

/// Compact mapping of active levels: assigns each active level a 0-based compact
/// index. Returns the global-to-compact map (`u32::MAX` for inactive levels) and
/// the number of active levels.
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

/// Build compact mapping for a channel pair using pre-computed active level
/// flags; `base_rows`/`base_cols` are the channels' global DOF offsets
/// ([`TermMeta::column_base`](crate::domain::TermMeta::column_base)).
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

/// A connected component in a bipartite factor-pair graph.
///
/// Indices are compact (0-based into the parent CrossTab's n_rows / n_cols).
pub(crate) struct BipartiteComponent {
    pub(crate) rows: Vec<usize>,
    pub(crate) cols: Vec<usize>,
}

// ---------------------------------------------------------------------------
// CrossTab — bipartite block representation of a local Gramian
// ---------------------------------------------------------------------------

/// Bipartite block representation of a local Gramian for a single factor pair.
///
/// Stores the cross-tabulation C (and its precomputed transpose C^T),
/// avoiding construction of the full symmetric Gramian CSR. The Gramian has
/// structure `G = [D_q, C; C^T, D_r]` where D_q and D_r are diagonal; those
/// diagonals are build-time-only (see [`BlockDiagonals`]) and are not stored
/// here, since the solve path never reads them.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct CrossTab {
    /// CSR(C): row-block rows (n_rows) x col-block cols (n_cols).
    pub(crate) c: CsrBlock,
    /// CSR(C^T): col-block rows (n_cols) x row-block cols (n_rows). Precomputed via
    /// `c.transpose()`.
    pub(crate) ct: CsrBlock,
}

/// Diagonal blocks `D_q`, `D_r` of a factor-pair Gramian.
///
/// These are consumed only during preconditioner assembly: [`Elimination::new`]
/// folds them into the reduced factor, after which they are never read again.
/// They therefore travel alongside the [`CrossTab`] through the build step
/// rather than being stored on (and serialized with) it.
///
/// [`Elimination::new`]: crate::block_elim
#[derive(Clone)]
pub(crate) struct BlockDiagonals {
    /// Diagonal block for the row factor (length n_rows).
    pub(crate) rows: Vec<f64>,
    /// Diagonal block for the col factor (length n_cols).
    pub(crate) cols: Vec<f64>,
}

impl BlockDiagonals {
    /// Gather the diagonal for a single bipartite component into the flat
    /// per-vertex order `[rows | cols]` that [`CrossTab::neighbors`] indexes,
    /// mirroring the CSR extraction in [`CrossTab::extract_component`].
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

    /// Build a CrossTab for one channel pair using pre-computed active level flags.
    ///
    /// Reuses active levels already determined via `find_all_active_levels`,
    /// avoiding a redundant observation scan.
    ///
    /// Returns the diagonal blocks separately ([`BlockDiagonals`]): they are
    /// build-time-only and so are not stored on the `CrossTab`.
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

    /// Symmetric adjacency of the bipartite Gram over local `[q | r]` node
    /// indexing: q-nodes walk `C`, r-nodes walk `Cᵀ`; neighbor indices come
    /// back in the same `[q | r]` indexing.
    pub(crate) fn neighbors(&self, i: usize) -> impl Iterator<Item = (usize, f64)> + '_ {
        let n_rows = self.n_rows();
        let (block, row, off) = if i < n_rows {
            (&self.c, i, n_rows)
        } else {
            (&self.ct, i - n_rows, 0)
        };
        block.row(row).map(move |(j, v)| (off + j, v))
    }

    /// Find connected components in the bipartite graph defined by C.
    ///
    /// DFS over [`Self::neighbors`]; components as sorted compact row/col indices.
    /// O(n_rows + n_cols + nnz_C).
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

            // Sort for deterministic ordering
            rows.sort_unstable();
            cols.sort_unstable();
            components.push(BipartiteComponent { rows, cols });
        }

        components
    }

    /// Extract a sub-CrossTab for a single bipartite component.
    ///
    /// Remaps row/col indices to the component's local 0-based indexing.
    /// O(nnz in the component).
    ///
    /// `row_remap`/`col_remap` are parent-sized scratch buffers (length `n_rows()` /
    /// `n_cols()`). They must arrive all-`u32::MAX` and are reset to that on exit,
    /// so a single pair can be reused across every component of one parent
    /// instead of allocating fresh per component.
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

        // Build reverse maps: parent compact index -> component compact index.
        for (new_idx, &old_idx) in comp.rows.iter().enumerate() {
            row_remap[old_idx] = to_u32(new_idx);
        }
        for (new_idx, &old_idx) in comp.cols.iter().enumerate() {
            col_remap[old_idx] = to_u32(new_idx);
        }

        // Extract CSR(C): only rows in comp.rows, remap columns
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

        // Reset only the touched entries so the buffers are all-`u32::MAX` again
        // for the next component.
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
