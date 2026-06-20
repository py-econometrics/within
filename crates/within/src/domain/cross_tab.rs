//! Cross-tabulation of a factor pair: the bipartite local Gramian.
//!
//! [`CrossTab`] holds `C` as a [`CsrBlock`] plus its precomputed transpose and
//! the two diagonals (rather than assembling the symmetric block matrix), and
//! supports bipartite connected-components splitting and per-component extraction.
//! Levels are stored compactly with a `local_to_global` map for active levels only.

use crate::csr_block::CsrBlock;
use crate::domain::{Design, FactorMeta};
use crate::observation::{factor_columns, level_at, Store};

mod accumulate;
use accumulate::accumulate_cross_block;

/// Checked `usize -> u32` for CSR indptr/index and compact-level values. A
/// silent `as u32` truncation above `u32::MAX` would corrupt the assembled
/// cross-tab with no diagnostic; these are build-path invariants, so panic
/// loudly instead.
#[inline]
fn to_u32(x: usize) -> u32 {
    u32::try_from(x).expect("CSR index exceeds u32::MAX")
}

// ---------------------------------------------------------------------------
// BipartiteComponent / SchurData — supporting types for CrossTab
// ---------------------------------------------------------------------------

/// Compact mapping of active levels for a factor pair.
///
/// Maps global level indices to local (compact) indices for factors q and r,
/// and provides the local-to-global index vector for the combined domain.
struct ActiveLevels {
    q_map: Vec<u32>,
    n_q: usize,
    r_map: Vec<u32>,
    n_r: usize,
    local_to_global: Vec<u32>,
}

/// Scan all observations once and mark which levels are active for each factor.
///
/// Returns `active[f][level]` = true if any observation uses that level of factor f.
pub(crate) fn find_all_active_levels<S: Store>(design: &Design<S>) -> Vec<Vec<bool>> {
    let n_obs = design.store.n_obs();
    let cols = factor_columns(&design.store);
    let mut active: Vec<Vec<bool>> = design
        .factors
        .iter()
        .map(|f| vec![false; f.n_levels])
        .collect();
    // Factor-outer / obs-inner: all writes for a factor land in one `active[f]`
    // buffer before moving on, instead of hopping between `n_factors` buffers
    // on every observation.
    for (f, col) in active.iter_mut().enumerate() {
        let levels = cols[f];
        for uid in 0..n_obs {
            col[level_at(&design.store, levels, uid, f)] = true;
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

/// Build compact mapping for a factor pair using pre-computed active level flags.
///
/// Extracts the mapping logic from `find_all_active_levels`, taking pre-computed
/// active booleans instead of scanning observations.
fn build_compact_mapping(
    active_q: &[bool],
    active_r: &[bool],
    fq: &FactorMeta,
    fr: &FactorMeta,
) -> Option<ActiveLevels> {
    let (q_map, n_q) = compact_map(active_q);
    let (r_map, n_r) = compact_map(active_r);

    if n_q == 0 || n_r == 0 {
        return None;
    }

    let mut local_to_global = Vec::with_capacity(n_q + n_r);
    for (j, &a) in active_q.iter().enumerate() {
        if a {
            local_to_global.push(to_u32(fq.offset + j));
        }
    }
    for (k, &a) in active_r.iter().enumerate() {
        if a {
            local_to_global.push(to_u32(fr.offset + k));
        }
    }

    Some(ActiveLevels {
        q_map,
        n_q,
        r_map,
        n_r,
        local_to_global,
    })
}

/// A connected component in a bipartite factor-pair graph.
///
/// Indices are compact (0-based into the parent CrossTab's n_q / n_r).
pub(crate) struct BipartiteComponent {
    pub(crate) q_indices: Vec<usize>,
    pub(crate) r_indices: Vec<usize>,
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
    /// CSR(C): q-block rows (n_q) x r-block cols (n_r).
    pub(crate) c: CsrBlock,
    /// CSR(C^T): r-block rows (n_r) x q-block cols (n_q). Precomputed via
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
    /// Diagonal block for factor q (length n_q).
    pub(crate) q: Vec<f64>,
    /// Diagonal block for factor r (length n_r).
    pub(crate) r: Vec<f64>,
}

impl BlockDiagonals {
    /// Slice the diagonals for a single bipartite component, mirroring the CSR
    /// extraction in [`CrossTab::extract_component`].
    pub(crate) fn extract_component(&self, comp: &BipartiteComponent) -> Self {
        Self {
            q: comp.q_indices.iter().map(|&i| self.q[i]).collect(),
            r: comp.r_indices.iter().map(|&i| self.r[i]).collect(),
        }
    }
}

impl CrossTab {
    /// Number of rows in the q-block.
    pub(crate) fn n_q(&self) -> usize {
        self.c.nrows
    }

    /// Number of rows in the r-block.
    pub(crate) fn n_r(&self) -> usize {
        self.c.ncols
    }

    /// Total number of DOFs (n_q + n_r).
    pub(crate) fn n_local(&self) -> usize {
        self.c.nrows + self.c.ncols
    }

    /// Build a CrossTab using pre-computed active level flags.
    ///
    /// Reuses active levels already determined via `find_all_active_levels`,
    /// avoiding a redundant observation scan.
    ///
    /// Returns the diagonal blocks separately ([`BlockDiagonals`]): they are
    /// build-time-only and so are not stored on the `CrossTab`.
    pub(crate) fn build_for_pair_with_active<S: Store>(
        design: &Design<S>,
        weights: Option<&[f64]>,
        q: usize,
        r: usize,
        all_active: &[Vec<bool>],
    ) -> Option<(Self, BlockDiagonals, Vec<u32>)> {
        let fq = &design.factors[q];
        let fr = &design.factors[r];
        let active = build_compact_mapping(&all_active[q], &all_active[r], fq, fr)?;

        let (c, diag_q, diag_r) = accumulate_cross_block(design, weights, q, r, &active);
        let ct = c.transpose();
        let cross_tab = CrossTab { c, ct };
        let diagonals = BlockDiagonals {
            q: diag_q,
            r: diag_r,
        };
        Some((cross_tab, diagonals, active.local_to_global))
    }

    /// Find connected components in the bipartite graph defined by C.
    ///
    /// Uses DFS on CSR(C) (q->r edges) and CSR(C^T) (r->q edges).
    /// Returns components as vectors of compact q-indices and r-indices.
    /// O(n_q + n_r + nnz_C).
    pub(crate) fn bipartite_connected_components(&self) -> Vec<BipartiteComponent> {
        let n_q = self.n_q();
        let n_r = self.n_r();
        let n = n_q + n_r;
        if n == 0 {
            return Vec::new();
        }

        // Node labels: 0..n_q are q-nodes, n_q..n_q+n_r are r-nodes
        let mut visited = vec![false; n];
        let mut components = Vec::new();
        let mut stack = Vec::new();

        for start in 0..n {
            if visited[start] {
                continue;
            }
            visited[start] = true;
            stack.push(start);
            let mut q_indices = Vec::new();
            let mut r_indices = Vec::new();

            while let Some(node) = stack.pop() {
                if node < n_q {
                    // q-node: follow C edges to r-nodes
                    let qi = node;
                    q_indices.push(qi);
                    let start_idx = self.c.indptr[qi] as usize;
                    let end_idx = self.c.indptr[qi + 1] as usize;
                    for idx in start_idx..end_idx {
                        let rj = self.c.indices[idx] as usize;
                        let global_rj = n_q + rj;
                        if !visited[global_rj] {
                            visited[global_rj] = true;
                            stack.push(global_rj);
                        }
                    }
                } else {
                    // r-node: follow C^T edges to q-nodes
                    let ri = node - n_q;
                    r_indices.push(ri);
                    let start_idx = self.ct.indptr[ri] as usize;
                    let end_idx = self.ct.indptr[ri + 1] as usize;
                    for idx in start_idx..end_idx {
                        let qj = self.ct.indices[idx] as usize;
                        if !visited[qj] {
                            visited[qj] = true;
                            stack.push(qj);
                        }
                    }
                }
            }

            // Sort for deterministic ordering
            q_indices.sort_unstable();
            r_indices.sort_unstable();
            components.push(BipartiteComponent {
                q_indices,
                r_indices,
            });
        }

        components
    }

    /// Extract a sub-CrossTab for a single bipartite component.
    ///
    /// Remaps q/r indices to the component's local 0-based indexing.
    /// O(nnz in the component).
    ///
    /// `q_remap`/`r_remap` are parent-sized scratch buffers (length `n_q()` /
    /// `n_r()`). They must arrive all-`u32::MAX` and are reset to that on exit,
    /// so a single pair can be reused across every component of one parent
    /// instead of allocating fresh per component.
    pub(crate) fn extract_component(
        &self,
        comp: &BipartiteComponent,
        q_remap: &mut [u32],
        r_remap: &mut [u32],
    ) -> Self {
        let n_q = comp.q_indices.len();
        let n_r = comp.r_indices.len();
        debug_assert_eq!(q_remap.len(), self.n_q());
        debug_assert_eq!(r_remap.len(), self.n_r());

        // Build reverse maps: parent compact index -> component compact index.
        for (new_idx, &old_idx) in comp.q_indices.iter().enumerate() {
            q_remap[old_idx] = to_u32(new_idx);
        }
        for (new_idx, &old_idx) in comp.r_indices.iter().enumerate() {
            r_remap[old_idx] = to_u32(new_idx);
        }

        // Extract CSR(C): only rows in comp.q_indices, remap columns
        let mut c_indptr = vec![0u32; n_q + 1];
        let mut c_indices = Vec::new();
        let mut c_data = Vec::new();
        for (new_qi, &old_qi) in comp.q_indices.iter().enumerate() {
            let start = self.c.indptr[old_qi] as usize;
            let end = self.c.indptr[old_qi + 1] as usize;
            for idx in start..end {
                let old_rj = self.c.indices[idx] as usize;
                let new_rj = r_remap[old_rj];
                if new_rj != u32::MAX {
                    c_indices.push(new_rj);
                    c_data.push(self.c.data[idx]);
                }
            }
            c_indptr[new_qi + 1] = to_u32(c_indices.len());
        }

        let c = CsrBlock {
            indptr: c_indptr,
            indices: c_indices,
            data: c_data,
            nrows: n_q,
            ncols: n_r,
        };
        let ct = c.transpose();

        // Reset only the touched entries so the buffers are all-`u32::MAX` again
        // for the next component.
        for &old_idx in &comp.q_indices {
            q_remap[old_idx] = u32::MAX;
        }
        for &old_idx in &comp.r_indices {
            r_remap[old_idx] = u32::MAX;
        }

        CrossTab { c, ct }
    }
}

#[cfg(test)]
mod tests;
