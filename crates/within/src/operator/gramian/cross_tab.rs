//! Cross-tabulation of a factor pair: the bipartite local Gramian.
//!
//! For a pair of factors `(q, r)`, the local Gramian has 2×2 block structure:
//!
//! ```text
//! G_local = [ D_q    C  ]
//!           [ C^T    D_r ]
//! ```
//!
//! where `D_q` and `D_r` are diagonal (weighted level counts) and `C` is the
//! cross-tabulation matrix (`C[j,k]` = weighted count of observations at level
//! `j` of factor `q` **and** level `k` of factor `r`).
//!
//! [`CrossTab`] stores this decomposed form — `C` as a [`CsrBlock`], its
//! precomputed transpose `C^T`, and the two diagonals — rather than assembling
//! the full symmetric CSR. This is more compact and directly supports:
//!
//! - **Connected components** ([`CrossTab::bipartite_connected_components`]) —
//!   DFS on the bipartite graph `C` to split disconnected subdomains
//! - **Component extraction** ([`CrossTab::extract_component`]) — build a
//!   sub-CrossTab for a single connected component
//!
//! # Compact indexing
//!
//! Not all levels of a factor may be active (observed). The cross-tab uses
//! *compact* indices: only active levels are numbered `0..n_q` and `0..n_r`.
//! A `local_to_global` vector maps these back to global DOF indices.

use super::super::csr_block::CsrBlock;
use super::explicit::DENSE_TABLE_MAX_ENTRIES;
use crate::domain::Design;
use crate::observation::Store;

// ---------------------------------------------------------------------------
// BipartiteComponent / SchurData — supporting types for CrossTab
// ---------------------------------------------------------------------------

/// Borrowed view of compact mapping parameters for a factor pair.
///
/// Bundles the global-to-compact index maps and compact dimensions,
/// reducing the parameter count of `accumulate_cross_block`.
struct CompactPair<'a> {
    q_map: &'a [u32],
    r_map: &'a [u32],
    n_q: usize,
    n_r: usize,
}

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

impl ActiveLevels {
    fn as_compact_pair(&self) -> CompactPair<'_> {
        CompactPair {
            q_map: &self.q_map,
            r_map: &self.r_map,
            n_q: self.n_q,
            n_r: self.n_r,
        }
    }
}

/// Scan all observations once and mark which levels are active for each factor.
///
/// Returns `active[f][level]` = true if any observation uses that level of factor f.
pub fn find_all_active_levels<S: Store>(design: &Design<S>) -> Vec<Vec<bool>> {
    let n_factors = design.factors.len();
    let n_obs = design.store.n_obs();
    let mut active: Vec<Vec<bool>> = design
        .factors
        .iter()
        .map(|f| vec![false; f.n_levels])
        .collect();
    for uid in 0..n_obs {
        for f in 0..n_factors {
            active[f][design.store.level(uid, f) as usize] = true;
        }
    }
    active
}

/// Build compact mapping for a factor pair using pre-computed active level flags.
///
/// Extracts the mapping logic from `find_active_levels`, taking pre-computed
/// active booleans instead of scanning observations.
fn build_compact_mapping(
    active_q: &[bool],
    active_r: &[bool],
    fq: &crate::observation::FactorMeta,
    fr: &crate::observation::FactorMeta,
) -> Option<ActiveLevels> {
    let mut q_map = vec![u32::MAX; fq.n_levels];
    let mut n_q = 0u32;
    for (j, &a) in active_q.iter().enumerate() {
        if a {
            q_map[j] = n_q;
            n_q += 1;
        }
    }

    let mut r_map = vec![u32::MAX; fr.n_levels];
    let mut n_r = 0u32;
    for (k, &a) in active_r.iter().enumerate() {
        if a {
            r_map[k] = n_r;
            n_r += 1;
        }
    }

    let n_q = n_q as usize;
    let n_r = n_r as usize;

    if n_q == 0 || n_r == 0 {
        return None;
    }

    let mut local_to_global = Vec::with_capacity(n_q + n_r);
    for (j, &a) in active_q.iter().enumerate() {
        if a {
            local_to_global.push((fq.offset + j) as u32);
        }
    }
    for (k, &a) in active_r.iter().enumerate() {
        if a {
            local_to_global.push((fr.offset + k) as u32);
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

/// Scan factors q and r to find active levels, then delegate to `build_compact_mapping`.
///
/// Returns `None` if either factor has no active levels.
#[cfg(test)]
fn find_active_levels<S: Store>(design: &Design<S>, q: usize, r: usize) -> Option<ActiveLevels> {
    let fq = &design.factors[q];
    let fr = &design.factors[r];
    let n_obs = design.store.n_obs();

    let mut active_q = vec![false; fq.n_levels];
    let mut active_r = vec![false; fr.n_levels];
    for uid in 0..n_obs {
        active_q[design.store.level(uid, q) as usize] = true;
        active_r[design.store.level(uid, r) as usize] = true;
    }

    build_compact_mapping(&active_q, &active_r, fq, fr)
}

/// A connected component in a bipartite factor-pair graph.
///
/// Indices are compact (0-based into the parent CrossTab's n_q / n_r).
pub(crate) struct BipartiteComponent {
    pub q_indices: Vec<usize>,
    pub r_indices: Vec<usize>,
}

// ---------------------------------------------------------------------------
// CrossTab — bipartite block representation of a local Gramian
// ---------------------------------------------------------------------------

/// Bipartite block representation of a local Gramian for a single factor pair.
///
/// Stores the cross-tabulation C (and its precomputed transpose C^T) plus
/// diagonal blocks, avoiding construction of the full symmetric Gramian CSR.
/// The Gramian has structure `G = [D_q, C; C^T, D_r]` where D_q and D_r are
/// diagonal.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct CrossTab {
    /// CSR(C): q-block rows (n_q) x r-block cols (n_r).
    pub(crate) c: CsrBlock,
    /// CSR(C^T): r-block rows (n_r) x q-block cols (n_q). Precomputed via
    /// `c.transpose()`.
    pub(crate) ct: CsrBlock,
    /// Diagonal block for factor q (length n_q).
    pub(crate) diag_q: Vec<f64>,
    /// Diagonal block for factor r (length n_r).
    pub(crate) diag_r: Vec<f64>,
}

impl CrossTab {
    /// Number of rows in the q-block.
    pub fn n_q(&self) -> usize {
        self.c.nrows
    }

    /// Number of rows in the r-block.
    pub fn n_r(&self) -> usize {
        self.c.ncols
    }

    /// Total number of DOFs (n_q + n_r).
    pub fn n_local(&self) -> usize {
        self.c.nrows + self.c.ncols
    }

    /// Build a CrossTab for an entire factor pair, discovering active levels
    /// from the observation data in a single scan.
    ///
    /// Returns `None` if either factor has no active levels (empty pair).
    /// Also returns `local_to_global`: q-levels first, then r-levels, matching
    /// the convention used by `ActiveLevels` and `SubdomainCore::global_indices`.
    #[cfg(test)]
    pub fn build_for_pair<S: Store>(
        design: &Design<S>,
        weights: Option<&[f64]>,
        q: usize,
        r: usize,
    ) -> Option<(Self, Vec<u32>)> {
        let active = find_active_levels(design, q, r)?;

        let (c, diag_q, diag_r) =
            accumulate_cross_block(design, weights, q, r, &active.as_compact_pair());
        let ct = c.transpose();
        let cross_tab = CrossTab {
            c,
            ct,
            diag_q,
            diag_r,
        };
        Some((cross_tab, active.local_to_global))
    }

    /// Build a CrossTab using pre-computed active level flags.
    ///
    /// Like `build_for_pair` but avoids redundant observation scans when
    /// active levels have already been determined via `find_all_active_levels`.
    pub fn build_for_pair_with_active<S: Store>(
        design: &Design<S>,
        weights: Option<&[f64]>,
        q: usize,
        r: usize,
        all_active: &[Vec<bool>],
    ) -> Option<(Self, Vec<u32>)> {
        let fq = &design.factors[q];
        let fr = &design.factors[r];
        let active = build_compact_mapping(&all_active[q], &all_active[r], fq, fr)?;

        let (c, diag_q, diag_r) =
            accumulate_cross_block(design, weights, q, r, &active.as_compact_pair());
        let ct = c.transpose();
        let cross_tab = CrossTab {
            c,
            ct,
            diag_q,
            diag_r,
        };
        Some((cross_tab, active.local_to_global))
    }

    /// Build a CrossTab for a factor pair by extracting blocks from an explicit Gramian.
    ///
    /// For factor pair (q, r), the Gramian encodes:
    /// - `G[fq.offset+j, fq.offset+j]` → `diag_q[j]`
    /// - `G[fr.offset+k, fr.offset+k]` → `diag_r[k]`
    /// - `G[fq.offset+j, fr.offset+k]` → `C[j, k]` (off-diagonal block)
    ///
    /// Extracts `C` as a `CsrBlock`, computes `C^T` via transpose, and builds
    /// the compact `local_to_global` mapping (skipping levels with zero diagonal).
    /// Returns `None` if no active levels in either factor.
    #[cfg(test)]
    pub fn from_gramian_block(
        gramian: &schwarz_precond::SparseMatrix,
        fq: &crate::observation::FactorMeta,
        fr: &crate::observation::FactorMeta,
    ) -> Option<(Self, Vec<u32>)> {
        let indptr = gramian.indptr();
        let indices = gramian.indices();
        let data = gramian.data();

        let r_lo = fr.offset as u32;
        let r_hi = (fr.offset + fr.n_levels) as u32;

        // Extract diagonals and detect active levels
        let (full_diag_q, active_q) =
            test_helpers::extract_factor_diagonal(indptr, indices, data, fq.offset, fq.n_levels);
        let (full_diag_r, active_r) =
            test_helpers::extract_factor_diagonal(indptr, indices, data, fr.offset, fr.n_levels);

        // Build compact index maps
        let mut q_map = vec![u32::MAX; fq.n_levels];
        let mut n_q = 0usize;
        for (j, &a) in active_q.iter().enumerate() {
            if a {
                q_map[j] = n_q as u32;
                n_q += 1;
            }
        }

        let mut r_map = vec![u32::MAX; fr.n_levels];
        let mut n_r = 0usize;
        for (k, &a) in active_r.iter().enumerate() {
            if a {
                r_map[k] = n_r as u32;
                n_r += 1;
            }
        }

        if n_q == 0 || n_r == 0 {
            return None;
        }

        // Build local_to_global: q-levels first, then r-levels
        let mut local_to_global = Vec::with_capacity(n_q + n_r);
        for (j, &a) in active_q.iter().enumerate() {
            if a {
                local_to_global.push((fq.offset + j) as u32);
            }
        }
        for (k, &a) in active_r.iter().enumerate() {
            if a {
                local_to_global.push((fr.offset + k) as u32);
            }
        }

        // Extract compact diagonals
        let diag_q: Vec<f64> = (0..fq.n_levels)
            .filter(|&j| active_q[j])
            .map(|j| full_diag_q[j])
            .collect();
        let diag_r: Vec<f64> = (0..fr.n_levels)
            .filter(|&k| active_r[k])
            .map(|k| full_diag_r[k])
            .collect();

        // Extract off-diagonal C block
        let (c_indptr, c_indices, c_data) = test_helpers::OffdiagExtractor {
            indptr,
            indices,
            data,
            fq,
            active_q: &active_q,
            q_map: &q_map,
            r_map: &r_map,
            r_lo,
            r_hi,
        }
        .extract(n_q);

        let c = CsrBlock {
            indptr: c_indptr,
            indices: c_indices,
            data: c_data,
            nrows: n_q,
            ncols: n_r,
        };
        let ct = c.transpose();

        Some((
            CrossTab {
                c,
                ct,
                diag_q,
                diag_r,
            },
            local_to_global,
        ))
    }

    /// Find connected components in the bipartite graph defined by C.
    ///
    /// Uses DFS on CSR(C) (q->r edges) and CSR(C^T) (r->q edges).
    /// Returns components as vectors of compact q-indices and r-indices.
    /// O(n_q + n_r + nnz_C).
    pub fn bipartite_connected_components(&self) -> Vec<BipartiteComponent> {
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
    pub fn extract_component(&self, comp: &BipartiteComponent) -> Self {
        let n_q = comp.q_indices.len();
        let n_r = comp.r_indices.len();

        // Build reverse maps: parent compact index -> component compact index
        let mut q_remap = vec![u32::MAX; self.n_q()];
        for (new_idx, &old_idx) in comp.q_indices.iter().enumerate() {
            q_remap[old_idx] = new_idx as u32;
        }
        let mut r_remap = vec![u32::MAX; self.n_r()];
        for (new_idx, &old_idx) in comp.r_indices.iter().enumerate() {
            r_remap[old_idx] = new_idx as u32;
        }

        // Extract diagonals
        let diag_q: Vec<f64> = comp.q_indices.iter().map(|&i| self.diag_q[i]).collect();
        let diag_r: Vec<f64> = comp.r_indices.iter().map(|&i| self.diag_r[i]).collect();

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
            c_indptr[new_qi + 1] = c_indices.len() as u32;
        }

        let c = CsrBlock {
            indptr: c_indptr,
            indices: c_indices,
            data: c_data,
            nrows: n_q,
            ncols: n_r,
        };
        let ct = c.transpose();

        CrossTab {
            c,
            ct,
            diag_q,
            diag_r,
        }
    }
}

// ---------------------------------------------------------------------------
// accumulate_cross_block — shared observation accumulation for CrossTab
// ---------------------------------------------------------------------------

/// Accumulate observation weights into a cross-tabulation block C plus diagonals.
///
/// Used by `CrossTab::build_for_pair`. Observations whose compact index is
/// `u32::MAX` are skipped.
///
/// Dispatches to a dense or sparse path based on the table size.
fn accumulate_cross_block<S: Store>(
    design: &Design<S>,
    weights: Option<&[f64]>,
    q: usize,
    r: usize,
    compact: &CompactPair<'_>,
) -> (CsrBlock, Vec<f64>, Vec<f64>) {
    let table_size = compact.n_q * compact.n_r;
    if table_size <= DENSE_TABLE_MAX_ENTRIES {
        accumulate_dense_cross_block(design, weights, q, r, compact)
    } else {
        accumulate_sparse_cross_block(design, weights, q, r, compact)
    }
}

#[inline]
fn obs_weight(weights: Option<&[f64]>, uid: usize) -> f64 {
    match weights {
        Some(w) => w[uid],
        None => 1.0,
    }
}

/// Dense path: flat table with O(1) accumulation per observation (n_q * n_r <= 5M).
fn accumulate_dense_cross_block<S: Store>(
    design: &Design<S>,
    weights: Option<&[f64]>,
    q: usize,
    r: usize,
    compact: &CompactPair<'_>,
) -> (CsrBlock, Vec<f64>, Vec<f64>) {
    let n_obs = design.store.n_obs();
    let n_q = compact.n_q;
    let n_r = compact.n_r;
    let q_compact = compact.q_map;
    let r_compact = compact.r_map;
    let mut diag_q = vec![0.0f64; n_q];
    let mut diag_r = vec![0.0f64; n_r];
    let mut table = vec![0.0f64; n_q * n_r];

    for uid in 0..n_obs {
        let j = design.store.level(uid, q) as usize;
        let k = design.store.level(uid, r) as usize;
        let cj = q_compact[j];
        let ck = r_compact[k];
        if cj == u32::MAX || ck == u32::MAX {
            continue;
        }
        let w = obs_weight(weights, uid);
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
    compact: &CompactPair<'_>,
) -> (CsrBlock, Vec<f64>, Vec<f64>) {
    let n_obs = design.store.n_obs();
    let n_q = compact.n_q;
    let n_r = compact.n_r;
    let q_compact = compact.q_map;
    let r_compact = compact.r_map;
    let mut diag_q = vec![0.0f64; n_q];
    let mut diag_r = vec![0.0f64; n_r];

    // Pass 1: accumulate diags + count entries per row
    let mut row_counts = vec![0u32; n_q];
    for uid in 0..n_obs {
        let j = design.store.level(uid, q) as usize;
        let k = design.store.level(uid, r) as usize;
        let cj = q_compact[j];
        let ck = r_compact[k];
        if cj == u32::MAX || ck == u32::MAX {
            continue;
        }
        let w = obs_weight(weights, uid);
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
        let j = design.store.level(uid, q) as usize;
        let k = design.store.level(uid, r) as usize;
        let cj = q_compact[j];
        let ck = r_compact[k];
        if cj == u32::MAX || ck == u32::MAX {
            continue;
        }
        let w = obs_weight(weights, uid);
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

    let c = CsrBlock {
        indptr: c_indptr,
        indices: c_indices,
        data: c_data,
        nrows: n_q,
        ncols: n_r,
    };
    (c, diag_q, diag_r)
}

// ---------------------------------------------------------------------------
// from_gramian_block helpers (test-only)
// ---------------------------------------------------------------------------

#[cfg(test)]
pub(crate) mod test_helpers {
    /// Extract full diagonals and active-level flags from a Gramian for one factor.
    ///
    /// Returns `(full_diag, active)` where `full_diag[j]` is the Gramian diagonal
    /// at the factor's j-th level, and `active[j]` is true if that diagonal is
    /// non-zero.
    pub(crate) fn extract_factor_diagonal(
        indptr: &[u32],
        indices: &[u32],
        data: &[f64],
        offset: usize,
        n_levels: usize,
    ) -> (Vec<f64>, Vec<bool>) {
        let mut full_diag = vec![0.0; n_levels];
        let mut active = vec![false; n_levels];
        for j in 0..n_levels {
            let row = offset + j;
            let start = indptr[row] as usize;
            let end = indptr[row + 1] as usize;
            for idx in start..end {
                let col = indices[idx] as usize;
                if col == row {
                    full_diag[j] = data[idx];
                    if data[idx] != 0.0 {
                        active[j] = true;
                    }
                }
            }
        }
        (full_diag, active)
    }

    /// Extracts the off-diagonal C block from a Gramian for a factor pair.
    ///
    /// Bundles the many parameters of the extraction into a single struct,
    /// iterating active q-rows, filtering columns in the r-factor range,
    /// and remapping indices via the compact maps.
    pub(crate) struct OffdiagExtractor<'a> {
        pub indptr: &'a [u32],
        pub indices: &'a [u32],
        pub data: &'a [f64],
        pub fq: &'a crate::observation::FactorMeta,
        pub active_q: &'a [bool],
        pub q_map: &'a [u32],
        pub r_map: &'a [u32],
        pub r_lo: u32,
        pub r_hi: u32,
    }

    impl OffdiagExtractor<'_> {
        /// Extract CSR components `(indptr, indices, data)` for the C block.
        pub fn extract(&self, n_q: usize) -> (Vec<u32>, Vec<u32>, Vec<f64>) {
            let mut c_indptr = vec![0u32; n_q + 1];
            let mut c_indices = Vec::new();
            let mut c_data = Vec::new();

            for j in 0..self.fq.n_levels {
                if !self.active_q[j] {
                    continue;
                }
                let compact_q = self.q_map[j] as usize;
                let row = self.fq.offset + j;
                let start = self.indptr[row] as usize;
                let end = self.indptr[row + 1] as usize;
                for idx in start..end {
                    let col = self.indices[idx];
                    if col >= self.r_lo && col < self.r_hi {
                        let k = (col - self.r_lo) as usize;
                        if self.r_map[k] != u32::MAX {
                            c_indices.push(self.r_map[k]);
                            c_data.push(self.data[idx]);
                        }
                    }
                }
                c_indptr[compact_q + 1] = c_indices.len() as u32;
            }
            debug_assert!(c_indptr.windows(2).all(|w| w[0] <= w[1]));

            (c_indptr, c_indices, c_data)
        }
    }
}
