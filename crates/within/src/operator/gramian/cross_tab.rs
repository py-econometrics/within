//! CrossTab — bipartite block representation of a local Gramian for a single factor pair.

use schwarz_precond::SparseMatrix;

use super::super::csr_block::CsrBlock;
use super::csr_assembly::accumulate_cross_block;
use crate::domain::WeightedDesign;
use crate::observation::ObservationStore;

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

/// Identify which levels in factors q and r are actually used, build compact
/// local→global mappings, and return the local-to-global index vector.
///
/// Returns `None` if either factor has no active levels.
fn find_active_levels<S: ObservationStore>(
    design: &WeightedDesign<S>,
    q: usize,
    r: usize,
) -> Option<ActiveLevels> {
    let fq = &design.factors[q];
    let fr = &design.factors[r];
    let n_obs = design.store.n_obs();

    let mut active_q = vec![false; fq.n_levels];
    let mut active_r = vec![false; fr.n_levels];
    for uid in 0..n_obs {
        active_q[design.store.level(uid, q) as usize] = true;
        active_r[design.store.level(uid, r) as usize] = true;
    }

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

    // Build local-to-global mapping: q levels first, then r levels
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
///
/// Used to build SDDM matrices directly (for ApproxChol).
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

    /// Size of the first (q) block.
    pub fn first_block_size(&self) -> usize {
        self.c.nrows
    }

    /// Build the SDDM matrix L = [D_q, -C; -C^T, D_r] directly in CSR format.
    ///
    /// Rows are already sorted: for q-block rows the diagonal (index i) comes
    /// before shifted C columns (>= n_q); for r-block rows C^T columns (< n_q)
    /// come before the diagonal (index n_q + i).
    pub fn to_sddm(&self) -> SparseMatrix {
        let n_q = self.n_q();
        let n_r = self.n_r();
        let n = n_q + n_r;
        let nnz_c = self.c.nnz();
        // Each row gets: 1 diagonal + off-diagonal entries from C or C^T
        // Total NNZ = n (diagonals) + 2 * nnz_c (C and C^T entries)
        let total_nnz = n + 2 * nnz_c;

        let mut indptr = vec![0u32; n + 1];
        let mut indices = Vec::with_capacity(total_nnz);
        let mut data = Vec::with_capacity(total_nnz);

        // Q-block rows (i = 0..n_q): diagonal at i, then C entries shifted by +n_q
        for i in 0..n_q {
            // Diagonal entry
            indices.push(i as u32);
            data.push(self.diag_q[i]);
            // C entries for this row, with negated values and shifted columns
            let c_start = self.c.indptr[i] as usize;
            let c_end = self.c.indptr[i + 1] as usize;
            for idx in c_start..c_end {
                indices.push(self.c.indices[idx] + n_q as u32);
                data.push(-self.c.data[idx]);
            }
            indptr[i + 1] = indices.len() as u32;
        }

        // R-block rows (i = 0..n_r): C^T entries first (cols < n_q), then diagonal at n_q+i
        for i in 0..n_r {
            // C^T entries for this row, with negated values (col indices already < n_q)
            let ct_start = self.ct.indptr[i] as usize;
            let ct_end = self.ct.indptr[i + 1] as usize;
            for idx in ct_start..ct_end {
                indices.push(self.ct.indices[idx]);
                data.push(-self.ct.data[idx]);
            }
            // Diagonal entry
            indices.push((n_q + i) as u32);
            data.push(self.diag_r[i]);
            indptr[n_q + i + 1] = indices.len() as u32;
        }

        debug_assert_eq!(indices.len(), total_nnz);

        SparseMatrix::new(indptr, indices, data, n)
    }

    /// Build a CrossTab for an entire factor pair, discovering active levels
    /// from the observation data in a single scan.
    ///
    /// Returns `None` if either factor has no active levels (empty pair).
    /// Also returns `local_to_global`: q-levels first, then r-levels, matching
    /// the convention used by `ActiveLevels` and `SubdomainCore::global_indices`.
    pub fn build_for_pair<S: ObservationStore>(
        design: &WeightedDesign<S>,
        q: usize,
        r: usize,
    ) -> Option<(Self, Vec<u32>)> {
        let active = find_active_levels(design, q, r)?;

        let (c, diag_q, diag_r) = accumulate_cross_block(
            design,
            q,
            r,
            &active.q_map,
            &active.r_map,
            active.n_q,
            active.n_r,
        );
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
        gramian: &SparseMatrix,
        fq: &crate::observation::FactorMeta,
        fr: &crate::observation::FactorMeta,
    ) -> Option<(Self, Vec<u32>)> {
        let indptr = gramian.indptr();
        let indices = gramian.indices();
        let data = gramian.data();

        let r_lo = fr.offset as u32;
        let r_hi = (fr.offset + fr.n_levels) as u32;

        // Extract diagonals and detect active levels
        let mut full_diag_q = vec![0.0; fq.n_levels];
        let mut full_diag_r = vec![0.0; fr.n_levels];
        let mut active_q = vec![false; fq.n_levels];
        let mut active_r = vec![false; fr.n_levels];

        for j in 0..fq.n_levels {
            let row = fq.offset + j;
            let start = indptr[row] as usize;
            let end = indptr[row + 1] as usize;
            for idx in start..end {
                let col = indices[idx] as usize;
                if col == row {
                    full_diag_q[j] = data[idx];
                    if data[idx] != 0.0 {
                        active_q[j] = true;
                    }
                }
            }
        }

        for k in 0..fr.n_levels {
            let row = fr.offset + k;
            let start = indptr[row] as usize;
            let end = indptr[row + 1] as usize;
            for idx in start..end {
                let col = indices[idx] as usize;
                if col == row {
                    full_diag_r[k] = data[idx];
                    if data[idx] != 0.0 {
                        active_r[k] = true;
                    }
                }
            }
        }

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

        // Extract C block: iterate q-rows, filter columns in r-range, remap both
        let mut c_indptr = vec![0u32; n_q + 1];
        let mut c_indices = Vec::new();
        let mut c_data = Vec::new();

        for j in 0..fq.n_levels {
            if !active_q[j] {
                continue;
            }
            let compact_q = q_map[j] as usize;
            let row = fq.offset + j;
            let start = indptr[row] as usize;
            let end = indptr[row + 1] as usize;
            for idx in start..end {
                let col = indices[idx];
                if col >= r_lo && col < r_hi {
                    let k = (col - r_lo) as usize;
                    if r_map[k] != u32::MAX {
                        c_indices.push(r_map[k]);
                        c_data.push(data[idx]);
                    }
                }
            }
            c_indptr[compact_q + 1] = c_indices.len() as u32;
        }

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{FixedEffectsDesign, WeightedDesign};
    use crate::observation::{FactorMajorStore, ObservationWeights};
    use crate::operator::gramian::Gramian;

    fn assert_cross_tabs_equal(a: &CrossTab, b: &CrossTab) {
        assert_eq!(a.n_q(), b.n_q(), "n_q mismatch");
        assert_eq!(a.n_r(), b.n_r(), "n_r mismatch");
        for (i, (av, bv)) in a.diag_q.iter().zip(&b.diag_q).enumerate() {
            assert!(
                (av - bv).abs() < 1e-12,
                "diag_q[{i}] mismatch: {av} vs {bv}"
            );
        }
        for (i, (av, bv)) in a.diag_r.iter().zip(&b.diag_r).enumerate() {
            assert!(
                (av - bv).abs() < 1e-12,
                "diag_r[{i}] mismatch: {av} vs {bv}"
            );
        }
        assert_eq!(a.c.indptr, b.c.indptr, "C indptr mismatch");
        assert_eq!(a.c.indices, b.c.indices, "C indices mismatch");
        for (i, (av, bv)) in a.c.data.iter().zip(&b.c.data).enumerate() {
            assert!(
                (av - bv).abs() < 1e-12,
                "C data[{i}] mismatch: {av} vs {bv}"
            );
        }
    }

    fn make_2fe_design() -> FixedEffectsDesign {
        let store = FactorMajorStore::new(
            vec![vec![0, 1, 2, 0, 1], vec![0, 1, 2, 3, 0]],
            ObservationWeights::Unit,
            5,
        )
        .expect("valid factor-major store");
        FixedEffectsDesign::from_store(store, &[3, 4]).expect("valid 2FE design")
    }

    fn make_3fe_design() -> FixedEffectsDesign {
        let store = FactorMajorStore::new(
            vec![
                vec![0, 1, 2, 0, 1, 2],
                vec![0, 1, 0, 1, 0, 1],
                vec![0, 0, 1, 1, 0, 1],
            ],
            ObservationWeights::Unit,
            6,
        )
        .expect("valid factor-major store");
        FixedEffectsDesign::from_store(store, &[3, 2, 2]).expect("valid 3FE design")
    }

    #[test]
    fn test_from_gramian_block_matches_build_for_pair_2fe() {
        let design = make_2fe_design();
        let gramian = Gramian::build(&design);

        let (ct_obs, l2g_obs) = CrossTab::build_for_pair(&design, 0, 1).unwrap();
        let (ct_gram, l2g_gram) =
            CrossTab::from_gramian_block(&gramian.matrix, &design.factors[0], &design.factors[1])
                .unwrap();

        assert_eq!(l2g_obs, l2g_gram, "local_to_global mismatch");
        assert_cross_tabs_equal(&ct_obs, &ct_gram);
    }

    #[test]
    fn test_from_gramian_block_matches_build_for_pair_3fe() {
        let design = make_3fe_design();
        let gramian = Gramian::build(&design);

        for q in 0..3 {
            for r in (q + 1)..3 {
                let obs_result = CrossTab::build_for_pair(&design, q, r);
                let gram_result = CrossTab::from_gramian_block(
                    &gramian.matrix,
                    &design.factors[q],
                    &design.factors[r],
                );

                match (obs_result, gram_result) {
                    (Some((ct_obs, l2g_obs)), Some((ct_gram, l2g_gram))) => {
                        assert_eq!(l2g_obs, l2g_gram, "l2g mismatch for pair ({q},{r})");
                        assert_cross_tabs_equal(&ct_obs, &ct_gram);
                    }
                    (None, None) => {}
                    _ => panic!("one returned None and the other Some for pair ({q},{r})"),
                }
            }
        }
    }

    #[test]
    fn test_from_gramian_block_single_component() {
        // Fully connected 2FE design: single component
        let store = FactorMajorStore::new(
            vec![vec![0, 1, 0, 1], vec![0, 0, 1, 1]],
            ObservationWeights::Unit,
            4,
        )
        .expect("valid factor-major store");
        let design = FixedEffectsDesign::from_store(store, &[2, 2]).expect("valid design");
        let gramian = Gramian::build(&design);

        let (ct_gram, _) =
            CrossTab::from_gramian_block(&gramian.matrix, &design.factors[0], &design.factors[1])
                .unwrap();

        let components = ct_gram.bipartite_connected_components();
        assert_eq!(components.len(), 1, "expected single component");
    }

    #[test]
    fn test_from_gramian_block_multiple_components() {
        // Design with two disconnected components:
        // factor 0: [0, 0, 1, 1]   factor 1: [0, 1, 2, 3]
        // levels (0,0), (0,1) form one component; (1,2), (1,3) form another
        let store = FactorMajorStore::new(
            vec![vec![0, 0, 1, 1], vec![0, 1, 2, 3]],
            ObservationWeights::Unit,
            4,
        )
        .expect("valid factor-major store");
        let design = FixedEffectsDesign::from_store(store, &[2, 4]).expect("valid design");
        let gramian = Gramian::build(&design);

        let (ct_obs, _) = CrossTab::build_for_pair(&design, 0, 1).unwrap();
        let (ct_gram, _) =
            CrossTab::from_gramian_block(&gramian.matrix, &design.factors[0], &design.factors[1])
                .unwrap();

        let comps_obs = ct_obs.bipartite_connected_components();
        let comps_gram = ct_gram.bipartite_connected_components();
        assert_eq!(comps_obs.len(), comps_gram.len());
        for (co, cg) in comps_obs.iter().zip(&comps_gram) {
            assert_eq!(co.q_indices, cg.q_indices);
            assert_eq!(co.r_indices, cg.r_indices);
        }
    }

    #[test]
    fn test_from_gramian_block_weighted() {
        let fl = vec![vec![0u32, 1, 0, 1], vec![0, 0, 1, 1]];
        let weights = vec![1.0, 2.0, 3.0, 4.0];
        let n_levels = vec![2, 2];

        let store = FactorMajorStore::new(fl, ObservationWeights::Dense(weights), 4)
            .expect("valid weighted store");
        let design = WeightedDesign::from_store(store, &n_levels).expect("valid weighted design");
        let gramian = Gramian::build(&design);

        let (ct_obs, l2g_obs) = CrossTab::build_for_pair(&design, 0, 1).unwrap();
        let (ct_gram, l2g_gram) =
            CrossTab::from_gramian_block(&gramian.matrix, &design.factors[0], &design.factors[1])
                .unwrap();

        assert_eq!(l2g_obs, l2g_gram);
        assert_cross_tabs_equal(&ct_obs, &ct_gram);
    }
}
