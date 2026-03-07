use std::sync::Arc;

use schwarz_precond::{ResidualUpdater, SparseMatrix};

use crate::domain::WeightedDesign;
use crate::observation::ObservationStore;

// ===========================================================================
// DofObservationIndex — inverted CSR index from DOFs to observations
// ===========================================================================

/// Inverted CSR index: for each DOF d, stores the observation indices that reference it.
///
/// Given a `WeightedDesign`, observation `i` references DOF `offset_q + level(i, q)` for
/// each factor `q`. This structure lets you efficiently look up all observations that
/// touch a given DOF — the key primitive for observation-space residual updates.
pub struct DofObservationIndex {
    offsets: Vec<u32>,
    indices: Vec<u32>,
}

impl DofObservationIndex {
    /// Build the inverted index from a `WeightedDesign`.
    ///
    /// Two-pass algorithm:
    /// 1. Count observations per DOF.
    /// 2. Prefix-sum to get offsets, then scatter observation indices.
    pub fn build<S: ObservationStore>(design: &WeightedDesign<S>) -> Self {
        let n_dofs = design.n_dofs;
        let n_obs = design.store.n_obs();
        let n_factors = design.store.n_factors();

        // Pass 1: count
        let mut counts = vec![0u32; n_dofs];
        for i in 0..n_obs {
            for q in 0..n_factors {
                let dof = design.factors[q].offset + design.store.level(i, q) as usize;
                debug_assert!(dof < n_dofs, "dof {dof} >= n_dofs {n_dofs}");
                counts[dof] += 1;
            }
        }

        // Build offsets via prefix sum
        let mut offsets = vec![0u32; n_dofs + 1];
        for d in 0..n_dofs {
            offsets[d + 1] = offsets[d] + counts[d];
        }

        // Pass 2: fill indices
        let total = offsets[n_dofs] as usize;
        let mut indices = vec![0u32; total];
        let mut pos = offsets[..n_dofs].to_vec(); // write cursors
        for i in 0..n_obs {
            for q in 0..n_factors {
                let dof = design.factors[q].offset + design.store.level(i, q) as usize;
                debug_assert!(dof < n_dofs, "dof {dof} >= n_dofs {n_dofs}");
                indices[pos[dof] as usize] = i as u32;
                pos[dof] += 1;
            }
        }

        Self { offsets, indices }
    }

    /// Observation indices that reference the given DOF.
    #[inline]
    pub fn obs_for_dof(&self, dof: u32) -> &[u32] {
        let start = self.offsets[dof as usize] as usize;
        let end = self.offsets[dof as usize + 1] as usize;
        &self.indices[start..end]
    }

    /// Total number of DOFs in the index.
    #[cfg(test)]
    pub(crate) fn n_dofs(&self) -> usize {
        self.offsets.len() - 1
    }
}

// ===========================================================================
// Residual updaters
// ===========================================================================

/// Observation-space residual updater for multiplicative Schwarz.
///
/// Instead of the naive `r -= G * delta` (full matvec, O(n_dofs) per subdomain),
/// this exploits FE structure to perform a sparse update via observation space:
///
/// 1. Find affected observations via `DofObservationIndex` lookups
/// 2. Compute `t_i = (D * delta)_i` for each affected observation
/// 3. Scatter back: `r -= D^T W t`
///
/// Cost: O(n_affected_obs * Q) instead of O(n_dofs).
///
/// All buffers are pre-allocated and reused across calls — zero heap allocation
/// in the hot loop. Uses a visited bitset for O(1) deduplication (no sort).
pub struct ObservationSpaceUpdater<'a, S: ObservationStore> {
    design: &'a WeightedDesign<S>,
    dof_obs_index: DofObservationIndex,
    /// Dense DOF → local position map (n_dofs-sized, u32::MAX = absent).
    dof_to_pos: Vec<u32>,
    /// Visited flag per observation for O(1) dedup (n_obs-sized).
    obs_visited: Vec<bool>,
    /// Indices of observations marked visited (for sparse cleanup).
    touched_obs: Vec<u32>,
}

const SENTINEL: u32 = u32::MAX;

impl<'a, S: ObservationStore> ObservationSpaceUpdater<'a, S> {
    pub fn new(design: &'a WeightedDesign<S>) -> Self {
        let dof_obs_index = DofObservationIndex::build(design);
        let n_obs = design.store.n_obs();
        Self {
            dof_obs_index,
            dof_to_pos: vec![SENTINEL; design.n_dofs],
            obs_visited: vec![false; n_obs],
            touched_obs: Vec::with_capacity(n_obs),
            design,
        }
    }
}

impl<S: ObservationStore> ResidualUpdater for ObservationSpaceUpdater<'_, S> {
    fn update(&mut self, global_indices: &[u32], weighted_correction: &[f64], r_work: &mut [f64]) {
        let store = &self.design.store;
        let factors = &self.design.factors;

        // Build dense DOF → local position map (O(1) lookup per DOF).
        let dof_to_pos = &mut self.dof_to_pos;
        for (pos, &gi) in global_indices.iter().enumerate() {
            dof_to_pos[gi as usize] = pos as u32;
        }

        // Collect affected observations via visited bitset — O(1) dedup, no sort.
        let obs_visited = &mut self.obs_visited;
        let touched_obs = &mut self.touched_obs;
        let dof_obs_index = &self.dof_obs_index;
        touched_obs.clear();
        for &gi in global_indices {
            for &obs in dof_obs_index.obs_for_dof(gi) {
                if !obs_visited[obs as usize] {
                    obs_visited[obs as usize] = true;
                    touched_obs.push(obs);
                }
            }
        }

        // For each affected observation, compute the observation-space value
        // t_i = (D * delta)_i = sum_q correction[local_pos(offset_q + level(i,q))]
        // then scatter back: r[offset_q + level(i,q)] -= weight(i) * t_i
        for &obs in touched_obs.iter() {
            let obs_idx = obs as usize;

            // Compute t_i = sum over factors of correction value for this observation's DOFs
            let mut t_i = 0.0;
            for (q, fq) in factors.iter().enumerate() {
                let dof = fq.offset + store.level(obs_idx, q) as usize;
                let pos = dof_to_pos[dof];
                if pos != SENTINEL {
                    t_i += weighted_correction[pos as usize];
                }
            }

            if t_i == 0.0 {
                continue;
            }

            // Scatter: r[dof] -= weight(i) * t_i for each factor
            let w_t = store.weight(obs_idx) * t_i;
            for (q, fq) in factors.iter().enumerate() {
                let dof = fq.offset + store.level(obs_idx, q) as usize;
                r_work[dof] -= w_t;
            }
        }

        // Sparse cleanup: only clear entries we touched.
        for &obs in touched_obs.iter() {
            obs_visited[obs as usize] = false;
        }
        for &gi in global_indices {
            dof_to_pos[gi as usize] = SENTINEL;
        }
    }

    fn reset(&mut self, _r_original: &[f64]) {
        // No-op: the observation-space updater is stateless — each update()
        // is a pure incremental r -= D^T W D delta, with no accumulator.
    }
}

/// Sparse Gramian residual updater for multiplicative Schwarz.
///
/// Uses the explicit Gramian CSR to perform residual updates via row scatter:
/// `r -= G * delta` restricted to the touched rows.
///
/// Cost: O(nnz_touched) with contiguous CSR reads. No buffers, no bookkeeping.
/// Trades O(nnz) memory (the Gramian) for faster per-iteration updates compared
/// to the observation-space path.
pub struct SparseGramianUpdater {
    gramian: Arc<SparseMatrix>,
}

impl SparseGramianUpdater {
    pub fn new(gramian: Arc<SparseMatrix>) -> Self {
        Self { gramian }
    }
}

impl ResidualUpdater for SparseGramianUpdater {
    fn update(&mut self, global_indices: &[u32], weighted_correction: &[f64], r_work: &mut [f64]) {
        let indptr = self.gramian.indptr();
        let indices = self.gramian.indices();
        let data = self.gramian.data();

        for (k, &gi) in global_indices.iter().enumerate() {
            let c = weighted_correction[k];
            if c == 0.0 {
                continue;
            }
            let row = gi as usize;
            let start = indptr[row] as usize;
            let end = indptr[row + 1] as usize;
            for idx in start..end {
                r_work[indices[idx] as usize] -= c * data[idx];
            }
        }
    }

    fn reset(&mut self, _r_original: &[f64]) {
        // No-op: each update is a pure incremental r -= G * delta.
    }
}
