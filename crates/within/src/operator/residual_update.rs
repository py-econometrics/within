use std::sync::Arc;

use schwarz_precond::{ResidualUpdater, SparseMatrix};

use super::dof_obs_index::DofObservationIndex;
use crate::domain::WeightedDesign;
use crate::observation::ObservationStore;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::FixedEffectsDesign;
    use crate::observation::{FactorMajorStore, ObservationWeights};
    use crate::operator::gramian::Gramian;
    use schwarz_precond::OperatorResidualUpdater;

    /// Helper: build a design, explicit Gramian, and both updaters.
    /// Returns (design, gramian, obs_updater).
    fn make_test_setup() -> (FixedEffectsDesign, Gramian) {
        // 2 factors, 5 observations
        // factor 0: [0, 1, 2, 0, 1] (3 levels)
        // factor 1: [0, 1, 2, 3, 0] (4 levels)
        // n_dofs = 7
        let store = FactorMajorStore::new(
            vec![vec![0, 1, 2, 0, 1], vec![0, 1, 2, 3, 0]],
            ObservationWeights::Unit,
            5,
        )
        .expect("valid factor-major store");
        let design =
            FixedEffectsDesign::from_store(store, &[3, 4]).expect("valid fixed-effects design");
        let gramian = Gramian::build(&design);
        (design, gramian)
    }

    #[test]
    fn test_obs_updater_matches_operator_updater_single_step() {
        let (design, gramian) = make_test_setup();
        let n_dofs = design.n_dofs;

        let mut obs_updater = ObservationSpaceUpdater::new(&design);
        let mut op_updater = OperatorResidualUpdater::new(&gramian, n_dofs);

        // Initial residual
        let r_original = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        op_updater.reset(&r_original);

        let mut r_obs = r_original.clone();
        let mut r_op = r_original.clone();

        // Correction on subdomain {0, 1, 3, 4} (factor0 levels 0,1 + factor1 levels 0,1)
        let global_indices: Vec<u32> = vec![0, 1, 3, 4];
        let correction = vec![0.5, -0.3, 0.2, 0.1];

        obs_updater.update(&global_indices, &correction, &mut r_obs);
        op_updater.update(&global_indices, &correction, &mut r_op);

        for i in 0..n_dofs {
            assert!(
                (r_obs[i] - r_op[i]).abs() < 1e-12,
                "mismatch at DOF {i}: obs={}, op={}",
                r_obs[i],
                r_op[i],
            );
        }
    }

    #[test]
    fn test_obs_updater_matches_operator_updater_two_steps() {
        let (design, gramian) = make_test_setup();
        let n_dofs = design.n_dofs;

        let mut obs_updater = ObservationSpaceUpdater::new(&design);
        let mut op_updater = OperatorResidualUpdater::new(&gramian, n_dofs);

        let r_original = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0];
        op_updater.reset(&r_original);

        let mut r_obs = r_original.clone();
        let mut r_op = r_original.clone();

        // First subdomain correction
        let gi1: Vec<u32> = vec![0, 1, 3, 4];
        let c1 = vec![0.5, -0.3, 0.2, 0.1];
        obs_updater.update(&gi1, &c1, &mut r_obs);
        op_updater.update(&gi1, &c1, &mut r_op);

        for i in 0..n_dofs {
            assert!(
                (r_obs[i] - r_op[i]).abs() < 1e-12,
                "step 1 mismatch at DOF {i}: obs={}, op={}",
                r_obs[i],
                r_op[i],
            );
        }

        // Second subdomain correction
        let gi2: Vec<u32> = vec![2, 5, 6];
        let c2 = vec![1.0, -0.5, 0.8];
        obs_updater.update(&gi2, &c2, &mut r_obs);
        op_updater.update(&gi2, &c2, &mut r_op);

        for i in 0..n_dofs {
            assert!(
                (r_obs[i] - r_op[i]).abs() < 1e-12,
                "step 2 mismatch at DOF {i}: obs={}, op={}",
                r_obs[i],
                r_op[i],
            );
        }
    }

    #[test]
    fn test_obs_updater_zero_correction_is_noop() {
        let (design, _) = make_test_setup();
        let mut updater = ObservationSpaceUpdater::new(&design);

        let r_original = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let mut r_work = r_original.clone();

        let gi: Vec<u32> = vec![0, 1, 3];
        let correction = vec![0.0, 0.0, 0.0];
        updater.update(&gi, &correction, &mut r_work);

        assert_eq!(r_work, r_original);
    }

    #[test]
    fn test_obs_updater_single_dof_correction() {
        let (design, gramian) = make_test_setup();
        let n_dofs = design.n_dofs;

        let mut obs_updater = ObservationSpaceUpdater::new(&design);
        let mut op_updater = OperatorResidualUpdater::new(&gramian, n_dofs);

        let r_original = vec![1.0; n_dofs];
        op_updater.reset(&r_original);

        let mut r_obs = r_original.clone();
        let mut r_op = r_original.clone();

        // Single DOF correction
        let gi: Vec<u32> = vec![0];
        let correction = vec![1.0];
        obs_updater.update(&gi, &correction, &mut r_obs);
        op_updater.update(&gi, &correction, &mut r_op);

        for i in 0..n_dofs {
            assert!(
                (r_obs[i] - r_op[i]).abs() < 1e-12,
                "single-DOF mismatch at {i}: obs={}, op={}",
                r_obs[i],
                r_op[i],
            );
        }
    }

    #[test]
    fn test_obs_updater_weighted_design() {
        use crate::observation::{FactorMajorStore, ObservationWeights};

        let fl = vec![vec![0u32, 1, 0, 1], vec![0, 0, 1, 1]];
        let weights = vec![1.0, 2.0, 3.0, 4.0];
        let n_levels = vec![2, 2];

        let store = FactorMajorStore::new(fl, ObservationWeights::Dense(weights), 4)
            .expect("valid weighted store");
        let design = WeightedDesign::from_store(store, &n_levels).expect("valid weighted design");
        let gramian = Gramian::build(&design);
        let n_dofs = design.n_dofs; // 4

        let mut obs_updater = ObservationSpaceUpdater::new(&design);
        let mut op_updater = OperatorResidualUpdater::new(&gramian, n_dofs);

        let r_original = vec![5.0, 3.0, 7.0, 1.0];
        op_updater.reset(&r_original);

        let mut r_obs = r_original.clone();
        let mut r_op = r_original.clone();

        let gi: Vec<u32> = vec![0, 2];
        let correction = vec![0.5, -0.3];
        obs_updater.update(&gi, &correction, &mut r_obs);
        op_updater.update(&gi, &correction, &mut r_op);

        for i in 0..n_dofs {
            assert!(
                (r_obs[i] - r_op[i]).abs() < 1e-12,
                "weighted mismatch at {i}: obs={}, op={}",
                r_obs[i],
                r_op[i],
            );
        }
    }

    // -----------------------------------------------------------------------
    // SparseGramianUpdater tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_sparse_gramian_updater_matches_obs_updater_single_step() {
        let (design, gramian) = make_test_setup();
        let n_dofs = design.n_dofs;

        let mut obs_updater = ObservationSpaceUpdater::new(&design);
        let mut sparse_updater = SparseGramianUpdater::new(gramian.matrix);

        let r_original = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let mut r_obs = r_original.clone();
        let mut r_sparse = r_original;

        let global_indices: Vec<u32> = vec![0, 1, 3, 4];
        let correction = vec![0.5, -0.3, 0.2, 0.1];

        obs_updater.update(&global_indices, &correction, &mut r_obs);
        sparse_updater.update(&global_indices, &correction, &mut r_sparse);

        for i in 0..n_dofs {
            assert!(
                (r_obs[i] - r_sparse[i]).abs() < 1e-12,
                "mismatch at DOF {i}: obs={}, sparse={}",
                r_obs[i],
                r_sparse[i],
            );
        }
    }

    #[test]
    fn test_sparse_gramian_updater_matches_obs_updater_two_steps() {
        let (design, gramian) = make_test_setup();
        let n_dofs = design.n_dofs;

        let mut obs_updater = ObservationSpaceUpdater::new(&design);
        let mut sparse_updater = SparseGramianUpdater::new(gramian.matrix);

        let r_original = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0];
        let mut r_obs = r_original.clone();
        let mut r_sparse = r_original;

        // First subdomain correction
        let gi1: Vec<u32> = vec![0, 1, 3, 4];
        let c1 = vec![0.5, -0.3, 0.2, 0.1];
        obs_updater.update(&gi1, &c1, &mut r_obs);
        sparse_updater.update(&gi1, &c1, &mut r_sparse);

        for i in 0..n_dofs {
            assert!(
                (r_obs[i] - r_sparse[i]).abs() < 1e-12,
                "step 1 mismatch at DOF {i}: obs={}, sparse={}",
                r_obs[i],
                r_sparse[i],
            );
        }

        // Second subdomain correction
        let gi2: Vec<u32> = vec![2, 5, 6];
        let c2 = vec![1.0, -0.5, 0.8];
        obs_updater.update(&gi2, &c2, &mut r_obs);
        sparse_updater.update(&gi2, &c2, &mut r_sparse);

        for i in 0..n_dofs {
            assert!(
                (r_obs[i] - r_sparse[i]).abs() < 1e-12,
                "step 2 mismatch at DOF {i}: obs={}, sparse={}",
                r_obs[i],
                r_sparse[i],
            );
        }
    }

    #[test]
    fn test_sparse_gramian_updater_zero_correction_is_noop() {
        let (_, gramian) = make_test_setup();
        let mut updater = SparseGramianUpdater::new(gramian.matrix);

        let r_original = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let mut r_work = r_original.clone();

        let gi: Vec<u32> = vec![0, 1, 3];
        let correction = vec![0.0, 0.0, 0.0];
        updater.update(&gi, &correction, &mut r_work);

        assert_eq!(r_work, r_original);
    }

    #[test]
    fn test_sparse_gramian_updater_single_dof_correction() {
        let (design, gramian) = make_test_setup();
        let n_dofs = design.n_dofs;

        let mut obs_updater = ObservationSpaceUpdater::new(&design);
        let mut sparse_updater = SparseGramianUpdater::new(gramian.matrix);

        let r_original = vec![1.0; n_dofs];
        let mut r_obs = r_original.clone();
        let mut r_sparse = r_original;

        let gi: Vec<u32> = vec![0];
        let correction = vec![1.0];
        obs_updater.update(&gi, &correction, &mut r_obs);
        sparse_updater.update(&gi, &correction, &mut r_sparse);

        for i in 0..n_dofs {
            assert!(
                (r_obs[i] - r_sparse[i]).abs() < 1e-12,
                "single-DOF mismatch at {i}: obs={}, sparse={}",
                r_obs[i],
                r_sparse[i],
            );
        }
    }

    #[test]
    fn test_sparse_gramian_updater_weighted_design() {
        use crate::observation::{FactorMajorStore, ObservationWeights};

        let fl = vec![vec![0u32, 1, 0, 1], vec![0, 0, 1, 1]];
        let weights = vec![1.0, 2.0, 3.0, 4.0];
        let n_levels = vec![2, 2];

        let store = FactorMajorStore::new(fl, ObservationWeights::Dense(weights), 4)
            .expect("valid weighted store");
        let design = WeightedDesign::from_store(store, &n_levels).expect("valid weighted design");
        let gramian = Gramian::build(&design);
        let n_dofs = design.n_dofs;

        let mut obs_updater = ObservationSpaceUpdater::new(&design);
        let mut sparse_updater = SparseGramianUpdater::new(gramian.matrix);

        let r_original = vec![5.0, 3.0, 7.0, 1.0];
        let mut r_obs = r_original.clone();
        let mut r_sparse = r_original;

        let gi: Vec<u32> = vec![0, 2];
        let correction = vec![0.5, -0.3];
        obs_updater.update(&gi, &correction, &mut r_obs);
        sparse_updater.update(&gi, &correction, &mut r_sparse);

        for i in 0..n_dofs {
            assert!(
                (r_obs[i] - r_sparse[i]).abs() < 1e-12,
                "weighted mismatch at {i}: obs={}, sparse={}",
                r_obs[i],
                r_sparse[i],
            );
        }
    }
}
