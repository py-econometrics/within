use crate::domain::WeightedDesign;
use crate::observation::ObservationStore;

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
    fn n_dofs(&self) -> usize {
        self.offsets.len() - 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::FixedEffectsDesign;

    // 2 factors, 5 observations:
    //   factor 0: levels [0, 1, 2, 0, 1]  (3 levels, offset 0)
    //   factor 1: levels [0, 1, 2, 3, 0]  (4 levels, offset 3)
    // DOFs: 0..3 for factor 0, 3..7 for factor 1
    fn make_test_design() -> FixedEffectsDesign {
        FixedEffectsDesign::new(
            vec![vec![0, 1, 2, 0, 1], vec![0, 1, 2, 3, 0]],
            vec![3, 4],
            5,
        )
        .expect("valid test design")
    }

    #[test]
    fn test_basic_dof_observation_mapping() {
        let design = make_test_design();
        let idx = DofObservationIndex::build(&design);

        assert_eq!(idx.n_dofs(), 7);

        // factor 0, level 0 (DOF 0): obs 0, 3
        let mut dof0: Vec<u32> = idx.obs_for_dof(0).to_vec();
        dof0.sort();
        assert_eq!(dof0, vec![0, 3]);

        // factor 0, level 1 (DOF 1): obs 1, 4
        let mut dof1: Vec<u32> = idx.obs_for_dof(1).to_vec();
        dof1.sort();
        assert_eq!(dof1, vec![1, 4]);

        // factor 0, level 2 (DOF 2): obs 2
        assert_eq!(idx.obs_for_dof(2), &[2]);

        // factor 1, level 0 (DOF 3): obs 0, 4
        let mut dof3: Vec<u32> = idx.obs_for_dof(3).to_vec();
        dof3.sort();
        assert_eq!(dof3, vec![0, 4]);

        // factor 1, level 1 (DOF 4): obs 1
        assert_eq!(idx.obs_for_dof(4), &[1]);

        // factor 1, level 2 (DOF 5): obs 2
        assert_eq!(idx.obs_for_dof(5), &[2]);

        // factor 1, level 3 (DOF 6): obs 3
        assert_eq!(idx.obs_for_dof(6), &[3]);
    }

    #[test]
    fn test_total_entries_equals_n_obs_times_n_factors() {
        let design = make_test_design();
        let idx = DofObservationIndex::build(&design);

        let total: usize = (0..idx.n_dofs())
            .map(|d| idx.obs_for_dof(d as u32).len())
            .sum();
        // Each observation contributes one entry per factor
        assert_eq!(total, 5 * 2);
    }

    #[test]
    fn test_single_factor_single_level() {
        // 1 factor, 3 observations, all same level
        let design = FixedEffectsDesign::new(vec![vec![0, 0, 0]], vec![1], 3)
            .expect("valid single-factor design");
        let idx = DofObservationIndex::build(&design);

        assert_eq!(idx.n_dofs(), 1);
        let mut obs: Vec<u32> = idx.obs_for_dof(0).to_vec();
        obs.sort();
        assert_eq!(obs, vec![0, 1, 2]);
    }

    #[test]
    fn test_dof_with_zero_observations() {
        // 1 factor, 2 observations using levels 0 and 2 — level 1 has no observations
        let design = FixedEffectsDesign::new(vec![vec![0, 2]], vec![3], 2)
            .expect("valid sparse-level design");
        let idx = DofObservationIndex::build(&design);

        assert_eq!(idx.n_dofs(), 3);
        assert_eq!(idx.obs_for_dof(0), &[0]);
        assert_eq!(idx.obs_for_dof(1), &[] as &[u32]); // empty
        assert_eq!(idx.obs_for_dof(2), &[1]);
    }
}
