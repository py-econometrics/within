#[cfg(feature = "ndarray")]
use ndarray::ArrayView2;

use super::{flatten_factor_major_levels, ObservationStore, ObservationWeights};

/// Compressed observation storage: deduplicates identical observation tuples,
/// storing only unique (level₁, level₂, …) tuples and a mapping from each
/// observation to its unique tuple index.
///
/// Wins in Gramian build and domain construction for panel data where many
/// observations share the same factor-level combination: iterating `n_unique`
/// instead of `n_obs` in those hot paths.
#[derive(Debug, Clone)]
pub struct CompressedStore {
    /// Row-major unique levels: `unique_levels[uid * n_factors + q]`.
    unique_levels: Vec<u32>,
    /// Aggregated total weight per unique tuple.
    unique_weights: Vec<f64>,
    /// Mapping: observation index → unique tuple index.
    obs_map: Vec<u32>,
    /// Per-observation weights (for D^T·w·y which needs per-obs weighting).
    obs_weights: ObservationWeights,
    n_obs: usize,
    n_unique: usize,
    n_factors: usize,
}

impl CompressedStore {
    /// Construct from factor-major data by sorting and deduplicating.
    /// O(n_obs log n_obs) from the sort.
    ///
    /// Used in tests where data originates as `Vec<Vec<u32>>`. Production code
    /// should use [`from_array`](Self::from_array).
    pub fn from_factor_major(
        factor_levels: Vec<Vec<u32>>,
        weights: ObservationWeights,
        n_obs: usize,
    ) -> Self {
        let n_factors = factor_levels.len();
        debug_assert!(
            factor_levels.iter().all(|col| col.len() == n_obs),
            "all factor columns must have n_obs entries"
        );
        let flat_levels = flatten_factor_major_levels(&factor_levels, n_obs);
        // Sort observation indices by their level tuple.
        let mut order: Vec<usize> = (0..n_obs).collect();
        order.sort_by(|&a, &b| {
            flat_levels[a * n_factors..(a + 1) * n_factors]
                .cmp(&flat_levels[b * n_factors..(b + 1) * n_factors])
        });
        Self::build(&flat_levels, &order, weights, n_obs, n_factors)
    }

    /// Construct from a 2D array view (n_obs × n_factors, usize).
    #[cfg(feature = "ndarray")]
    pub fn from_array(categories: ArrayView2<usize>, weights: ObservationWeights) -> Self {
        let n_obs = categories.nrows();
        let n_factors = categories.ncols();
        // Flatten to row-major, converting usize→u32 at the boundary.
        let flat_levels: Vec<u32> = if let Some(slice) = categories.as_slice() {
            slice.iter().map(|&v| v as u32).collect()
        } else {
            categories.iter().map(|&v| v as u32).collect()
        };
        // Sort observation indices by their level tuple.
        let mut order: Vec<usize> = (0..n_obs).collect();
        order.sort_by(|&a, &b| {
            flat_levels[a * n_factors..(a + 1) * n_factors]
                .cmp(&flat_levels[b * n_factors..(b + 1) * n_factors])
        });
        Self::build(&flat_levels, &order, weights, n_obs, n_factors)
    }

    /// Sort rows by factor-level tuple, then linear-scan to group duplicates.
    ///
    /// After sorting, consecutive rows with identical level tuples are merged
    /// into one unique entry whose weight is the sum of the originals. Each
    /// observation records its unique-tuple index in `obs_map`.
    ///
    /// `flat_levels` is row-major: `flat_levels[i * n_factors + q]` is the level
    /// for observation `i` in factor `q`. `sorted_order` is a permutation of
    /// `0..n_obs` sorted by level tuple.
    fn build(
        flat_levels: &[u32],
        sorted_order: &[usize],
        weights: ObservationWeights,
        n_obs: usize,
        n_factors: usize,
    ) -> Self {
        debug_assert_eq!(flat_levels.len(), n_obs * n_factors);
        debug_assert_eq!(sorted_order.len(), n_obs);
        weights.debug_assert_valid_for(n_obs);

        let row = |i: usize| &flat_levels[i * n_factors..(i + 1) * n_factors];

        let mut unique_levels = Vec::new();
        let mut aggregated_weights = Vec::new();
        let mut obs_map = vec![0u32; n_obs];
        let mut current_uid: u32 = 0;

        if n_obs > 0 {
            // Seed the first group from the first sorted observation.
            let first_obs = sorted_order[0];
            unique_levels.extend_from_slice(row(first_obs));
            aggregated_weights.push(weights.get(first_obs));
            obs_map[first_obs] = 0;

            // Scan remaining observations: start a new group when the tuple
            // changes, otherwise accumulate the weight into the current group.
            for k in 1..n_obs {
                let obs_idx = sorted_order[k];
                let prev_idx = sorted_order[k - 1];
                if row(obs_idx) != row(prev_idx) {
                    // New unique tuple — start a new group.
                    current_uid += 1;
                    unique_levels.extend_from_slice(row(obs_idx));
                    aggregated_weights.push(weights.get(obs_idx));
                } else {
                    // Duplicate tuple — merge weight into current group.
                    aggregated_weights[current_uid as usize] += weights.get(obs_idx);
                }
                obs_map[obs_idx] = current_uid;
            }
        }

        let n_unique = if n_obs == 0 {
            0
        } else {
            (current_uid + 1) as usize
        };
        debug_assert!(
            n_unique <= u32::MAX as usize,
            "n_unique ({n_unique}) exceeds u32::MAX — obs_map uses u32 indices"
        );

        Self {
            unique_levels,
            unique_weights: aggregated_weights,
            obs_map,
            obs_weights: weights,
            n_obs,
            n_unique,
            n_factors,
        }
    }

    /// Ratio of unique tuples to total observations.
    pub fn compression_ratio(&self) -> f64 {
        if self.n_obs == 0 {
            return 1.0;
        }
        self.n_unique as f64 / self.n_obs as f64
    }
}

impl ObservationStore for CompressedStore {
    #[inline]
    fn n_obs(&self) -> usize {
        self.n_obs
    }

    #[inline]
    fn n_factors(&self) -> usize {
        self.n_factors
    }

    #[inline]
    fn level(&self, obs: usize, factor: usize) -> u32 {
        let uid = self.obs_map[obs] as usize;
        self.unique_levels[uid * self.n_factors + factor]
    }

    #[inline]
    fn weight(&self, obs: usize) -> f64 {
        self.obs_weights.get(obs)
    }

    #[inline]
    fn is_unweighted(&self) -> bool {
        self.obs_weights.is_unit()
    }

    // -- Compressed overrides ------------------------------------------------

    #[inline]
    fn n_unique(&self) -> usize {
        self.n_unique
    }

    #[inline]
    fn unique_level(&self, uid: usize, factor: usize) -> u32 {
        self.unique_levels[uid * self.n_factors + factor]
    }

    #[inline]
    fn unique_total_weight(&self, uid: usize) -> f64 {
        self.unique_weights[uid]
    }

    #[inline]
    fn is_unique_unweighted(&self) -> bool {
        // Even if per-obs weights are unit, aggregated weights are counts > 1
        // when duplicates exist, so unique weights are NOT unit.
        self.n_unique == self.n_obs && self.obs_weights.is_unit()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observation::sample_factor_levels;
    use crate::observation::FactorMajorStore;

    #[test]
    fn test_compressed_levels_match() {
        let fl = sample_factor_levels();
        let fm = FactorMajorStore::new(fl.clone(), ObservationWeights::Unit, 4)
            .expect("valid factor-major store");
        let cs = CompressedStore::from_factor_major(fl, ObservationWeights::Unit, 4);

        assert_eq!(cs.n_obs(), fm.n_obs());
        assert_eq!(cs.n_factors(), fm.n_factors());
        for i in 0..4 {
            for q in 0..2 {
                assert_eq!(
                    cs.level(i, q),
                    fm.level(i, q),
                    "mismatch at obs={i}, factor={q}"
                );
            }
        }
    }

    #[test]
    fn test_compressed_dedup_count() {
        // obs: (0,0), (1,1), (2,0), (0,1) — all unique
        let fl = vec![vec![0u32, 1, 2, 0], vec![0, 1, 0, 1]];
        let cs = CompressedStore::from_factor_major(fl, ObservationWeights::Unit, 4);
        assert_eq!(cs.n_unique(), 4);
        assert!((cs.compression_ratio() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_compressed_dedup_with_duplicates() {
        // obs: (0,0), (1,1), (0,0), (1,1) — 2 unique tuples
        let fl = vec![vec![0u32, 1, 0, 1], vec![0, 1, 0, 1]];
        let cs = CompressedStore::from_factor_major(fl, ObservationWeights::Unit, 4);
        assert_eq!(cs.n_unique(), 2);
        assert!((cs.compression_ratio() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_compressed_unique_total_weight_unweighted() {
        // obs: (0,0), (1,1), (0,0), (1,1) — 2 unique with count 2 each
        let fl = vec![vec![0u32, 1, 0, 1], vec![0, 1, 0, 1]];
        let cs = CompressedStore::from_factor_major(fl, ObservationWeights::Unit, 4);
        let mut total = 0.0;
        for uid in 0..cs.n_unique() {
            total += cs.unique_total_weight(uid);
        }
        assert!(
            (total - 4.0).abs() < 1e-12,
            "total unique weight should equal n_obs"
        );
        // Each unique tuple has weight 2.0 (two obs of weight 1.0 each)
        for uid in 0..cs.n_unique() {
            assert!((cs.unique_total_weight(uid) - 2.0).abs() < 1e-12);
        }
    }

    #[test]
    fn test_compressed_unique_total_weight_weighted() {
        // obs: (0,0) w=1, (1,1) w=2, (0,0) w=3, (1,1) w=4
        let fl = vec![vec![0u32, 1, 0, 1], vec![0, 1, 0, 1]];
        let w = vec![1.0, 2.0, 3.0, 4.0];
        let cs = CompressedStore::from_factor_major(fl, ObservationWeights::Dense(w), 4);
        assert_eq!(cs.n_unique(), 2);
        // (0,0) aggregated weight = 1+3 = 4, (1,1) = 2+4 = 6
        let mut weights: Vec<(Vec<u32>, f64)> = (0..cs.n_unique())
            .map(|uid| {
                let levels: Vec<u32> = (0..2).map(|q| cs.unique_level(uid, q)).collect();
                (levels, cs.unique_total_weight(uid))
            })
            .collect();
        weights.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(weights[0].0, vec![0, 0]);
        assert!((weights[0].1 - 4.0).abs() < 1e-12);
        assert_eq!(weights[1].0, vec![1, 1]);
        assert!((weights[1].1 - 6.0).abs() < 1e-12);
    }

    #[test]
    fn test_compressed_unique_levels() {
        // obs: (0,0), (1,1), (0,0), (1,1)
        let fl = vec![vec![0u32, 1, 0, 1], vec![0, 1, 0, 1]];
        let cs = CompressedStore::from_factor_major(fl, ObservationWeights::Unit, 4);
        let mut tuples: Vec<Vec<u32>> = (0..cs.n_unique())
            .map(|uid| (0..2).map(|q| cs.unique_level(uid, q)).collect())
            .collect();
        tuples.sort();
        assert_eq!(tuples, vec![vec![0, 0], vec![1, 1]]);
    }

    #[test]
    fn test_compressed_is_unique_unweighted() {
        // With duplicates: unique weights are counts > 1, so NOT unweighted
        let fl = vec![vec![0u32, 1, 0, 1], vec![0, 1, 0, 1]];
        let cs = CompressedStore::from_factor_major(fl, ObservationWeights::Unit, 4);
        assert!(!cs.is_unique_unweighted()); // has duplicates

        // Without duplicates: all counts = 1, so unique IS unweighted
        let fl2 = vec![vec![0u32, 1, 2, 3], vec![0, 1, 2, 3]];
        let cs2 = CompressedStore::from_factor_major(fl2, ObservationWeights::Unit, 4);
        assert!(cs2.is_unique_unweighted());
    }

    #[test]
    fn test_compressed_per_obs_weight() {
        // Per-obs weights are preserved (needed for D^T·w·y)
        let fl = vec![vec![0u32, 1, 0, 1], vec![0, 1, 0, 1]];
        let w = vec![1.0, 2.0, 3.0, 4.0];
        let cs = CompressedStore::from_factor_major(fl, ObservationWeights::Dense(w.clone()), 4);
        for (i, &expected) in w.iter().enumerate() {
            assert_eq!(cs.weight(i), expected);
        }
        assert!(!cs.is_unweighted());
    }

    #[test]
    fn test_compressed_empty() {
        let cs = CompressedStore::from_factor_major(
            vec![Vec::<u32>::new(), Vec::new()],
            ObservationWeights::Unit,
            0,
        );
        assert_eq!(cs.n_obs(), 0);
        assert_eq!(cs.n_unique(), 0);
        assert!((cs.compression_ratio() - 1.0).abs() < 1e-12);
    }
}
