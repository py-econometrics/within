//! Factor-pair domain construction and partition-of-unity weights.
//!
//! In the additive Schwarz method, the global problem is split into overlapping
//! local subproblems that are solved independently and then combined. This
//! module constructs those subdomains from the structure of the fixed-effects
//! problem.
//!
//! # Factor pairs as subdomains
//!
//! The Gramian `G = D^T W D` has a block structure dictated by factor pairs.
//! For Q factors there are `Q(Q-1)/2` off-diagonal blocks, one per pair
//! `(q, r)`. Each pair defines a natural subdomain: the DOFs (coefficient
//! indices) are the union of active levels in factors q and r, and the local
//! operator is the corresponding principal submatrix of G.
//!
//! When a factor pair's bipartite graph (levels of q on one side, levels of r
//! on the other, edges from co-occurring observations) has multiple connected
//! components, each component becomes a separate, smaller subdomain. This
//! decomposition is computed via [`CrossTab::bipartite_connected_components`].
//!
//! # Partition of unity
//!
//! Because a single level can appear in multiple factor pairs (e.g., level j
//! of factor q participates in pairs (q,r1), (q,r2), ...), the subdomains
//! overlap. The two-sided additive Schwarz formula:
//!
//! ```text
//! M^{-1} = sum_i  R_i^T  D_i  A_i^{-1}  D_i  R_i
//! ```
//!
//! requires that the squared partition-of-unity weights sum to 1 at every DOF:
//! `sum_i (D_i)^2 = I` (restricted to covered DOFs). If a DOF appears in `c`
//! subdomains, each weight is set to `1/sqrt(c)` so that `c * (1/sqrt(c))^2 = 1`.
//!
//! In the common case where every DOF belongs to exactly one subdomain (no
//! overlap), all weights are 1.0 and the compact [`PartitionWeights::Uniform`]
//! representation avoids per-DOF storage.
//!
//! # Entry points
//!
//! - [`build_local_domains`] — builds subdomains only (for preconditioner-only paths).
//! - [`build_domains_and_gramian_blocks`] — builds subdomains *and* collects
//!   [`PairBlockData`] for composing an explicit Gramian, in a single observation
//!   scan per pair. This fused path avoids redundant work when both the
//!   preconditioner and an explicit Gramian are needed.

use std::sync::Arc;

use super::{PartitionWeights, Subdomain, WeightedDesign};
use crate::observation::ObservationStore;
use crate::operator::gramian::{find_all_active_levels, BipartiteComponent, CrossTab};

/// Build local subdomains (with pre-built CrossTabs) for pairs of factors.
///
/// For each factor pair, builds a fused CrossTab via a single observation scan,
/// detects connected components on the bipartite structure, and creates one
/// subdomain per component. The CrossTab travels with each subdomain to avoid
/// a rebuild.
///
/// Factor pairs are processed in parallel via Rayon. The
/// `compute_partition_weights` step remains sequential after the parallel
/// collect.
pub(crate) fn build_local_domains<S: ObservationStore>(
    design: &WeightedDesign<S>,
) -> Vec<(Subdomain, CrossTab)> {
    use rayon::prelude::*;

    let n_factors = design.n_factors();
    let pairs = build_pairs(n_factors);
    let all_active = find_all_active_levels(design);

    let mut domain_pairs: Vec<(Subdomain, CrossTab)> = pairs
        .par_iter()
        .flat_map(|&(q, r)| domains_for_pair(design, q, r, &all_active))
        .collect();

    compute_partition_weights(&mut domain_pairs, design.n_dofs);

    domain_pairs
}

/// Per-pair block data sufficient to compose the full Gramian CSR.
///
/// Produced alongside subdomain construction so that the same observation scan
/// serves both domain decomposition and Gramian assembly.
/// The `cross_tab` field is wrapped in `Arc` to avoid cloning the full CrossTab
/// when the same data is shared with subdomain entries.
pub(crate) struct PairBlockData {
    pub q: usize,
    pub r: usize,
    /// Shared reference to the full-pair CrossTab (diag_q, diag_r, C, C^T).
    pub cross_tab: Arc<CrossTab>,
    /// Global DOF indices of active q-levels (length = c.nrows).
    pub q_global: Vec<u32>,
    /// Global DOF indices of active r-levels (length = c.ncols).
    pub r_global: Vec<u32>,
}

/// Build local subdomains AND collect per-pair block data for Gramian composition.
///
/// Combines the parallel observation scan (for domain construction) with block
/// extraction (for Gramian assembly) in a single pass per pair. This avoids a
/// double observation scan when both domains and an explicit Gramian are needed.
pub(crate) fn build_domains_and_gramian_blocks<S: ObservationStore>(
    design: &WeightedDesign<S>,
) -> (Vec<(Subdomain, CrossTab)>, Vec<PairBlockData>) {
    use rayon::prelude::*;

    let n_factors = design.n_factors();
    let pairs = build_pairs(n_factors);
    let all_active = find_all_active_levels(design);

    type PairResult = (Vec<(Subdomain, CrossTab)>, Option<PairBlockData>);
    let results: Vec<PairResult> = pairs
        .par_iter()
        .map(|&(q, r)| domains_and_block_for_pair(design, q, r, &all_active))
        .collect();

    let mut domain_pairs = Vec::new();
    let mut blocks = Vec::new();
    for (domains, block) in results {
        domain_pairs.extend(domains);
        if let Some(b) = block {
            blocks.push(b);
        }
    }

    compute_partition_weights(&mut domain_pairs, design.n_dofs);
    (domain_pairs, blocks)
}

fn domains_and_block_for_pair<S: ObservationStore>(
    design: &WeightedDesign<S>,
    q: usize,
    r: usize,
    all_active: &[Vec<bool>],
) -> (Vec<(Subdomain, CrossTab)>, Option<PairBlockData>) {
    let (full_ct, l2g) = match CrossTab::build_for_pair_with_active(design, q, r, all_active) {
        Some(pair) => pair,
        None => return (Vec::new(), None),
    };

    let n_q_full = full_ct.n_q();
    let ct_arc = Arc::new(full_ct);

    let block_data = PairBlockData {
        q,
        r,
        cross_tab: Arc::clone(&ct_arc),
        q_global: l2g[..n_q_full].to_vec(),
        r_global: l2g[n_q_full..].to_vec(),
    };

    let domains = split_into_subdomains_arc(ct_arc, &l2g, n_q_full, (q, r));

    (domains, Some(block_data))
}

fn domains_for_pair<S: ObservationStore>(
    design: &WeightedDesign<S>,
    q: usize,
    r: usize,
    all_active: &[Vec<bool>],
) -> Vec<(Subdomain, CrossTab)> {
    let (full_ct, l2g) = match CrossTab::build_for_pair_with_active(design, q, r, all_active) {
        Some(pair) => pair,
        None => return Vec::new(),
    };

    let n_q_full = full_ct.n_q();
    split_into_subdomains(full_ct, &l2g, n_q_full, (q, r))
}

/// Split a full CrossTab into per-component subdomains.
///
/// Finds bipartite connected components, extracts a sub-CrossTab for each,
/// and builds a `Subdomain` with uniform partition-of-unity weights.
fn split_into_subdomains(
    full_ct: CrossTab,
    l2g: &[u32],
    n_q_full: usize,
    factor_pair: (usize, usize),
) -> Vec<(Subdomain, CrossTab)> {
    let components = full_ct.bipartite_connected_components();

    let cross_tabs: Vec<CrossTab> = if components.len() == 1 {
        vec![full_ct]
    } else {
        components
            .iter()
            .map(|comp| full_ct.extract_component(comp))
            .collect()
    };

    components
        .iter()
        .zip(cross_tabs)
        .map(|(comp, comp_ct)| {
            let comp_l2g = component_global_indices(comp, l2g, n_q_full);
            let core =
                super::SubdomainCore::uniform(comp_l2g.into_iter().map(|g| g as u32).collect());
            (Subdomain { factor_pair, core }, comp_ct)
        })
        .collect()
}

/// Like `split_into_subdomains` but accepts an `Arc<CrossTab>`.
///
/// When there is a single component, unwraps or clones the Arc to avoid
/// full CrossTab cloning. Multiple components extract sub-CrossTabs as usual.
fn split_into_subdomains_arc(
    full_ct: Arc<CrossTab>,
    l2g: &[u32],
    n_q_full: usize,
    factor_pair: (usize, usize),
) -> Vec<(Subdomain, CrossTab)> {
    let components = full_ct.bipartite_connected_components();

    let cross_tabs: Vec<CrossTab> = if components.len() == 1 {
        vec![Arc::try_unwrap(full_ct).unwrap_or_else(|arc| (*arc).clone())]
    } else {
        components
            .iter()
            .map(|comp| full_ct.extract_component(comp))
            .collect()
    };

    components
        .iter()
        .zip(cross_tabs)
        .map(|(comp, comp_ct)| {
            let comp_l2g = component_global_indices(comp, l2g, n_q_full);
            let core =
                super::SubdomainCore::uniform(comp_l2g.into_iter().map(|g| g as u32).collect());
            (Subdomain { factor_pair, core }, comp_ct)
        })
        .collect()
}

/// Compute global DOF indices for a bipartite component.
///
/// Maps the component's compact q/r indices through the local-to-global vector,
/// returning global indices with q-levels first, then r-levels.
fn component_global_indices(comp: &BipartiteComponent, l2g: &[u32], n_q_full: usize) -> Vec<usize> {
    comp.q_indices
        .iter()
        .map(|&i| l2g[i] as usize)
        .chain(comp.r_indices.iter().map(|&i| l2g[n_q_full + i] as usize))
        .collect()
}

fn build_pairs(n_factors: usize) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    for q in 0..n_factors {
        for r in (q + 1)..n_factors {
            pairs.push((q, r));
        }
    }
    pairs
}

/// Compute partition-of-unity weights for overlapping Schwarz subdomains.
///
/// The two-sided additive Schwarz formula `M⁻¹ = Σ Rᵢᵀ D̃ᵢ Aᵢ⁻¹ D̃ᵢ Rᵢ`
/// requires that the squared weights sum to identity at every DOF:
/// `Σ Rᵢᵀ D̃ᵢ² Rᵢ = I`. For a DOF appearing in `c` subdomains, each weight
/// is set to `1/√c`, so that `c × (1/√c)² = 1`.
///
/// In the common (non-overlapping) case where every DOF belongs to exactly one
/// subdomain, all weights are 1.0 and the compact `PartitionWeights::Uniform`
/// representation is used to avoid per-DOF storage.
fn compute_partition_weights(domain_pairs: &mut [(Subdomain, CrossTab)], n_dofs: usize) {
    let mut counts = vec![0u32; n_dofs];
    for (d, _) in domain_pairs.iter() {
        for &idx in d.core.global_indices() {
            debug_assert!((idx as usize) < n_dofs);
            counts[idx as usize] += 1;
        }
    }
    for (d, _) in domain_pairs.iter_mut() {
        let all_unique = d
            .core
            .global_indices()
            .iter()
            .all(|&idx| counts[idx as usize] <= 1);
        if all_unique {
            d.core.set_uniform_partition_weights();
        } else {
            let weights: Vec<f64> = d
                .core
                .global_indices()
                .iter()
                .map(|&idx| {
                    let c = counts[idx as usize];
                    debug_assert!(c > 0);
                    1.0 / (c as f64).sqrt()
                })
                .collect();
            d.core
                .set_partition_weights(PartitionWeights::NonUniform(weights))
                .expect("partition weight count must match index count");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::FixedEffectsDesign;
    use crate::observation::{FactorMajorStore, ObservationWeights};

    fn make_test_design() -> FixedEffectsDesign {
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
        FixedEffectsDesign::from_store(store).expect("valid test design")
    }

    #[test]
    fn test_full_cover_domain_count() {
        let dm = make_test_design();
        let domain_pairs = build_local_domains(&dm);
        // 3 factor pairs; each pair may produce multiple components
        assert!(domain_pairs.len() >= 3);
    }

    #[test]
    fn test_partition_of_unity() {
        let dm = make_test_design();
        let domain_pairs = build_local_domains(&dm);
        let n_dofs = dm.n_dofs;
        // Two-sided PoU: squared weights must sum to 1 at every DOF.
        let mut weight_sq_sum = vec![0.0; n_dofs];
        for (d, _) in &domain_pairs {
            for (i, &idx) in d.core.global_indices().iter().enumerate() {
                let w = d.core.partition_weights().get(i);
                weight_sq_sum[idx as usize] += w * w;
            }
        }
        for &ws in &weight_sq_sum {
            if ws > 0.0 {
                assert!((ws - 1.0).abs() < 1e-12, "Weight² sum {ws} != 1.0");
            }
        }
    }

    #[test]
    fn test_domains_cover_all_dofs() {
        let dm = make_test_design();
        let domain_pairs = build_local_domains(&dm);
        let mut covered = vec![false; dm.n_dofs];
        for (d, _) in &domain_pairs {
            for &idx in d.core.global_indices() {
                covered[idx as usize] = true;
            }
        }
        assert!(covered.iter().all(|&c| c), "Not all DOFs covered");
    }

    // -----------------------------------------------------------------------
    // build_domains_and_gramian_blocks + from_pair_blocks tests
    // -----------------------------------------------------------------------

    use crate::operator::gramian::Gramian;

    fn assert_domain_sets_equal(
        obs_domains: &[(Subdomain, crate::operator::gramian::CrossTab)],
        gram_domains: &[(Subdomain, crate::operator::gramian::CrossTab)],
    ) {
        assert_eq!(
            obs_domains.len(),
            gram_domains.len(),
            "domain count mismatch"
        );

        // Sort both by global_indices for deterministic comparison
        let mut obs_sorted: Vec<_> = obs_domains
            .iter()
            .map(|(d, _)| d.core.global_indices().to_vec())
            .collect();
        obs_sorted.sort();
        let mut gram_sorted: Vec<_> = gram_domains
            .iter()
            .map(|(d, _)| d.core.global_indices().to_vec())
            .collect();
        gram_sorted.sort();

        for (o, g) in obs_sorted.iter().zip(&gram_sorted) {
            assert_eq!(o, g, "global_indices mismatch");
        }
    }

    #[test]
    fn test_composed_gramian_matches_observation_gramian() {
        use schwarz_precond::Operator;

        for design in [make_test_design(), {
            let store = FactorMajorStore::new(
                vec![vec![0, 1, 2, 0, 1], vec![0, 1, 2, 3, 0]],
                ObservationWeights::Unit,
                5,
            )
            .unwrap();
            FixedEffectsDesign::from_store(store).unwrap()
        }] {
            let obs_gramian = Gramian::build(&design);
            let (_domains, blocks) = build_domains_and_gramian_blocks(&design);
            let composed =
                Gramian::from_pair_blocks(&blocks, &design.factors, design.n_dofs).unwrap();

            // Compare matvec output for several test vectors
            let n = design.n_dofs;
            for seed in 0..3 {
                let x: Vec<f64> = (0..n).map(|i| (i * 7 + seed) as f64 * 0.1).collect();
                let mut y_obs = vec![0.0; n];
                let mut y_composed = vec![0.0; n];
                obs_gramian.apply(&x, &mut y_obs);
                composed.apply(&x, &mut y_composed);
                for i in 0..n {
                    assert!(
                        (y_obs[i] - y_composed[i]).abs() < 1e-12,
                        "matvec mismatch at DOF {i}: obs={}, composed={}",
                        y_obs[i],
                        y_composed[i],
                    );
                }
            }
        }
    }

    #[test]
    fn test_composed_gramian_domains_match() {
        let design = make_test_design();
        let obs_domains = build_local_domains(&design);
        let (composed_domains, _blocks) = build_domains_and_gramian_blocks(&design);
        assert_domain_sets_equal(&obs_domains, &composed_domains);
    }
}
