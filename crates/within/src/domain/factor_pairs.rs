//! Factor-pair domain construction and partition-of-unity weights.

use super::{PartitionWeights, Subdomain, WeightedDesign};
use crate::observation::ObservationStore;
use crate::operator::csr_block::CsrBlock;
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

/// Build local subdomains from an explicit Gramian instead of scanning observations.
///
/// Same output as `build_local_domains` but uses `CrossTab::from_gramian_block()`
/// to extract factor-pair blocks from the Gramian CSR. No `ObservationStore`
/// access needed — only the Gramian and factor metadata.
#[cfg(test)]
pub(crate) fn build_local_domains_from_gramian(
    gramian: &schwarz_precond::SparseMatrix,
    factors: &[crate::observation::FactorMeta],
    n_dofs: usize,
) -> Vec<(Subdomain, CrossTab)> {
    use rayon::prelude::*;

    let n_factors = factors.len();
    let pairs = build_pairs(n_factors);

    let mut domain_pairs: Vec<(Subdomain, CrossTab)> = pairs
        .par_iter()
        .flat_map(|&(q, r)| domains_for_pair_from_gramian(gramian, factors, q, r))
        .collect();

    compute_partition_weights(&mut domain_pairs, n_dofs);

    domain_pairs
}

/// Per-pair block data sufficient to compose the full Gramian CSR.
///
/// Produced alongside subdomain construction so that the same observation scan
/// serves both domain decomposition and Gramian assembly.
pub(crate) struct PairBlockData {
    pub q: usize,
    pub r: usize,
    pub diag_q: Vec<f64>,
    pub diag_r: Vec<f64>,
    /// Off-diagonal block C_qr (compact indices).
    pub c: CsrBlock,
    /// Transpose C_qr^T (compact indices).
    pub ct: CsrBlock,
    /// Global DOF indices of active q-levels (length = c.nrows).
    pub q_global: Vec<u32>,
    /// Global DOF indices of active r-levels (length = c.ncols).
    pub r_global: Vec<u32>,
}

/// Build local subdomains AND collect per-pair block data for Gramian composition.
///
/// Combines the parallel observation scan (for domain construction) with block
/// extraction (for Gramian assembly) in a single pass per pair. This avoids the
/// sequential `Gramian::build()` bottleneck while still producing the explicit
/// Gramian CSR needed for operator apply and residual updates.
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

    // Extract block data for Gramian composition before component splitting
    let block_data = PairBlockData {
        q,
        r,
        diag_q: full_ct.diag_q.clone(),
        diag_r: full_ct.diag_r.clone(),
        c: full_ct.c.clone(),
        ct: full_ct.ct.clone(),
        q_global: l2g[..n_q_full].to_vec(),
        r_global: l2g[n_q_full..].to_vec(),
    };

    let domains = split_into_subdomains(full_ct, &l2g, n_q_full, (q, r));

    (domains, Some(block_data))
}

#[cfg(test)]
fn domains_for_pair_from_gramian(
    gramian: &schwarz_precond::SparseMatrix,
    factors: &[crate::observation::FactorMeta],
    q: usize,
    r: usize,
) -> Vec<(Subdomain, CrossTab)> {
    let (full_ct, l2g) = match CrossTab::from_gramian_block(gramian, &factors[q], &factors[r]) {
        Some(pair) => pair,
        None => return Vec::new(),
    };

    let n_q_full = full_ct.n_q();
    split_into_subdomains(full_ct, &l2g, n_q_full, (q, r))
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

fn compute_partition_weights(domain_pairs: &mut [(Subdomain, CrossTab)], n_dofs: usize) {
    let mut counts = vec![0u32; n_dofs];
    for (d, _) in domain_pairs.iter() {
        for &idx in &d.core.global_indices {
            debug_assert!((idx as usize) < n_dofs);
            counts[idx as usize] += 1;
        }
    }
    for (d, _) in domain_pairs.iter_mut() {
        let all_unique = d
            .core
            .global_indices
            .iter()
            .all(|&idx| counts[idx as usize] <= 1);
        if all_unique {
            d.core.partition_weights = PartitionWeights::Uniform(d.core.global_indices.len());
        } else {
            let weights: Vec<f64> = d
                .core
                .global_indices
                .iter()
                .map(|&idx| {
                    let c = counts[idx as usize];
                    debug_assert!(c > 0);
                    1.0 / c as f64
                })
                .collect();
            d.core.partition_weights = PartitionWeights::NonUniform(weights);
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
        FixedEffectsDesign::from_store(store, &[3, 2, 2]).expect("valid test design")
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
        let mut weight_sum = vec![0.0; n_dofs];
        for (d, _) in &domain_pairs {
            for (i, &idx) in d.core.global_indices.iter().enumerate() {
                weight_sum[idx as usize] += d.core.partition_weights.get(i);
            }
        }
        for &ws in &weight_sum {
            if ws > 0.0 {
                assert!((ws - 1.0).abs() < 1e-12, "Weight sum {ws} != 1.0");
            }
        }
    }

    #[test]
    fn test_domains_cover_all_dofs() {
        let dm = make_test_design();
        let domain_pairs = build_local_domains(&dm);
        let mut covered = vec![false; dm.n_dofs];
        for (d, _) in &domain_pairs {
            for &idx in &d.core.global_indices {
                covered[idx as usize] = true;
            }
        }
        assert!(covered.iter().all(|&c| c), "Not all DOFs covered");
    }

    // -----------------------------------------------------------------------
    // build_local_domains_from_gramian tests
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
            .map(|(d, _)| d.core.global_indices.clone())
            .collect();
        obs_sorted.sort();
        let mut gram_sorted: Vec<_> = gram_domains
            .iter()
            .map(|(d, _)| d.core.global_indices.clone())
            .collect();
        gram_sorted.sort();

        for (o, g) in obs_sorted.iter().zip(&gram_sorted) {
            assert_eq!(o, g, "global_indices mismatch");
        }
    }

    #[test]
    fn test_gramian_domains_match_obs_domains() {
        for design in [make_test_design(), {
            let store = FactorMajorStore::new(
                vec![vec![0, 1, 2, 0, 1], vec![0, 1, 2, 3, 0]],
                ObservationWeights::Unit,
                5,
            )
            .unwrap();
            FixedEffectsDesign::from_store(store, &[3, 4]).unwrap()
        }] {
            let gramian = Gramian::build(&design);
            let obs_domains = build_local_domains(&design);
            let gram_domains =
                build_local_domains_from_gramian(&gramian.matrix, &design.factors, design.n_dofs);
            assert_domain_sets_equal(&obs_domains, &gram_domains);
        }
    }

    #[test]
    fn test_gramian_domains_partition_of_unity() {
        let design = make_test_design();
        let gramian = Gramian::build(&design);
        let domain_pairs =
            build_local_domains_from_gramian(&gramian.matrix, &design.factors, design.n_dofs);
        let n_dofs = design.n_dofs;
        let mut weight_sum = vec![0.0; n_dofs];
        for (d, _) in &domain_pairs {
            for (i, &idx) in d.core.global_indices.iter().enumerate() {
                weight_sum[idx as usize] += d.core.partition_weights.get(i);
            }
        }
        for &ws in &weight_sum {
            if ws > 0.0 {
                assert!((ws - 1.0).abs() < 1e-12, "Weight sum {ws} != 1.0");
            }
        }
    }

    #[test]
    fn test_gramian_domains_cover_all_dofs() {
        let design = make_test_design();
        let gramian = Gramian::build(&design);
        let domain_pairs =
            build_local_domains_from_gramian(&gramian.matrix, &design.factors, design.n_dofs);
        let mut covered = vec![false; design.n_dofs];
        for (d, _) in &domain_pairs {
            for &idx in &d.core.global_indices {
                covered[idx as usize] = true;
            }
        }
        assert!(covered.iter().all(|&c| c), "Not all DOFs covered");
    }

    // -----------------------------------------------------------------------
    // build_domains_and_gramian_blocks + from_pair_blocks tests
    // -----------------------------------------------------------------------

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
            FixedEffectsDesign::from_store(store, &[3, 4]).unwrap()
        }] {
            let obs_gramian = Gramian::build(&design);
            let (_domains, blocks) = build_domains_and_gramian_blocks(&design);
            let composed = Gramian::from_pair_blocks(&blocks, &design.factors, design.n_dofs);

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
