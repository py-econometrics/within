//! Factor-pair domain construction and partition-of-unity weights.

use super::{PartitionWeights, Subdomain, WeightedDesign};
use crate::observation::ObservationStore;
use crate::operator::gramian::CrossTab;

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
///
/// `star_ref`: if Some(q), use star cover centered on factor q.
/// Otherwise, use full cover (all C(Q,2) pairs).
pub(crate) fn build_local_domains<S: ObservationStore>(
    design: &WeightedDesign<S>,
    star_ref: Option<usize>,
) -> Vec<(Subdomain, CrossTab)> {
    use rayon::prelude::*;

    let n_factors = design.n_factors();
    let pairs = build_pairs(n_factors, star_ref);

    let mut domain_pairs: Vec<(Subdomain, CrossTab)> = pairs
        .par_iter()
        .flat_map(|&(q, r)| domains_for_pair(design, q, r))
        .collect();

    compute_partition_weights(&mut domain_pairs, design.n_dofs);

    domain_pairs
}

fn domains_for_pair<S: ObservationStore>(
    design: &WeightedDesign<S>,
    q: usize,
    r: usize,
) -> Vec<(Subdomain, CrossTab)> {
    let (full_ct, l2g) = match CrossTab::build_for_pair(design, q, r) {
        Some(pair) => pair,
        None => return Vec::new(),
    };

    let n_q_full = full_ct.n_q();
    let factor_pair = (q, r);
    let components = full_ct.bipartite_connected_components();

    // Build per-component CrossTabs (avoid cloning full_ct for single-component case)
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
            // Build local_to_global for this component: q-levels first, then r-levels
            let comp_l2g: Vec<usize> = comp
                .q_indices
                .iter()
                .map(|&i| l2g[i] as usize)
                .chain(comp.r_indices.iter().map(|&i| l2g[n_q_full + i] as usize))
                .collect();

            let core =
                super::SubdomainCore::uniform(comp_l2g.into_iter().map(|g| g as u32).collect());
            (Subdomain { factor_pair, core }, comp_ct)
        })
        .collect()
}

fn build_pairs(n_factors: usize, star_ref: Option<usize>) -> Vec<(usize, usize)> {
    match star_ref {
        Some(q_star) => (0..n_factors)
            .filter(|&r| r != q_star)
            .map(|r| (q_star.min(r), q_star.max(r)))
            .collect(),
        None => {
            let mut pairs = Vec::new();
            for q in 0..n_factors {
                for r in (q + 1)..n_factors {
                    pairs.push((q, r));
                }
            }
            pairs
        }
    }
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

    fn make_test_design() -> FixedEffectsDesign {
        FixedEffectsDesign::new(
            vec![
                vec![0, 1, 2, 0, 1, 2],
                vec![0, 1, 0, 1, 0, 1],
                vec![0, 0, 1, 1, 0, 1],
            ],
            vec![3, 2, 2],
            6,
        )
        .expect("valid test design")
    }

    #[test]
    fn test_full_cover_domain_count() {
        let dm = make_test_design();
        let domain_pairs = build_local_domains(&dm, None);
        // 3 factor pairs; each pair may produce multiple components
        assert!(domain_pairs.len() >= 3);
    }

    #[test]
    fn test_star_cover_domain_count() {
        let dm = make_test_design();
        let domain_pairs = build_local_domains(&dm, Some(0));
        assert!(domain_pairs.len() >= 2);
    }

    #[test]
    fn test_partition_of_unity() {
        let dm = make_test_design();
        let domain_pairs = build_local_domains(&dm, None);
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
        let domain_pairs = build_local_domains(&dm, None);
        let mut covered = vec![false; dm.n_dofs];
        for (d, _) in &domain_pairs {
            for &idx in &d.core.global_indices {
                covered[idx as usize] = true;
            }
        }
        assert!(covered.iter().all(|&c| c), "Not all DOFs covered");
    }
}
