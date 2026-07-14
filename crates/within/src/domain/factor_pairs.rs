//! Channel-pair subdomain construction.
//!
//! Each cross-factor channel pair becomes a Schwarz subdomain (one per
//! connected component of its bipartite cross-tab). Overlap is handled by
//! partition-of-unity weights — see [`schwarz_precond::domain`] for the math.
//!
//! Entry point: [`build_local_domains`].

use schwarz_precond::{PartitionWeights, SubdomainCore};

use crate::config::ScalingConfig;
use crate::{BuildError, BuildWarning, SignedPair};

use super::{find_all_active_levels, BlockDiagonals, Channel, ChannelPair, CrossTab, Design};

mod sddm;
use sddm::{convert, NotScalable};
pub(crate) use sddm::{CoordinateMap, GroundEdges, LocalComponent, SchurReduction, SolveSpace};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ComponentClass {
    KnownLaplacian,
    General,
}

/// A factor-pair Schwarz domain paired with its validated local operator.
#[derive(Clone)]
pub(crate) struct LocalDomain {
    pub(crate) core: SubdomainCore,
    pub(crate) component: LocalComponent,
}

/// Build local subdomains (with pre-built CrossTabs) for cross-factor channel
/// pairs.
///
/// For each pair of distinct terms `q < r`, every channel of `q` is paired
/// with every channel of `r` (same-factor channel pairs are exactly
/// orthogonal after whitening, so they are never enumerated). Each channel
/// pair builds a fused CrossTab via one observation scan, detects connected
/// components on the bipartite structure, and creates one subdomain per
/// component. The converted SDDM component travels with each subdomain to avoid
/// rebuilding or re-inferring its numerical structure later.
///
/// Channel pairs are processed in parallel via Rayon. The
/// `compute_partition_weights` step remains sequential after the parallel
/// collect.
pub(crate) fn build_local_domains(
    design: &Design<'_>,
    weights: Option<&[f64]>,
    scaling: &ScalingConfig,
) -> Result<(Vec<LocalDomain>, Vec<BuildWarning>), BuildError> {
    use rayon::prelude::*;

    let channels: Vec<Channel> = (0..design.n_factors())
        .flat_map(|term| design.channels(term))
        .collect();
    let pairs: Vec<ChannelPair> = channels
        .iter()
        .enumerate()
        .flat_map(|(i, &q)| {
            channels[i + 1..]
                .iter()
                .filter(move |r| r.term != q.term)
                .map(move |&r| ChannelPair { q, r })
        })
        .collect();
    let all_active = find_all_active_levels(design);

    let per_pair: Vec<(Vec<LocalDomain>, Vec<BuildWarning>)> = pairs
        .par_iter()
        .map(|&pair| {
            let Some((full_ct, full_diag, l2g)) =
                CrossTab::build_for_pair_with_active(design, weights, pair, &all_active)
            else {
                return Ok((Vec::new(), Vec::new()));
            };
            split_into_subdomains(pair, full_ct, full_diag, &l2g, scaling)
        })
        .collect::<Result<_, BuildError>>()?;
    let mut domain_pairs = Vec::new();
    let mut warnings = Vec::new();
    for (domains, pair_warnings) in per_pair {
        domain_pairs.extend(domains);
        warnings.extend(pair_warnings);
    }

    // 1/√c reweighting assumes every subdomain sharing a DOF is equally
    // informative about it; a slope channel breaks that assumption and
    // collapses convergence on weakly-connected designs (#94). Slope-carrying
    // designs keep every subdomain at uniform weight instead — the plain path
    // is measurably indifferent to this weighting, so this changes nothing there.
    if !channels.iter().any(|c| c.loading.is_some()) {
        compute_partition_weights(&mut domain_pairs, design.n_dofs);
    }

    Ok((domain_pairs, warnings))
}

/// Split a full CrossTab into per-component subdomains.
///
/// Finds bipartite connected components, extracts a sub-CrossTab and its
/// sliced [`BlockDiagonals`] for each, and converts every component into a
/// validated SDDM representation. Dead
/// singletons (zero diagonal — an exact-zero design column) produce no
/// subdomain, matching the uncovered-inactive-level invariant.
fn split_into_subdomains(
    pair: ChannelPair,
    full_ct: CrossTab,
    full_diag: BlockDiagonals,
    l2g: &[u32],
    scaling: &ScalingConfig,
) -> Result<(Vec<LocalDomain>, Vec<BuildWarning>), BuildError> {
    let n_q_full = full_ct.n_q();
    let components = full_ct.bipartite_connected_components();

    let (cross_tabs, diagonals): (Vec<CrossTab>, Vec<BlockDiagonals>) = if components.len() == 1 {
        (vec![full_ct], vec![full_diag])
    } else {
        // One reusable remap buffer pair for the whole parent; `extract_component`
        // resets it per component, avoiding a fresh parent-sized allocation each.
        let mut q_remap = vec![u32::MAX; full_ct.n_q()];
        let mut r_remap = vec![u32::MAX; full_ct.n_r()];
        let cross_tabs = components
            .iter()
            .map(|comp| full_ct.extract_component(comp, &mut q_remap, &mut r_remap))
            .collect();
        let diagonals = components
            .iter()
            .map(|comp| full_diag.extract_component(comp))
            .collect();
        (cross_tabs, diagonals)
    };

    let mut domains = Vec::with_capacity(components.len());
    let mut warnings = Vec::new();
    for ((comp, comp_ct), comp_diag) in components.iter().zip(cross_tabs).zip(diagonals) {
        if comp_diag.q.iter().chain(&comp_diag.r).all(|&v| v == 0.0) {
            continue;
        }
        let class = if pair.q.loading.is_none() && pair.r.loading.is_none() {
            ComponentClass::KnownLaplacian
        } else {
            ComponentClass::General
        };
        let signed_pair = SignedPair {
            term_q: pair.q.term,
            column_q: pair.q.column,
            term_r: pair.r.term,
            column_r: pair.r.column,
        };
        let (component, uncertified) = convert(comp_ct, comp_diag, class, scaling)
            .map_err(|NotScalable| BuildError::UnscalableComponent { pair: signed_pair })?;
        if let Some(uncertified) = uncertified {
            warnings.push(BuildWarning::UnscalableComponent {
                pair: signed_pair,
                sweeps: uncertified.sweeps,
                violation: uncertified.violation,
            });
        }
        let comp_l2g: Vec<u32> = comp
            .q_indices
            .iter()
            .map(|&i| l2g[i])
            .chain(comp.r_indices.iter().map(|&i| l2g[n_q_full + i]))
            .collect();
        domains.push(LocalDomain {
            core: schwarz_precond::SubdomainCore::uniform(comp_l2g),
            component,
        });
    }
    Ok((domains, warnings))
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
fn compute_partition_weights(domain_pairs: &mut [LocalDomain], n_dofs: usize) {
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    // Pass 1: histogram how many subdomains each DOF appears in. Atomic
    // increments commute, so the parallel accumulation matches the serial scan.
    let counts: Vec<AtomicU32> = (0..n_dofs).map(|_| AtomicU32::new(0)).collect();
    domain_pairs.par_iter().for_each(|ld| {
        for &idx in ld.core.global_indices() {
            debug_assert!((idx as usize) < n_dofs);
            counts[idx as usize].fetch_add(1, Ordering::Relaxed);
        }
    });
    let counts: Vec<u32> = counts.into_iter().map(AtomicU32::into_inner).collect();

    // Pass 2: each subdomain's weights depend only on the shared counts, so the
    // per-domain work is independent.
    domain_pairs.par_iter_mut().for_each(|ld| {
        let all_unique = ld
            .core
            .global_indices()
            .iter()
            .all(|&idx| counts[idx as usize] <= 1);
        if all_unique {
            ld.core.set_uniform_partition_weights();
        } else {
            let weights: Vec<f64> = ld
                .core
                .global_indices()
                .iter()
                .map(|&idx| {
                    let c = counts[idx as usize];
                    debug_assert!(c > 0);
                    1.0 / (c as f64).sqrt()
                })
                .collect();
            ld.core
                .set_partition_weights(PartitionWeights::NonUniform(weights))
                .expect("partition weight count must match index count");
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::Design;
    use crate::observation::ObservationFrame;

    fn make_test_design() -> Design<'static> {
        let frame = ObservationFrame::new(
            vec![
                vec![0u32, 1, 2, 0, 1, 2].into(),
                vec![0u32, 1, 0, 1, 0, 1].into(),
                vec![0u32, 0, 1, 1, 0, 1].into(),
            ],
            Vec::new(),
        )
        .expect("valid frame");
        Design::from_frame(frame).expect("valid test design")
    }

    #[test]
    fn test_full_cover_domain_count() {
        let dm = make_test_design();
        let (domain_pairs, _) =
            build_local_domains(&dm, None, &ScalingConfig::default()).expect("plain domains build");
        // 3 factor pairs; each pair may produce multiple components
        assert!(domain_pairs.len() >= 3);
    }

    #[test]
    fn test_partition_of_unity() {
        let dm = make_test_design();
        let (domain_pairs, _) =
            build_local_domains(&dm, None, &ScalingConfig::default()).expect("plain domains build");
        let n_dofs = dm.n_dofs;
        // Two-sided PoU: squared weights must sum to 1 at every DOF.
        let mut weight_sq_sum = vec![0.0; n_dofs];
        for ld in &domain_pairs {
            for (i, &idx) in ld.core.global_indices().iter().enumerate() {
                let w = ld.core.partition_weights().get(i);
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
        let (domain_pairs, _) =
            build_local_domains(&dm, None, &ScalingConfig::default()).expect("plain domains build");
        let mut covered = vec![false; dm.n_dofs];
        for ld in &domain_pairs {
            for &idx in ld.core.global_indices() {
                covered[idx as usize] = true;
            }
        }
        assert!(covered.iter().all(|&c| c), "Not all DOFs covered");
    }
}
