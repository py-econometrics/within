//! Channel-pair subdomain construction.
//!
//! Each cross-factor channel pair becomes a Schwarz subdomain (one per
//! connected component of its bipartite cross-tab). Overlap is handled by
//! partition-of-unity weights — see [`schwarz_precond::domain`] for the math.
//!
//! Entry point: [`build_local_domains`].

use schwarz_precond::{PartitionWeights, SubdomainCore};

use crate::channel::{Channel, ChannelPair};
use crate::config::ScalingConfig;
use crate::{BuildError, BuildWarning};

use super::{find_all_active_levels, BlockDiagonals, CrossTab, Design};

mod sddm;
use crate::domain::Loading;
use sddm::{convert, NotScalable};
pub(crate) use sddm::{CoordinateMap, Grounding, LocalComponent, MatrixForm, SddmMatrix};

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

/// Same-factor channel pairs are exactly orthogonal after whitening, so never enumerated.
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
        .flat_map(|(i, &rows)| {
            channels[i + 1..]
                .iter()
                .filter(move |cols| cols.term != rows.term)
                .map(move |&cols| ChannelPair { rows, cols })
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
            let class = if matches!(design.loading(pair.rows), Loading::Constant)
                && matches!(design.loading(pair.cols), Loading::Constant)
            {
                ComponentClass::KnownLaplacian
            } else {
                ComponentClass::General
            };
            split_into_subdomains(pair, class, full_ct, full_diag, &l2g, scaling)
        })
        .collect::<Result<_, BuildError>>()?;
    let mut domain_pairs = Vec::new();
    let mut warnings = Vec::new();
    for (domains, pair_warnings) in per_pair {
        domain_pairs.extend(domains);
        warnings.extend(pair_warnings);
    }

    // A slope channel breaks `1/√c`'s equal-informativeness assumption (#94), so stay uniform.
    if !channels
        .iter()
        .any(|&c| design.loading(c).covariate().is_some())
    {
        compute_partition_weights(&mut domain_pairs, design.n_dofs);
    }

    Ok((domain_pairs, warnings))
}

/// Dead singletons (zero diagonal, an exact-zero design column) produce no subdomain.
fn split_into_subdomains(
    pair: ChannelPair,
    class: ComponentClass,
    full_ct: CrossTab,
    full_diag: BlockDiagonals,
    l2g: &[u32],
    scaling: &ScalingConfig,
) -> Result<(Vec<LocalDomain>, Vec<BuildWarning>), BuildError> {
    let n_rows_full = full_ct.n_rows();
    let components = full_ct.bipartite_connected_components();

    let (cross_tabs, diagonals): (Vec<CrossTab>, Vec<Vec<f64>>) = if components.len() == 1 {
        let flat = full_diag.rows.into_iter().chain(full_diag.cols).collect();
        (vec![full_ct], vec![flat])
    } else {
        let mut row_remap = vec![u32::MAX; full_ct.n_rows()];
        let mut col_remap = vec![u32::MAX; full_ct.n_cols()];
        let cross_tabs = components
            .iter()
            .map(|comp| full_ct.extract_component(comp, &mut row_remap, &mut col_remap))
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
        if comp_diag.iter().all(|&v| v == 0.0) {
            continue;
        }
        let comp_globals: Vec<u32> = comp
            .rows
            .iter()
            .map(|&i| l2g[i])
            .chain(comp.cols.iter().map(|&i| l2g[n_rows_full + i]))
            .collect();
        let (comp_ct, comp_diag, comp_globals) =
            sddm::orient_for_elimination(comp_ct, comp_diag, comp_globals);
        let (component, uncertified) = convert(comp_ct, comp_diag, class, scaling)
            .map_err(|NotScalable| BuildError::UnscalableComponent { pair })?;
        if let Some(uncertified) = uncertified {
            warnings.push(BuildWarning::UnscalableComponent {
                pair,
                sweeps: uncertified.sweeps,
                violation: uncertified.violation,
            });
        }
        domains.push(LocalDomain {
            core: schwarz_precond::SubdomainCore::uniform(comp_globals),
            component,
        });
    }
    Ok((domains, warnings))
}

/// Two-sided Schwarz needs `Σ Rᵢᵀ D̃ᵢ² Rᵢ = I`, so a DOF in `c` subdomains gets `1/√c`.
fn compute_partition_weights(domain_pairs: &mut [LocalDomain], n_dofs: usize) {
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    // Atomic increments commute, so the parallel accumulation matches the serial scan.
    let counts: Vec<AtomicU32> = (0..n_dofs).map(|_| AtomicU32::new(0)).collect();
    domain_pairs.par_iter().for_each(|ld| {
        for &idx in ld.core.global_indices() {
            debug_assert!((idx as usize) < n_dofs);
            counts[idx as usize].fetch_add(1, Ordering::Relaxed);
        }
    });
    let counts: Vec<u32> = counts.into_iter().map(AtomicU32::into_inner).collect();

    // Each subdomain's weights depend only on shared counts, so per-domain work is independent.
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
    use crate::Effect;

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
    fn slope_design_keeps_uniform_partition_weights() {
        // Slope designs keep uniform weights; 1/√c reweighting collapses their convergence (#94).
        let levels_a = [0u32, 1, 2, 0, 1, 2];
        let levels_b = [0u32, 1, 0, 1, 0, 1];
        let levels_c = [0u32, 0, 1, 1, 0, 1];
        let z = [1.0, -2.0, 0.5, 3.0, -1.5, 2.5];
        let design = Design::new(vec![
            Effect::new(&levels_a, true, [&z[..]]).expect("slope effect"),
            Effect::new(&levels_b, true, []).expect("effect b"),
            Effect::new(&levels_c, true, []).expect("effect c"),
        ])
        .expect("valid slope design");

        let (domain_pairs, _) = build_local_domains(&design, None, &ScalingConfig::default())
            .expect("slope domains build");

        for ld in &domain_pairs {
            for i in 0..ld.core.global_indices().len() {
                assert_eq!(
                    ld.core.partition_weights().get(i),
                    1.0,
                    "slope design must keep uniform partition weights"
                );
            }
        }

        // Non-vacuity: without a shared DOF, uniform vs 1/√c weights are indistinguishable.
        let mut counts = vec![0u32; design.n_dofs];
        for ld in &domain_pairs {
            for &idx in ld.core.global_indices() {
                counts[idx as usize] += 1;
            }
        }
        assert!(
            counts.iter().any(|&c| c > 1),
            "test is vacuous: build a design whose subdomains share a DOF"
        );
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
