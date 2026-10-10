//! Channel-pair subdomain construction.
//!
//! Each cross-factor channel pair becomes a Schwarz subdomain (one per
//! connected component of its bipartite cross-tab; isolated levels share one
//! edgeless subdomain per side). Overlap is handled by
//! partition-of-unity weights — see [`schwarz_precond::domain`] for the math.
//!
//! Entry point: [`build_local_domains`].

use schwarz_precond::{PartitionWeights, SubdomainCore};

use crate::channel::{Channel, ChannelPair};
use crate::config::LocalSolverConfig;
use crate::csr_block::to_u32;
use crate::{BuildError, BuildWarning};

use super::{CrossTab, PreparedDesign};

mod grounding;
mod sddm;
use crate::domain::cross_tab::{BipartiteComponent, PairColumns};
use crate::domain::Column;
use grounding::scaled_groundings;
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
    prepared: &PreparedDesign<'_>,
    config: &LocalSolverConfig,
) -> Result<(Vec<LocalDomain>, Vec<BuildWarning>), BuildError> {
    use rayon::prelude::*;

    let design = &prepared.design;

    config.validate()?;

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
    let per_pair: Vec<(Vec<LocalDomain>, Vec<BuildWarning>)> = pairs
        .par_iter()
        .map(|&pair| {
            let (full_ct, l2g) = CrossTab::build_for_pair(prepared, pair);
            let class = if design.column(pair.rows) == Column::Intercept
                && design.column(pair.cols) == Column::Intercept
            {
                ComponentClass::KnownLaplacian
            } else {
                ComponentClass::General
            };
            split_into_subdomains(prepared, pair, class, full_ct, &l2g, config)
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
        .any(|&c| design.column(c) != Column::Intercept)
    {
        compute_partition_weights(&mut domain_pairs, design.n_dofs);
    }

    Ok((domain_pairs, warnings))
}

/// Dead singletons (zero diagonal, an exact-zero design column) produce no subdomain.
fn split_into_subdomains(
    prepared: &PreparedDesign<'_>,
    pair: ChannelPair,
    class: ComponentClass,
    full_ct: CrossTab,
    l2g: &[u32],
    config: &LocalSolverConfig,
) -> Result<(Vec<LocalDomain>, Vec<BuildWarning>), BuildError> {
    let row_diag = prepared.channel_diagonal(pair.rows);
    let col_diag = prepared.channel_diagonal(pair.cols);
    debug_assert_eq!(
        (row_diag.len(), col_diag.len()),
        (full_ct.n_rows(), full_ct.n_cols())
    );
    let n_rows_full = full_ct.n_rows();
    // An isolated level's Gram row is diagonal, so batching changes no local solve, only overhead.
    let mut components = Vec::new();
    let (mut isolated_rows, mut isolated_cols) = (Vec::new(), Vec::new());
    for comp in full_ct.bipartite_connected_components() {
        match (&comp.rows[..], &comp.cols[..]) {
            (&[i], []) if row_diag[i] != 0.0 => isolated_rows.push(i),
            ([], &[j]) if col_diag[j] != 0.0 => isolated_cols.push(j),
            _ => components.push(comp),
        }
    }
    for (rows, cols) in [(isolated_rows, Vec::new()), (Vec::new(), isolated_cols)] {
        if !rows.is_empty() || !cols.is_empty() {
            components.push(BipartiteComponent { rows, cols });
        }
    }

    let cross_tabs: Vec<CrossTab> = if components.len() == 1 {
        vec![full_ct]
    } else {
        let mut row_remap = vec![u32::MAX; full_ct.n_rows()];
        let mut col_remap = vec![u32::MAX; full_ct.n_cols()];
        components
            .iter()
            .map(|comp| full_ct.extract_component(comp, &mut row_remap, &mut col_remap))
            .collect()
    };

    // Only folded components are tested: a frustrated block is nonsingular, so it never floats.
    let mut levels: Option<Vec<Option<(usize, f64)>>> = None;
    let mut domains = Vec::with_capacity(components.len());
    let mut warnings = Vec::new();
    for (comp, comp_ct) in components.iter().zip(cross_tabs) {
        let comp_diag: Vec<f64> = comp
            .rows
            .iter()
            .map(|&i| row_diag[i])
            .chain(comp.cols.iter().map(|&i| col_diag[i]))
            .collect();
        if comp_diag.iter().all(|&v| v == 0.0) {
            continue;
        }
        // Pair-local `[rows | cols]` positions, mapped to globals once oriented.
        let comp_indices: Vec<u32> = comp
            .rows
            .iter()
            .map(|&i| to_u32(i))
            .chain(comp.cols.iter().map(|&j| to_u32(n_rows_full + j)))
            .collect();
        let (comp_ct, comp_diag, mut comp_indices) =
            sddm::orient_for_elimination(comp_ct, comp_diag, comp_indices);
        let (component, uncertified) = convert(comp_ct, comp_diag, class, &config.scaling)
            .map_err(|NotScalable| BuildError::UnscalableComponent { pair })?;
        if let Some(uncertified) = uncertified {
            warnings.push(BuildWarning::UnscalableComponent {
                pair,
                iterations: uncertified.iterations,
                violation: uncertified.violation,
            });
        }
        if class == ComponentClass::General && component.form == MatrixForm::Laplacian {
            let levels = levels.get_or_insert_with(|| vec![None; n_rows_full + col_diag.len()]);
            let mut factors = vec![1.0; comp_indices.len()];
            component
                .coordinates
                .fold(&mut factors, component.matrix.n_eliminated());
            for (&p, &f) in comp_indices.iter().zip(&factors) {
                levels[p as usize] = Some((domains.len(), f));
            }
        }
        for index in &mut comp_indices {
            *index = l2g[*index as usize];
        }
        domains.push(LocalDomain {
            core: schwarz_precond::SubdomainCore::uniform(comp_indices),
            component,
        });
    }

    // A folded component's dropped surplus is `fᵀGf` exactly, so only those read the observations.
    if let Some(levels) = levels {
        let columns = PairColumns::new(prepared, pair);
        let groundings = scaled_groundings(&columns, &levels, n_rows_full, domains.len());
        for (ld, grounding) in domains.iter_mut().zip(groundings) {
            if grounding == Grounding::Floating {
                ld.component.matrix.float();
            }
        }
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
    use crate::domain::{Design, PreparedDesign};
    use crate::Effect;

    fn make_test_design() -> PreparedDesign<'static> {
        PreparedDesign::from_levels_for_test(vec![
            vec![0, 1, 2, 0, 1, 2],
            vec![0, 1, 0, 1, 0, 1],
            vec![0, 0, 1, 1, 0, 1],
        ])
    }

    #[test]
    fn test_full_cover_domain_count() {
        let dm = make_test_design();
        let (domain_pairs, _) =
            build_local_domains(&dm, &LocalSolverConfig::default()).expect("plain domains build");
        // 3 factor pairs; each pair may produce multiple components
        assert!(domain_pairs.len() >= 3);
    }

    #[test]
    fn test_partition_of_unity() {
        let dm = make_test_design();
        let (domain_pairs, _) =
            build_local_domains(&dm, &LocalSolverConfig::default()).expect("plain domains build");
        let n_dofs = dm.design.n_dofs;
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
        let design = PreparedDesign::unweighted_for_test(design);

        let (domain_pairs, _) = build_local_domains(&design, &LocalSolverConfig::default())
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
        let mut counts = vec![0u32; design.design.n_dofs];
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

    /// `z = a_row · b_col` aliases the intercept; a floated block keeps no surplus.
    #[rstest::rstest]
    #[case::aliased(0.0, Grounding::Floating)]
    #[case::identified(1e-4, Grounding::Grounded)]
    fn a_folded_alias_floats_without_surplus(#[case] delta: f64, #[case] expected: Grounding) {
        let (rows, cols) = ([0u32, 0, 1, 1, 2, 2], [0u32, 1, 0, 1, 0, 1]);
        let z = [1.0, 3.0, -2.0, -6.0 * (1.0 + delta), 0.5, 1.5];
        let weights = [0.3, 2.0, 1.0, 7.5, 0.01, 4.0];
        let design = Design::new(vec![
            Effect::new(&rows, false, [&z[..]]).expect("slope effect"),
            Effect::new(&cols, true, []).expect("plain effect"),
        ])
        .expect("valid design");
        let prepared = PreparedDesign::new(design, Some(&weights)).expect("valid weights");
        let (domains, _) =
            build_local_domains(&prepared, &LocalSolverConfig::default()).expect("domains");
        let [ld] = &domains[..] else {
            panic!("one component expected, got {}", domains.len())
        };
        let matrix = &ld.component.matrix;
        assert_eq!(ld.component.form, MatrixForm::Laplacian);
        assert_eq!(matrix.grounding, expected);
        if expected == Grounding::Floating {
            assert!(matrix.ground_edges.iter().all(|&g| g == 0.0));
            for (i, &d) in matrix.diagonal.iter().enumerate() {
                let sum: f64 = matrix.cross_tab.neighbors(i).map(|(_, v)| v.abs()).sum();
                assert_eq!(d, sum, "row {i}");
            }
        }
    }

    #[test]
    fn isolated_levels_share_one_subdomain() {
        // Workers 0 and 1 load ±z within one firm, so their cross cells cancel to exactly zero.
        let workers = [0u32, 0, 1, 1, 2, 2];
        let firms = [0u32, 0, 1, 1, 0, 1];
        let z = [1.0, -1.0, 2.0, -2.0, 1.0, 1.0];
        let design = Design::new(vec![
            Effect::new(&workers, false, [&z[..]]).expect("slope effect"),
            Effect::new(&firms, true, []).expect("firm effect"),
        ])
        .expect("valid design");
        let prepared = PreparedDesign::unweighted_for_test(design);
        let (domains, _) =
            build_local_domains(&prepared, &LocalSolverConfig::default()).expect("domains");
        let mut cores: Vec<Vec<u32>> = domains
            .iter()
            .map(|d| d.core.global_indices().to_vec())
            .collect();
        cores.iter_mut().for_each(|core| core.sort_unstable());
        cores.sort_unstable();
        assert_eq!(cores, [vec![0, 1], vec![2, 3, 4]]);
    }

    #[test]
    fn test_domains_cover_all_dofs() {
        let dm = make_test_design();
        let (domain_pairs, _) =
            build_local_domains(&dm, &LocalSolverConfig::default()).expect("plain domains build");
        let mut covered = vec![false; dm.design.n_dofs];
        for ld in &domain_pairs {
            for &idx in ld.core.global_indices() {
                covered[idx as usize] = true;
            }
        }
        assert!(covered.iter().all(|&c| c), "Not all DOFs covered");
    }
}
