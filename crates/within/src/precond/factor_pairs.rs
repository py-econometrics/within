//! Channel-pair subdomain construction.
//!
//! Each cross-factor channel pair hands approx-chol its Gram factor `V`, one column per
//! observation, and its block-diagonal approximate inverse becomes one Schwarz subdomain.
//! Overlap is handled by partition-of-unity weights — see [`schwarz_precond::domain`] for
//! the math.
//!
//! Entry point: [`build_local_domains`].

use approx_chol::ScalingOutcome;
use schwarz_precond::{PartitionWeights, SubdomainCore};

use crate::config::{LocalSolverConfig, ScalingFailure};
use crate::design::Column;
use crate::design::{Channel, ChannelPair};
use crate::precond::block_solver::BlockSolver;
use crate::{BuildError, BuildWarning};

use super::Gramian;

/// A factor-pair Schwarz domain paired with its factored local operator.
#[derive(Clone)]
pub(crate) struct LocalDomain {
    pub(crate) core: SubdomainCore,
    pub(crate) solver: BlockSolver,
}

/// Same-factor channel pairs are exactly orthogonal after orthogonalization, so never enumerated.
pub(crate) fn build_local_domains(
    gramian: Gramian<'_>,
    config: &LocalSolverConfig,
) -> Result<(Vec<LocalDomain>, Vec<BuildWarning>), BuildError> {
    use rayon::prelude::*;

    let layout = gramian.layout();

    config.validate()?;

    let channels: Vec<Channel> = (0..layout.n_terms())
        .flat_map(|term| layout.channels(term))
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
    let per_pair: Vec<(Option<LocalDomain>, Vec<BuildWarning>)> = pairs
        .par_iter()
        .map(|&pair| factor_pair(gramian, pair, config))
        .collect::<Result<_, BuildError>>()?;
    let mut domain_pairs = Vec::new();
    let mut warnings = Vec::new();
    for (domain, pair_warnings) in per_pair {
        domain_pairs.extend(domain);
        warnings.extend(pair_warnings);
    }

    // A slope channel breaks `1/√c`'s equal-informativeness assumption (#94), so stay uniform.
    if !channels
        .iter()
        .any(|&c| layout.column(c) != Column::Intercept)
    {
        compute_partition_weights(&mut domain_pairs, gramian.unknowns().len());
    }

    Ok((domain_pairs, warnings))
}

/// `G = V Vᵀ`, observation `k` adding `√w_k (z_q(k) e_{a_k} + z_r(k) e_{n_q + b_k})`.
fn factor_pair(
    gramian: Gramian<'_>,
    pair: ChannelPair,
    config: &LocalSolverConfig,
) -> Result<(Option<LocalDomain>, Vec<BuildWarning>), BuildError> {
    let v = gramian.pair_factor(pair);
    let inverse = approx_chol::factorize_with(v, config.factor_config(), config.scaling_policy()?)
        .map_err(|e| BuildError::LocalSolverBuild(e.to_string()))?;

    let mut warnings = Vec::new();
    for clamp in inverse.components().filter_map(|c| c.clamp()) {
        if clamp.scaling == ScalingOutcome::Certified {
            continue;
        }
        match config.scaling.on_failure {
            ScalingFailure::Warn => warnings.push(BuildWarning::UnscalableComponent {
                pair,
                violation: clamp.violation,
            }),
            ScalingFailure::Error => return Err(BuildError::UnscalableComponent { pair }),
        }
    }
    if inverse.components().len() == 0 {
        return Ok((None, warnings));
    }

    let (unknowns, layout) = (gramian.unknowns(), gramian.layout());
    let (rows, cols) = (
        unknowns.dofs(layout, pair.rows),
        unknowns.dofs(layout, pair.cols),
    );
    let (n_rows, row_base, col_base) = (rows.len(), rows.start, cols.start);
    let global = |vertex: usize| {
        let dof = if vertex < n_rows {
            row_base + vertex
        } else {
            col_base + vertex - n_rows
        };
        u32::try_from(dof).expect("the DOF space fits u32")
    };
    // The factor's components are disjoint blocks, so one subdomain per pair solves each alone.
    let domain = LocalDomain {
        core: SubdomainCore::uniform((0..inverse.n()).map(global).collect()),
        solver: BlockSolver::new(inverse),
    };
    Ok((Some(domain), warnings))
}

/// Two-sided Schwarz needs `Σ Rᵢᵀ D̃ᵢ² Rᵢ = I`, so a DOF in `c` subdomains gets `1/√c`.
fn compute_partition_weights(domain_pairs: &mut [LocalDomain], n_dofs: usize) {
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    // Atomic increments commute, so the parallel accumulation matches the serial scan.
    let counts: Vec<AtomicU32> = (0..n_dofs).map(|_| AtomicU32::new(0)).collect();
    domain_pairs.par_iter().for_each(|ld| {
        for &idx in ld.core.global_indices() {
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
    use crate::design::Design;
    use crate::system::OrthogonalDesign;
    use crate::Effect;

    fn make_test_design() -> OrthogonalDesign<'static> {
        Design::from_levels_for_test(vec![
            vec![0, 1, 2, 0, 1, 2],
            vec![0, 1, 0, 1, 0, 1],
            vec![0, 0, 1, 1, 0, 1],
        ])
        .orthogonalize(None)
        .unwrap()
    }

    #[test]
    fn test_full_cover_domain_count() {
        let design = make_test_design();
        let (domain_pairs, _) =
            build_local_domains(Gramian::new(&design), &LocalSolverConfig::default())
                .expect("plain domains build");
        // 3 factor pairs; each pair may produce multiple components
        assert!(domain_pairs.len() >= 3);
    }

    #[test]
    fn test_partition_of_unity() {
        let design = make_test_design();
        let (domain_pairs, _) =
            build_local_domains(Gramian::new(&design), &LocalSolverConfig::default())
                .expect("plain domains build");
        let n_dofs = design.unknowns().len();
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
        let design = design.orthogonalize(None).unwrap();

        let (domain_pairs, _) =
            build_local_domains(Gramian::new(&design), &LocalSolverConfig::default())
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
        let mut counts = vec![0u32; design.unknowns().len()];
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
    fn one_subdomain_per_pair_solves_dead_levels_to_zero() {
        use schwarz_precond::{LocalSolver, MemBuffer, MemStack, Threads};
        // ±z within one firm cancels a cross cell to exactly zero; worker 4's zero z is dead.
        let workers = [0u32, 0, 1, 1, 2, 2, 3, 3, 4];
        let firms = [0u32, 0, 2, 2, 0, 1, 3, 3, 0];
        let z = [1.0, -1.0, 2.0, -2.0, 1.0, 1.0, 3.0, -3.0, 0.0];
        let design = Design::new(vec![
            Effect::new(&workers, false, [&z[..]]).expect("slope effect"),
            Effect::new(&firms, true, []).expect("firm effect"),
        ])
        .expect("valid design");
        let design = design.orthogonalize(None).unwrap();
        let (domains, _) =
            build_local_domains(Gramian::new(&design), &LocalSolverConfig::default())
                .expect("domains");
        let [domain] = &domains[..] else {
            panic!("one subdomain per pair, got {}", domains.len())
        };
        // Workers occupy DOFs 0..5 and firms 5..9, in the pair's own numbering.
        assert_eq!(domain.core.global_indices(), (0..9).collect::<Vec<u32>>());
        let mut x = vec![1.0; 9];
        let mut buffer = MemBuffer::new(domain.solver.scratch(Threads::All));
        domain
            .solver
            .solve_local(&mut x, Threads::All, MemStack::new(&mut buffer))
            .expect("solve");
        assert_eq!(x[4], 0.0);
        assert!(
            x.iter().enumerate().all(|(i, v)| i == 4 || *v != 0.0),
            "{x:?}"
        );
    }

    #[test]
    fn test_domains_cover_all_dofs() {
        let design = make_test_design();
        let (domain_pairs, _) =
            build_local_domains(Gramian::new(&design), &LocalSolverConfig::default())
                .expect("plain domains build");
        let mut covered = vec![false; design.unknowns().len()];
        for ld in &domain_pairs {
            for &idx in ld.core.global_indices() {
                covered[idx as usize] = true;
            }
        }
        assert!(covered.iter().all(|&c| c), "Not all DOFs covered");
    }
}
