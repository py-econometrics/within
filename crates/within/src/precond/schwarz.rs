//! The additive Schwarz rung over channel-pair subdomains.
//!
//! Each cross-factor channel pair hands approx-chol its Gram factor `V`, one column per
//! observation, and its block-diagonal approximate inverse becomes one Schwarz subdomain.
//! Overlap is handled by partition-of-unity weights — see [`schwarz_precond::domain`] for
//! the math.

use approx_chol::{ApproxInverse, ScalingOutcome};
use rayon::prelude::*;
use schwarz_precond::{
    LocalSolveError, LocalSolver, MemStack, Operator, PartitionWeights, SchwarzPreconditioner,
    StackReq, SubdomainCore, SubdomainEntry, Threads,
};
use serde::{Deserialize, Serialize};

use crate::config::{LocalSolverConfig, ScalingFailure};
use crate::system::{Gramian, PairBlock};
use crate::{BuildError, BuildWarning};

/// The rung's map: additive Schwarz over the channel-pair subdomains.
#[derive(Clone, Serialize, Deserialize)]
pub(super) struct FeSchwarz {
    inner: SchwarzPreconditioner<BlockSolver>,
}

impl std::fmt::Debug for FeSchwarz {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FeSchwarz")
            .field("n_subdomains", &self.inner.subdomains().len())
            .finish()
    }
}

impl FeSchwarz {
    pub(super) fn operator(&self) -> &dyn Operator {
        &self.inner
    }
}

/// The Schwarz map over every cross-term channel pair; `None` without a pair to act on.
pub(super) fn build(
    gramian: Gramian<'_>,
    config: &LocalSolverConfig,
) -> Result<(Option<FeSchwarz>, Vec<BuildWarning>), BuildError> {
    let (domains, warnings) = build_local_domains(gramian, config)?;
    if domains.is_empty() {
        return Ok((None, warnings));
    }
    Ok((Some(build_additive(domains, gramian.n())?), warnings))
}

/// `n_dofs` may exceed the span of subdomain indices; an uncovered column resolves to `0`.
fn build_additive(domains: Vec<LocalDomain>, n_dofs: usize) -> Result<FeSchwarz, BuildError> {
    let entries = domains
        .into_par_iter()
        .map(build_entry)
        .collect::<Result<Vec<_>, BuildError>>()?;
    Ok(FeSchwarz {
        inner: SchwarzPreconditioner::with_n_dofs(entries, n_dofs),
    })
}

fn build_entry(domain: LocalDomain) -> Result<SubdomainEntry<BlockSolver>, BuildError> {
    let LocalDomain { core, solver } = domain;
    SubdomainEntry::try_new(core, solver).map_err(BuildError::Preconditioner)
}

/// A factor-pair Schwarz domain paired with its factored local operator.
#[derive(Clone)]
struct LocalDomain {
    core: SubdomainCore,
    solver: BlockSolver,
}

/// One subdomain per pair block of `S` with a component to act on.
fn build_local_domains(
    gramian: Gramian<'_>,
    config: &LocalSolverConfig,
) -> Result<(Vec<LocalDomain>, Vec<BuildWarning>), BuildError> {
    let (approx, scaling) = (config.factor_config(), config.scaling_policy()?);
    let on_failure = config.scaling.on_failure;
    let per_pair: Vec<(Option<LocalDomain>, Vec<BuildWarning>)> = gramian
        .pair_blocks()
        .into_par_iter()
        .map(|block| factor_pair(block, approx, scaling, on_failure))
        .collect::<Result<_, BuildError>>()?;
    let mut domain_pairs = Vec::new();
    let mut warnings = Vec::new();
    for (domain, pair_warnings) in per_pair {
        domain_pairs.extend(domain);
        warnings.extend(pair_warnings);
    }

    // A slope channel breaks `1/√c`'s equal-informativeness assumption (#94), so stay uniform;
    // the complement pairs read `e`'s channels too.
    if !gramian.has_slopes() {
        compute_partition_weights(&mut domain_pairs, gramian.n());
    }

    Ok((domain_pairs, warnings))
}

/// The approximate inverse of `block`, one subdomain over its vertices.
fn factor_pair(
    block: PairBlock<'_>,
    approx: approx_chol::Config,
    scaling: approx_chol::ScalingPolicy,
    on_failure: ScalingFailure,
) -> Result<(Option<LocalDomain>, Vec<BuildWarning>), BuildError> {
    let pair = block.pair;
    let inverse = (block.factorize(approx, scaling))
        .map_err(|e| BuildError::LocalSolverBuild(e.to_string()))?;

    let mut warnings = Vec::new();
    for clamp in inverse.components().filter_map(|c| c.clamp()) {
        if clamp.scaling == ScalingOutcome::Certified {
            continue;
        }
        match on_failure {
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

    let global = |vertex| u32::try_from(block.unknown(vertex)).expect("the DOF space fits u32");
    // The factor's components are disjoint blocks, so one subdomain per pair solves each alone.
    let domain = LocalDomain {
        core: SubdomainCore::uniform((0..inverse.n()).map(global).collect()),
        solver: BlockSolver { inverse },
    };
    Ok((Some(domain), warnings))
}

/// Two-sided Schwarz needs `Σ Rᵢᵀ D̃ᵢ² Rᵢ = I`, so a DOF in `c` subdomains gets `1/√c`.
fn compute_partition_weights(domain_pairs: &mut [LocalDomain], n_dofs: usize) {
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

/// One pair's factor; the subdomain's local order is the factored matrix's numbering.
#[derive(Clone, Serialize, Deserialize)]
struct BlockSolver {
    inverse: ApproxInverse,
}

fn budget(threads: Threads) -> approx_chol::Threads {
    match threads {
        Threads::All => approx_chol::Threads::All,
        Threads::Count(n) => approx_chol::Threads::Count(n),
    }
}

impl LocalSolver for BlockSolver {
    fn n_local(&self) -> usize {
        self.inverse.n()
    }

    fn scratch(&self, threads: Threads) -> StackReq {
        self.inverse.scratch(1, budget(threads))
    }

    fn solve_local(
        &self,
        x: &mut [f64],
        threads: Threads,
        stack: &mut MemStack,
    ) -> Result<(), LocalSolveError> {
        self.inverse
            .apply_with(x, budget(threads), stack)
            .map_err(|e| LocalSolveError::BackendFailed {
                context: "BlockSolver::solve_local",
                message: e.to_string(),
            })
    }

    fn inner_parallelism_work_estimate(&self) -> usize {
        self.inverse.n_steps()
    }
}

#[cfg(test)]
mod build_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::design::Design;
    use crate::system::ReducedDesign;
    use crate::Effect;

    fn make_test_design() -> ReducedDesign<'static> {
        Design::from_levels_for_test(vec![
            vec![0, 1, 2, 0, 1, 2],
            vec![0, 1, 0, 1, 0, 1],
            vec![0, 0, 1, 1, 0, 1],
        ])
        .reduce(None)
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
        let design = design.reduce(None).unwrap();

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
        // Firms tie workers at five levels and come last, so they are eliminated; worker 4's zero
        // z leaves its column of `S` dead.
        let workers = [0u32, 0, 1, 1, 2, 2, 3, 3, 4];
        let firms = [0u32, 0, 2, 2, 0, 1, 3, 3, 4];
        let z = [1.0, -1.0, 2.0, -2.0, 1.0, 1.0, 3.0, -3.0, 0.0];
        let design = Design::new(vec![
            Effect::new(&workers, false, [&z[..]]).expect("slope effect"),
            Effect::new(&firms, true, []).expect("firm effect"),
        ])
        .expect("valid design");
        let design = design.reduce(None).unwrap();
        let (domains, _) =
            build_local_domains(Gramian::new(&design), &LocalSolverConfig::default())
                .expect("domains");
        let [domain] = &domains[..] else {
            panic!("one subdomain per pair, got {}", domains.len())
        };
        // The workers' slopes are the only unknowns.
        assert_eq!(domain.core.global_indices(), (0..5).collect::<Vec<u32>>());
        let mut x = vec![1.0; 5];
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
