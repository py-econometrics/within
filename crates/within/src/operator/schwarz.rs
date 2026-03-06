//! Schwarz preconditioner: FE-specific construction helpers.
//!
//! Re-exports `SchwarzPreconditioner` from the `schwarz-precond` crate and
//! provides helpers that bridge FE types (design, subdomains, Gramian) to the crate's generic
//! `SubdomainEntry<BlockElimSolver>` API.

pub use schwarz_precond::MultiplicativeSchwarzPreconditioner;
pub use schwarz_precond::SchwarzPreconditioner;

use approx_chol::low_level::Builder;
use approx_chol::{Config, CsrRef};
use rayon::prelude::*;
use schwarz_precond::SubdomainEntry;

use super::gramian::CrossTab;
use super::local_solver::{
    ApproxCholSolver, BlockElimSolver, FeLocalSolver, LocalSolveStrategy, ReducedFactor,
};
use super::residual_update::{ObservationSpaceUpdater, SparseGramianUpdater};
use super::schur_complement::{
    ApproxSchurComplement, EliminationInfo, ExactSchurComplement, SchurComplement, SchurResult,
};
use crate::config::{ApproxSchurConfig, LocalSolverConfig, SchwarzConfig};
use crate::domain::{build_local_domains, Subdomain, WeightedDesign};
use crate::observation::ObservationStore;
use crate::{WithinError, WithinResult};

/// Concrete Schwarz type used in the parent crate.
pub type FeSchwarz = SchwarzPreconditioner<FeLocalSolver>;

/// Concrete multiplicative Schwarz type: one-level with observation-space residual updates.
pub type FeMultSchwarz<'a, S> =
    MultiplicativeSchwarzPreconditioner<FeLocalSolver, ObservationSpaceUpdater<'a, S>>;

/// Concrete multiplicative Schwarz type: one-level with explicit Gramian CSR residual updates.
pub type FeMultSchwarzSparse<'a> =
    MultiplicativeSchwarzPreconditioner<FeLocalSolver, SparseGramianUpdater<'a>>;

// ---------------------------------------------------------------------------
// Domain source abstraction
// ---------------------------------------------------------------------------

/// Abstracts over how domain entries are acquired.
pub(crate) enum DomainSource<'a, S: ObservationStore> {
    /// Scan observations to build from scratch.
    FromDesign(&'a WeightedDesign<S>),
    /// Reuse pre-built pairs (from fused domain+gramian pass).
    FromParts(Vec<(Subdomain, CrossTab)>),
}

// ---------------------------------------------------------------------------
// Public convenience builder
// ---------------------------------------------------------------------------

/// Build additive Schwarz from FE design with default domain decomposition.
pub fn build_schwarz<S: ObservationStore>(
    design: &WeightedDesign<S>,
    config: &SchwarzConfig,
) -> WithinResult<FeSchwarz> {
    build_additive(DomainSource::FromDesign(design), design.n_dofs, config)
}

// ---------------------------------------------------------------------------
// Crate-internal consolidated builders
// ---------------------------------------------------------------------------

/// Build additive Schwarz from any domain source.
pub(crate) fn build_additive<S: ObservationStore>(
    source: DomainSource<'_, S>,
    n_dofs: usize,
    config: &SchwarzConfig,
) -> WithinResult<FeSchwarz> {
    let entries = build_entries_from_source(source, config)?;
    Ok(SchwarzPreconditioner::new(entries, n_dofs)?)
}

/// Build multiplicative Schwarz with observation-space updater.
pub(crate) fn build_multiplicative_obs<'a, S: ObservationStore>(
    source: DomainSource<'_, S>,
    design: &'a WeightedDesign<S>,
    config: &SchwarzConfig,
    symmetric: bool,
) -> WithinResult<FeMultSchwarz<'a, S>> {
    let entries = build_entries_from_source(source, config)?;
    let updater = ObservationSpaceUpdater::new(design);
    Ok(MultiplicativeSchwarzPreconditioner::new(
        entries,
        updater,
        design.n_dofs,
        symmetric,
    )?)
}

/// Build multiplicative Schwarz with sparse Gramian updater.
pub(crate) fn build_multiplicative_sparse<'a, S: ObservationStore>(
    source: DomainSource<'_, S>,
    gramian: &'a super::gramian::Gramian,
    n_dofs: usize,
    config: &SchwarzConfig,
    symmetric: bool,
) -> WithinResult<FeMultSchwarzSparse<'a>> {
    let entries = build_entries_from_source(source, config)?;
    let updater = SparseGramianUpdater::new(&gramian.matrix);
    Ok(MultiplicativeSchwarzPreconditioner::new(
        entries, updater, n_dofs, symmetric,
    )?)
}

// ---------------------------------------------------------------------------
// Internal: build entries from source
// ---------------------------------------------------------------------------

fn build_entries_from_source<S: ObservationStore>(
    source: DomainSource<'_, S>,
    config: &SchwarzConfig,
) -> WithinResult<Vec<SubdomainEntry<FeLocalSolver>>> {
    match source {
        DomainSource::FromDesign(design) => {
            let domain_pairs = build_local_domains(design, None);
            build_entries_from_pairs(domain_pairs, config)
        }
        DomainSource::FromParts(domain_pairs) => build_entries_from_pairs(domain_pairs, config),
    }
}

fn build_entries_from_pairs(
    domain_pairs: Vec<(Subdomain, CrossTab)>,
    config: &SchwarzConfig,
) -> WithinResult<Vec<SubdomainEntry<FeLocalSolver>>> {
    domain_pairs
        .into_par_iter()
        .map(|(domain, cross_tab)| build_entry(domain, cross_tab, config))
        .collect()
}

// ---------------------------------------------------------------------------
// Helper: build SubdomainEntry from FE types
// ---------------------------------------------------------------------------

/// Build a single `SubdomainEntry<FeLocalSolver>` from a pre-built CrossTab.
///
/// Dispatches to either full-SDDM or Schur complement path based on config.
pub(crate) fn build_entry(
    domain: Subdomain,
    cross_tab: CrossTab,
    config: &SchwarzConfig,
) -> WithinResult<SubdomainEntry<FeLocalSolver>> {
    let solver = match &config.local_solver {
        LocalSolverConfig::FullSddm => {
            let first_block_size = cross_tab.first_block_size();
            let matrix = cross_tab.to_sddm();
            let n_local = matrix.n();
            let csr = CsrRef::new(
                matrix.indptr(),
                matrix.indices(),
                matrix.data(),
                n_local as u32,
            )
            .map_err(|e| {
                WithinError::LocalSolverBuild(format!("invalid local SDDM CSR structure: {e}"))
            })?;
            let builder = Builder::new(config.smoother);
            let factor = builder.build(csr).map_err(|e| {
                WithinError::LocalSolverBuild(format!("failed local SDDM factorization: {e}"))
            })?;
            let was_augmented = factor.n() > n_local;
            let strategy = LocalSolveStrategy::from_flags(Some(first_block_size), was_augmented);
            FeLocalSolver::FullSddm {
                solver: ApproxCholSolver::new(factor, strategy, n_local),
            }
        }
        LocalSolverConfig::SchurComplement {
            approx_chol,
            approx_schur,
            dense_threshold,
        } => {
            let reduced = build_reduced_schur_factor(
                &cross_tab,
                *approx_chol,
                *approx_schur,
                *dense_threshold,
            )?;
            FeLocalSolver::SchurComplement(Box::new(BlockElimSolver::new(
                cross_tab,
                reduced.elimination.inv_diag_elim,
                reduced.factor,
                reduced.elimination.eliminate_q,
            )))
        }
    };
    Ok(SubdomainEntry::new(domain.core, solver))
}

struct ReducedSchurBuild {
    factor: ReducedFactor,
    elimination: EliminationInfo,
}

fn dense_fast_path_enabled(n_keep: usize, threshold: usize) -> bool {
    threshold > 0 && n_keep <= threshold
}

fn compute_schur(cross_tab: &CrossTab, approx_schur: Option<ApproxSchurConfig>) -> SchurResult {
    match approx_schur {
        None => ExactSchurComplement.compute(cross_tab),
        Some(cfg) => ApproxSchurComplement::new(cfg).compute(cross_tab),
    }
}

fn build_sparse_reduced_factor(
    matrix: &schwarz_precond::SparseMatrix,
    approx_chol: Config,
) -> WithinResult<ReducedFactor> {
    let schur_builder = Builder::new(approx_chol);
    let csr = CsrRef::new(
        matrix.indptr(),
        matrix.indices(),
        matrix.data(),
        matrix.n() as u32,
    )
    .map_err(|e| WithinError::LocalSolverBuild(format!("invalid Schur complement CSR: {e}")))?;
    schur_builder
        .build(csr)
        .map(ReducedFactor::approx)
        .map_err(|e| {
            WithinError::LocalSolverBuild(format!("failed Schur complement factorization: {e}"))
        })
}

fn build_reduced_schur_factor(
    cross_tab: &CrossTab,
    approx_chol: Config,
    approx_schur: Option<ApproxSchurConfig>,
    dense_threshold: usize,
) -> WithinResult<ReducedSchurBuild> {
    let n_keep = cross_tab.n_q().min(cross_tab.n_r());
    let prefer_dense = dense_fast_path_enabled(n_keep, dense_threshold);

    // Fastest path for tiny exact Schur: build dense directly and factor dense.
    if prefer_dense && approx_schur.is_none() {
        let dense = ExactSchurComplement.compute_dense_anchored(cross_tab);
        if let Some(factor) =
            ReducedFactor::try_dense_laplacian_minor(dense.anchored_minor, dense.n)
        {
            return Ok(ReducedSchurBuild {
                factor,
                elimination: dense.elimination,
            });
        }
    }

    // General path (exact or approximate): sparse Schur assembly once.
    let schur = compute_schur(cross_tab, approx_schur);
    if prefer_dense {
        if let Some(factor) = ReducedFactor::try_dense_laplacian(&schur.matrix) {
            return Ok(ReducedSchurBuild {
                factor,
                elimination: schur.elimination,
            });
        }
    }

    let factor = build_sparse_reduced_factor(&schur.matrix, approx_chol)?;
    Ok(ReducedSchurBuild {
        factor,
        elimination: schur.elimination,
    })
}

/// Compute how many DOFs in a domain belong to the first factor of its factor pair.
pub fn compute_first_block_size<S: ObservationStore>(
    design: &WeightedDesign<S>,
    domain: &Subdomain,
) -> usize {
    let (q, _) = domain.factor_pair;
    let fq = &design.factors[q];
    let lo = fq.offset;
    let hi = fq.offset + fq.n_levels;
    domain
        .core
        .global_indices
        .iter()
        .filter(|&&idx| {
            let idx = idx as usize;
            idx >= lo && idx < hi
        })
        .count()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::cmp::Ordering;
    use std::hint::black_box;
    use std::time::Instant;

    use super::*;
    use crate::config::{ApproxSchurConfig, DEFAULT_DENSE_SCHUR_THRESHOLD};
    use crate::domain::{build_local_domains, FixedEffectsDesign};
    use crate::observation::FactorMajorStore;
    use crate::operator::csr_block::CsrBlock;
    use schwarz_precond::{LocalSolver, Operator};

    fn make_test_data() -> (FixedEffectsDesign, Vec<(Subdomain, CrossTab)>) {
        let design = FixedEffectsDesign::new(
            vec![vec![0, 1, 0, 1, 2], vec![0, 0, 1, 1, 0]],
            vec![3, 2],
            5,
        )
        .expect("valid fixed-effects design");
        let domain_pairs = build_local_domains(&design, None);
        (design, domain_pairs)
    }

    fn synthetic_cross_tab(n_keep: usize, elim_ratio: usize) -> CrossTab {
        let n_q = n_keep * elim_ratio;
        let n_r = n_keep;
        let mut table = vec![0.0; n_q * n_r];

        // Deterministic 3-edge pattern per eliminated vertex:
        // ring edges + hashed jump to keep graph connected and nontrivial.
        for i in 0..n_q {
            let j0 = i % n_r;
            let j1 = (i + 1) % n_r;
            let j2 = (i.wrapping_mul(7).wrapping_add(3)) % n_r;
            table[i * n_r + j0] += 1.0;
            table[i * n_r + j1] += 0.8;
            table[i * n_r + j2] += 0.6;
        }

        let mut diag_q = vec![0.0; n_q];
        let mut diag_r = vec![0.0; n_r];
        for i in 0..n_q {
            let row = &table[i * n_r..(i + 1) * n_r];
            let mut s = 0.0;
            for (j, &w) in row.iter().enumerate() {
                s += w;
                diag_r[j] += w;
            }
            diag_q[i] = s;
        }

        let c = CsrBlock::from_dense_table(&table, n_q, n_r);
        let ct = c.transpose();
        CrossTab {
            c,
            ct,
            diag_q,
            diag_r,
        }
    }

    fn benchmark_build_path(
        cross_tab: &CrossTab,
        approx_schur: Option<ApproxSchurConfig>,
        dense_threshold: usize,
        iters: usize,
    ) -> f64 {
        let approx_chol = Config {
            split_merge: Some(8),
            seed: 42,
            ..Default::default()
        };
        let mut samples = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            let reduced =
                build_reduced_schur_factor(cross_tab, approx_chol, approx_schur, dense_threshold)
                    .expect("reduced Schur build failed");
            black_box(reduced.factor);
            samples.push(t0.elapsed().as_secs_f64() * 1e6);
        }
        samples.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        samples[samples.len() / 2]
    }

    fn build_local_solver_for_bench(
        n_keep: usize,
        approx_schur: Option<ApproxSchurConfig>,
        dense_threshold: usize,
    ) -> BlockElimSolver {
        let cross_tab = synthetic_cross_tab(n_keep, 8);
        let approx_chol = Config {
            split_merge: Some(8),
            seed: 42,
            ..Default::default()
        };
        let reduced =
            build_reduced_schur_factor(&cross_tab, approx_chol, approx_schur, dense_threshold)
                .expect("reduced Schur build failed");
        BlockElimSolver::new(
            cross_tab,
            reduced.elimination.inv_diag_elim,
            reduced.factor,
            reduced.elimination.eliminate_q,
        )
    }

    fn benchmark_local_solve_path(solver: &BlockElimSolver, iters: usize) -> f64 {
        let n_local = solver.n_local();
        let scratch = solver.scratch_size();
        let mut rhs_template = vec![0.0; n_local];
        for (i, v) in rhs_template.iter_mut().enumerate() {
            *v = ((i.wrapping_mul(13) % 31) as f64 - 15.0) * 0.1;
        }

        let mut rhs = vec![0.0; scratch];
        let mut sol = vec![0.0; scratch];
        let t0 = Instant::now();
        let mut checksum = 0.0;
        for _ in 0..iters {
            rhs[..n_local].copy_from_slice(&rhs_template);
            solver
                .solve_local(&mut rhs, &mut sol)
                .expect("benchmark local solve");
            checksum += sol[0];
        }
        black_box(checksum);
        (t0.elapsed().as_secs_f64() * 1e6) / iters as f64
    }

    #[test]
    fn test_build_schwarz() {
        let (design, domain_pairs) = make_test_data();
        let config = SchwarzConfig::default();
        let schwarz = build_additive::<FactorMajorStore>(
            DomainSource::FromParts(domain_pairs),
            design.n_dofs,
            &config,
        )
        .expect("build schwarz with explicit domains");
        assert!(!schwarz.subdomains().is_empty());

        let r = vec![1.0; design.n_dofs];
        let mut z = vec![0.0; design.n_dofs];
        schwarz.apply(&r, &mut z);
        assert!(z.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_build_default() {
        let (design, _) = make_test_data();
        let config = SchwarzConfig::default();
        let schwarz = build_schwarz(&design, &config).expect("build default schwarz");
        assert!(!schwarz.subdomains().is_empty());
    }

    #[test]
    fn test_first_block_size_computation() {
        let design =
            FixedEffectsDesign::new(vec![vec![0, 1, 0, 1], vec![0, 0, 1, 1]], vec![2, 2], 4)
                .expect("valid design");
        let domain_pairs = build_local_domains(&design, None);

        assert!(!domain_pairs.is_empty());
        let fbs = compute_first_block_size(&design, &domain_pairs[0].0);
        assert_eq!(fbs, 2);
    }

    #[test]
    fn test_exact_schur_uses_dense_fast_path_for_tiny_reduced_system() {
        let (_, mut domain_pairs) = make_test_data();
        let (domain, cross_tab) = domain_pairs.swap_remove(0);

        let config = SchwarzConfig {
            smoother: Config::default(),
            local_solver: LocalSolverConfig::SchurComplement {
                approx_chol: Config::default(),
                approx_schur: None,
                dense_threshold: crate::config::DEFAULT_DENSE_SCHUR_THRESHOLD,
            },
        };
        let entry =
            build_entry(domain, cross_tab, &config).expect("exact Schur entry build failed");

        match entry.solver {
            FeLocalSolver::SchurComplement(solver) => {
                assert!(solver.uses_dense_reduced_factor());
            }
            FeLocalSolver::FullSddm { .. } => panic!("expected SchurComplement solver"),
        }
    }

    #[test]
    fn test_approximate_schur_uses_dense_fast_path_for_tiny_reduced_system() {
        let (_, mut domain_pairs) = make_test_data();
        let (domain, cross_tab) = domain_pairs.swap_remove(0);

        let config = SchwarzConfig {
            smoother: Config::default(),
            local_solver: LocalSolverConfig::SchurComplement {
                approx_chol: Config::default(),
                approx_schur: Some(crate::config::ApproxSchurConfig { seed: 7 }),
                dense_threshold: crate::config::DEFAULT_DENSE_SCHUR_THRESHOLD,
            },
        };
        let entry =
            build_entry(domain, cross_tab, &config).expect("approximate Schur entry build failed");

        match entry.solver {
            FeLocalSolver::SchurComplement(solver) => {
                assert!(solver.uses_dense_reduced_factor());
            }
            FeLocalSolver::FullSddm { .. } => panic!("expected SchurComplement solver"),
        }
    }

    #[test]
    fn test_dense_threshold_zero_disables_dense_fast_path() {
        let (_, mut domain_pairs) = make_test_data();
        let (domain, cross_tab) = domain_pairs.swap_remove(0);

        let config = SchwarzConfig {
            smoother: Config::default(),
            local_solver: LocalSolverConfig::SchurComplement {
                approx_chol: Config::default(),
                approx_schur: None,
                dense_threshold: 0,
            },
        };
        let entry =
            build_entry(domain, cross_tab, &config).expect("exact Schur entry build failed");

        match entry.solver {
            FeLocalSolver::SchurComplement(solver) => {
                assert!(!solver.uses_dense_reduced_factor());
            }
            FeLocalSolver::FullSddm { .. } => panic!("expected SchurComplement solver"),
        }
    }

    #[test]
    #[ignore] // run with: cargo test -p within bench_isolated_schur_dense_vs_sparse_paths --lib -- --ignored --nocapture
    fn bench_isolated_schur_dense_vs_sparse_paths() {
        let sizes = [4usize, 8, 12, 16, 20, 24, 28, 32, 40, 48, 64];
        println!(
            "{:>5} | {:>11} {:>11} {:>7} | {:>11} {:>11} {:>7} | {:>10} {:>10} {:>7}",
            "n_keep",
            "exact_dense",
            "exact_sparse",
            "ratio",
            "approx_dense",
            "approx_sparse",
            "ratio",
            "solve_dense",
            "solve_sparse",
            "ratio"
        );
        println!("{}", "-".repeat(118));

        for &n_keep in &sizes {
            let cross_tab = synthetic_cross_tab(n_keep, 8);
            let build_iters = if n_keep <= 32 { 100 } else { 40 };

            let exact_dense = benchmark_build_path(&cross_tab, None, usize::MAX, build_iters);
            let exact_sparse = benchmark_build_path(&cross_tab, None, 0, build_iters);
            let approx_dense = benchmark_build_path(
                &cross_tab,
                Some(ApproxSchurConfig { seed: 42 }),
                usize::MAX,
                build_iters,
            );
            let approx_sparse = benchmark_build_path(
                &cross_tab,
                Some(ApproxSchurConfig { seed: 42 }),
                0,
                build_iters,
            );

            let solve_iters = if n_keep <= 32 { 8_000 } else { 3_000 };
            let solver_dense = build_local_solver_for_bench(n_keep, None, usize::MAX);
            let solver_sparse = build_local_solver_for_bench(n_keep, None, 0);
            let solve_dense = benchmark_local_solve_path(&solver_dense, solve_iters);
            let solve_sparse = benchmark_local_solve_path(&solver_sparse, solve_iters);

            println!(
                "{:>5} | {:>11.2} {:>11.2} {:>7.2} | {:>11.2} {:>11.2} {:>7.2} | {:>10.3} {:>10.3} {:>7.2}",
                n_keep,
                exact_dense,
                exact_sparse,
                exact_dense / exact_sparse,
                approx_dense,
                approx_sparse,
                approx_dense / approx_sparse,
                solve_dense,
                solve_sparse,
                solve_dense / solve_sparse
            );
        }

        println!(
            "\nDefault dense threshold currently: {}",
            DEFAULT_DENSE_SCHUR_THRESHOLD
        );
    }

    #[test]
    fn test_gramian_multiplicative_schwarz_matches_obs_schwarz() {
        let design = FixedEffectsDesign::new(
            vec![vec![0, 1, 0, 1, 2], vec![0, 0, 1, 1, 0]],
            vec![3, 2],
            5,
        )
        .expect("valid design");

        let config = SchwarzConfig::default();
        let gramian = crate::operator::gramian::Gramian::build(&design);

        let domain_pairs = build_local_domains(&design, None);

        let obs_schwarz =
            build_multiplicative_obs(DomainSource::FromDesign(&design), &design, &config, false)
                .unwrap();
        let gram_schwarz = build_multiplicative_sparse(
            DomainSource::<FactorMajorStore>::FromParts(domain_pairs),
            &gramian,
            design.n_dofs,
            &config,
            false,
        )
        .unwrap();

        let r = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut z_obs = vec![0.0; 5];
        let mut z_gram = vec![0.0; 5];

        obs_schwarz.apply(&r, &mut z_obs);
        gram_schwarz.apply(&r, &mut z_gram);

        for i in 0..5 {
            assert!(
                (z_obs[i] - z_gram[i]).abs() < 1e-12,
                "mismatch at DOF {i}: obs={}, gram={}",
                z_obs[i],
                z_gram[i],
            );
        }
    }
}
