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
use crate::config::{ApproxSchurConfig, LocalSolverConfig};
use crate::domain::{build_local_domains, Subdomain, WeightedDesign};
use crate::observation::ObservationStore;
use crate::{WithinError, WithinResult};

/// Concrete Schwarz type used in the parent crate.
pub type FeSchwarz = SchwarzPreconditioner<FeLocalSolver>;

/// Concrete multiplicative Schwarz type: one-level with observation-space residual updates.
pub type FeMultSchwarz<'a, S> =
    MultiplicativeSchwarzPreconditioner<FeLocalSolver, ObservationSpaceUpdater<'a, S>>;

/// Concrete multiplicative Schwarz type: one-level with explicit Gramian CSR residual updates.
pub type FeMultSchwarzSparse =
    MultiplicativeSchwarzPreconditioner<FeLocalSolver, SparseGramianUpdater>;

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
    config: &LocalSolverConfig,
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
    config: &LocalSolverConfig,
) -> WithinResult<FeSchwarz> {
    let entries = build_entries_from_source(source, config)?;
    Ok(SchwarzPreconditioner::new(entries, n_dofs)?)
}

/// Build multiplicative Schwarz with observation-space updater.
///
/// Always non-symmetric (GMRES-only).
#[cfg(test)]
#[allow(dead_code)]
pub(crate) fn build_multiplicative_obs<'a, S: ObservationStore>(
    source: DomainSource<'_, S>,
    design: &'a WeightedDesign<S>,
    config: &LocalSolverConfig,
) -> WithinResult<FeMultSchwarz<'a, S>> {
    let entries = build_entries_from_source(source, config)?;
    let updater = ObservationSpaceUpdater::new(design);
    Ok(MultiplicativeSchwarzPreconditioner::new(
        entries,
        updater,
        design.n_dofs,
        false,
    )?)
}

/// Build multiplicative Schwarz with sparse Gramian updater.
///
/// Always non-symmetric (GMRES-only).
pub(crate) fn build_multiplicative_sparse<S: ObservationStore>(
    source: DomainSource<'_, S>,
    gramian: &super::gramian::Gramian,
    n_dofs: usize,
    config: &LocalSolverConfig,
) -> WithinResult<FeMultSchwarzSparse> {
    let entries = build_entries_from_source(source, config)?;
    let updater = SparseGramianUpdater::new(gramian.matrix.clone());
    Ok(MultiplicativeSchwarzPreconditioner::new(
        entries, updater, n_dofs, false,
    )?)
}

// ---------------------------------------------------------------------------
// Internal: build entries from source
// ---------------------------------------------------------------------------

fn build_entries_from_source<S: ObservationStore>(
    source: DomainSource<'_, S>,
    config: &LocalSolverConfig,
) -> WithinResult<Vec<SubdomainEntry<FeLocalSolver>>> {
    match source {
        DomainSource::FromDesign(design) => {
            let domain_pairs = build_local_domains(design);
            build_entries_from_pairs(domain_pairs, config)
        }
        DomainSource::FromParts(domain_pairs) => build_entries_from_pairs(domain_pairs, config),
    }
}

fn build_entries_from_pairs(
    domain_pairs: Vec<(Subdomain, CrossTab)>,
    config: &LocalSolverConfig,
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
    config: &LocalSolverConfig,
) -> WithinResult<SubdomainEntry<FeLocalSolver>> {
    let solver = match config {
        LocalSolverConfig::FullSddm { approx_chol } => {
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
            let builder = Builder::new(*approx_chol);
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
            let schur_config = ReducedSchurConfig {
                approx_chol: *approx_chol,
                approx_schur: *approx_schur,
                dense_threshold: *dense_threshold,
            };
            let reduced = build_reduced_schur_factor(&cross_tab, &schur_config)?;
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

pub(crate) struct ReducedSchurBuild {
    pub(crate) factor: ReducedFactor,
    pub(crate) elimination: EliminationInfo,
}

fn dense_fast_path_enabled(n_keep: usize, threshold: usize) -> bool {
    threshold > 0 && n_keep <= threshold
}

fn compute_schur(
    cross_tab: &CrossTab,
    approx_schur: Option<ApproxSchurConfig>,
) -> WithinResult<SchurResult> {
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

/// Configuration for building a reduced Schur factor.
pub(crate) struct ReducedSchurConfig {
    pub approx_chol: Config,
    pub approx_schur: Option<ApproxSchurConfig>,
    pub dense_threshold: usize,
}

pub(crate) fn build_reduced_schur_factor(
    cross_tab: &CrossTab,
    config: &ReducedSchurConfig,
) -> WithinResult<ReducedSchurBuild> {
    let n_keep = cross_tab.n_q().min(cross_tab.n_r());
    let prefer_dense = dense_fast_path_enabled(n_keep, config.dense_threshold);

    // Fastest path for tiny exact Schur: build dense directly and factor dense.
    if prefer_dense && config.approx_schur.is_none() {
        let dense = ExactSchurComplement.compute_dense_anchored(cross_tab)?;
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
    let schur = compute_schur(cross_tab, config.approx_schur)?;
    if prefer_dense {
        if let Some(factor) = ReducedFactor::try_dense_laplacian(&schur.matrix) {
            return Ok(ReducedSchurBuild {
                factor,
                elimination: schur.elimination,
            });
        }
    }

    let factor = build_sparse_reduced_factor(&schur.matrix, config.approx_chol)?;
    Ok(ReducedSchurBuild {
        factor,
        elimination: schur.elimination,
    })
}

/// Compute how many DOFs in a domain belong to the first factor of its factor pair.
#[cfg(test)]
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
