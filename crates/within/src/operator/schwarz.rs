//! Schwarz preconditioner: FE-specific construction helpers.
//!
//! This module bridges the fixed-effects domain types ([`Design`],
//! [`Subdomain`], `CrossTab`) to the generic `schwarz-precond` crate API.
//! The generic crate knows nothing about panel data — it operates on abstract
//! [`SubdomainEntry`] values containing a local solver and a set of global DOF
//! indices. This module handles the translation.
//!
//! # Local solver
//!
//! Each subdomain needs a local solver that can approximately invert the
//! restricted Gramian on that subdomain. The solver eliminates one factor
//! block via exact diagonal inversion, then factors the reduced Schur
//! complement (see `schur_complement`).
//!
//! # Builder pattern
//!
//! Construction flows through a layered builder:
//!
//! 1. **Domain acquisition** — accept pre-built `(Subdomain, CrossTab)`
//!    pairs (enabling fused build paths that scan observations only once)
//! 2. **Entry construction** — each `(Subdomain, CrossTab)` pair is
//!    converted into a `SubdomainEntry<BlockElimSolver>` in parallel via
//!    `build_entry`, which dispatches on the config
//! 3. **Schwarz assembly** — entries are passed to the generic
//!    `SchwarzPreconditioner` (additive) or
//!    `MultiplicativeSchwarzPreconditioner` constructor from `schwarz-precond`

pub use schwarz_precond::MultiplicativeSchwarzPreconditioner;
pub use schwarz_precond::SchwarzPreconditioner;

use std::sync::Arc;

use approx_chol::low_level::Builder;
use approx_chol::CsrRef;
use rayon::prelude::*;
use schwarz_precond::SubdomainEntry;

use super::gramian::CrossTab;
use super::local_solver::{BlockElimSolver, ReducedFactor};
use super::residual_update::SparseGramianUpdater;
use super::schur_complement::{
    ApproxSchurComplement, EliminationInfo, ExactSchurComplement, SchurComplement, SchurResult,
};
use crate::config::{ApproxCholConfig, ApproxSchurConfig, LocalSolverConfig};
use crate::domain::Subdomain;
use crate::{WithinError, WithinResult};

/// Concrete additive Schwarz type used in the parent crate.
pub type FeSchwarz = SchwarzPreconditioner<BlockElimSolver>;

/// Concrete multiplicative Schwarz type: one-level with explicit Gramian CSR residual updates.
pub type FeMultSchwarzSparse =
    MultiplicativeSchwarzPreconditioner<BlockElimSolver, SparseGramianUpdater>;

// ---------------------------------------------------------------------------
// Crate-internal consolidated builders
// ---------------------------------------------------------------------------

/// Build additive Schwarz with an explicit reduction strategy.
pub(crate) fn build_additive_with_strategy(
    domains: Vec<(Subdomain, Arc<CrossTab>)>,
    n_dofs: usize,
    config: &LocalSolverConfig,
    strategy: schwarz_precond::ReductionStrategy,
) -> WithinResult<FeSchwarz> {
    let entries = build_entries_from_pairs(domains, config)?;
    Ok(SchwarzPreconditioner::with_strategy(
        entries, n_dofs, strategy,
    )?)
}

/// Build multiplicative Schwarz with sparse Gramian updater.
///
/// Always non-symmetric (GMRES-only).
pub(crate) fn build_multiplicative_sparse(
    domains: Vec<(Subdomain, Arc<CrossTab>)>,
    gramian: &super::gramian::Gramian,
    n_dofs: usize,
    config: &LocalSolverConfig,
) -> WithinResult<FeMultSchwarzSparse> {
    let entries = build_entries_from_pairs(domains, config)?;
    let updater = SparseGramianUpdater::new(gramian.matrix.clone());
    Ok(MultiplicativeSchwarzPreconditioner::new(
        entries, updater, n_dofs,
    )?)
}

fn build_entries_from_pairs(
    domain_pairs: Vec<(Subdomain, Arc<CrossTab>)>,
    config: &LocalSolverConfig,
) -> WithinResult<Vec<SubdomainEntry<BlockElimSolver>>> {
    domain_pairs
        .into_par_iter()
        .map(|(domain, cross_tab)| build_entry(domain, cross_tab, config))
        .collect()
}

// ---------------------------------------------------------------------------
// Helper: build SubdomainEntry from FE types
// ---------------------------------------------------------------------------

/// Build a single `SubdomainEntry<BlockElimSolver>` from a pre-built CrossTab.
pub(crate) fn build_entry(
    domain: Subdomain,
    cross_tab: Arc<CrossTab>,
    config: &LocalSolverConfig,
) -> WithinResult<SubdomainEntry<BlockElimSolver>> {
    let schur_config = ReducedSchurConfig {
        approx_chol: config.approx_chol,
        approx_schur: config.approx_schur,
        dense_threshold: config.dense_threshold,
    };
    let reduced = build_reduced_schur_factor(&cross_tab, &schur_config)?;
    let solver = BlockElimSolver::new(
        cross_tab,
        reduced.elimination.inv_diag_elim,
        reduced.factor,
        reduced.elimination.eliminate_q,
    );
    SubdomainEntry::try_new(domain.core, solver)
        .map_err(|e| WithinError::LocalSolverBuild(format!("invalid subdomain entry: {e}")))
}

pub(crate) struct ReducedSchurBuild {
    pub(crate) factor: ReducedFactor,
    pub(crate) elimination: EliminationInfo,
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
    approx_chol: ApproxCholConfig,
) -> WithinResult<ReducedFactor> {
    let schur_builder = Builder::new(approx_chol.to_approx_chol());
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
    pub approx_chol: ApproxCholConfig,
    pub approx_schur: Option<ApproxSchurConfig>,
    pub dense_threshold: usize,
}

pub(crate) fn build_reduced_schur_factor(
    cross_tab: &CrossTab,
    config: &ReducedSchurConfig,
) -> WithinResult<ReducedSchurBuild> {
    let n_keep = cross_tab.n_q().min(cross_tab.n_r());
    let prefer_dense = config.dense_threshold > 0 && n_keep <= config.dense_threshold;

    // Below the dense threshold the reduced system is tiny — always use exact
    // Schur complement (cheap at this size) and dense Cholesky factorization.
    if prefer_dense {
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

    // General path (exact or approximate): sparse Schur assembly.
    let schur = compute_schur(cross_tab, config.approx_schur)?;

    let factor = build_sparse_reduced_factor(&schur.matrix, config.approx_chol)?;
    Ok(ReducedSchurBuild {
        factor,
        elimination: schur.elimination,
    })
}
