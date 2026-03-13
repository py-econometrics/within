//! Schwarz preconditioner: FE-specific construction helpers.
//!
//! This module bridges the fixed-effects domain types ([`WeightedDesign`],
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
//! 1. **Domain acquisition** — either scan observations from a
//!    [`WeightedDesign`] or accept pre-built `(Subdomain, CrossTab)` pairs
//!    via the `DomainSource` enum (the latter enables fused build paths
//!    that scan observations only once)
//! 2. **Entry construction** — each `(Subdomain, CrossTab)` pair is
//!    converted into a `SubdomainEntry<BlockElimSolver>` in parallel via
//!    `build_entry`, which dispatches on the config
//! 3. **Schwarz assembly** — entries are passed to the generic
//!    `SchwarzPreconditioner` (additive) or
//!    `MultiplicativeSchwarzPreconditioner` constructor from `schwarz-precond`
//!
//! The public entry point is [`build_schwarz`] for the additive variant.

pub use schwarz_precond::MultiplicativeSchwarzPreconditioner;
pub use schwarz_precond::SchwarzPreconditioner;

use approx_chol::low_level::Builder;
use approx_chol::CsrRef;
use rayon::prelude::*;
use schwarz_precond::SubdomainEntry;
use serde::{Deserialize, Serialize};

use super::gramian::CrossTab;
use super::local_solver::{BlockElimSolver, FeLocalSolveInvoker, ReducedFactor};
use super::residual_update::SparseGramianUpdater;
use super::schur_complement::{
    ApproxSchurComplement, EliminationInfo, ExactSchurComplement, SchurComplement, SchurResult,
};
use crate::config::{ApproxCholConfig, ApproxSchurConfig, LocalSolverConfig};
use crate::domain::{build_local_domains, Subdomain, WeightedDesign};
use crate::observation::ObservationStore;
use crate::{WithinError, WithinResult};

/// Concrete additive Schwarz type used in the parent crate.
#[derive(Clone, Serialize, Deserialize)]
pub struct FeSchwarz(SchwarzPreconditioner<BlockElimSolver, FeLocalSolveInvoker>);

impl FeSchwarz {
    pub(crate) fn new(inner: SchwarzPreconditioner<BlockElimSolver, FeLocalSolveInvoker>) -> Self {
        Self(inner)
    }

    /// Subdomain entries with their local solvers.
    pub fn subdomains(&self) -> &[SubdomainEntry<BlockElimSolver>] {
        self.0.subdomains()
    }

    /// Current reduction strategy (may be `Auto`).
    pub fn reduction_strategy(&self) -> schwarz_precond::ReductionStrategy {
        self.0.reduction_strategy()
    }

    /// Resolved reduction strategy (`Auto` replaced by the detected choice).
    pub fn resolved_reduction_strategy(&self) -> schwarz_precond::ReductionStrategy {
        self.0.resolved_reduction_strategy()
    }

    /// Subdomain diagnostics (sizes, overlap counts).
    pub fn diagnostics(&self) -> schwarz_precond::AdditiveSchwarzDiagnostics {
        self.0.diagnostics()
    }

    /// Clone with a different reduction strategy.
    pub fn with_reduction_strategy(&self, strategy: schwarz_precond::ReductionStrategy) -> Self {
        Self(self.0.with_reduction_strategy(strategy))
    }

    /// Apply the preconditioner, returning an error on local-solver failure.
    pub fn try_apply(&self, r: &[f64], z: &mut [f64]) -> Result<(), schwarz_precond::ApplyError> {
        self.0.try_apply(r, z)
    }
}

impl schwarz_precond::Operator for FeSchwarz {
    fn nrows(&self) -> usize {
        self.0.nrows()
    }

    fn ncols(&self) -> usize {
        self.0.ncols()
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) {
        self.0.apply(x, y);
    }

    fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) {
        self.0.apply_adjoint(x, y);
    }

    fn try_apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), schwarz_precond::ApplyError> {
        self.0.try_apply(x, y)
    }

    fn try_apply_adjoint(
        &self,
        x: &[f64],
        y: &mut [f64],
    ) -> Result<(), schwarz_precond::ApplyError> {
        self.0.try_apply_adjoint(x, y)
    }
}

/// Concrete multiplicative Schwarz type: one-level with explicit Gramian CSR residual updates.
pub type FeMultSchwarzSparse =
    MultiplicativeSchwarzPreconditioner<BlockElimSolver, SparseGramianUpdater>;

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
    build_additive_with_strategy(
        source,
        n_dofs,
        config,
        schwarz_precond::ReductionStrategy::default(),
    )
}

/// Build additive Schwarz from any domain source with an explicit reduction strategy.
pub(crate) fn build_additive_with_strategy<S: ObservationStore>(
    source: DomainSource<'_, S>,
    n_dofs: usize,
    config: &LocalSolverConfig,
    strategy: schwarz_precond::ReductionStrategy,
) -> WithinResult<FeSchwarz> {
    let entries = build_entries_from_source(source, config)?;
    Ok(FeSchwarz::new(
        SchwarzPreconditioner::with_strategy_and_invoker(
            entries,
            n_dofs,
            strategy,
            FeLocalSolveInvoker,
        )?,
    ))
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
) -> WithinResult<Vec<SubdomainEntry<BlockElimSolver>>> {
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
    cross_tab: CrossTab,
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
        .global_indices()
        .iter()
        .filter(|&&idx| {
            let idx = idx as usize;
            idx >= lo && idx < hi
        })
        .count()
}
