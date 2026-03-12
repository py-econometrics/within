//! Pre-built preconditioner for reuse across multiple solves.

use schwarz_precond::{AdditiveSchwarzDiagnostics, LocalSolver, Operator, ReductionStrategy};
use serde::{Deserialize, Serialize};

use crate::config::Preconditioner;
use crate::domain::{Subdomain, WeightedDesign};
use crate::observation::ObservationStore;
use crate::operator::gramian::{CrossTab, Gramian};
use crate::operator::schwarz::{
    build_additive_with_strategy, build_multiplicative_sparse, DomainSource, FeMultSchwarzSparse,
    FeSchwarz,
};
use crate::{WithinError, WithinResult};

/// A pre-built preconditioner ready for use in Krylov solves.
///
/// Implements [`Operator`] via enum dispatch to the inner variant.
#[derive(Clone, Serialize, Deserialize)]
pub enum FePreconditioner {
    /// Additive Schwarz (symmetric -- valid for CG and GMRES).
    Additive(FeSchwarz),
    /// Multiplicative Schwarz with sparse Gramian updater (GMRES only).
    Multiplicative(FeMultSchwarzSparse),
}

impl FePreconditioner {
    /// Number of Schwarz subdomains in the built preconditioner.
    pub fn n_subdomains(&self) -> usize {
        match self {
            Self::Additive(p) => p.subdomains().len(),
            Self::Multiplicative(p) => p.subdomains().len(),
        }
    }

    /// Estimated nested-parallel work per subdomain.
    pub fn subdomain_inner_parallel_work(&self) -> Vec<usize> {
        match self {
            Self::Additive(p) => p
                .subdomains()
                .iter()
                .map(|entry| entry.solver().inner_parallelism_work_estimate())
                .collect(),
            Self::Multiplicative(p) => p
                .subdomains()
                .iter()
                .map(|entry| entry.solver().inner_parallelism_work_estimate())
                .collect(),
        }
    }

    /// Configured additive reduction strategy, if this is an additive preconditioner.
    pub fn reduction_strategy(&self) -> Option<ReductionStrategy> {
        match self {
            Self::Additive(p) => Some(p.reduction_strategy()),
            Self::Multiplicative(_) => None,
        }
    }

    /// Concrete additive backend selected for the current Rayon thread-pool width.
    pub fn resolved_reduction_strategy(&self) -> Option<ReductionStrategy> {
        match self {
            Self::Additive(p) => Some(p.resolved_reduction_strategy()),
            Self::Multiplicative(_) => None,
        }
    }

    /// Build-time additive Schwarz scheduling diagnostics.
    pub fn additive_schwarz_diagnostics(&self) -> Option<AdditiveSchwarzDiagnostics> {
        match self {
            Self::Additive(p) => Some(p.diagnostics()),
            Self::Multiplicative(_) => None,
        }
    }
}

impl Operator for FePreconditioner {
    fn nrows(&self) -> usize {
        match self {
            Self::Additive(p) => p.nrows(),
            Self::Multiplicative(p) => p.nrows(),
        }
    }

    fn ncols(&self) -> usize {
        match self {
            Self::Additive(p) => p.ncols(),
            Self::Multiplicative(p) => p.ncols(),
        }
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) {
        match self {
            Self::Additive(p) => p.apply(x, y),
            Self::Multiplicative(p) => p.apply(x, y),
        }
    }

    fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) {
        match self {
            Self::Additive(p) => p.apply_adjoint(x, y),
            Self::Multiplicative(p) => p.apply_adjoint(x, y),
        }
    }

    fn try_apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), schwarz_precond::ApplyError> {
        match self {
            Self::Additive(p) => p.try_apply(x, y),
            Self::Multiplicative(p) => p.try_apply(x, y),
        }
    }

    fn try_apply_adjoint(
        &self,
        x: &[f64],
        y: &mut [f64],
    ) -> Result<(), schwarz_precond::ApplyError> {
        match self {
            Self::Additive(p) => p.try_apply_adjoint(x, y),
            Self::Multiplicative(p) => p.try_apply_adjoint(x, y),
        }
    }
}

/// Build a [`FePreconditioner`] from pre-built domains and configuration.
///
/// Shared dispatch logic used by both `build_preconditioner` and
/// `build_preconditioner_fused`.
fn build_from_domains<S: ObservationStore>(
    domains: Vec<(Subdomain, CrossTab)>,
    n_dofs: usize,
    gramian: Option<&Gramian>,
    config: &Preconditioner,
) -> WithinResult<FePreconditioner> {
    match config {
        Preconditioner::Additive(solver_config, strategy) => {
            let p = build_additive_with_strategy(
                DomainSource::<S>::FromParts(domains),
                n_dofs,
                solver_config,
                *strategy,
            )?;
            Ok(FePreconditioner::Additive(p))
        }
        Preconditioner::Multiplicative(solver_config) => {
            let gramian = gramian.ok_or_else(|| {
                WithinError::LocalSolverBuild(
                    "multiplicative preconditioner requires an explicit Gramian".to_string(),
                )
            })?;
            let p = build_multiplicative_sparse(
                DomainSource::<S>::FromParts(domains),
                gramian,
                n_dofs,
                solver_config,
            )?;
            Ok(FePreconditioner::Multiplicative(p))
        }
    }
}

/// Build a [`FePreconditioner`] from a design and configuration.
///
/// When `Multiplicative` is requested, a Gramian is required for the sparse
/// residual updater.
pub fn build_preconditioner<S: ObservationStore>(
    design: &WeightedDesign<S>,
    gramian: Option<&Gramian>,
    preconditioner_config: &Preconditioner,
) -> WithinResult<FePreconditioner> {
    use crate::domain::build_local_domains;

    let domains = build_local_domains(design);
    build_from_domains::<S>(domains, design.n_dofs, gramian, preconditioner_config)
}

/// Fused build: single observation scan -> domains + Gramian.
///
/// Uses `build_domains_and_gramian_blocks` to scan observations once, producing
/// both the domain decomposition (for the preconditioner) and per-pair block
/// data (for composing the explicit Gramian CSR). Avoids the double scan that
/// would occur if `Gramian::build` and `build_local_domains` were called
/// separately.
pub(crate) fn build_preconditioner_fused<S: ObservationStore>(
    design: &WeightedDesign<S>,
    preconditioner_config: &Preconditioner,
) -> WithinResult<(Gramian, FePreconditioner)> {
    use crate::domain::build_domains_and_gramian_blocks;

    let (domains, blocks) = build_domains_and_gramian_blocks(design);
    let gramian = Gramian::from_pair_blocks(&blocks, &design.factors, design.n_dofs)?;
    let precond = build_from_domains::<S>(
        domains,
        design.n_dofs,
        Some(&gramian),
        preconditioner_config,
    )?;

    Ok((gramian, precond))
}
