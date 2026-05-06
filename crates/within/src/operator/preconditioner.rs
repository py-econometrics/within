//! Preconditioner enum dispatch and fused build paths.
//!
//! [`FePreconditioner`] is the top-level preconditioner type used by the
//! [`orchestrate`](crate::orchestrate) layer. It dispatches via enum to one
//! of two Schwarz variants:
//!
//! - **Additive** ([`FeSchwarz`]) — symmetric, valid for both CG and GMRES.
//!   Subdomains contribute independently and their corrections are summed.
//! - **Multiplicative** ([`FeMultSchwarzSparse`]) — sequential updates with
//!   a sparse Gramian residual updater. Non-symmetric, so requires GMRES.
//!   Generally converges in fewer iterations but cannot be parallelized
//!   across subdomains within a single sweep.
//!
//! # Fused build path
//!
//! Constructing both the explicit Gramian and the preconditioner requires
//! scanning the observation data to build cross-tabulations. Doing this
//! separately for each would double the I/O cost. The
//! `build_preconditioner_fused` function scans observations once, producing
//! both the domain decomposition (for the preconditioner) and the per-pair
//! block data (for assembling the Gramian CSR). This fused path is the
//! default in the orchestrate layer.
//!
//! # Integration with `schwarz-precond`
//!
//! The enum implements the [`Operator`] trait from
//! the `schwarz-precond` crate, so it can be passed directly to CG/GMRES as
//! a preconditioner. Error handling flows through `try_apply` for graceful
//! reporting of local-solver failures.

use std::sync::Arc;

use schwarz_precond::{AdditiveSchwarzDiagnostics, Operator, ReductionStrategy};
use serde::{Deserialize, Serialize};

use crate::config::Preconditioner;
use crate::domain::{Design, Subdomain};
use crate::observation::Store;
use crate::operator::gramian::{CrossTab, Gramian};
use crate::operator::schwarz::{
    build_additive_with_strategy, build_multiplicative_sparse, FeMultSchwarzSparse, FeSchwarz,
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

    /// Configured additive reduction strategy, if this is an additive preconditioner.
    pub fn additive_reduction_strategy(&self) -> Option<ReductionStrategy> {
        match self {
            Self::Additive(p) => Some(p.reduction_strategy()),
            Self::Multiplicative(_) => None,
        }
    }

    /// Concrete additive backend selected for the current Rayon thread-pool width.
    pub fn resolved_additive_reduction_strategy(&self) -> Option<ReductionStrategy> {
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

    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), schwarz_precond::SolveError> {
        match self {
            Self::Additive(p) => p.apply(x, y),
            Self::Multiplicative(p) => p.apply(x, y),
        }
    }

    fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) -> Result<(), schwarz_precond::SolveError> {
        match self {
            Self::Additive(p) => p.apply_adjoint(x, y),
            Self::Multiplicative(p) => p.apply_adjoint(x, y),
        }
    }
}

/// Build a [`FePreconditioner`] from pre-built domains and configuration.
///
/// Shared dispatch logic used by both `build_preconditioner` and
/// `build_preconditioner_fused`.
fn build_from_domains(
    domains: Vec<(Subdomain, Arc<CrossTab>)>,
    n_dofs: usize,
    gramian: Option<&Gramian>,
    config: &Preconditioner,
) -> WithinResult<FePreconditioner> {
    match config {
        Preconditioner::Additive(solver_config, strategy) => {
            let p = build_additive_with_strategy(domains, n_dofs, solver_config, *strategy)?;
            Ok(FePreconditioner::Additive(p))
        }
        Preconditioner::Multiplicative(solver_config) => {
            let gramian = gramian.ok_or_else(|| {
                WithinError::LocalSolverBuild(
                    "multiplicative preconditioner requires an explicit Gramian".to_string(),
                )
            })?;
            let p = build_multiplicative_sparse(domains, gramian, n_dofs, solver_config)?;
            Ok(FePreconditioner::Multiplicative(p))
        }
    }
}

/// Build a [`FePreconditioner`] from a design and configuration.
///
/// When `Multiplicative` is requested, a Gramian is required for the sparse
/// residual updater.
pub fn build_preconditioner<S: Store>(
    design: &Design<S>,
    weights: Option<&[f64]>,
    gramian: Option<&Gramian>,
    preconditioner_config: &Preconditioner,
) -> WithinResult<FePreconditioner> {
    use crate::domain::build_local_domains;

    let domains = build_local_domains(design, weights);
    build_from_domains(domains, design.n_dofs, gramian, preconditioner_config)
}

/// Fused build: single observation scan -> domains + Gramian.
///
/// Uses `build_domains_and_gramian_blocks` to scan observations once, producing
/// both the domain decomposition (for the preconditioner) and per-pair block
/// data (for composing the explicit Gramian CSR). Avoids the double scan that
/// would occur if `Gramian::build` and `build_local_domains` were called
/// separately.
pub(crate) fn build_preconditioner_fused<S: Store>(
    design: &Design<S>,
    weights: Option<&[f64]>,
    preconditioner_config: &Preconditioner,
) -> WithinResult<(Gramian, FePreconditioner)> {
    use crate::domain::build_domains_and_gramian_blocks;

    let (domains, blocks) = build_domains_and_gramian_blocks(design, weights);
    let gramian = Gramian::from_pair_blocks(&blocks, &design.factors, design.n_dofs)?;
    let precond = build_from_domains(
        domains,
        design.n_dofs,
        Some(&gramian),
        preconditioner_config,
    )?;

    Ok((gramian, precond))
}
