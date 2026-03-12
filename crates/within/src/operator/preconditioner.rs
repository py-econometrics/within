//! Pre-built preconditioner for reuse across multiple solves.

use schwarz_precond::Operator;
use serde::{Deserialize, Serialize};

use crate::config::Preconditioner;
use crate::domain::WeightedDesign;
use crate::observation::ObservationStore;
use crate::operator::gramian::Gramian;
use crate::operator::schwarz::{
    build_additive, build_multiplicative_sparse, DomainSource, FeMultSchwarzSparse, FeSchwarz,
};
use crate::{WithinError, WithinResult};

/// A pre-built preconditioner ready for use in Krylov solves.
///
/// Implements [`Operator`] via enum dispatch to the inner variant.
#[derive(Serialize, Deserialize)]
pub enum FePreconditioner {
    /// Additive Schwarz (symmetric -- valid for CG and GMRES).
    Additive(FeSchwarz),
    /// Multiplicative Schwarz with sparse Gramian updater (GMRES only).
    Multiplicative(FeMultSchwarzSparse),
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

    match preconditioner_config {
        Preconditioner::Additive(config) => {
            let domains = build_local_domains(design);
            let precond = build_additive(
                DomainSource::<S>::FromParts(domains),
                design.n_dofs,
                config,
                schwarz_precond::ReductionStrategy::default(),
            )?;
            Ok(FePreconditioner::Additive(precond))
        }
        Preconditioner::Multiplicative(config) => {
            let gramian = gramian.ok_or_else(|| {
                WithinError::LocalSolverBuild(
                    "multiplicative preconditioner requires an explicit Gramian".to_string(),
                )
            })?;
            let domains = build_local_domains(design);
            let precond = build_multiplicative_sparse(
                DomainSource::<S>::FromParts(domains),
                gramian,
                design.n_dofs,
                config,
            )?;
            Ok(FePreconditioner::Multiplicative(precond))
        }
    }
}

/// Fused build: single observation scan → domains + Gramian.
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

    let precond = match preconditioner_config {
        Preconditioner::Additive(config) => {
            let p = build_additive(
                DomainSource::<S>::FromParts(domains),
                design.n_dofs,
                config,
                schwarz_precond::ReductionStrategy::default(),
            )?;
            FePreconditioner::Additive(p)
        }
        Preconditioner::Multiplicative(config) => {
            let p = build_multiplicative_sparse(
                DomainSource::<S>::FromParts(domains),
                &gramian,
                design.n_dofs,
                config,
            )?;
            FePreconditioner::Multiplicative(p)
        }
    };

    Ok((gramian, precond))
}
