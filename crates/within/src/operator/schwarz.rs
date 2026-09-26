//! Schwarz preconditioner: bridges FE domain types to the generic
//! `schwarz-precond` API, plus the opaque public [`Preconditioner`] handle.

use rayon::prelude::*;
use schwarz_precond::{Operator, SchwarzPreconditioner, SubdomainEntry};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::block_elim::BlockElimSolver;
use crate::config::{LocalSolverConfig, PreconditionerConfig, ReductionStrategy, Staleness};
use crate::domain::{LocalDomain, PreparedDesign};
use crate::operator::gauge::GaugeConstraint;
use crate::{BuildError, BuildWarning};

#[cfg(test)]
mod tests;

/// The additive Schwarz map's description: what the builder builds and a built map records.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct SchwarzConfig {
    pub(crate) local_solver: LocalSolverConfig,
    pub(crate) reduction: ReductionStrategy,
}

/// Concrete additive Schwarz type used in the parent crate.
#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct FeSchwarz {
    inner: SchwarzPreconditioner<BlockElimSolver>,
    config: SchwarzConfig,
}

impl std::fmt::Debug for FeSchwarz {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FeSchwarz")
            .field("n_subdomains", &self.inner.subdomains().len())
            .field("config", &self.config)
            .finish()
    }
}

impl Operator for FeSchwarz {
    fn nrows(&self) -> usize {
        self.inner.nrows()
    }

    fn ncols(&self) -> usize {
        self.inner.ncols()
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), schwarz_precond::SolveError> {
        self.inner.apply(x, y)
    }

    fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) -> Result<(), schwarz_precond::SolveError> {
        self.inner.apply_adjoint(x, y)
    }
}

/// Diagonal/Jacobi preconditioner for the fixed-effects Gramian.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct DiagonalPreconditioner {
    inv_diag: Arc<[f64]>,
}

impl Operator for DiagonalPreconditioner {
    fn nrows(&self) -> usize {
        self.inv_diag.len()
    }

    fn ncols(&self) -> usize {
        self.inv_diag.len()
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), schwarz_precond::SolveError> {
        let n = self.inv_diag.len();
        if x.len() != n || y.len() != n {
            return Err(schwarz_precond::SolveError::InvalidInput {
                context: "DiagonalPreconditioner::apply",
                message: format!(
                    "x.len()={}, y.len()={}, expected n_dofs={}",
                    x.len(),
                    y.len(),
                    n
                ),
            });
        }
        for ((yi, &xi), &di) in y.iter_mut().zip(x.iter()).zip(self.inv_diag.iter()) {
            *yi = di * xi;
        }
        Ok(())
    }

    /// A diagonal operator is symmetric (`M^T = M`), so the adjoint is `apply`.
    fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) -> Result<(), schwarz_precond::SolveError> {
        self.apply(x, y)
    }
}

/// `n_dofs` may exceed the span of subdomain indices; an uncovered column resolves to `0`.
pub(crate) fn build_additive(
    domains: Vec<LocalDomain>,
    config: &SchwarzConfig,
    n_dofs: usize,
) -> Result<FeSchwarz, BuildError> {
    let entries = domains
        .into_par_iter()
        .map(|domain| build_entry(domain, &config.local_solver))
        .collect::<Result<Vec<_>, BuildError>>()?;
    Ok(FeSchwarz {
        inner: SchwarzPreconditioner::with_n_dofs(entries, n_dofs, config.reduction),
        config: config.clone(),
    })
}

/// Build a single `SubdomainEntry<BlockElimSolver>` from a pre-built CrossTab.
pub(crate) fn build_entry(
    domain: LocalDomain,
    config: &LocalSolverConfig,
) -> Result<SubdomainEntry<BlockElimSolver>, BuildError> {
    let LocalDomain { core, component } = domain;
    let solver = BlockElimSolver::build(component, config)?;
    SubdomainEntry::try_new(core, solver).map_err(BuildError::Preconditioner)
}

/// Opaque handle to a pre-built preconditioner; cloning is O(1) via `Arc`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Preconditioner {
    inner: Variant,
    build_duration: Duration,
    /// Cross-term nulls every apply keeps out of the solve space, `P M⁻¹ P`; a property of the
    /// design the solver attaches, so it is rebuilt rather than serialized.
    #[serde(skip)]
    pub(crate) gauge: Option<Arc<GaugeConstraint>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
enum Variant {
    // Append only: postcard encodes by declaration order and the fixtures depend on it.
    Additive(FeSchwarz),
    Diagonal(DiagonalPreconditioner),
    Adaptive(AdaptiveLadder),
}

/// An unescalated `Adaptive` map: the diagonal plus what a reusing solver needs to escalate.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct AdaptiveLadder {
    base: DiagonalPreconditioner,
    pub(crate) stall: Staleness,
    pub(crate) escalated: SchwarzConfig,
}

impl Preconditioner {
    /// Stable display name for the concrete preconditioner variant.
    pub fn variant_name(&self) -> &'static str {
        match &self.inner {
            Variant::Additive(_) => "Additive",
            Variant::Diagonal(_) => "Diagonal",
            Variant::Adaptive(_) => "Adaptive",
        }
    }

    /// The ladder a reusing solver resumes, while this map has not escalated.
    pub(crate) fn ladder(&self) -> Option<&AdaptiveLadder> {
        match &self.inner {
            Variant::Adaptive(ladder) => Some(ladder),
            Variant::Additive(_) | Variant::Diagonal(_) => None,
        }
    }

    /// Drop the ladder for a design with no factor pair to escalate to, keeping its diagonal.
    pub(crate) fn settle(self) -> Self {
        match self.inner {
            Variant::Adaptive(ladder) => Self {
                inner: Variant::Diagonal(ladder.base),
                ..self
            },
            Variant::Additive(_) | Variant::Diagonal(_) => self,
        }
    }

    /// The configuration that rebuilds this map.
    pub fn config(&self) -> PreconditionerConfig {
        match &self.inner {
            Variant::Additive(p) => PreconditionerConfig::Additive {
                local_solver: p.config.local_solver.clone(),
                reduction: p.config.reduction,
            },
            Variant::Diagonal(_) => PreconditionerConfig::Diagonal,
            Variant::Adaptive(ladder) => PreconditionerConfig::Adaptive {
                local_solver: ladder.escalated.local_solver.clone(),
                reduction: ladder.escalated.reduction,
                stall: ladder.stall,
            },
        }
    }

    // `within` does not re-export `schwarz_precond::Operator`, so these are the only public way.
    /// Number of rows of the underlying linear operator.
    pub fn nrows(&self) -> usize {
        <Self as schwarz_precond::Operator>::nrows(self)
    }

    /// Number of columns of the underlying linear operator.
    pub fn ncols(&self) -> usize {
        <Self as schwarz_precond::Operator>::ncols(self)
    }

    /// Apply the preconditioner: writes `M^{-1} x` into `y`.
    pub fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), schwarz_precond::SolveError> {
        <Self as schwarz_precond::Operator>::apply(self, x, y)
    }

    /// Time spent building this preconditioner.
    pub fn build_duration(&self) -> Duration {
        self.build_duration
    }
}

impl Preconditioner {
    fn base(&self) -> &dyn Operator {
        match &self.inner {
            Variant::Additive(p) => p,
            Variant::Diagonal(p) => p,
            Variant::Adaptive(ladder) => &ladder.base,
        }
    }
}

impl Operator for Preconditioner {
    fn nrows(&self) -> usize {
        self.base().nrows()
    }

    fn ncols(&self) -> usize {
        self.base().ncols()
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), schwarz_precond::SolveError> {
        match &self.gauge {
            Some(gauge) => gauge.constrain(x, y, |p, y| self.base().apply(p, y)),
            None => self.base().apply(x, y),
        }
    }

    fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) -> Result<(), schwarz_precond::SolveError> {
        self.apply(x, y)
    }
}

/// Build the diagonal/Jacobi map.
pub(crate) fn build_diagonal(prepared: &PreparedDesign<'_>) -> Result<Preconditioner, BuildError> {
    let build_started = Instant::now();
    let diagonal = diagonal_map(prepared)?;
    Ok(Preconditioner {
        inner: Variant::Diagonal(diagonal),
        build_duration: build_started.elapsed(),
        gauge: None,
    })
}

/// Build the diagonal base of an `Adaptive` strategy; the Schwarz rung is left to a stalled solve.
pub(crate) fn build_adaptive(
    prepared: &PreparedDesign<'_>,
    stall: Staleness,
    escalated: SchwarzConfig,
) -> Result<Preconditioner, BuildError> {
    let build_started = Instant::now();
    let base = diagonal_map(prepared)?;
    Ok(Preconditioner {
        inner: Variant::Adaptive(AdaptiveLadder {
            base,
            stall,
            escalated,
        }),
        build_duration: build_started.elapsed(),
        gauge: None,
    })
}

fn diagonal_map(prepared: &PreparedDesign<'_>) -> Result<DiagonalPreconditioner, BuildError> {
    let mut diag = prepared.gram_diagonal();

    // A zero diagonal is an unidentified DOF, so the pseudo-inverse keeps it in the null space.
    for (index, d) in diag.iter_mut().enumerate() {
        if *d == 0.0 {
            continue;
        }
        let inv = 1.0 / *d;
        if !inv.is_finite() {
            return Err(BuildError::SingularDiagonal { index });
        }
        *d = inv;
    }

    Ok(DiagonalPreconditioner {
        inv_diag: Arc::from(diag),
    })
}

/// Build the additive Schwarz map plus its warnings; `None` when the design has no factor-pair
/// subdomain to build on, where plain LSMR is the fallback.
pub(crate) fn build_schwarz(
    prepared: &PreparedDesign<'_>,
    config: &SchwarzConfig,
) -> Result<(Option<Preconditioner>, Vec<BuildWarning>), BuildError> {
    let build_started = Instant::now();
    let (domains, warnings) = crate::domain::build_local_domains(prepared, &config.local_solver)?;
    if domains.is_empty() {
        return Ok((None, warnings));
    }
    let schwarz = build_additive(domains, config, prepared.design.n_dofs)?;
    let preconditioner = Preconditioner {
        inner: Variant::Additive(schwarz),
        build_duration: build_started.elapsed(),
        gauge: None,
    };
    Ok((Some(preconditioner), warnings))
}
