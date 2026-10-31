//! Schwarz preconditioner: bridges FE domain types to the generic
//! `schwarz-precond` API, plus the opaque public [`Preconditioner`] handle.

use rayon::prelude::*;
use schwarz_precond::{Operator, SchwarzPreconditioner, SubdomainEntry};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::block_elim::BlockElimSolver;
use crate::config::{LocalSolverConfig, PreconditionerConfig, Staleness};
use crate::domain::{LocalDomain, PreparedDesign};
use crate::{BuildError, BuildWarning};

#[cfg(test)]
mod tests;

/// `n_dofs` may exceed the span of subdomain indices; an uncovered column resolves to `0`.
pub(crate) fn build_additive(
    domains: Vec<LocalDomain>,
    local_solver: &LocalSolverConfig,
    n_dofs: usize,
) -> Result<SchwarzPreconditioner<BlockElimSolver>, BuildError> {
    let entries = domains
        .into_par_iter()
        .map(|LocalDomain { core, component }| {
            let solver = BlockElimSolver::build(component, local_solver)?;
            SubdomainEntry::try_new(core, solver).map_err(BuildError::Preconditioner)
        })
        .collect::<Result<Vec<_>, BuildError>>()?;
    Ok(SchwarzPreconditioner::with_n_dofs(entries, n_dofs))
}

/// Opaque handle to a pre-built preconditioner; cloning is O(1) via `Arc`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Preconditioner {
    inner: Variant,
    build_duration: Duration,
}

#[derive(Clone, Serialize, Deserialize)]
enum Variant {
    // Append only: postcard encodes by declaration order and the fixtures depend on it.
    Additive {
        schwarz: SchwarzPreconditioner<BlockElimSolver>,
        local_solver: LocalSolverConfig,
    },
    Diagonal(Arc<[f64]>),
    Adaptive(AdaptiveLadder),
}

impl std::fmt::Debug for Variant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Additive {
                schwarz,
                local_solver,
            } => f
                .debug_struct("Additive")
                .field("n_subdomains", &schwarz.subdomains().len())
                .field("local_solver", local_solver)
                .finish(),
            Self::Diagonal(inv_diag) => f.debug_tuple("Diagonal").field(inv_diag).finish(),
            Self::Adaptive(ladder) => f.debug_tuple("Adaptive").field(ladder).finish(),
        }
    }
}

/// An unescalated `Adaptive` map: the diagonal plus what a reusing solver needs to escalate.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct AdaptiveLadder {
    base: Arc<[f64]>,
    pub(crate) stall: Staleness,
    pub(crate) escalated: LocalSolverConfig,
}

impl Preconditioner {
    /// Stable display name for the concrete preconditioner variant.
    pub fn variant_name(&self) -> &'static str {
        match &self.inner {
            Variant::Additive { .. } => "Additive",
            Variant::Diagonal(_) => "Diagonal",
            Variant::Adaptive(_) => "Adaptive",
        }
    }

    /// The ladder a reusing solver resumes, while this map has not escalated.
    pub(crate) fn ladder(&self) -> Option<&AdaptiveLadder> {
        match &self.inner {
            Variant::Adaptive(ladder) => Some(ladder),
            Variant::Additive { .. } | Variant::Diagonal(_) => None,
        }
    }

    /// Drop the ladder for a design with no factor pair to escalate to, keeping its diagonal.
    pub(crate) fn settle(self) -> Self {
        match self.inner {
            Variant::Adaptive(ladder) => Self {
                inner: Variant::Diagonal(ladder.base),
                ..self
            },
            Variant::Additive { .. } | Variant::Diagonal(_) => self,
        }
    }

    /// The configuration that rebuilds this map.
    pub fn config(&self) -> PreconditionerConfig {
        match &self.inner {
            Variant::Additive { local_solver, .. } => PreconditionerConfig::Additive {
                local_solver: local_solver.clone(),
            },
            Variant::Diagonal(_) => PreconditionerConfig::Diagonal,
            Variant::Adaptive(ladder) => PreconditionerConfig::Adaptive {
                local_solver: ladder.escalated.clone(),
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

impl Operator for Preconditioner {
    fn nrows(&self) -> usize {
        match &self.inner {
            Variant::Additive { schwarz, .. } => schwarz.nrows(),
            Variant::Diagonal(d) | Variant::Adaptive(AdaptiveLadder { base: d, .. }) => d.len(),
        }
    }

    fn ncols(&self) -> usize {
        self.nrows()
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), schwarz_precond::SolveError> {
        let inv_diag = match &self.inner {
            Variant::Additive { schwarz, .. } => return schwarz.apply(x, y),
            Variant::Diagonal(d) | Variant::Adaptive(AdaptiveLadder { base: d, .. }) => d,
        };
        let n = inv_diag.len();
        if x.len() != n || y.len() != n {
            return Err(schwarz_precond::SolveError::InvalidInput {
                context: "Preconditioner::apply",
                message: format!(
                    "x.len()={}, y.len()={}, expected n_dofs={n}",
                    x.len(),
                    y.len()
                ),
            });
        }
        for ((yi, &xi), &di) in y.iter_mut().zip(x).zip(inv_diag.iter()) {
            *yi = di * xi;
        }
        Ok(())
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
    })
}

/// Build the diagonal base of an `Adaptive` strategy; the Schwarz rung is left to a stalled solve.
pub(crate) fn build_adaptive(
    prepared: &PreparedDesign<'_>,
    stall: Staleness,
    escalated: LocalSolverConfig,
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
    })
}

fn diagonal_map(prepared: &PreparedDesign<'_>) -> Result<Arc<[f64]>, BuildError> {
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

    Ok(Arc::from(diag))
}

/// What one preconditioner build yields: the map (`None` adds nothing below it) and its warnings.
pub(crate) struct PreconditionerBuild {
    pub(crate) map: Option<Preconditioner>,
    pub(crate) warnings: Vec<BuildWarning>,
}

/// The additive Schwarz build; no map when the design has no factor-pair subdomain.
pub(crate) fn build_schwarz(
    prepared: &PreparedDesign<'_>,
    local_solver: &LocalSolverConfig,
) -> Result<PreconditionerBuild, BuildError> {
    let build_started = Instant::now();
    let (domains, warnings) = crate::domain::build_local_domains(prepared, local_solver)?;
    if domains.is_empty() {
        return Ok(PreconditionerBuild {
            map: None,
            warnings,
        });
    }
    let schwarz = build_additive(domains, local_solver, prepared.design.n_dofs)?;
    let preconditioner = Preconditioner {
        inner: Variant::Additive {
            schwarz,
            local_solver: local_solver.clone(),
        },
        build_duration: build_started.elapsed(),
    };
    Ok(PreconditionerBuild {
        map: Some(preconditioner),
        warnings,
    })
}
