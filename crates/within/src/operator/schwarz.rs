//! Schwarz preconditioner: bridges FE domain types to the generic
//! `schwarz-precond` API, plus the opaque public [`Preconditioner`] handle.

use rayon::prelude::*;
use schwarz_precond::{Operator, SchwarzPreconditioner, SubdomainEntry};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;

use crate::block_solver::BlockSolver;
use crate::config::{LocalSolverConfig, PreconditionerConfig, Rung};
use crate::domain::{LocalDomain, PreparedDesign, Unknowns};
use crate::{BuildError, BuildWarning};

#[cfg(test)]
mod tests;

/// Concrete additive Schwarz type used in the parent crate.
#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct FeSchwarz {
    inner: SchwarzPreconditioner<BlockSolver>,
}

impl std::fmt::Debug for FeSchwarz {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FeSchwarz")
            .field("n_subdomains", &self.inner.subdomains().len())
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
pub(crate) struct DiagonalPreconditioner {
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
    n_dofs: usize,
) -> Result<FeSchwarz, BuildError> {
    let entries = domains
        .into_par_iter()
        .map(build_entry)
        .collect::<Result<Vec<_>, BuildError>>()?;
    Ok(FeSchwarz {
        inner: SchwarzPreconditioner::with_n_dofs(entries, n_dofs),
    })
}

pub(crate) fn build_entry(domain: LocalDomain) -> Result<SubdomainEntry<BlockSolver>, BuildError> {
    let LocalDomain { core, solver } = domain;
    SubdomainEntry::try_new(core, solver).map_err(BuildError::Preconditioner)
}

/// Opaque handle to a pre-built preconditioner; cloning is O(1) via `Arc`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(try_from = "Decoded")]
pub struct Preconditioner {
    pub(crate) map: RungMap,
    /// The config this map is a rung of, as written.
    pub(crate) config: PreconditionerConfig,
    pub(crate) build_duration: Duration,
}

/// [`Preconditioner`] as decoded, before its map is known to be a rung of its config.
#[derive(Deserialize)]
struct Decoded {
    map: RungMap,
    config: PreconditionerConfig,
    build_duration: Duration,
}

impl TryFrom<Decoded> for Preconditioner {
    type Error = &'static str;

    fn try_from(decoded: Decoded) -> Result<Self, Self::Error> {
        let preconditioner = Self {
            map: decoded.map,
            config: decoded.config,
            build_duration: decoded.build_duration,
        };
        match preconditioner.config.position(preconditioner.rung()) {
            Some(_) => Ok(preconditioner),
            None => Err("the map is not a rung of its config"),
        }
    }
}

/// The linear map of one rung.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) enum RungMap {
    // Append only: postcard encodes by declaration order and the fixtures depend on it.
    Diagonal(DiagonalPreconditioner),
    Schwarz(FeSchwarz),
}

/// One rung built on one system, and the build's warnings.
pub(crate) struct RungBuild<M = RungMap> {
    /// `None` when the system's operator is zero: every solve is exact at `x = 0`.
    pub(crate) map: Option<M>,
    pub(crate) warnings: Vec<BuildWarning>,
}

impl Preconditioner {
    /// The rung of its config this map is.
    pub fn rung(&self) -> Rung {
        match &self.map {
            RungMap::Diagonal(_) => Rung::Diagonal,
            RungMap::Schwarz(_) => Rung::Schwarz,
        }
    }

    /// The config this map is a rung of, as written.
    pub fn config(&self) -> &PreconditionerConfig {
        &self.config
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
    fn operator(&self) -> &dyn Operator {
        match &self.map {
            RungMap::Diagonal(p) => p,
            RungMap::Schwarz(p) => p,
        }
    }
}

impl Operator for Preconditioner {
    fn nrows(&self) -> usize {
        self.operator().nrows()
    }

    fn ncols(&self) -> usize {
        self.operator().ncols()
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), schwarz_precond::SolveError> {
        self.operator().apply(x, y)
    }

    fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) -> Result<(), schwarz_precond::SolveError> {
        self.apply(x, y)
    }
}

/// The diagonal/Jacobi map of a normal matrix with diagonal `diag`.
pub(crate) fn diagonal_map(mut diag: Vec<f64>) -> Result<DiagonalPreconditioner, BuildError> {
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

/// The additive Schwarz map over `unknowns`; `None` when no factor pair yields a subdomain, which
/// on a design of two or more terms happens only when the operator over `unknowns` is zero.
pub(crate) fn build_schwarz(
    prepared: &PreparedDesign<'_>,
    local_solver: &LocalSolverConfig,
    unknowns: Unknowns,
) -> Result<RungBuild, BuildError> {
    let (domains, warnings) = crate::domain::build_local_domains(prepared, local_solver, unknowns)?;
    let map = if domains.is_empty() {
        None
    } else {
        Some(RungMap::Schwarz(build_additive(
            domains,
            prepared.design.n_unknowns(unknowns),
        )?))
    };
    Ok(RungBuild { map, warnings })
}
