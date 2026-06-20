//! Schwarz preconditioner: bridges FE domain types to the generic
//! `schwarz-precond` API, plus the opaque public [`Preconditioner`] handle.

use std::sync::Arc;

use rayon::prelude::*;
use schwarz_precond::{Operator, SchwarzPreconditioner, SubdomainEntry};
use serde::{Deserialize, Serialize};

use crate::block_elim::BlockElimSolver;
use crate::config::{LocalSolverConfig, PreconditionerConfig};
use crate::domain::{Design, LocalDomain};
use crate::observation::{factor_columns, level_at, validate_weights, Store};
use crate::BuildError;

/// Concrete additive Schwarz type used in the parent crate.
#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct FeSchwarz(SchwarzPreconditioner<BlockElimSolver>);

impl std::fmt::Debug for FeSchwarz {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FeSchwarz")
            .field("n_subdomains", &self.0.subdomains().len())
            .finish()
    }
}

impl Operator for FeSchwarz {
    fn nrows(&self) -> usize {
        self.0.nrows()
    }

    fn ncols(&self) -> usize {
        self.0.ncols()
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), schwarz_precond::SolveError> {
        self.0.apply(x, y)
    }

    fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) -> Result<(), schwarz_precond::SolveError> {
        self.0.apply_adjoint(x, y)
    }
}

/// Diagonal/Jacobi preconditioner for the fixed-effects Gramian.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct DiagonalPreconditioner {
    inv_diag: Arc<[f64]>,
}

impl DiagonalPreconditioner {
    fn new(inv_diag: Vec<f64>) -> Self {
        Self {
            inv_diag: Arc::from(inv_diag),
        }
    }
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

// ---------------------------------------------------------------------------
// Crate-internal builders
// ---------------------------------------------------------------------------

/// Build additive Schwarz with an explicit reduction strategy.
pub(crate) fn build_additive_with_strategy(
    domains: Vec<LocalDomain>,
    config: &LocalSolverConfig,
    strategy: schwarz_precond::ReductionStrategy,
) -> Result<FeSchwarz, BuildError> {
    let entries = domains
        .into_par_iter()
        .map(|domain| build_entry(domain, config))
        .collect::<Result<Vec<_>, BuildError>>()?;
    Ok(FeSchwarz(SchwarzPreconditioner::new(entries, strategy)))
}

/// Build a single `SubdomainEntry<BlockElimSolver>` from a pre-built CrossTab.
pub(crate) fn build_entry(
    domain: LocalDomain,
    config: &LocalSolverConfig,
) -> Result<SubdomainEntry<BlockElimSolver>, BuildError> {
    let LocalDomain {
        subdomain,
        cross_tab,
        block_diagonals,
    } = domain;
    let solver = BlockElimSolver::build(cross_tab, &block_diagonals, config)?;
    SubdomainEntry::try_new(subdomain.core, solver).map_err(BuildError::Preconditioner)
}

/// Opaque handle to a pre-built fixed-effects preconditioner.
///
/// Cloning is O(1): the inner factorization is `Arc`-backed, so passing
/// `&precond` to [`Solver::new`] never duplicates it.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Preconditioner {
    inner: Variant,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
enum Variant {
    // Keep Additive first: postcard encodes enum discriminants by declaration order,
    // and the v3 fixture depends on Additive remaining discriminant 0.
    Additive(FeSchwarz),
    Diagonal(DiagonalPreconditioner),
}

impl Preconditioner {
    /// Stable display name for the concrete preconditioner variant.
    pub fn variant_name(&self) -> &'static str {
        match &self.inner {
            Variant::Additive(_) => "Additive",
            Variant::Diagonal(_) => "Diagonal",
        }
    }

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
}

impl Operator for Preconditioner {
    fn nrows(&self) -> usize {
        match &self.inner {
            Variant::Additive(p) => p.nrows(),
            Variant::Diagonal(p) => p.nrows(),
        }
    }

    fn ncols(&self) -> usize {
        match &self.inner {
            Variant::Additive(p) => p.ncols(),
            Variant::Diagonal(p) => p.ncols(),
        }
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), schwarz_precond::SolveError> {
        match &self.inner {
            Variant::Additive(p) => p.apply(x, y),
            Variant::Diagonal(p) => p.apply(x, y),
        }
    }

    fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) -> Result<(), schwarz_precond::SolveError> {
        match &self.inner {
            Variant::Additive(p) => p.apply_adjoint(x, y),
            Variant::Diagonal(p) => p.apply_adjoint(x, y),
        }
    }
}

fn build_diagonal<S: Store>(
    design: &Design<S>,
    weights: Option<&[f64]>,
) -> Result<DiagonalPreconditioner, BuildError> {
    let mut diag = vec![0.0; design.n_dofs];
    let cols = factor_columns(&design.store);

    for (factor_idx, factor) in design.factors.iter().enumerate() {
        let slice = &mut diag[factor.offset..factor.offset + factor.n_levels];
        for uid in 0..design.n_obs {
            let level = level_at(&design.store, cols[factor_idx], uid, factor_idx);
            slice[level] += weights.map_or(1.0, |w| w[uid]);
        }
    }

    // Invert in place. A zero diagonal entry is an unidentified DOF — a
    // structural zero column of `D` from an unobserved or fully zero-weighted
    // level. Take the pseudo-inverse (`inv = 0`) so the coordinate stays in the
    // preconditioner's null space and resolves to 0, matching the
    // unpreconditioned path. A non-finite reciprocal (a diagonal so small that
    // `1/d` overflows) is genuinely degenerate and still rejected.
    for (index, d) in diag.iter_mut().enumerate() {
        if *d == 0.0 {
            continue;
        }
        let inv = 1.0 / *d;
        if !inv.is_finite() {
            return Err(BuildError::SingularDiagonal {
                block: "diagonal",
                index,
            });
        }
        *d = inv;
    }

    Ok(DiagonalPreconditioner::new(diag))
}

/// Build a [`Preconditioner`] from a design and optional observation weights.
pub(crate) fn build_preconditioner<S: Store>(
    design: &Design<S>,
    weights: Option<&[f64]>,
    config: Option<&PreconditionerConfig>,
) -> Result<Option<Preconditioner>, BuildError> {
    use crate::domain::build_local_domains;

    validate_weights(weights, design.n_obs)?;

    let default_cfg = PreconditionerConfig::default();
    let resolved = config.unwrap_or(&default_cfg);
    match resolved {
        PreconditionerConfig::Off => Ok(None),
        PreconditionerConfig::Additive {
            local_solver,
            reduction,
        } => {
            let domains = build_local_domains(design, weights);
            if domains.is_empty() {
                // Single-factor designs (and other configurations with no
                // factor-pair subdomains) have no useful additive Schwarz
                // preconditioner. Fall back to unpreconditioned LSMR.
                return Ok(None);
            }
            let p = build_additive_with_strategy(domains, local_solver, *reduction)?;
            Ok(Some(Preconditioner {
                inner: Variant::Additive(p),
            }))
        }
        PreconditionerConfig::Diagonal => {
            let p = build_diagonal(design, weights)?;
            Ok(Some(Preconditioner {
                inner: Variant::Diagonal(p),
            }))
        }
    }
}
