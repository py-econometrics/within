//! Schwarz preconditioner: bridges FE domain types to the generic
//! `schwarz-precond` API, plus the opaque public [`Preconditioner`] handle.

use std::sync::Arc;

use rayon::prelude::*;
use schwarz_precond::{Operator, SchwarzPreconditioner, SubdomainEntry};
use serde::{Deserialize, Serialize};

use crate::block_elim::BlockElimSolver;
use crate::config::{LocalSolverConfig, PreconditionerConfig};
use crate::domain::{Design, LocalDomain};
use crate::{BuildError, BuildWarning};

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
///
/// `n_dofs` is the operator's column count, which can exceed the span of the
/// subdomains' indices: an unidentified direction (e.g. a singleton level's
/// slope) is a structural-zero column no subdomain covers, yet must count
/// toward the shape. Uncovered DOFs resolve to `0`, like `Off`/`Diagonal`.
pub(crate) fn build_additive_with_strategy(
    domains: Vec<LocalDomain>,
    config: &LocalSolverConfig,
    strategy: schwarz_precond::ReductionStrategy,
    n_dofs: usize,
) -> Result<FeSchwarz, BuildError> {
    let entries = domains
        .into_par_iter()
        .map(|domain| build_entry(domain, config))
        .collect::<Result<Vec<_>, BuildError>>()?;
    Ok(FeSchwarz(SchwarzPreconditioner::with_n_dofs(
        entries, n_dofs, strategy,
    )))
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

/// Opaque handle to a pre-built fixed-effects preconditioner.
///
/// Cloning is O(1): the inner factorization is `Arc`-backed, so passing
/// `&precond` to [`Solver::new`](crate::Solver::new) never duplicates it.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Preconditioner {
    inner: Variant,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
enum Variant {
    // Keep Additive first: postcard encodes enum discriminants by declaration order,
    // and the wire fixture depends on Additive remaining discriminant 0.
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

fn build_diagonal(
    design: &Design<'_>,
    weights: Option<&[f64]>,
) -> Result<DiagonalPreconditioner, BuildError> {
    let mut diag = vec![0.0; design.n_dofs];

    // diag(DᵀWD): each column contributes w·loading² per observation — the
    // loading is 1 for an intercept column and the slope value otherwise.
    for (factor_idx, term) in design.terms.iter().enumerate() {
        let levels = design.frame.level_column(factor_idx);
        let w = |uid: usize| weights.map_or(1.0, |ws| ws[uid]);
        let mut column = 0;
        if term.intercept {
            let slice = &mut diag[term.offset..term.offset + term.n_levels];
            for (uid, &level) in levels.iter().enumerate() {
                slice[level as usize] += w(uid);
            }
            column = 1;
        }
        for &z_col in &term.slopes {
            let z = design.frame.loading_column(z_col);
            let base = term.column_base(column);
            let slice = &mut diag[base..base + term.n_levels];
            for (uid, &level) in levels.iter().enumerate() {
                // Keep `w * z * z` left-to-right: a zero weight then kills a
                // huge `z` before the square can overflow (0 * inf = NaN).
                slice[level as usize] += w(uid) * z[uid] * z[uid];
            }
            column += 1;
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

    Ok(DiagonalPreconditioner {
        inv_diag: Arc::from(diag),
    })
}

/// Build a [`Preconditioner`] from a design and optional observation weights,
/// plus any non-fatal [`BuildWarning`]s the build produced.
pub(crate) fn build_preconditioner(
    design: &Design<'_>,
    weights: Option<&[f64]>,
    config: Option<&PreconditionerConfig>,
) -> Result<(Option<Preconditioner>, Vec<BuildWarning>), BuildError> {
    use crate::domain::build_local_domains;

    design.validate_weights(weights)?;

    let default_cfg = PreconditionerConfig::default();
    let resolved = config.unwrap_or(&default_cfg);
    match resolved {
        PreconditionerConfig::Off => Ok((None, Vec::new())),
        PreconditionerConfig::Additive {
            local_solver,
            reduction,
        } => {
            let (domains, warnings) = build_local_domains(design, weights, &local_solver.scaling)?;
            if domains.is_empty() {
                // Single-factor designs (and other configurations with no
                // factor-pair subdomains) have no useful additive Schwarz
                // preconditioner. Fall back to unpreconditioned LSMR.
                return Ok((None, warnings));
            }
            let p = build_additive_with_strategy(domains, local_solver, *reduction, design.n_dofs)?;
            Ok((
                Some(Preconditioner {
                    inner: Variant::Additive(p),
                }),
                warnings,
            ))
        }
        PreconditionerConfig::Diagonal => {
            let p = build_diagonal(design, weights)?;
            Ok((
                Some(Preconditioner {
                    inner: Variant::Diagonal(p),
                }),
                Vec::new(),
            ))
        }
    }
}
