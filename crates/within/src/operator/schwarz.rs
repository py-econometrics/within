//! Schwarz preconditioner: bridges FE domain types to the generic
//! `schwarz-precond` API, plus the opaque public [`Preconditioner`] handle.

use std::sync::Arc;

use rayon::prelude::*;
use schwarz_precond::{Operator, SchwarzPreconditioner, SubdomainEntry};
use serde::{Deserialize, Serialize};

use crate::block_elim::BlockElimSolver;
use crate::config::{LocalSolverConfig, PreconditionerConfig};
use crate::domain::Loading;
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

/// `n_dofs` may exceed the span of subdomain indices; an uncovered column resolves to `0`.
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

/// Opaque handle to a pre-built preconditioner; cloning is O(1) via `Arc`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Preconditioner {
    inner: Variant,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
enum Variant {
    // Keep Additive first: postcard encodes by declaration order and the fixture depends on it.
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

    for (factor_idx, term) in design.terms.iter().enumerate() {
        let levels = design.frame.level_column(factor_idx);
        let w = |uid: usize| weights.map_or(1.0, |ws| ws[uid]);
        for (column, loading) in term.columns.iter().enumerate() {
            let base = term.column_base(column);
            let slice = &mut diag[base..base + term.n_levels];
            match loading {
                Loading::Constant => {
                    for (uid, &level) in levels.iter().enumerate() {
                        slice[level as usize] += w(uid);
                    }
                }
                Loading::Covariate(z_col) => {
                    let z = design.frame.loading_column(*z_col as usize);
                    for (uid, &level) in levels.iter().enumerate() {
                        // Keep `w * z * z` left-to-right: a zero weight kills a huge `z` first.
                        slice[level as usize] += w(uid) * z[uid] * z[uid];
                    }
                }
            }
        }
    }

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

/// Build a [`Preconditioner`] from a design and optional weights, plus any warnings.
pub(crate) fn build_preconditioner(
    design: &Design<'_>,
    weights: Option<&[f64]>,
    config: Option<&PreconditionerConfig>,
) -> Result<(Option<Preconditioner>, Vec<BuildWarning>), BuildError> {
    use crate::domain::build_local_domains;

    // Weights are pre-validated by the sole caller, whose permutation preserves length and sign.
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
                // No factor-pair subdomains means no useful Schwarz; fall back to plain LSMR.
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
