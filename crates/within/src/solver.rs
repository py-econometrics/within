//! The solve API: the persistent [`Solver`] (caches the preconditioner across
//! multiple solves on the same design) and the one-shot [`solve`] / [`solve_batch`]
//! convenience wrappers built on top of it.

use std::borrow::Cow;
use std::sync::Arc;
use std::time::Instant;

use ndarray::ArrayView2;
use rayon::prelude::*;
use schwarz_precond::{lsmr as lsmr_solve, mlsmr, LsmrResult, MlsmrOptions};

use crate::channel::CoefficientAddress;
use crate::config::{LsmrOptions, PreconditionerConfig};
use crate::domain::collinearity::{detect_collinear_slopes, CollinearSlope};
use crate::domain::{Design, Effect, PreparedDesign};
use crate::operator::design::gather_apply;
use crate::operator::gauge::GaugeConstraint;
use crate::operator::schwarz::{build_diagonal, build_schwarz, Preconditioner, SchwarzConfig};
use crate::operator::DesignOperator;
use crate::{BuildError, BuildWarning, SolveError, WithinError};

mod layout;
#[cfg(test)]
mod tests;

pub use layout::CoefficientLayout;

/// Fallible conversion into a [`Design`] for [`Solver::new`]: a categories
/// matrix (`ArrayView2<u32>`), a list of [`Effect`] terms, a pass-through
/// [`Design`], or a `&Design` whose storage the solver then shares.
pub trait IntoDesign<'a> {
    /// Build the [`Design`], validating inputs along the way.
    fn into_design(self) -> Result<Design<'a>, BuildError>;
}

impl<'a> IntoDesign<'a> for ArrayView2<'a, u32> {
    fn into_design(self) -> Result<Design<'a>, BuildError> {
        Design::from_categories(self)
    }
}

impl<'a> IntoDesign<'a> for Design<'a> {
    fn into_design(self) -> Result<Design<'a>, BuildError> {
        Ok(self)
    }
}

impl<'a> IntoDesign<'a> for &Design<'a> {
    fn into_design(self) -> Result<Design<'a>, BuildError> {
        Ok(self.clone())
    }
}

impl<'a> IntoDesign<'a> for Vec<Effect<'a>> {
    fn into_design(self) -> Result<Design<'a>, BuildError> {
        Design::new(self)
    }
}

/// Preconditioner input for [`Solver::new`].
///
/// Constructed implicitly via `From`/`Into` from any of:
/// - bare `None` — build the library default Schwarz preconditioner
/// - `&PreconditionerConfig` or `Some(&PreconditionerConfig)` — build from a tuned config
/// - `PreconditionerConfig` (owned) — same as above
/// - [`Preconditioner`] (owned or `&`) — reuse a previously built (or deserialized) preconditioner
///
/// `None` resolves unambiguously because there is exactly one `From<Option<X>>`
/// impl (with `X = &PreconditionerConfig`).
pub enum PreconditionerInput {
    /// Library default: an additive Schwarz preconditioner with default tuning.
    Default,
    /// Build from this config (`PreconditionerConfig::Off` ⇒ unpreconditioned).
    Config(PreconditionerConfig),
    /// Reuse this pre-built preconditioner (e.g. deserialized or pulled off a previous solver).
    Prebuilt(Preconditioner),
}

impl From<PreconditionerConfig> for PreconditionerInput {
    fn from(c: PreconditionerConfig) -> Self {
        Self::Config(c)
    }
}

impl From<&PreconditionerConfig> for PreconditionerInput {
    fn from(c: &PreconditionerConfig) -> Self {
        Self::Config(c.clone())
    }
}

impl From<Option<&PreconditionerConfig>> for PreconditionerInput {
    fn from(opt: Option<&PreconditionerConfig>) -> Self {
        opt.map_or(Self::Default, |c| Self::Config(c.clone()))
    }
}

impl From<Preconditioner> for PreconditionerInput {
    fn from(p: Preconditioner) -> Self {
        Self::Prebuilt(p)
    }
}

impl From<&Preconditioner> for PreconditionerInput {
    /// Reuse by reference; clone is O(1).
    fn from(p: &Preconditioner) -> Self {
        Self::Prebuilt(p.clone())
    }
}

/// Build the map a strategy names; `None` is unpreconditioned LSMR.
fn build_map(
    prepared: &PreparedDesign<'_>,
    config: PreconditionerConfig,
) -> Result<(Option<Preconditioner>, Vec<BuildWarning>), BuildError> {
    Ok(match config {
        PreconditionerConfig::Off => (None, Vec::new()),
        PreconditionerConfig::Diagonal => (Some(build_diagonal(prepared)?), Vec::new()),
        PreconditionerConfig::Additive {
            local_solver,
            reduction,
        } => build_schwarz(
            prepared,
            &SchwarzConfig {
                local_solver,
                reduction,
            },
        )?,
    })
}

/// Common solve output for all orchestration entry points.
#[derive(Debug, Clone)]
#[must_use]
pub struct SolveResult {
    /// Fixed-effect coefficients (length = total DOFs across all factors).
    ///
    /// Term-major by compact level position `p`: coefficient column `c` sits at
    /// `term_offset + c * n_levels + p`, with columns ordered
    /// `[intercept?, slopes…]`. Use [`SolveResult::layout`] to translate caller
    /// labels to these slots. Slots for unidentified directions hold the
    /// minimal-norm value `0`, never NaN; see [`SolveResult::unidentified`].
    pub x: Vec<f64>,
    /// Per-level directions the data cannot identify.
    pub unidentified: Vec<CoefficientAddress>,
    /// Non-fatal build warnings (see [`Solver::warnings`]).
    pub warnings: Vec<BuildWarning>,
    /// Address ↔ flat-`x`-index translation for this design's coefficients.
    pub layout: CoefficientLayout,
    /// Demeaned response: `y - D x` (length = n_obs), in caller order.
    ///
    /// Invariant: any per-observation field added here must be translated
    /// back from internal order via `Design::permute_obs_out` before being
    /// stored, or it leaks the locality-sorted row order to the caller.
    pub demeaned: Vec<f64>,
    /// Whether the iterative solver converged within `maxiter` iterations.
    pub converged: bool,
    /// Number of LSMR iterations used.
    pub iterations: usize,
    /// Relative normal-equation residual `||D^T W (y - Dx)|| / ||D^T W y||`,
    /// estimated from the LSMR recurrence (Fong & Saunders) at no extra cost.
    /// Exact for an unpreconditioned solve; for a preconditioned solve it is
    /// measured in the preconditioner's metric and typically sits a modest
    /// factor below the true-metric value.
    pub residual: f64,
    /// Wall-clock time for the entire solve (setup + LSMR), in seconds.
    pub time_total: f64,
    /// Wall-clock time for preconditioner construction, in seconds.
    pub time_setup: f64,
    /// Wall-clock time for the LSMR solve phase, in seconds.
    pub time_solve: f64,
}

/// Result of a batch solve across multiple RHS vectors.
#[derive(Debug, Clone)]
pub struct BatchSolveResult {
    /// All coefficient vectors concatenated (length = n_dofs * n_rhs), each
    /// block laid out as in [`SolveResult::x`].
    ///
    /// Slots for unidentified directions hold the minimal-norm value `0`,
    /// never NaN; see [`BatchSolveResult::unidentified`].
    pub x: Vec<f64>,
    /// Per-level directions the data cannot identify, shared across all RHS:
    /// identification depends only on the design and weights, never on `y`.
    pub unidentified: Vec<CoefficientAddress>,
    /// Non-fatal build warnings, shared across all RHS; see [`SolveResult::warnings`].
    pub warnings: Vec<BuildWarning>,
    /// Address ↔ flat-`x`-index translation for this design's coefficients.
    pub layout: CoefficientLayout,
    /// All demeaned responses concatenated (length = n_obs * n_rhs).
    pub demeaned: Vec<f64>,
    /// Per-RHS convergence flags.
    pub converged: Vec<bool>,
    /// Per-RHS iteration counts.
    pub iterations: Vec<usize>,
    /// Per-RHS relative normal-equation residual estimate; see
    /// [`SolveResult::residual`].
    pub residual: Vec<f64>,
    /// Per-RHS solve times in seconds.
    pub time_solve: Vec<f64>,
    /// Wall-clock time for the shared batch setup -- solver and preconditioner
    /// construction (`Solver::new`) -- in seconds; 0 when a pre-built
    /// preconditioner was reused.
    pub time_setup: f64,
    /// Total wall-clock time for the entire batch (setup + all solves), in seconds.
    pub time_total: f64,
    /// Number of coefficients per RHS (rows of the underlying design).
    pub n_dofs: usize,
    /// Number of observations (columns of the underlying design).
    pub n_obs: usize,
}

impl BatchSolveResult {
    /// Coefficient vector for the `i`-th RHS.
    pub fn x(&self, i: usize) -> &[f64] {
        &self.x[i * self.n_dofs..(i + 1) * self.n_dofs]
    }
    /// Demeaned response for the `i`-th RHS.
    pub fn demeaned(&self, i: usize) -> &[f64] {
        &self.demeaned[i * self.n_obs..(i + 1) * self.n_obs]
    }
}

/// Persistent solver that owns its preconditioner for reuse across multiple solves.
///
/// Build once with [`Solver::new`], then call [`Solver::solve`] or
/// [`Solver::solve_batch`] repeatedly with different RHS vectors. The expensive
/// preconditioner factorization happens only at construction time; LSMR tuning
/// ([`LsmrOptions`]) is supplied per call.
///
/// Ownership: each observation column is borrowed or owned independently
/// (`Cow`); a solver that outlives its inputs — e.g. one returned across the
/// Python boundary — uses owned columns.
pub struct Solver<'a> {
    prepared: PreparedDesign<'a>,
    preconditioner: Option<Preconditioner>,
    warnings: Vec<BuildWarning>,
}

impl std::fmt::Debug for Solver<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Solver")
            .field("n_obs", &self.prepared.design.n_obs)
            .field("n_dofs", &self.prepared.design.n_dofs)
            .field("has_weights", &self.prepared.sqrt_weights().is_some())
            .field("has_preconditioner", &self.preconditioner.is_some())
            .finish()
    }
}

/// Per-RHS solve output shared by [`Solver::solve`] and [`Solver::solve_batch`].
///
/// The design-level fields (`layout`, `warnings`, `unidentified`) are identical
/// across RHS, so the batch path attaches them once instead of cloning them per
/// RHS as it would if each worker returned a full [`SolveResult`].
struct RhsSolution {
    x: Vec<f64>,
    demeaned: Vec<f64>,
    converged: bool,
    iterations: usize,
    residual: f64,
    gather_secs: f64,
    time_solve: f64,
}

/// One RHS in the internal observation frame: `y` to subtract the fit from, `b` for LSMR.
struct PreparedRhs<'s> {
    op: DesignOperator<'s>,
    y: Cow<'s, [f64]>,
    /// `W^{1/2} y`; `None` when unweighted, where `b` is `y` itself.
    weighted: Option<Vec<f64>>,
    /// The gather is a recurring per-solve cost of the locality sort, so it counts as setup.
    gather_secs: f64,
}

impl PreparedRhs<'_> {
    fn b(&self) -> &[f64] {
        self.weighted.as_deref().unwrap_or(&self.y)
    }
}

/// One finished LSMR run and the seconds it alone took.
struct Run {
    result: LsmrResult,
    solve_secs: f64,
}

impl Run {
    fn timed(run: impl FnOnce() -> Result<LsmrResult, SolveError>) -> Result<Self, SolveError> {
        let t_start = Instant::now();
        let result = run()?;
        Ok(Self {
            result,
            solve_secs: t_start.elapsed().as_secs_f64(),
        })
    }
}

impl<'a> Solver<'a> {
    /// Construct a solver.
    ///
    /// `design` accepts raw categories (`ArrayView2<u32>`), a pre-built
    /// [`Design`], or `&Design` to share one design across solvers (an O(1)
    /// clone of its storage). `preconditioner` accepts:
    /// - `None` — build the library default Schwarz preconditioner
    /// - `&PreconditionerConfig` / `Some(&PreconditionerConfig)` — build from a tuned config
    /// - `PreconditionerConfig::Off` — solve unpreconditioned
    /// - `PreconditionerConfig::Diagonal` — use diagonal/Jacobi preconditioning
    /// - [`Preconditioner`] or `&Preconditioner` — reuse a previously built one
    ///
    /// `weights` is `None` for an unweighted solve. Supplied weights are validated
    /// in caller observation order, then retained internally only as `√w` in the
    /// design's internal observation order.
    ///
    /// LSMR tuning ([`LsmrOptions`]) is supplied per call to [`Solver::solve`] /
    /// [`Solver::solve_batch`], not at construction; preconditioner factorization
    /// state is the only expensive thing built here.
    pub fn new(
        design: impl IntoDesign<'a>,
        weights: Option<&[f64]>,
        preconditioner: impl Into<PreconditionerInput>,
    ) -> Result<Self, BuildError> {
        // Whiten the slope columns (if any) before the preconditioner reads them.
        let prepared = PreparedDesign::new(design.into_design()?, weights)?;
        let screened = detect_collinear_slopes(&prepared);
        let mut warnings: Vec<BuildWarning> = screened.iter().map(CollinearSlope::warn).collect();
        let n_dofs = prepared.design.n_dofs;

        let (mut preconditioner, build_warnings) = match preconditioner.into() {
            PreconditionerInput::Default => build_map(&prepared, PreconditionerConfig::default())?,
            PreconditionerInput::Config(c) => build_map(&prepared, c)?,
            PreconditionerInput::Prebuilt(p) => {
                if p.nrows() != n_dofs || p.ncols() != n_dofs {
                    return Err(BuildError::PreconditionerDimensionMismatch {
                        expected: n_dofs,
                        actual_rows: p.nrows(),
                        actual_cols: p.ncols(),
                    });
                }
                (Some(p), Vec::new())
            }
        };

        // Only `M⁻¹` can inject a null of `A`; unpreconditioned LSMR never leaves `range(Aᵀ)`.
        if let Some(p) = preconditioner.as_mut() {
            p.gauge = GaugeConstraint::build(&prepared, &screened).map(Arc::new);
        }
        warnings.extend(build_warnings);

        Ok(Self {
            prepared,
            preconditioner,
            warnings,
        })
    }

    /// Non-fatal events from design screening and the preconditioner build; a reused
    /// pre-built preconditioner contributes none (its own were reported when built).
    pub fn warnings(&self) -> &[BuildWarning] {
        &self.warnings
    }

    /// Validate `y` and move it into the internal observation frame.
    fn prepare<'s>(&'s self, y: &'s [f64]) -> Result<PreparedRhs<'s>, SolveError> {
        // `weighted_rhs` zips y with sqrt-weights, silently truncating when `y.len() > n_rows`.
        if y.len() != self.prepared.design.n_obs {
            return Err(SolveError::InvalidInput {
                context: "Solver::solve",
                message: format!(
                    "response vector length ({}) does not match number of observations ({})",
                    y.len(),
                    self.prepared.design.n_obs
                ),
            });
        }
        if let Some((index, &value)) = y.iter().enumerate().find(|&(_, &v)| !v.is_finite()) {
            return Err(SolveError::InvalidInput {
                context: "Solver::solve",
                message: format!("response at index {index} must be finite, got {value}"),
            });
        }

        let t_start = Instant::now();
        let y = self.prepared.design.permute_obs_in(y);
        let op = DesignOperator::new(&self.prepared);
        let weighted = match op.weighted_rhs(&y) {
            Cow::Borrowed(_) => None,
            Cow::Owned(b) => Some(b),
        };
        Ok(PreparedRhs {
            op,
            y,
            weighted,
            gather_secs: t_start.elapsed().as_secs_f64(),
        })
    }

    /// Demean and back-transform one finished run. Excludes the design-level `layout` /
    /// `warnings` / `unidentified`, which the public entry points attach once.
    fn finish(&self, rhs: PreparedRhs<'_>, run: Run) -> RhsSolution {
        let r = run.result;
        // Shapes are guaranteed here, so the bare `D x` matvec is infallible.
        let mut demeaned = vec![0.0; self.prepared.design.n_obs];
        gather_apply(&self.prepared, &r.x, &mut demeaned, None);
        for (d, &yi) in demeaned.iter_mut().zip(rhs.y.iter()) {
            *d = yi - *d;
        }

        let mut x = r.x;
        if let Some(rp) = &self.prepared.reparam {
            rp.back_transform(&mut x);
        }

        RhsSolution {
            x,
            // Back to the caller's observation order (no-op if not reordered).
            demeaned: self.prepared.design.permute_obs_out(demeaned),
            converged: r.converged,
            iterations: r.iterations,
            // Read from the LSMR recurrence at no extra cost; see `SolveResult::residual`.
            residual: r.normal_eq_residual,
            gather_secs: rhs.gather_secs,
            time_solve: run.solve_secs,
        }
    }

    /// Every RHS in one pass on the preconditioner; both entry points come through here.
    fn solve_all(&self, ys: &[&[f64]], lsmr: &LsmrOptions) -> Result<Vec<RhsSolution>, SolveError> {
        // Collecting into `Result` fails fast on the first per-RHS error, not during the fold.
        ys.par_iter()
            .map(|y| {
                let rhs = self.prepare(y)?;
                let options = MlsmrOptions {
                    local_size: lsmr.local_size,
                    ..Default::default()
                };
                let run = Run::timed(|| match self.preconditioner.as_ref() {
                    Some(m) => mlsmr(&rhs.op, rhs.b(), m, lsmr.tol, lsmr.maxiter, options),
                    None => lsmr_solve(&rhs.op, rhs.b(), lsmr.tol, lsmr.maxiter, lsmr.local_size),
                })?;
                Ok(self.finish(rhs, run))
            })
            .collect()
    }

    /// Per-level directions the data cannot identify, shared across all RHS:
    /// identification depends only on the design and weights, never on `y`.
    fn unidentified(&self) -> Vec<CoefficientAddress> {
        let Some(reparam) = &self.prepared.reparam else {
            return Vec::new();
        };
        reparam
            .unidentified
            .iter()
            .copied()
            .map(|position| position.to_caller_address(&self.prepared.design))
            .collect()
    }

    /// Solve for a single RHS vector with the given LSMR tuning.
    pub fn solve<'o>(
        &self,
        y: &[f64],
        lsmr: impl Into<Option<&'o LsmrOptions>>,
    ) -> Result<SolveResult, WithinError> {
        let default = LsmrOptions::default();
        let lsmr = lsmr.into().unwrap_or(&default);

        let t_start = Instant::now();
        let solution = self
            .solve_all(&[y], lsmr)?
            .into_iter()
            .next()
            .expect("one RHS yields one solution");

        Ok(SolveResult {
            x: solution.x,
            unidentified: self.unidentified(),
            warnings: self.warnings.clone(),
            layout: CoefficientLayout::from_design(&self.prepared.design),
            demeaned: solution.demeaned,
            converged: solution.converged,
            iterations: solution.iterations,
            residual: solution.residual,
            time_total: t_start.elapsed().as_secs_f64(),
            time_setup: solution.gather_secs,
            time_solve: solution.time_solve,
        })
    }

    /// Solve for multiple RHS vectors in parallel.
    pub fn solve_batch<'o>(
        &self,
        ys: &[&[f64]],
        lsmr: impl Into<Option<&'o LsmrOptions>>,
    ) -> Result<BatchSolveResult, WithinError> {
        let t_start = Instant::now();
        let default = LsmrOptions::default();
        let lsmr = lsmr.into().unwrap_or(&default);
        let n_rhs = ys.len();

        let solutions = self.solve_all(ys, lsmr)?;

        let mut x = Vec::with_capacity(self.prepared.design.n_dofs * n_rhs);
        let mut demeaned = Vec::with_capacity(self.prepared.design.n_obs * n_rhs);
        let mut converged = Vec::with_capacity(n_rhs);
        let mut iterations = Vec::with_capacity(n_rhs);
        let mut residual = Vec::with_capacity(n_rhs);
        let mut time_solve = Vec::with_capacity(n_rhs);

        for solution in solutions {
            x.extend_from_slice(&solution.x);
            demeaned.extend_from_slice(&solution.demeaned);
            converged.push(solution.converged);
            iterations.push(solution.iterations);
            residual.push(solution.residual);
            time_solve.push(solution.time_solve);
        }

        Ok(BatchSolveResult {
            x,
            unidentified: self.unidentified(),
            warnings: self.warnings.clone(),
            layout: CoefficientLayout::from_design(&self.prepared.design),
            demeaned,
            converged,
            iterations,
            residual,
            time_solve,
            time_setup: 0.0,
            time_total: t_start.elapsed().as_secs_f64(),
            n_dofs: self.prepared.design.n_dofs,
            n_obs: self.prepared.design.n_obs,
        })
    }

    /// Access the preconditioner (for serialization or reuse across solvers).
    pub fn preconditioner(&self) -> Option<&Preconditioner> {
        self.preconditioner.as_ref()
    }

    /// Number of DOFs (coefficients).
    pub fn n_dofs(&self) -> usize {
        self.prepared.design.n_dofs
    }

    /// Number of observations.
    pub fn n_obs(&self) -> usize {
        self.prepared.design.n_obs
    }
}

/// Solve fixed-effects least squares for a design input.
///
/// `design` is anything implementing [`IntoDesign`]: an observation-major
/// `(n_obs, n_factors)` categories array (arbitrary `u32` labels per factor,
/// compacted internally), a list of [`Effect`] terms, or an owned or borrowed
/// [`Design`].
/// `y` is the response vector (length = n_obs).
///
/// Zero-copy for F-order category arrays whose dominant factor is already
/// sorted; otherwise columns are copied once (per column at ingest, or
/// whole-frame by the locality sort).
///
/// `preconditioner` accepts the same input shapes as [`Solver::new`]:
/// `None`, a [`crate::PreconditionerConfig`] by reference or value, an owned
/// [`crate::Preconditioner`], or a `&Preconditioner` for amortized reuse.
///
/// This is a convenience wrapper around [`Solver::new`] + [`Solver::solve`].
pub fn solve<'a, 'o>(
    design: impl IntoDesign<'a>,
    y: &[f64],
    weights: Option<&[f64]>,
    lsmr: impl Into<Option<&'o LsmrOptions>>,
    preconditioner: impl Into<PreconditionerInput>,
) -> Result<SolveResult, WithinError> {
    let t_start = Instant::now();
    let solver = Solver::new(design, weights, preconditioner)?;
    let time_setup = t_start.elapsed().as_secs_f64();
    let mut result = solver.solve(y, lsmr)?;
    // Include solver construction (preconditioner build) in setup time
    result.time_setup += time_setup;
    result.time_total = t_start.elapsed().as_secs_f64();
    Ok(result)
}

/// Solve fixed-effects least squares for multiple response vectors.
///
/// Same as [`solve`] but solves all RHS vectors in parallel (via rayon),
/// reusing the preconditioner across all solves.
pub fn solve_batch<'a, 'o>(
    design: impl IntoDesign<'a>,
    ys: &[&[f64]],
    weights: Option<&[f64]>,
    lsmr: impl Into<Option<&'o LsmrOptions>>,
    preconditioner: impl Into<PreconditionerInput>,
) -> Result<BatchSolveResult, WithinError> {
    let t_start = Instant::now();
    let solver = Solver::new(design, weights, preconditioner)?;
    let time_setup = t_start.elapsed().as_secs_f64();
    let mut result = solver.solve_batch(ys, lsmr)?;
    result.time_setup += time_setup;
    result.time_total = t_start.elapsed().as_secs_f64();
    Ok(result)
}
