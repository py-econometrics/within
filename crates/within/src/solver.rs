//! The solve API: the persistent [`Solver`] (caches the preconditioner across
//! multiple solves on the same design) and the one-shot [`solve`] / [`solve_batch`]
//! convenience wrappers built on top of it.

use std::borrow::Cow;
use std::time::Instant;

use ndarray::{ArrayView2, Axis};
use rayon::prelude::*;
use schwarz_precond::{lsmr as lsmr_solve, mlsmr, MlsmrOptions};

use crate::channel::Channel;
use crate::config::{LsmrOptions, PreconditionerConfig};
use crate::domain::{Design, Effect};
use crate::observation::ObservationFrame;
use crate::operator::design::gather_apply;
use crate::operator::schwarz::{build_preconditioner, Preconditioner};
use crate::operator::DesignOperator;
use crate::{BuildError, BuildWarning, SolveError, WithinError};

mod reparam;
#[cfg(test)]
mod tests;
use reparam::SlopeReparam;

/// Fallible conversion into a [`Design`] for [`Solver::new`]: a categories
/// matrix (`ArrayView2<u32>`), a list of [`Effect`] terms, or a pass-through
/// [`Design`].
pub trait IntoDesign<'a> {
    /// Build the [`Design`], validating inputs along the way.
    fn into_design(self) -> Result<Design<'a>, BuildError>;
}

impl<'a> IntoDesign<'a> for ArrayView2<'a, u32> {
    fn into_design(self) -> Result<Design<'a>, BuildError> {
        // Gather strided (C-order) columns once so every downstream read is contiguous.
        let categorical = (0..self.ncols())
            .map(|factor| {
                let col = self.index_axis_move(Axis(1), factor);
                match col.to_slice() {
                    Some(s) => Cow::Borrowed(s),
                    None => Cow::Owned(col.to_vec()),
                }
            })
            .collect();
        Design::from_frame(ObservationFrame::new(categorical, Vec::new())?)
    }
}

impl<'a> IntoDesign<'a> for Design<'a> {
    fn into_design(self) -> Result<Design<'a>, BuildError> {
        Ok(self)
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

/// One coefficient of the design: a [`Channel`] at one level of its term.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoefficientAddress {
    /// The coefficient column this address sits in.
    pub channel: Channel,
    /// Level index within the term (`0..n_levels`).
    pub level: usize,
}

/// Translates a [`CoefficientAddress`] to its flat index in [`SolveResult::x`]
/// and back, so callers need not reconstruct the term-major offset formula
/// (`offset + column * n_levels + level`) by hand.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoefficientLayout {
    terms: Vec<TermLayout>,
    n_dofs: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TermLayout {
    offset: usize,
    n_levels: usize,
    n_columns: usize,
}

impl CoefficientLayout {
    pub(crate) fn from_design(design: &Design) -> Self {
        let terms = design
            .terms
            .iter()
            .map(|t| TermLayout {
                offset: t.offset,
                n_levels: t.n_levels,
                n_columns: t.n_columns(),
            })
            .collect();
        Self {
            terms,
            n_dofs: design.n_dofs,
        }
    }

    /// Total number of coefficients (the length of [`SolveResult::x`]).
    pub fn n_dofs(&self) -> usize {
        self.n_dofs
    }

    /// Number of terms in the design.
    pub fn n_terms(&self) -> usize {
        self.terms.len()
    }

    /// Level count of `term`, or `None` if `term` is out of range.
    pub fn n_levels(&self, term: usize) -> Option<usize> {
        self.terms.get(term).map(|t| t.n_levels)
    }

    /// Coefficient-column count of `term` (`intercept? + slopes`, ordered
    /// `[intercept?, slopes…]`), or `None` if `term` is out of range.
    pub fn n_columns(&self, term: usize) -> Option<usize> {
        self.terms.get(term).map(|t| t.n_columns)
    }

    /// Flat [`SolveResult::x`] index of `at`, or `None` if any coordinate is
    /// out of range.
    pub fn index(&self, at: CoefficientAddress) -> Option<usize> {
        let t = self.terms.get(at.channel.term)?;
        (at.level < t.n_levels && at.channel.column < t.n_columns)
            .then(|| t.offset + at.channel.column * t.n_levels + at.level)
    }

    /// The address of flat index `i`, or `None` if `i >= n_dofs`.
    pub fn address(&self, i: usize) -> Option<CoefficientAddress> {
        if i >= self.n_dofs {
            return None;
        }
        // Term blocks ascend by offset, so the owner is the last one not exceeding `i`.
        let term = self.terms.partition_point(|t| t.offset <= i) - 1;
        let t = &self.terms[term];
        let within = i - t.offset;
        Some(CoefficientAddress {
            channel: Channel {
                term,
                column: within / t.n_levels,
            },
            level: within % t.n_levels,
        })
    }
}

/// Common solve output for all orchestration entry points.
#[derive(Debug, Clone)]
#[must_use]
pub struct SolveResult {
    /// Fixed-effect coefficients (length = total DOFs across all factors).
    ///
    /// Term-major: coefficient column `c` of level `level` sits at
    /// `term_offset + c * n_levels + level`, columns ordered
    /// `[intercept?, slopes…]`. Slots for unidentified directions hold the
    /// minimal-norm value `0`, never NaN; see [`SolveResult::unidentified`].
    pub x: Vec<f64>,
    /// Per-level directions the data cannot identify.
    pub unidentified: Vec<CoefficientAddress>,
    /// Non-fatal preconditioner-build warnings (see [`Solver::warnings`]);
    /// empty when a pre-built preconditioner was reused.
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
    /// Non-fatal preconditioner-build warnings, shared across all RHS; see
    /// [`SolveResult::warnings`].
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
/// Python boundary — uses owned columns. Weights are always owned; for a
/// one-shot weighted solve from a borrowed slice, use the free [`solve`] function.
pub struct Solver<'a> {
    design: Design<'a>,
    /// `sqrt(W)` in the design's internal observation order, computed once and
    /// borrowed by the per-RHS [`DesignOperator`]s (raw weights are needed only
    /// during construction).
    sqrt_weights: Option<Vec<f64>>,
    preconditioner: Option<Preconditioner>,
    reparam: Option<SlopeReparam>,
    warnings: Vec<BuildWarning>,
}

impl std::fmt::Debug for Solver<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Solver")
            .field("n_obs", &self.design.n_obs)
            .field("n_dofs", &self.design.n_dofs)
            .field("has_weights", &self.sqrt_weights.is_some())
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
    time_setup: f64,
    time_solve: f64,
}

impl<'a> Solver<'a> {
    /// Construct a solver.
    ///
    /// `design` accepts raw categories (`ArrayView2<u32>`) or a pre-built
    /// [`Design`]. `preconditioner` accepts:
    /// - `None` — build the library default Schwarz preconditioner
    /// - `&PreconditionerConfig` / `Some(&PreconditionerConfig)` — build from a tuned config
    /// - `PreconditionerConfig::Off` — solve unpreconditioned
    /// - `PreconditionerConfig::Diagonal` — use diagonal/Jacobi preconditioning
    /// - [`Preconditioner`] or `&Preconditioner` — reuse a previously built one
    ///
    /// `weights` is `None` for unweighted, or an owned `Vec<f64>` that the
    /// solver takes ownership of (it re-reads the weights on every solve). To
    /// solve once from a borrowed slice, use the free [`solve`] function.
    ///
    /// LSMR tuning ([`LsmrOptions`]) is supplied per call to [`Solver::solve`] /
    /// [`Solver::solve_batch`], not at construction; preconditioner factorization
    /// state is the only expensive thing built here.
    pub fn new(
        design: impl IntoDesign<'a>,
        weights: Option<Vec<f64>>,
        preconditioner: impl Into<PreconditionerInput>,
    ) -> Result<Self, BuildError> {
        let mut design = design.into_design()?;
        design.validate_weights(weights.as_deref())?;

        // The match keeps the unpermuted arm a plain move rather than a borrow-and-copy.
        let weights = match &design.obs_perm {
            Some(_) => weights.map(|w| design.permute_obs_in(&w).into_owned()),
            None => weights,
        };

        // Reparametrize the slope columns (if any) before the preconditioner reads the frame.
        let reparam = SlopeReparam::build(&mut design, weights.as_deref());

        let (preconditioner, warnings) = match preconditioner.into() {
            PreconditionerInput::Default => {
                build_preconditioner(&design, weights.as_deref(), None)?
            }
            PreconditionerInput::Config(c) => {
                build_preconditioner(&design, weights.as_deref(), Some(&c))?
            }
            PreconditionerInput::Prebuilt(p) => {
                if p.nrows() != design.n_dofs || p.ncols() != design.n_dofs {
                    return Err(BuildError::PreconditionerDimensionMismatch {
                        expected: design.n_dofs,
                        actual_rows: p.nrows(),
                        actual_cols: p.ncols(),
                    });
                }
                (Some(p), Vec::new())
            }
        };

        let sqrt_weights = weights.map(|mut w| {
            for wi in &mut w {
                *wi = wi.sqrt();
            }
            w
        });

        Ok(Self {
            design,
            sqrt_weights,
            preconditioner,
            reparam,
            warnings,
        })
    }

    /// Non-fatal events from the preconditioner build; empty when reusing a
    /// pre-built preconditioner (its warnings were reported when it was built).
    pub fn warnings(&self) -> &[BuildWarning] {
        &self.warnings
    }

    /// Shared per-RHS solve: validate `y`, run (m)lsmr, demean, and
    /// back-transform slopes. Excludes the design-level `layout` / `warnings` /
    /// `unidentified`, which the public entry points attach once (see
    /// [`RhsSolution`]).
    fn solve_rhs(&self, y: &[f64], lsmr: &LsmrOptions) -> Result<RhsSolution, SolveError> {
        // `weighted_rhs` zips y with sqrt-weights, silently truncating when `y.len() > n_rows`.
        if y.len() != self.design.n_obs {
            return Err(SolveError::InvalidInput {
                context: "Solver::solve",
                message: format!(
                    "response vector length ({}) does not match number of observations ({})",
                    y.len(),
                    self.design.n_obs
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

        // The gather is a recurring per-solve cost of the locality sort, so it counts as setup.
        let y_internal = self.design.permute_obs_in(y);
        let y: &[f64] = &y_internal;

        let rect_op = DesignOperator::new(&self.design, self.sqrt_weights.as_deref());
        let b = rect_op.weighted_rhs(y);
        let b: &[f64] = &b;

        let t_solve_start = Instant::now();
        let time_setup = t_solve_start.duration_since(t_start).as_secs_f64();

        let r = match self.preconditioner.as_ref() {
            Some(p) => {
                let options = MlsmrOptions {
                    local_size: lsmr.local_size,
                    ..Default::default()
                };
                mlsmr(&rect_op, b, p, lsmr.tol, lsmr.maxiter, options)?
            }
            None => lsmr_solve(&rect_op, b, lsmr.tol, lsmr.maxiter, lsmr.local_size)?,
        };

        let time_solve = t_solve_start.elapsed().as_secs_f64();

        // Shapes are guaranteed here, so the bare `D x` matvec is infallible.
        let mut demeaned = vec![0.0; self.design.n_obs];
        gather_apply(&self.design, &r.x, &mut demeaned, None);
        for (d, &yi) in demeaned.iter_mut().zip(y.iter()) {
            *d = yi - *d;
        }

        let mut x = r.x;
        if let Some(rp) = &self.reparam {
            rp.back_transform(&mut x);
        }

        Ok(RhsSolution {
            x,
            // Back to the caller's observation order (no-op if not reordered).
            demeaned: self.design.permute_obs_out(demeaned),
            converged: r.converged,
            iterations: r.iterations,
            // Read from the LSMR recurrence at no extra cost; see `SolveResult::residual`.
            residual: r.normal_eq_residual,
            time_setup,
            time_solve,
        })
    }

    /// Per-level directions the data cannot identify, shared across all RHS:
    /// identification depends only on the design and weights, never on `y`.
    fn unidentified(&self) -> Vec<CoefficientAddress> {
        self.reparam
            .as_ref()
            .map(|rp| rp.unidentified.clone())
            .unwrap_or_default()
    }

    /// Solve for a single RHS vector with the given LSMR tuning.
    pub fn solve<'o>(
        &self,
        y: &[f64],
        lsmr: impl Into<Option<&'o LsmrOptions>>,
    ) -> Result<SolveResult, SolveError> {
        let default = LsmrOptions::default();
        let lsmr = lsmr.into().unwrap_or(&default);

        let t_start = Instant::now();
        let solution = self.solve_rhs(y, lsmr)?;

        Ok(SolveResult {
            x: solution.x,
            unidentified: self.unidentified(),
            warnings: self.warnings.clone(),
            layout: CoefficientLayout::from_design(&self.design),
            demeaned: solution.demeaned,
            converged: solution.converged,
            iterations: solution.iterations,
            residual: solution.residual,
            time_total: t_start.elapsed().as_secs_f64(),
            time_setup: solution.time_setup,
            time_solve: solution.time_solve,
        })
    }

    /// Solve for multiple RHS vectors in parallel.
    pub fn solve_batch<'o>(
        &self,
        ys: &[&[f64]],
        lsmr: impl Into<Option<&'o LsmrOptions>>,
    ) -> Result<BatchSolveResult, SolveError> {
        let t_start = Instant::now();
        let default = LsmrOptions::default();
        let lsmr = lsmr.into().unwrap_or(&default);
        let n_rhs = ys.len();

        // Collecting into `Result` fails fast on the first per-RHS error, not during the fold.
        let solutions: Vec<RhsSolution> = ys
            .par_iter()
            .map(|y| self.solve_rhs(y, lsmr))
            .collect::<Result<Vec<_>, _>>()?;

        let mut x = Vec::with_capacity(self.design.n_dofs * n_rhs);
        let mut demeaned = Vec::with_capacity(self.design.n_obs * n_rhs);
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
            layout: CoefficientLayout::from_design(&self.design),
            demeaned,
            converged,
            iterations,
            residual,
            time_solve,
            time_setup: 0.0,
            time_total: t_start.elapsed().as_secs_f64(),
            n_dofs: self.design.n_dofs,
            n_obs: self.design.n_obs,
        })
    }

    /// Access the preconditioner (for serialization or reuse across solvers).
    pub fn preconditioner(&self) -> Option<&Preconditioner> {
        self.preconditioner.as_ref()
    }

    /// Number of DOFs (coefficients).
    pub fn n_dofs(&self) -> usize {
        self.design.n_dofs
    }

    /// Number of observations.
    pub fn n_obs(&self) -> usize {
        self.design.n_obs
    }
}

/// Solve fixed-effects least squares for a design input.
///
/// `design` is anything implementing [`IntoDesign`]: an observation-major
/// `(n_obs, n_factors)` categories array (levels `0..max_level` per factor,
/// count inferred) or a list of [`Effect`] terms.
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
    let solver = Solver::new(design, weights.map(|w| w.to_vec()), preconditioner)?;
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
    let solver = Solver::new(design, weights.map(|w| w.to_vec()), preconditioner)?;
    let time_setup = t_start.elapsed().as_secs_f64();
    let mut result = solver.solve_batch(ys, lsmr)?;
    result.time_setup += time_setup;
    result.time_total = t_start.elapsed().as_secs_f64();
    Ok(result)
}
