//! The solve API: the persistent [`Solver`] (caches the preconditioner across
//! multiple solves on the same design) and the one-shot [`solve`] / [`solve_batch`]
//! convenience wrappers built on top of it.

use std::borrow::Cow;
use std::time::Instant;

use ndarray::{ArrayView2, Axis};
use rayon::prelude::*;
use schwarz_precond::{lsmr as lsmr_solve, mlsmr, Operator as _};

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

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Fallible conversion into a [`Design`] for [`Solver::new`]: a categories
/// matrix (`ArrayView2<u32>`), a list of [`Effect`] terms, or a pass-through
/// [`Design`].
pub trait IntoDesign<'a> {
    /// Build the [`Design`], validating inputs along the way.
    fn into_design(self) -> Result<Design<'a>, BuildError>;
}

impl<'a> IntoDesign<'a> for ArrayView2<'a, u32> {
    fn into_design(self) -> Result<Design<'a>, BuildError> {
        // Borrow F-contiguous columns zero-copy; gather strided (C-order)
        // columns once here so every downstream read is a contiguous slice.
        let categorical = (0..self.ncols())
            .map(|q| {
                let col = self.index_axis_move(Axis(1), q);
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

/// A per-level direction of the design that the data cannot identify.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnidentifiedDirection {
    /// Index into the design's term list.
    pub term: usize,
    /// Level index within the term (`0..n_levels`).
    pub level: usize,
    /// Column within the term's per-level block: intercept first (when
    /// present), then slopes in declaration order.
    pub column: usize,
}

/// Translates a `(term, level, column)` coefficient address to its flat index
/// in [`SolveResult::x`] and back, so callers need not reconstruct the
/// term-major offset formula (`offset + column * n_levels + level`) by hand.
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

    /// Flat [`SolveResult::x`] index of coefficient `column` of `level` within
    /// `term`, or `None` if any coordinate is out of range.
    pub fn index(&self, term: usize, level: usize, column: usize) -> Option<usize> {
        let t = self.terms.get(term)?;
        (level < t.n_levels && column < t.n_columns).then(|| t.offset + column * t.n_levels + level)
    }

    /// The `(term, level, column)` address of flat index `i`, or `None` if
    /// `i >= n_dofs`.
    pub fn address(&self, i: usize) -> Option<(usize, usize, usize)> {
        if i >= self.n_dofs {
            return None;
        }
        // Term blocks are contiguous in ascending offset order, so the owning
        // term is the last one whose offset does not exceed `i`.
        let term = self.terms.partition_point(|t| t.offset <= i) - 1;
        let t = &self.terms[term];
        let within = i - t.offset;
        Some((term, within % t.n_levels, within / t.n_levels))
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
    pub unidentified: Vec<UnidentifiedDirection>,
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
    /// Final relative residual norm `‖r‖ / ‖b‖`.
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
    pub unidentified: Vec<UnidentifiedDirection>,
    /// Address ↔ flat-`x`-index translation for this design's coefficients.
    pub layout: CoefficientLayout,
    /// All demeaned responses concatenated (length = n_obs * n_rhs).
    pub demeaned: Vec<f64>,
    /// Per-RHS convergence flags.
    pub converged: Vec<bool>,
    /// Per-RHS iteration counts.
    pub iterations: Vec<usize>,
    /// Per-RHS final relative residual norms.
    pub residual: Vec<f64>,
    /// Per-RHS solve times in seconds.
    pub time_solve: Vec<f64>,
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

// ---------------------------------------------------------------------------
// Solver
// ---------------------------------------------------------------------------

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
    weights: Option<Vec<f64>>,
    preconditioner: Option<Preconditioner>,
    reparam: Option<SlopeReparam>,
    warnings: Vec<BuildWarning>,
}

impl std::fmt::Debug for Solver<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Solver")
            .field("n_obs", &self.design.n_obs)
            .field("n_dofs", &self.design.n_dofs)
            .field("has_weights", &self.weights.is_some())
            .field("has_preconditioner", &self.preconditioner.is_some())
            .finish()
    }
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
    /// - [`Preconditioner`] or `&Preconditioner` — reuse a previously built (or deserialized) preconditioner
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

        // Align weights with the design's internal (possibly locality-sorted)
        // observation order. The match keeps the unpermuted arm a plain move
        // (`permute_obs_in` would borrow and `into_owned` would copy).
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

        Ok(Self {
            design,
            weights,
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

    /// Solve for a single RHS vector with the given LSMR tuning.
    pub fn solve<'o>(
        &self,
        y: &[f64],
        lsmr: impl Into<Option<&'o LsmrOptions>>,
    ) -> Result<SolveResult, SolveError> {
        let default = LsmrOptions::default();
        let lsmr = lsmr.into().unwrap_or(&default);
        // Guard the silent-truncation hole: weighted_rhs zips y with sqrt-weights,
        // which would otherwise discard trailing values when y.len() > n_rows.
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

        // All matvecs run in the design's internal (possibly locality-sorted)
        // observation order; `demeaned` is translated back on return. The
        // gather is a recurring per-solve cost of the locality sort, so it
        // counts toward `time_setup` (and `time_total`).
        let y_internal = self.design.permute_obs_in(y);
        let y: &[f64] = &y_internal;

        let rect_op = DesignOperator::new(&self.design, self.weights.as_deref());
        let b = rect_op.weighted_rhs(y);
        let b: &[f64] = &b;

        let t_solve_start = Instant::now();
        let time_setup = t_solve_start.duration_since(t_start).as_secs_f64();

        let r = match self.preconditioner.as_ref() {
            Some(p) => mlsmr(&rect_op, b, p, lsmr.tol, lsmr.maxiter, lsmr.local_size)?,
            None => lsmr_solve(&rect_op, b, lsmr.tol, lsmr.maxiter, lsmr.local_size)?,
        };

        let time_solve = t_solve_start.elapsed().as_secs_f64();

        // demeaned = y - D x. The bare unweighted `D x` matvec is `gather_apply`
        // without a scale; shapes are guaranteed here, so it is infallible —
        // no DesignOperator wrapper (and its scatter scratch) needed.
        let mut demeaned = vec![0.0; self.design.n_obs];
        gather_apply(&self.design, &r.x, &mut demeaned, None);
        for (d, &yi) in demeaned.iter_mut().zip(y.iter()) {
            *d = yi - *d;
        }

        // Relative normal-equation residual: ||D^T W (y - Dx)|| / ||D^T W y||.
        // Compute D^T W v as rect_op.apply_adjoint(W^{1/2} v): apply_adjoint
        // delivers D^T W^{1/2} (·), so feeding W^{1/2} v gives D^T W v.
        let mut rhs = vec![0.0; self.design.n_dofs];
        rect_op.apply_adjoint(b, &mut rhs)?;
        let rhs_norm = norm(&rhs).max(1e-15);
        let weighted_demeaned = rect_op.weighted_rhs(&demeaned);
        let mut residual_dof = vec![0.0; self.design.n_dofs];
        rect_op.apply_adjoint(weighted_demeaned.as_ref(), &mut residual_dof)?;
        let residual = norm(&residual_dof) / rhs_norm;

        let mut x = r.x;
        let unidentified = match &self.reparam {
            Some(rp) => {
                rp.back_transform(&mut x);
                rp.unidentified.clone()
            }
            None => Vec::new(),
        };

        Ok(SolveResult {
            x,
            unidentified,
            layout: CoefficientLayout::from_design(&self.design),
            // Back to the caller's observation order (no-op if not reordered).
            demeaned: self.design.permute_obs_out(demeaned),
            converged: r.converged,
            iterations: r.iterations,
            residual,
            time_total: t_start.elapsed().as_secs_f64(),
            time_setup,
            time_solve,
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

        // Fail fast on the first per-RHS error rather than materializing a
        // `Vec<Result<..>>` and only surfacing the failure during the fold.
        let results: Vec<SolveResult> = ys
            .par_iter()
            .map(|y| self.solve(y, lsmr))
            .collect::<Result<Vec<_>, _>>()?;

        let mut x = Vec::with_capacity(self.design.n_dofs * n_rhs);
        let mut demeaned = Vec::with_capacity(self.design.n_obs * n_rhs);
        let mut converged = Vec::with_capacity(n_rhs);
        let mut iterations = Vec::with_capacity(n_rhs);
        let mut residual = Vec::with_capacity(n_rhs);
        let mut time_solve = Vec::with_capacity(n_rhs);

        // Identical for every RHS: identification depends only on the design
        // and weights, never on `y`.
        let unidentified = results
            .first()
            .map(|r| r.unidentified.clone())
            .unwrap_or_default();

        for r in results {
            x.extend_from_slice(&r.x);
            demeaned.extend_from_slice(&r.demeaned);
            converged.push(r.converged);
            iterations.push(r.iterations);
            residual.push(r.residual);
            time_solve.push(r.time_solve);
        }

        Ok(BatchSolveResult {
            x,
            unidentified,
            layout: CoefficientLayout::from_design(&self.design),
            demeaned,
            converged,
            iterations,
            residual,
            time_solve,
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

// ===========================================================================
// High-level one-shot API
// ===========================================================================

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
    let mut result = solver.solve_batch(ys, lsmr)?;
    result.time_total = t_start.elapsed().as_secs_f64();
    Ok(result)
}
