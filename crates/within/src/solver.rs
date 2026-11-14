//! The solve API: the persistent [`Solver`] (caches the preconditioner across
//! multiple solves on the same design) and the one-shot [`solve`] / [`solve_batch`]
//! convenience wrappers built on top of it.

use std::time::Instant;

use ndarray::ArrayView2;
use rayon::prelude::*;
use schwarz_precond::{
    mlsmr, EscalationPolicy, IdentityOp, LsmrResult, LsmrStopReason, MlsmrOptions, Operator,
};

use crate::channel::CoefficientAddress;
use crate::config::{LsmrOptions, PreconditionerConfig};
use crate::domain::{Design, Effect, PreparedDesign};
use crate::operator::design::ReducedDesign;
use crate::operator::schwarz::Preconditioner;
use crate::{BuildError, BuildWarning, SolveError, WithinError};

mod ladder;
mod layout;
mod system;
#[cfg(test)]
mod tests;

use ladder::PrecondSlot;
pub use layout::CoefficientLayout;
use system::{Fit, LeastSquaresSystem};

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
/// - bare `None` — build the library default preconditioner
/// - `&PreconditionerConfig` or `Some(&PreconditionerConfig)` — build from a tuned config
/// - `PreconditionerConfig` (owned) — same as above
/// - [`Preconditioner`] (owned or `&`) — reuse a previously built (or deserialized) preconditioner
///
/// `None` resolves unambiguously because there is exactly one `From<Option<X>>`
/// impl (with `X = &PreconditionerConfig`).
pub enum PreconditionerInput {
    /// Library default: diagonal, escalating to additive Schwarz on a stalled solve.
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
    /// Number of LSMR iterations used (summed across both rungs when escalated).
    pub iterations: usize,
    /// Relative normal-equation residual `‖Aᵀ(b − Ax)‖ / ‖Aᵀb‖` of the iterated system,
    /// estimated from the LSMR recurrence (Fong & Saunders) at no extra cost.
    /// Exact for an unpreconditioned solve; for a preconditioned solve it is
    /// measured in the preconditioner's metric and typically sits a modest
    /// factor below the true-metric value.
    pub residual: f64,
    /// Wall-clock time for the entire solve (setup + LSMR), in seconds.
    pub time_total: f64,
    /// Setup seconds in this call: RHS gather, deferred Adaptive build, `Solver::new` via [`solve`].
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
    /// Setup seconds in this call: deferred Adaptive build, `Solver::new` via [`solve_batch`]; else 0.
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
/// The least-squares system LSMR iterates on.
enum System<'a> {
    /// Every coefficient.
    Full(PreparedDesign<'a>),
    /// The term with the most coefficients is eliminated exactly; the rest iterate.
    Reduced(ReducedDesign<'a>),
}

impl<'a> System<'a> {
    fn prepared(&self) -> &PreparedDesign<'a> {
        match self {
            Self::Full(prepared) => prepared,
            Self::Reduced(reduced) => reduced.prepared(),
        }
    }
}

/// Persistent solver that owns its preconditioner for reuse across multiple solves.
///
/// Build once with [`Solver::new`], then call [`Solver::solve`] or
/// [`Solver::solve_batch`] repeatedly with different RHS vectors. The expensive
/// preconditioner factorization happens at construction time, except under
/// [`PreconditionerConfig::Adaptive`], which defers it to the first stalled solve;
/// LSMR tuning ([`LsmrOptions`]) is supplied per call.
///
/// Ownership: each observation column is borrowed or owned independently
/// (`Cow`); a solver that outlives its inputs — e.g. one returned across the
/// Python boundary — uses owned columns.
pub struct Solver<'a> {
    system: System<'a>,
    slot: PrecondSlot,
}

impl std::fmt::Debug for Solver<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let prepared = self.system.prepared();
        let system = match self.system {
            System::Full(_) => "Full",
            System::Reduced(_) => "Reduced",
        };
        f.debug_struct("Solver")
            .field("n_obs", &prepared.design.n_obs)
            .field("n_dofs", &prepared.design.n_dofs)
            .field("has_weights", &prepared.sqrt_weights().is_some())
            .field("system", &format_args!("{system}"))
            .field("has_preconditioner", &self.preconditioner().is_some())
            .field("has_escalated", &self.has_escalated())
            .finish()
    }
}

/// Seconds one RHS spent being prepared and in LSMR, across every rung it ran on.
#[derive(Clone, Copy, Default)]
struct Seconds {
    /// Validation, gather and weighting recur per solve, so they count as setup.
    prepare: f64,
    lsmr: f64,
}

/// Per-RHS output shared by [`Solver::solve`] and [`Solver::solve_batch`].
///
/// The shared fields (`layout`, `warnings`, `unidentified`) are identical
/// across RHS, so the batch path attaches them once instead of cloning them per
/// RHS as it would if each worker returned a full [`SolveResult`].
struct RhsSolution {
    fit: Fit,
    converged: bool,
    iterations: usize,
    residual: f64,
    seconds: Seconds,
}

/// One RHS between rungs.
enum Pass<'s, S: LeastSquaresSystem + 's> {
    /// No rung has run it yet; the first that does prepares it, inside its fan-out.
    Unstarted(&'s [f64]),
    /// Handed off with budget left; the next rung resumes from `probe.x`.
    Stalled {
        rhs: S::Rhs<'s>,
        probe: LsmrResult,
        seconds: Seconds,
    },
    Done(RhsSolution),
}

/// Runs every pending RHS on one map; each finishes, and is released, here or stalls.
fn rung<'s, S: LeastSquaresSystem>(
    system: &'s S,
    passes: Vec<Pass<'s, S>>,
    map: Option<&Preconditioner>,
    stall: Option<&dyn EscalationPolicy>,
    lsmr: &LsmrOptions,
) -> Result<Vec<Pass<'s, S>>, SolveError> {
    // Collecting into `Result` fails fast on the first per-RHS error, not during the fold.
    passes
        .into_par_iter()
        .map(|pass| {
            let (rhs, probe, mut seconds) = match pass {
                Pass::Done(solution) => return Ok(Pass::Done(solution)),
                Pass::Unstarted(y) => {
                    let started = Instant::now();
                    let rhs = system.prepare(y)?;
                    let prepare = started.elapsed().as_secs_f64();
                    (rhs, None, Seconds { prepare, lsmr: 0.0 })
                }
                Pass::Stalled {
                    rhs,
                    probe,
                    seconds,
                } => (rhs, Some(probe), seconds),
            };
            // The whole ladder shares the caller's budget; a resumed rung gets what the last left.
            let spent = probe.as_ref().map_or(0, |p| p.iterations);
            let options = MlsmrOptions {
                warm_start: probe.as_ref().map(|p| p.x.as_slice()),
                escalation: stall,
                local_size: lsmr.local_size,
            };
            let identity;
            let m: &dyn Operator = match map {
                Some(m) => m,
                None => {
                    identity = IdentityOp {
                        n: system.matrix().ncols(),
                    };
                    &identity
                }
            };
            let started = Instant::now();
            let maxiter = lsmr.maxiter - spent;
            let mut result = mlsmr(
                system.matrix(),
                system.b(&rhs),
                m,
                lsmr.tol,
                maxiter,
                options,
            )?;
            seconds.lsmr += started.elapsed().as_secs_f64();
            // The next rung starts from `result.x`; the probe it replaces is released here.
            drop(probe);
            result.iterations += spent;
            // A stall on the last permitted iteration leaves the next rung nothing to spend.
            if result.stop_reason == LsmrStopReason::Escalated && result.iterations < lsmr.maxiter {
                return Ok(Pass::Stalled {
                    rhs,
                    probe: result,
                    seconds,
                });
            }
            let LsmrResult {
                x,
                true_residual,
                converged,
                iterations,
                normal_eq_residual,
                ..
            } = result;
            Ok(Pass::Done(RhsSolution {
                fit: system.finish(rhs, x, true_residual)?,
                converged,
                iterations,
                // Read from the LSMR recurrence at no extra cost; see `SolveResult::residual`.
                residual: normal_eq_residual,
                seconds,
            }))
        })
        .collect()
}

/// Every RHS in one pass; an unsettled ladder then builds Schwarz once and resumes the stalled.
fn solve_all<S: LeastSquaresSystem>(
    system: &S,
    slot: &PrecondSlot,
    ys: &[&[f64]],
    lsmr: &LsmrOptions,
) -> Result<(Vec<RhsSolution>, f64), WithinError> {
    let (map, ladder) = match slot {
        PrecondSlot::Static { map, .. } => (map.as_ref(), None),
        // A settled ladder never re-probes: its map, or its kept build error, is final.
        PrecondSlot::Adaptive(a) => match a.built.get() {
            Some(Ok(_)) => (Some(a.rung()), None),
            Some(Err(e)) => return Err(e.clone().into()),
            None => (Some(&a.base), Some(a.as_ref())),
        },
    };
    let stall = ladder.map(|a| &a.ladder().stall as &dyn EscalationPolicy);
    let passes = ys.iter().map(|&y| Pass::Unstarted(y)).collect();
    let mut passes = rung(system, passes, map, stall, lsmr)?;

    let mut build_secs = 0.0;
    if let Some(a) = ladder {
        if passes.iter().any(|p| matches!(p, Pass::Stalled { .. })) {
            // Built outside the fan-out, so sibling RHS never race for it.
            build_secs = a.escalate(system)?;
            passes = rung(system, passes, Some(a.rung()), None, lsmr)?;
        }
    }
    let solutions = passes
        .into_iter()
        .map(|pass| match pass {
            Pass::Done(solution) => solution,
            _ => unreachable!("a rung without a stall policy finishes every RHS"),
        })
        .collect();
    Ok((solutions, build_secs))
}

/// The preconditioner slot `input` resolves to on `system`.
fn resolve<S: LeastSquaresSystem>(
    system: &S,
    input: PreconditionerInput,
) -> Result<PrecondSlot, BuildError> {
    match input {
        PreconditionerInput::Default => PrecondSlot::build(system, PreconditionerConfig::default()),
        PreconditionerInput::Config(c) => PrecondSlot::build(system, c),
        PreconditionerInput::Prebuilt(p) => {
            let n = system.matrix().ncols();
            if p.nrows() != n || p.ncols() != n {
                return Err(BuildError::PreconditionerDimensionMismatch {
                    expected: n,
                    actual_rows: p.nrows(),
                    actual_cols: p.ncols(),
                });
            }
            PrecondSlot::reuse(system, p)
        }
    }
}

impl<'a> Solver<'a> {
    /// Construct a solver.
    ///
    /// `design` accepts raw categories (`ArrayView2<u32>`), a pre-built
    /// [`Design`], or `&Design` to share one design across solvers (an O(1)
    /// clone of its storage). `preconditioner` accepts:
    /// - `None` — build the library default preconditioner
    /// - `&PreconditionerConfig` / `Some(&PreconditionerConfig)` — build from a
    ///   [`PreconditionerConfig`] variant
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
        let slot = resolve(&prepared, preconditioner.into())?;
        Ok(Self {
            system: System::Full(prepared),
            slot,
        })
    }

    /// [`Solver::new`] on the reduced system: the term with the most coefficients is eliminated
    /// exactly, LSMR iterates on the remaining terms, and the eliminated coefficients are
    /// recovered from the rest. Every [`PreconditionerConfig`] means the same on both systems.
    pub fn reduced(
        design: impl IntoDesign<'a>,
        weights: Option<&[f64]>,
        preconditioner: impl Into<PreconditionerInput>,
    ) -> Result<Self, BuildError> {
        let reduced = ReducedDesign::new(design.into_design()?, weights)?;
        let slot = resolve(&reduced, preconditioner.into())?;
        Ok(Self {
            system: System::Reduced(reduced),
            slot,
        })
    }

    /// Every RHS's solution, and the seconds a deferred Schwarz build took.
    fn solutions(
        &self,
        ys: &[&[f64]],
        lsmr: &LsmrOptions,
    ) -> Result<(Vec<RhsSolution>, f64), WithinError> {
        match &self.system {
            System::Full(prepared) => solve_all(prepared, &self.slot, ys, lsmr),
            System::Reduced(reduced) => solve_all(reduced, &self.slot, ys, lsmr),
        }
    }

    /// Non-fatal events from the preconditioner build; a reused pre-built preconditioner
    /// contributes none (its own were reported when built). An Adaptive solver reports its
    /// deferred build's once a solve escalates.
    pub fn warnings(&self) -> &[BuildWarning] {
        match &self.slot {
            PrecondSlot::Static { warnings, .. } => warnings,
            PrecondSlot::Adaptive(a) => a.build().map_or(&[], |b| &b.warnings),
        }
    }

    /// Whether an [`Adaptive`](crate::PreconditionerConfig::Adaptive) solve has built Schwarz.
    pub fn has_escalated(&self) -> bool {
        match &self.slot {
            PrecondSlot::Adaptive(a) => a.schwarz().is_some(),
            PrecondSlot::Static { .. } => false,
        }
    }

    /// Per-level directions the data cannot identify, shared across all RHS:
    /// identification depends only on the design and weights, never on `y`.
    fn unidentified(&self) -> Vec<CoefficientAddress> {
        let prepared = self.system.prepared();
        prepared
            .unidentified()
            .map(|position| position.to_caller_address(&prepared.design))
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
        let (solutions, build_secs) = self.solutions(&[y], lsmr)?;
        let solution = solutions
            .into_iter()
            .next()
            .expect("one RHS yields one solution");

        Ok(SolveResult {
            x: solution.fit.x,
            unidentified: self.unidentified(),
            // Read after solving so a deferred Adaptive build's warnings are included.
            warnings: self.warnings().to_vec(),
            layout: CoefficientLayout::from_design(&self.system.prepared().design),
            demeaned: solution.fit.demeaned,
            converged: solution.converged,
            iterations: solution.iterations,
            residual: solution.residual,
            time_total: t_start.elapsed().as_secs_f64(),
            // A deferred Schwarz build happens between two runs: it is setup, not solve.
            time_setup: solution.seconds.prepare + build_secs,
            time_solve: solution.seconds.lsmr,
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

        let (solutions, build_secs) = self.solutions(ys, lsmr)?;

        let mut x = Vec::with_capacity(self.system.prepared().design.n_dofs * n_rhs);
        let mut demeaned = Vec::with_capacity(self.system.prepared().design.n_obs * n_rhs);
        let mut converged = Vec::with_capacity(n_rhs);
        let mut iterations = Vec::with_capacity(n_rhs);
        let mut residual = Vec::with_capacity(n_rhs);
        let mut time_solve = Vec::with_capacity(n_rhs);

        for solution in solutions {
            x.extend_from_slice(&solution.fit.x);
            demeaned.extend_from_slice(&solution.fit.demeaned);
            converged.push(solution.converged);
            iterations.push(solution.iterations);
            residual.push(solution.residual);
            time_solve.push(solution.seconds.lsmr);
        }

        Ok(BatchSolveResult {
            x,
            unidentified: self.unidentified(),
            // Read after solving so a deferred Adaptive build's warnings are included.
            warnings: self.warnings().to_vec(),
            layout: CoefficientLayout::from_design(&self.system.prepared().design),
            demeaned,
            converged,
            iterations,
            residual,
            time_solve,
            // The deferred build is the batch's only setup, so this is zero when it built nothing.
            time_setup: build_secs,
            time_total: t_start.elapsed().as_secs_f64(),
            n_dofs: self.system.prepared().design.n_dofs,
            n_obs: self.system.prepared().design.n_obs,
        })
    }

    /// Access the preconditioner (for serialization or reuse across solvers).
    /// Under Adaptive: the Schwarz map once built, otherwise the diagonal base carrying the ladder.
    pub fn preconditioner(&self) -> Option<&Preconditioner> {
        match &self.slot {
            PrecondSlot::Static { map, .. } => map.as_ref(),
            PrecondSlot::Adaptive(a) => Some(a.rung()),
        }
    }

    /// Number of DOFs (coefficients).
    pub fn n_dofs(&self) -> usize {
        self.system.prepared().design.n_dofs
    }

    /// Number of observations.
    pub fn n_obs(&self) -> usize {
        self.system.prepared().design.n_obs
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
/// Zero-copy for F-order category arrays with labels `0..n_levels` whose dominant
/// factor is already sorted. Label compaction copies a factor's column at ingest;
/// the locality sort copies every column once more.
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
