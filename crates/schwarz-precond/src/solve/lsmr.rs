//! LSMR for rectangular least-squares.
//!
//! Solves `min ‖b − A x‖₂` with an optional preconditioner `M ≈ AᵀA`.
//! Without `M`, the standard Golub-Kahan bidiagonalization is used. With
//! `M`, the Modified Golub-Kahan variant requires only **one** `M⁻¹`
//! application per iteration — no square-root factorization of `M` is
//! needed.
//!
//! # Modified Golub-Kahan Bidiagonalization
//!
//! The standard preconditioned Golub-Kahan process requires two triangular
//! solves per iteration (`L⁻¹` and `L⁻ᵀ` where `M = LᵀL`). The modified
//! version introduces an auxiliary vector `p̃ = M ṽ` and exploits the
//! identity `‖ṽ‖_M = √⟨ṽ, p̃⟩` to merge both solves into a single
//! `M⁻¹` application:
//!
//! 1. Forward matvec: `A ṽ_k`
//! 2. Adjoint matvec: `Aᵀ u_{k+1}`
//! 3. Preconditioner solve: `M⁻¹ p̃` (the only place `M` appears)
//! 4. M-norm via dot product: `α = √⟨ṽ, p̃⟩`
//!
//! # Architecture
//!
//! Mirrors Algorithm 2.8 of Fong & Saunders:
//!
//! - [`Bidiagonalization`] — trait emitting one [`BidiagStep`] per call.
//!   Two impls: [`GolubKahan`] (no preconditioner) and
//!   [`ModifiedGolubKahan`] (preconditioned). The choice of bidiagonalization
//!   is the only place `M` appears; the rest of the solver is generic.
//! - [`Givens`] — a single rotation `(c, s, r)` constructed from `(a, b)`.
//!   Built twice per iteration: P̂_k eliminates `β_{k+1}`, P̄_k
//!   eliminates `θ_{k+1}`.
//! - [`LsmrRecurrenceState`] — applies P̂_k and P̄_k to advance the
//!   transformed-RHS scalars (`φ̄`, `ζ̄`); emits a [`RotationStep`] of
//!   natural rotation outputs `(ρ, ρ̄, θ_new, θ̄, ζ)`.
//! - [`SolutionState`] — the `(x, h, h̄)` vector recurrence that assembles
//!   `x` without storing the full `V_k` basis.
//! - [`ConvergenceState`] — Fong & Saunders' two stops plus the running
//!   `‖A‖_F²` estimate.
//!
//! # References
//!
//! - Fong & Saunders (2011). "LSMR: An Iterative Algorithm for Sparse
//!   Least-Squares Problems." *SIAM J. Sci. Comput.* 33(5).
//! - Arridge, Betcke, Harhanen (2014). "Iterated preconditioned LSQR
//!   method for inverse problems on unstructured grids." *Inverse Problems* 30(7).
//! - Hamarik, Huang, Kaltenbacher, Kangro (2024). "Flexible Modified LSMR
//!   for Least Squares Problems." arXiv:2408.16652.

mod bidiag;
mod recurrence;
#[cfg(test)]
mod tests;

use super::vec_norm;
use crate::{Operator, SolveError};
use bidiag::{BidiagStep, Bidiagonalization, GolubKahan, ModifiedGolubKahan};
use recurrence::{ConvergenceState, LsmrRecurrenceState, RotationStep, SolutionState, Stop};

/// Below this count the per-iteration vector kernels run sequentially —
/// rayon wake/steal overhead would dominate otherwise. Matches the threshold
/// used by `sparse_matrix::CsrMatrix::matvec_add`.
const LSMR_PAR_THRESHOLD: usize = 10_000;
/// Per-worker chunk size for the parallel vector kernels. Tuned to keep each
/// chunk's work above rayon dispatch overhead while staying L1-resident —
/// sizing chunks to `n / n_threads` instead regresses at 5M+ DOFs because
/// per-thread chunks blow L1/L2 and workers stream at DRAM bandwidth.
const LSMR_UPDATE_CHUNK: usize = 4096;

/// Result of an LSMR solve.
#[must_use]
pub struct LsmrResult {
    /// Solution vector.
    pub x: Vec<f64>,
    /// Whether the solver converged within the tolerance.
    pub converged: bool,
    /// Total number of iterations performed.
    pub iterations: usize,
    /// Final residual norm estimate `‖b − A x‖`.
    pub residual_norm: f64,
    /// Reason the solver stopped.
    pub stop_reason: LsmrStopReason,
}

/// Reason an LSMR solve stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LsmrStopReason {
    /// The right-hand side was exactly zero.
    ZeroRhs,
    /// The initial normal-equation residual was exactly zero (`Aᵀb = 0`).
    InitialNormalEquationResidualZero,
    /// The least-squares residual estimate met the absolute tolerance.
    ResidualTolerance,
    /// The normal-equation residual estimate met the relative tolerance.
    NormalEquationTolerance,
    /// The bidiagonalization reached a lucky breakdown.
    BidiagonalizationBreakdown,
    /// The iteration budget was exhausted before convergence.
    MaxIterations,
}

/// Unpreconditioned LSMR.
///
/// Solves `min ‖b − A x‖₂` over the standard Golub-Kahan bidiagonalization.
/// Minimizes the normal-equation residual `‖Aᵀ r‖`, giving smoother
/// convergence than LSQR. `operator` is rectangular (m × n).
///
/// See [`mlsmr`] for the preconditioned variant. `local_size` is the
/// optional windowed reorthogonalization size (see [`mlsmr`] for details);
/// memory cost when enabled is `local_size · n` doubles.
pub fn lsmr<A: Operator + ?Sized>(
    operator: &A,
    b: &[f64],
    tol: f64,
    maxiter: usize,
    local_size: Option<usize>,
) -> Result<LsmrResult, SolveError> {
    mlsmr_impl::<A, A>(operator, b, None, tol, maxiter, local_size)
}

/// Modified LSMR with preconditioner `M ≈ AᵀA`.
///
/// Solves `min ‖b − A x‖₂` using the Modified Golub-Kahan bidiagonalization,
/// requiring one `M⁻¹` apply per iteration. `operator` is rectangular
/// (m × n); `preconditioner` is square (n × n) and SPD.
///
/// `local_size` is the number of past `v` vectors to reorthogonalize
/// against via windowed modified Gram-Schmidt. `None` (default) disables
/// the correction — the short-recurrence bidiagonalization is used as is.
/// `Some(N)` enables a window of `N` past `v` vectors. `Some(5..20)` is
/// cheap insurance for ill-conditioned problems where rounding causes the
/// `v_k` sequence to lose orthogonality and convergence to stall. Values
/// up to `min(m, n)` approach full reorthogonalization. Memory cost is
/// `2 · local_size · n` doubles (storing `(v_j, M v_j)` pairs).
pub fn mlsmr<A: Operator + ?Sized, M: Operator + ?Sized>(
    operator: &A,
    b: &[f64],
    preconditioner: &M,
    tol: f64,
    maxiter: usize,
    local_size: Option<usize>,
) -> Result<LsmrResult, SolveError> {
    mlsmr_impl(operator, b, Some(preconditioner), tol, maxiter, local_size)
}

fn mlsmr_impl<A: Operator + ?Sized, M: Operator + ?Sized>(
    operator: &A,
    b: &[f64],
    preconditioner: Option<&M>,
    tol: f64,
    maxiter: usize,
    local_size: Option<usize>,
) -> Result<LsmrResult, SolveError> {
    validate_lsmr_inputs(operator, b, tol)?;
    let n = operator.ncols();

    let b_norm = vec_norm(b);
    if b_norm == 0.0 {
        return Ok(zero_rhs_result(n));
    }

    let local_size = local_size.unwrap_or(0);
    match preconditioner {
        None => {
            let (bidiag, step1) = GolubKahan::init(operator, b, local_size)?;
            lsmr_from_bidiag(bidiag, step1, b_norm, tol, maxiter)
        }
        Some(m) => {
            validate_lsmr_preconditioner(operator, m)?;
            let (bidiag, step1) = ModifiedGolubKahan::init(operator, m, b, local_size)?;
            lsmr_from_bidiag(bidiag, step1, b_norm, tol, maxiter)
        }
    }
}

fn zero_rhs_result(n: usize) -> LsmrResult {
    LsmrResult {
        x: vec![0.0; n],
        converged: true,
        iterations: 0,
        residual_norm: 0.0,
        stop_reason: LsmrStopReason::ZeroRhs,
    }
}

/// Run the LSMR scalar/vector recurrences over a bidiagonalization stream.
/// Generic over the bidiagonalization, which is the only place the choice
/// of preconditioner enters.
fn lsmr_from_bidiag<B: Bidiagonalization>(
    mut bidiag: B,
    step1: BidiagStep,
    b_norm: f64,
    tol: f64,
    maxiter: usize,
) -> Result<LsmrResult, SolveError> {
    let n = bidiag.v().len();
    if step1.alpha == 0.0 {
        return Ok(LsmrResult {
            x: vec![0.0; n],
            converged: true,
            iterations: 0,
            residual_norm: b_norm,
            stop_reason: LsmrStopReason::InitialNormalEquationResidualZero,
        });
    }

    let mut recurrence = LsmrRecurrenceState::init(step1);
    let mut solution = SolutionState::init(bidiag.v());
    let mut convergence = ConvergenceState::new(b_norm, tol, step1.alpha);
    let mut prev_rot = RotationStep::initial();

    for itn in 1..=maxiter {
        let step = bidiag.step()?;
        convergence.observe(step);
        let curr_rot = recurrence.step(step);
        solution.update(bidiag.v(), curr_rot, prev_rot);

        // Convergence wins over breakdown when both fire on the same step:
        // the user-specified tolerance is the contract, breakdown is an
        // internal property of the bidiagonalization.
        let stop = convergence.check(&recurrence);
        if let Some(stop_reason) = lsmr_stop_reason(stop) {
            return Ok(LsmrResult {
                x: solution.into_x(),
                converged: true,
                iterations: itn,
                residual_norm: recurrence.residual_estimate(),
                stop_reason,
            });
        }
        if step.alpha == 0.0 {
            return Ok(LsmrResult {
                x: solution.into_x(),
                converged: true,
                iterations: itn,
                residual_norm: recurrence.residual_estimate(),
                stop_reason: LsmrStopReason::BidiagonalizationBreakdown,
            });
        }
        prev_rot = curr_rot;
    }

    Ok(LsmrResult {
        x: solution.into_x(),
        converged: false,
        iterations: maxiter,
        residual_norm: recurrence.residual_estimate(),
        stop_reason: LsmrStopReason::MaxIterations,
    })
}

fn validate_lsmr_inputs<A: Operator + ?Sized>(
    operator: &A,
    b: &[f64],
    tol: f64,
) -> Result<(), SolveError> {
    if b.len() != operator.nrows() {
        return Err(SolveError::InvalidInput {
            context: "lsmr",
            message: format!(
                "rhs length {} does not match operator row count {}",
                b.len(),
                operator.nrows()
            ),
        });
    }
    if !tol.is_finite() || tol < 0.0 {
        return Err(SolveError::InvalidInput {
            context: "lsmr",
            message: format!("tolerance must be finite and nonnegative, got {tol}"),
        });
    }
    if let Some((index, value)) = b.iter().copied().enumerate().find(|(_, v)| !v.is_finite()) {
        return Err(SolveError::InvalidInput {
            context: "lsmr",
            message: format!("rhs entry {index} must be finite, got {value}"),
        });
    }
    Ok(())
}

fn validate_lsmr_preconditioner<A: Operator + ?Sized, M: Operator + ?Sized>(
    operator: &A,
    preconditioner: &M,
) -> Result<(), SolveError> {
    let n = operator.ncols();
    if preconditioner.nrows() != n || preconditioner.ncols() != n {
        return Err(SolveError::InvalidInput {
            context: "lsmr",
            message: format!(
                "preconditioner shape {}x{} must match operator column count {n}",
                preconditioner.nrows(),
                preconditioner.ncols(),
            ),
        });
    }
    Ok(())
}

fn lsmr_stop_reason(stop: Stop) -> Option<LsmrStopReason> {
    match stop {
        Stop::Continue => None,
        Stop::ResidualTolerance => Some(LsmrStopReason::ResidualTolerance),
        Stop::NormalEquationTolerance => Some(LsmrStopReason::NormalEquationTolerance),
    }
}
