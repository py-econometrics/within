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

use rayon::prelude::*;

use super::{dot, vec_norm};
use crate::{Operator, SolveError};

/// Below this count the per-iteration vector kernels run sequentially —
/// rayon wake/steal overhead would dominate otherwise. Matches the threshold
/// used by `sparse_matrix::CsrMatrix::matvec_add`.
const LSMR_PAR_THRESHOLD: usize = 10_000;
/// Per-worker chunk size for the parallel vector kernels. Tuned to keep each
/// chunk's work above rayon dispatch overhead while staying L1-resident —
/// sizing chunks to `n / n_threads` instead regresses at 5M+ DOFs because
/// per-thread chunks blow L1/L2 and workers stream at DRAM bandwidth.
const LSMR_UPDATE_CHUNK: usize = 4096;

// ---------------------------------------------------------------------------
// Vector kernels
// ---------------------------------------------------------------------------

/// Fused `y = x + scale * y` with `‖y_new‖²` returned. Parallel above the
/// threshold; each chunk accumulates its own partial sum to avoid cross-thread
/// traffic on the reduction.
#[inline]
fn axpy_with_sq_norm(y: &mut [f64], x: &[f64], scale: f64) -> f64 {
    debug_assert_eq!(x.len(), y.len());
    let seq = |y_c: &mut [f64], x_c: &[f64]| -> f64 {
        let mut s = 0.0;
        for (yi, &xi) in y_c.iter_mut().zip(x_c.iter()) {
            let val = xi + scale * *yi;
            *yi = val;
            s += val * val;
        }
        s
    };
    if y.len() >= LSMR_PAR_THRESHOLD {
        y.par_chunks_mut(LSMR_UPDATE_CHUNK)
            .zip(x.par_chunks(LSMR_UPDATE_CHUNK))
            .map(|(y_c, x_c)| seq(y_c, x_c))
            .sum()
    } else {
        seq(y, x)
    }
}

/// `y = alpha * x + beta * y`. Parallel above the threshold.
#[inline]
fn axpby(y: &mut [f64], x: &[f64], alpha: f64, beta: f64) {
    debug_assert_eq!(x.len(), y.len());
    let seq = |y_c: &mut [f64], x_c: &[f64]| {
        for (yi, &xi) in y_c.iter_mut().zip(x_c.iter()) {
            *yi = alpha * xi + beta * *yi;
        }
    };
    if y.len() >= LSMR_PAR_THRESHOLD {
        y.par_chunks_mut(LSMR_UPDATE_CHUNK)
            .zip(x.par_chunks(LSMR_UPDATE_CHUNK))
            .for_each(|(y_c, x_c)| seq(y_c, x_c));
    } else {
        seq(y, x);
    }
}

/// In-place scalar multiply `y *= s`. Parallel above the threshold.
#[inline]
fn scale_in_place(y: &mut [f64], s: f64) {
    let seq = |c: &mut [f64]| {
        for yi in c {
            *yi *= s;
        }
    };
    if y.len() >= LSMR_PAR_THRESHOLD {
        y.par_chunks_mut(LSMR_UPDATE_CHUNK).for_each(seq);
    } else {
        seq(y);
    }
}

/// Parallel dot product; falls back to the sequential `super::dot` below
/// the threshold.
#[inline]
fn par_dot(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    if a.len() >= LSMR_PAR_THRESHOLD {
        a.par_chunks(LSMR_UPDATE_CHUNK)
            .zip(b.par_chunks(LSMR_UPDATE_CHUNK))
            .map(|(ac, bc)| ac.iter().zip(bc).map(|(x, y)| x * y).sum::<f64>())
            .sum()
    } else {
        dot(a, b)
    }
}

// ---------------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------------

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
}

// ---------------------------------------------------------------------------
// Givens rotation
// ---------------------------------------------------------------------------

/// Givens rotation `[[c, s], [-s, c]]` constructed from a column `(a, b)`
/// such that the rotation applied to that column yields `(r, 0)` with
/// `r = hypot(a, b)`.
#[derive(Clone, Copy)]
struct Givens {
    c: f64,
    s: f64,
    r: f64,
}

impl Givens {
    /// Construct the rotation that zeros `b` against `a`.
    fn new(a: f64, b: f64) -> Self {
        let r = f64::hypot(a, b);
        let (c, s) = if r > 0.0 { (a / r, b / r) } else { (1.0, 0.0) };
        Self { c, s, r }
    }
}

// ---------------------------------------------------------------------------
// Bidiagonalization step + trait
// ---------------------------------------------------------------------------

/// One step of the bidiagonal sequence: the freshly computed `(α_{k+1},
/// β_{k+1})` scalars emitted by the bidiagonalization.
#[derive(Clone, Copy)]
struct BidiagStep {
    alpha: f64,
    beta: f64,
}

/// A bidiagonalization stream feeding LSMR with `(α, β)` pairs and the
/// matching normalized basis vector `v_k`.
trait Bidiagonalization {
    /// Advance one step. After the call, `v()` is the normalized `v_{k+1}`.
    fn step(&mut self) -> Result<BidiagStep, SolveError>;
    /// Most recent normalized basis vector.
    fn v(&self) -> &[f64];
}

// ---------------------------------------------------------------------------
// Standard Golub-Kahan bidiagonalization (no preconditioner)
// ---------------------------------------------------------------------------

/// Workspaces used by [`GolubKahan`].
struct GolubKahanBuffers {
    /// `u_k` in observation space (length m), kept normalized.
    u: Vec<f64>,
    /// `v_k` in DOF space (length n), kept normalized.
    v: Vec<f64>,
    /// Scratch for `A · v` (length m).
    av: Vec<f64>,
    /// Scratch for `Aᵀ · u` (length n).
    atu: Vec<f64>,
}

impl GolubKahanBuffers {
    fn new(m: usize, n: usize) -> Self {
        Self {
            u: vec![0.0; m],
            v: vec![0.0; n],
            av: vec![0.0; m],
            atu: vec![0.0; n],
        }
    }
}

/// Standard Golub-Kahan bidiagonalization. No preconditioner, no `p̃`
/// buffer, two normalizations per step (`u` and `v`).
struct GolubKahan<'a, A: Operator + ?Sized> {
    operator: &'a A,
    bufs: GolubKahanBuffers,
    /// Last `α` emitted; needed by the next step's u-update.
    alpha: f64,
}

impl<'a, A: Operator + ?Sized> GolubKahan<'a, A> {
    /// Initialize the bidiagonalization. Returns `Self` and the first step
    /// `(α₁, β₁)`.
    fn init(operator: &'a A, b: &[f64]) -> Result<(Self, BidiagStep), SolveError> {
        let m = operator.nrows();
        let n = operator.ncols();
        let mut bufs = GolubKahanBuffers::new(m, n);

        // β₁ = ‖b‖, u₁ = b / β₁
        let beta = vec_norm(b);
        if beta > 0.0 {
            let inv = 1.0 / beta;
            for (ui, &bi) in bufs.u.iter_mut().zip(b) {
                *ui = bi * inv;
            }
        }

        // α₁ v₁ = Aᵀ u₁
        operator.try_apply_adjoint(&bufs.u, &mut bufs.v)?;
        let alpha = vec_norm(&bufs.v);
        if alpha > 0.0 {
            scale_in_place(&mut bufs.v, 1.0 / alpha);
        }

        Ok((
            Self {
                operator,
                bufs,
                alpha,
            },
            BidiagStep { alpha, beta },
        ))
    }
}

impl<A: Operator + ?Sized> Bidiagonalization for GolubKahan<'_, A> {
    fn step(&mut self) -> Result<BidiagStep, SolveError> {
        // β_{k+1} u_{k+1} = A v_k − α_k u_k
        self.operator.try_apply(&self.bufs.v, &mut self.bufs.av)?;
        let beta_sq = axpy_with_sq_norm(&mut self.bufs.u, &self.bufs.av, -self.alpha);
        let beta = beta_sq.sqrt();
        if beta > 0.0 {
            scale_in_place(&mut self.bufs.u, 1.0 / beta);
        }

        // α_{k+1} v_{k+1} = Aᵀ u_{k+1} − β_{k+1} v_k
        self.operator
            .try_apply_adjoint(&self.bufs.u, &mut self.bufs.atu)?;
        let alpha_sq = axpy_with_sq_norm(&mut self.bufs.v, &self.bufs.atu, -beta);
        let alpha = alpha_sq.sqrt();
        if alpha > 0.0 {
            scale_in_place(&mut self.bufs.v, 1.0 / alpha);
        }

        self.alpha = alpha;
        Ok(BidiagStep { alpha, beta })
    }

    fn v(&self) -> &[f64] {
        &self.bufs.v
    }
}

// ---------------------------------------------------------------------------
// Modified Golub-Kahan bidiagonalization (preconditioned)
// ---------------------------------------------------------------------------

/// Workspaces used by [`ModifiedGolubKahan`].
struct ModifiedGolubKahanBuffers {
    /// `u` in observation space (length m). Left unnormalized between steps;
    /// `‖u‖ = β_{k+1}` after the forward stage. The next step's adjoint matvec
    /// produces `β · Aᵀ u_norm`, and the `β/α_prev` coefficient applied to
    /// `p_tilde` cancels the unnormalization.
    u: Vec<f64>,
    /// `ṽ` in DOF space (length n). **Normalized** at the end of each step.
    v: Vec<f64>,
    /// `p̃` recurrence vector (length n).
    /// Invariant: `p_tilde_stored = α · M · v_normalized`.
    p_tilde: Vec<f64>,
    /// Scratch for `A · v` (length m).
    av: Vec<f64>,
    /// Scratch for `Aᵀ · u` (length n).
    atu: Vec<f64>,
}

impl ModifiedGolubKahanBuffers {
    fn new(m: usize, n: usize) -> Self {
        Self {
            u: vec![0.0; m],
            v: vec![0.0; n],
            p_tilde: vec![0.0; n],
            av: vec![0.0; m],
            atu: vec![0.0; n],
        }
    }
}

/// Modified Golub-Kahan bidiagonalization with `M ≈ AᵀA`. Stores `p̃`
/// scaled by `α` between steps so the preconditioner solve is a single
/// `M⁻¹` application per iteration.
struct ModifiedGolubKahan<'a, A: Operator + ?Sized, M: Operator + ?Sized> {
    operator: &'a A,
    preconditioner: &'a M,
    bufs: ModifiedGolubKahanBuffers,
    /// Last `α` emitted; needed by the next step to scale `p_tilde`.
    alpha: f64,
    /// `1/β_k`; cancels the unnormalization of `u` in the next step.
    beta_prev_inv: f64,
}

impl<'a, A: Operator + ?Sized, M: Operator + ?Sized> ModifiedGolubKahan<'a, A, M> {
    /// Initialize the bidiagonalization. Returns `Self` and the first step
    /// `(α₁, β₁)`.
    fn init(
        operator: &'a A,
        preconditioner: &'a M,
        b: &[f64],
    ) -> Result<(Self, BidiagStep), SolveError> {
        let m = operator.nrows();
        let n = operator.ncols();
        let mut bufs = ModifiedGolubKahanBuffers::new(m, n);

        // β₁ = ‖b‖, u₁ = b / β₁
        let beta = vec_norm(b);
        if beta > 0.0 {
            let inv = 1.0 / beta;
            for (ui, &bi) in bufs.u.iter_mut().zip(b) {
                *ui = bi * inv;
            }
        }

        // p̃ = Aᵀ u₁
        operator.try_apply_adjoint(&bufs.u, &mut bufs.p_tilde)?;

        // ṽ₁ = M⁻¹ p̃
        preconditioner.try_apply(&bufs.p_tilde, &mut bufs.v)?;

        // α₁ = √⟨ṽ₁, p̃⟩ via the M-norm dot product trick.
        let vp = dot(&bufs.v, &bufs.p_tilde);
        let alpha = if vp > 0.0 { vp.sqrt() } else { 0.0 };

        // Normalize v; leave p_tilde at α₁·p̃_norm.
        if alpha > 0.0 {
            scale_in_place(&mut bufs.v, 1.0 / alpha);
        }

        Ok((
            Self {
                operator,
                preconditioner,
                bufs,
                alpha,
                beta_prev_inv: 1.0, // u was normalized by init
            },
            BidiagStep { alpha, beta },
        ))
    }
}

impl<A: Operator + ?Sized, M: Operator + ?Sized> Bidiagonalization
    for ModifiedGolubKahan<'_, A, M>
{
    fn step(&mut self) -> Result<BidiagStep, SolveError> {
        // ũ_{k+1} = A ṽ_k − (α_k / β_k) ũ_k, β_{k+1} = ‖ũ_{k+1}‖
        let scale = -(self.alpha * self.beta_prev_inv);
        self.operator.try_apply(&self.bufs.v, &mut self.bufs.av)?;
        let beta_sq = axpy_with_sq_norm(&mut self.bufs.u, &self.bufs.av, scale);
        let beta = beta_sq.sqrt();
        let beta_inv = if beta > 0.0 { 1.0 / beta } else { 0.0 };

        // Aᵀ on unnormalized u: result is β · Aᵀ u_normalized.
        // p̃_stored = α_prev · p̃_norm, so β/α_prev cancels α_prev in p̃_stored.
        self.operator
            .try_apply_adjoint(&self.bufs.u, &mut self.bufs.atu)?;
        let p_coeff = beta / self.alpha;
        axpby(&mut self.bufs.p_tilde, &self.bufs.atu, beta_inv, -p_coeff);

        // ṽ_{k+1} = M⁻¹ p̃   (= α_new · v_norm before normalization)
        self.preconditioner
            .try_apply(&self.bufs.p_tilde, &mut self.bufs.v)?;

        // α_{k+1} = √⟨ṽ, p̃⟩
        let vp = par_dot(&self.bufs.v, &self.bufs.p_tilde);
        let alpha_new = if vp > 0.0 { vp.sqrt() } else { 0.0 };

        // Normalize v in place. p_tilde stays at α_new · p̃_norm for next step.
        if alpha_new > 0.0 {
            scale_in_place(&mut self.bufs.v, 1.0 / alpha_new);
        }

        self.alpha = alpha_new;
        self.beta_prev_inv = beta_inv;

        Ok(BidiagStep {
            alpha: alpha_new,
            beta,
        })
    }

    fn v(&self) -> &[f64] {
        &self.bufs.v
    }
}

// ---------------------------------------------------------------------------
// LSMR scalar recurrence: two interleaved Givens rotation chains
// ---------------------------------------------------------------------------

/// Natural outputs of one rotation step. Carries the scalars that
/// Algorithm 2.8 produces in the "construct rotation P̂_k / P̄_k" blocks
/// and feeds straight into the `(x, h, h̄)` recurrence.
#[derive(Clone, Copy)]
struct RotationStep {
    /// `ρ_k`, output of P̂_k.
    rho: f64,
    /// `ρ̄_k`, output of P̄_k.
    rho_bar: f64,
    /// `θ_{k+1}`, off-diagonal carried forward by P̂_k.
    theta_new: f64,
    /// `θ̄_k`, off-diagonal carried forward by P̄_k.
    theta_bar: f64,
    /// `ζ_k`, transformed-RHS scalar after P̄_k.
    zeta: f64,
}

impl RotationStep {
    /// Seed value used as `prev` on the first iteration. With
    /// `theta_bar = 0`, the `t_hbar` ratio in the solution recurrence
    /// vanishes, matching the `h̄₀ = 0` initial condition in Algorithm 2.8.
    fn initial() -> Self {
        Self {
            rho: 1.0,
            rho_bar: 1.0,
            theta_new: 0.0,
            theta_bar: 0.0,
            zeta: 0.0,
        }
    }
}

/// LSMR scalar state. Carries the state both Givens rotation chains need
/// between iterations: `α̅` and `φ̄` for P̂_k (LSQR-side), and `c̅, s̅,
/// ζ̄` for P̄_k (LSMR-side).
struct LsmrRecurrenceState {
    // P̂ chain
    alpha_bar: f64,
    phi_bar: f64,
    // P̄ chain
    c_bar: f64,
    s_bar: f64,
    zeta_bar: f64,
}

impl LsmrRecurrenceState {
    fn init(s1: BidiagStep) -> Self {
        Self {
            alpha_bar: s1.alpha,
            phi_bar: s1.beta,
            c_bar: 1.0,
            s_bar: 0.0,
            zeta_bar: s1.alpha * s1.beta,
        }
    }

    /// Construct and apply both rotations for the current step.
    fn step(&mut self, s: BidiagStep) -> RotationStep {
        // Construct rotation P̂_k: eliminates β_{k+1} against α̅_k.
        let p_hat = Givens::new(self.alpha_bar, s.beta);
        let theta_new = p_hat.s * s.alpha;
        let alpha_bar_new = -p_hat.c * s.alpha;
        let phi_bar_new = p_hat.s * self.phi_bar;

        // Construct rotation P̄_k: eliminates θ_{k+1} against c̅_{k-1}·ρ_k.
        // (s̅_{k-1} carries the previous step's value — read before commit.)
        let theta_bar = self.s_bar * p_hat.r;
        let p_bar = Givens::new(self.c_bar * p_hat.r, theta_new);
        let zeta = p_bar.c * self.zeta_bar;
        let zeta_bar_new = -p_bar.s * self.zeta_bar;

        // Commit chain state.
        self.alpha_bar = alpha_bar_new;
        self.phi_bar = phi_bar_new;
        self.c_bar = p_bar.c;
        self.s_bar = p_bar.s;
        self.zeta_bar = zeta_bar_new;

        RotationStep {
            rho: p_hat.r,
            rho_bar: p_bar.r,
            theta_new,
            theta_bar,
            zeta,
        }
    }

    /// `|φ̄|` — conservative estimate of `‖r_k‖`. The LSMR residual is
    /// bounded by the LSQR residual, which equals `|φ̄|`.
    fn residual_estimate(&self) -> f64 {
        self.phi_bar.abs()
    }

    /// `|ζ̄|` — running estimate of `‖Aᵀ r_k‖` (Fong & Saunders).
    fn normal_eq_residual_estimate(&self) -> f64 {
        self.zeta_bar.abs()
    }
}

// ---------------------------------------------------------------------------
// Solution recurrence vectors
// ---------------------------------------------------------------------------

/// Vectors carried by the LSMR solution recurrence.
///
/// `(h, h̄)` are the auxiliary recurrence vectors that let us assemble `x`
/// without storing the full `V_k` basis.
struct SolutionState {
    x: Vec<f64>,
    h: Vec<f64>,
    h_bar: Vec<f64>,
}

impl SolutionState {
    /// Initialize from the first normalized basis vector: `h₁ = v₁`,
    /// `x = 0`, `h̄₀ = 0`.
    fn init(v1: &[f64]) -> Self {
        Self {
            x: vec![0.0; v1.len()],
            h: v1.to_vec(),
            h_bar: vec![0.0; v1.len()],
        }
    }

    /// Apply one step of the `(x, h, h̄)` recurrence. `v` must be the
    /// normalized `v_{k+1}` from the bidiagonalization. `prev` carries
    /// `(ρ_{k-1}, ρ̄_{k-1})` from the previous rotation step (seeded with
    /// [`RotationStep::initial`] on the first iteration).
    fn update(&mut self, v: &[f64], curr: RotationStep, prev: RotationStep) {
        // Ratios consumed by the recurrence (Algorithm 2.8, "Update h̄, x, h").
        let t_x_denom = curr.rho * curr.rho_bar;
        let t_x = if t_x_denom.abs() > f64::EPSILON {
            curr.zeta / t_x_denom
        } else {
            0.0
        };
        let t_hbar_denom = prev.rho * prev.rho_bar;
        let t_hbar = if t_hbar_denom.abs() > f64::EPSILON {
            curr.theta_bar * curr.rho / t_hbar_denom
        } else {
            0.0
        };
        let t_h = if curr.rho.abs() > f64::EPSILON {
            curr.theta_new / curr.rho
        } else {
            0.0
        };

        let n = self.x.len();
        debug_assert_eq!(v.len(), n);

        let chunk = |hb_c: &mut [f64], h_c: &mut [f64], x_c: &mut [f64], v_c: &[f64]| {
            for (((hbi, hi), xi), vi) in hb_c
                .iter_mut()
                .zip(h_c.iter_mut())
                .zip(x_c.iter_mut())
                .zip(v_c.iter())
            {
                let h_old = *hi;
                let hb = h_old - t_hbar * *hbi;
                *hbi = hb;
                *xi += t_x * hb;
                *hi = *vi - t_h * h_old;
            }
        };

        if n >= LSMR_PAR_THRESHOLD {
            self.h_bar
                .par_chunks_mut(LSMR_UPDATE_CHUNK)
                .zip(self.h.par_chunks_mut(LSMR_UPDATE_CHUNK))
                .zip(self.x.par_chunks_mut(LSMR_UPDATE_CHUNK))
                .zip(v.par_chunks(LSMR_UPDATE_CHUNK))
                .for_each(|(((hb_c, h_c), x_c), v_c)| chunk(hb_c, h_c, x_c, v_c));
        } else {
            chunk(&mut self.h_bar, &mut self.h, &mut self.x, v);
        }
    }

    fn into_x(self) -> Vec<f64> {
        self.x
    }
}

// ---------------------------------------------------------------------------
// Convergence
// ---------------------------------------------------------------------------

enum Stop {
    Continue,
    Converged,
}

/// Fong & Saunders' two stop criteria for LSMR plus the running `‖A‖_F²`
/// accumulator that the second criterion needs.
struct ConvergenceState {
    /// `tol · ‖b‖`.
    abs_tol: f64,
    /// `tol`.
    rel_tol: f64,
    /// Running `‖A‖_F²` from the bidiagonal entries.
    a_norm_sq: f64,
}

impl ConvergenceState {
    fn new(b_norm: f64, tol: f64, alpha1: f64) -> Self {
        Self {
            abs_tol: tol * b_norm,
            rel_tol: tol,
            a_norm_sq: alpha1 * alpha1,
        }
    }

    /// Fold a fresh bidiagonal step into the `‖A‖_F²` estimate.
    fn observe(&mut self, s: BidiagStep) {
        self.a_norm_sq += s.alpha * s.alpha + s.beta * s.beta;
    }

    /// Check both stop criteria against the current scalar state.
    fn check(&self, r: &LsmrRecurrenceState) -> Stop {
        let residual = r.residual_estimate();
        if residual <= self.abs_tol {
            return Stop::Converged;
        }
        let a_norm = self.a_norm_sq.sqrt().max(f64::MIN_POSITIVE);
        let normar = r.normal_eq_residual_estimate();
        if normar / (a_norm * residual.max(f64::MIN_POSITIVE)) <= self.rel_tol {
            return Stop::Converged;
        }
        Stop::Continue
    }
}

// ---------------------------------------------------------------------------
// LSMR
// ---------------------------------------------------------------------------

/// LSMR with optional preconditioner `M ≈ AᵀA`.
///
/// Solves `min ‖b − A x‖₂`. Without `M`, runs standard LSMR over the
/// Golub-Kahan bidiagonalization. With `M`, runs the Modified variant
/// requiring one `M⁻¹` apply per iteration. Minimizes the normal-equation
/// residual `‖Aᵀ r‖`, giving smoother convergence than LSQR.
///
/// `operator` is rectangular (m × n). `preconditioner` is square
/// (n × n) and symmetric positive definite.
pub fn mlsmr<A: Operator + ?Sized, M: Operator + ?Sized>(
    operator: &A,
    b: &[f64],
    preconditioner: Option<&M>,
    tol: f64,
    maxiter: usize,
) -> Result<LsmrResult, SolveError> {
    let n = operator.ncols();
    debug_assert_eq!(b.len(), operator.nrows());

    let b_norm = vec_norm(b);
    if b_norm < f64::EPSILON {
        return Ok(LsmrResult {
            x: vec![0.0; n],
            converged: true,
            iterations: 0,
            residual_norm: 0.0,
        });
    }

    match preconditioner {
        None => {
            let (bidiag, step1) = GolubKahan::init(operator, b)?;
            lsmr_from_bidiag(bidiag, step1, b_norm, tol, maxiter)
        }
        Some(m) => {
            let (bidiag, step1) = ModifiedGolubKahan::init(operator, m, b)?;
            lsmr_from_bidiag(bidiag, step1, b_norm, tol, maxiter)
        }
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

        let converged = matches!(convergence.check(&recurrence), Stop::Converged);
        if converged || step.alpha == 0.0 {
            return Ok(LsmrResult {
                x: solution.into_x(),
                converged: true,
                iterations: itn,
                residual_norm: recurrence.residual_estimate(),
            });
        }
        prev_rot = curr_rot;
    }

    Ok(LsmrResult {
        x: solution.into_x(),
        converged: false,
        iterations: maxiter,
        residual_norm: recurrence.residual_estimate(),
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::IdentityOperator;

    /// Simple 4×3 overdetermined system.
    /// A = [1 0 0; 0 1 0; 0 0 1; 1 1 0]
    struct OverdeterminedOp;

    impl Operator for OverdeterminedOp {
        fn nrows(&self) -> usize {
            4
        }
        fn ncols(&self) -> usize {
            3
        }
        fn apply(&self, x: &[f64], y: &mut [f64]) {
            y[0] = x[0];
            y[1] = x[1];
            y[2] = x[2];
            y[3] = x[0] + x[1];
        }
        fn apply_adjoint(&self, u: &[f64], x: &mut [f64]) {
            x[0] = u[0] + u[3];
            x[1] = u[1] + u[3];
            x[2] = u[2];
        }
    }

    /// Diagonal preconditioner: M⁻¹ = diag(1/2, 1/2, 1)
    /// Approximates (Aᵀ A)⁻¹ = diag(2, 2, 1)⁻¹
    struct DiagPrecond;

    impl Operator for DiagPrecond {
        fn nrows(&self) -> usize {
            3
        }
        fn ncols(&self) -> usize {
            3
        }
        fn apply(&self, x: &[f64], y: &mut [f64]) {
            y[0] = x[0] / 2.0;
            y[1] = x[1] / 2.0;
            y[2] = x[2];
        }
        fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) {
            self.apply(x, y);
        }
    }

    #[test]
    fn test_mlsmr_unpreconditioned() {
        let b = vec![1.0, 2.0, 3.0, 3.0];
        let result = mlsmr(&OverdeterminedOp, &b, None::<&IdentityOperator>, 1e-10, 100)
            .expect("mlsmr solve");
        assert!(result.converged, "MLSMR did not converge");
        let err: f64 = result
            .x
            .iter()
            .zip([1.0, 2.0, 3.0].iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(err < 1e-6, "MLSMR solution error: {err}");
    }

    #[test]
    fn test_mlsmr_preconditioned() {
        let b = vec![1.0, 2.0, 3.0, 3.0];
        let result = mlsmr(&OverdeterminedOp, &b, Some(&DiagPrecond), 1e-10, 100)
            .expect("preconditioned mlsmr solve");
        assert!(result.converged, "Preconditioned MLSMR did not converge");
        let err: f64 = result
            .x
            .iter()
            .zip([1.0, 2.0, 3.0].iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(err < 1e-6, "Preconditioned MLSMR solution error: {err}");
    }

    #[test]
    fn test_mlsmr_inconsistent_system() {
        let b = vec![1.0, 2.0, 3.0, 0.0];
        let result = mlsmr(&OverdeterminedOp, &b, None::<&IdentityOperator>, 1e-10, 100)
            .expect("mlsmr solve");
        assert!(
            result.converged,
            "MLSMR did not converge on inconsistent system"
        );
        let mut ax = vec![0.0; 4];
        OverdeterminedOp.apply(&result.x, &mut ax);
        let residual: Vec<f64> = b.iter().zip(ax.iter()).map(|(bi, ai)| bi - ai).collect();
        let mut atr = vec![0.0; 3];
        OverdeterminedOp.apply_adjoint(&residual, &mut atr);
        let normal_resid = vec_norm(&atr);
        assert!(
            normal_resid < 1e-6,
            "Normal equation residual too large: {normal_resid}"
        );
    }

    #[test]
    fn test_mlsmr_maxiter_exhaustion() {
        let b = vec![1.0, 2.0, 3.0, 3.0];
        let result =
            mlsmr(&OverdeterminedOp, &b, None::<&IdentityOperator>, 1e-15, 1).expect("mlsmr solve");
        assert!(
            !result.converged,
            "should not converge in 1 iteration at 1e-15 tol"
        );
        assert_eq!(result.iterations, 1);
    }

    /// `None` (GolubKahan path) and `Some(&IdentityOperator)`
    /// (ModifiedGolubKahan with M = I) are mathematically the same algorithm.
    /// They should produce numerically equivalent solutions and iteration
    /// counts; this guards against future drift between the two
    /// bidiagonalization implementations.
    #[test]
    fn test_mlsmr_none_matches_identity_precond() {
        let b = vec![1.0, 2.0, 3.0, 3.0];
        let id = IdentityOperator::new(3);

        let none_result = mlsmr(&OverdeterminedOp, &b, None::<&IdentityOperator>, 1e-12, 100)
            .expect("mlsmr None solve");
        let id_result =
            mlsmr(&OverdeterminedOp, &b, Some(&id), 1e-12, 100).expect("mlsmr Identity solve");

        assert!(none_result.converged && id_result.converged);
        assert_eq!(none_result.iterations, id_result.iterations);

        let diff: f64 = none_result
            .x
            .iter()
            .zip(id_result.x.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(
            diff < 1e-10,
            "GolubKahan vs ModifiedGolubKahan-with-identity solutions disagree: {diff}"
        );
        assert!(
            (none_result.residual_norm - id_result.residual_norm).abs() < 1e-10,
            "residual norm estimates disagree: {} vs {}",
            none_result.residual_norm,
            id_result.residual_norm
        );
    }
}
