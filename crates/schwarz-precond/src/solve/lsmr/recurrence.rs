//! LSMR scalar/vector recurrence consuming the bidiagonalization stream.
//!
//! Given `(α, β)` pairs from a [`super::bidiag::Bidiagonalization`], this
//! module builds the two interleaved Givens rotation chains (P̂_k, P̄_k)
//! that yield Algorithm 2.8 of Fong & Saunders, advances the `(x, h, h̄)`
//! solution recurrence, and tracks the dual stopping criterion.

use crate::solve::lsmr::bidiag::BidiagStep;
use crate::solve::lsmr::{LSMR_PAR_THRESHOLD, LSMR_UPDATE_CHUNK};
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::prelude::{ParallelSlice, ParallelSliceMut};

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

/// Natural outputs of one rotation step. Carries the scalars that
/// Algorithm 2.8 produces in the "construct rotation P̂_k / P̄_k" blocks
/// and feeds straight into the `(x, h, h̄)` recurrence.
#[derive(Clone, Copy)]
pub(super) struct RotationStep {
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
    pub(super) fn initial() -> Self {
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
pub(super) struct LsmrRecurrenceState {
    // P̂ chain
    alpha_bar: f64,
    phi_bar: f64,
    // P̄ chain
    c_bar: f64,
    s_bar: f64,
    zeta_bar: f64,
}

impl LsmrRecurrenceState {
    pub(super) fn init(s1: BidiagStep) -> Self {
        Self {
            alpha_bar: s1.alpha,
            phi_bar: s1.beta,
            c_bar: 1.0,
            s_bar: 0.0,
            zeta_bar: s1.alpha * s1.beta,
        }
    }

    /// Construct and apply both rotations for the current step.
    pub(super) fn step(&mut self, s: BidiagStep) -> RotationStep {
        // Construct rotation P̂_k (Algorithm 2.8, LSQR side): eliminates
        // β_{k+1} against α̅_k. Outputs:
        //   * `p_hat.r` is the diagonal entry `ρ_k` of the upper-bidiagonal
        //     factor (returned below as `RotationStep::rho`).
        //   * `theta_new = ŝ_k · α_{k+1}` is the superdiagonal `θ_{k+1}`
        //     carried forward into P̄_k and the next iteration's P̂.
        //   * `alpha_bar_new = −ĉ_k · α_{k+1}` becomes `α̅_{k+1}`, the
        //     diagonal seed consumed by the next P̂.
        //   * `phi_bar_new = ŝ_k · φ̄_k` advances the LSQR transformed-RHS
        //     scalar used by the residual estimate `|φ̄|`.
        let p_hat = Givens::new(self.alpha_bar, s.beta);
        let theta_new = p_hat.s * s.alpha;
        let alpha_bar_new = -p_hat.c * s.alpha;
        let phi_bar_new = p_hat.s * self.phi_bar;

        // Construct rotation P̄_k (Algorithm 2.8, LSMR side): eliminates
        // θ_{k+1} against c̅_{k-1}·ρ_k. The off-diagonal feeding this
        // rotation is `θ̄_k = s̄_{k-1} · ρ_k`, which reads `self.s_bar` —
        // the *previous* iteration's `s̄`. The read MUST happen before we
        // commit `p_bar.s` into `self.s_bar` further down; otherwise we
        // would mix s̄_k into θ̄_k and break the recurrence.
        let theta_bar = self.s_bar * p_hat.r;
        let p_bar = Givens::new(self.c_bar * p_hat.r, theta_new);
        // `p_bar.r` is `ρ̄_k`. `zeta = c̄_k · ζ̄_k` is the committed
        // transformed-RHS scalar `ζ_k`.
        let zeta = p_bar.c * self.zeta_bar;
        // Sign comes from applying P̄_k to the transformed RHS: the
        // off-diagonal entry of `[[c̄, s̄], [−s̄, c̄]]` acting on `(ζ̄, 0)`
        // yields `−s̄_k · ζ̄_k` as the new ζ̄.
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
    pub(super) fn residual_estimate(&self) -> f64 {
        self.phi_bar.abs()
    }

    /// `|ζ̄|` — running estimate of `‖Aᵀ r_k‖` (Fong & Saunders).
    fn normal_eq_residual_estimate(&self) -> f64 {
        self.zeta_bar.abs()
    }
}

/// Vectors carried by the LSMR solution recurrence.
///
/// `(h, h̄)` are the auxiliary recurrence vectors that let us assemble `x`
/// without storing the full `V_k` basis.
pub(super) struct SolutionState {
    x: Vec<f64>,
    h: Vec<f64>,
    h_bar: Vec<f64>,
}

impl SolutionState {
    /// Initialize from the first normalized basis vector: `h₁ = v₁`,
    /// `x = 0`, `h̄₀ = 0`.
    pub(super) fn init(v1: &[f64]) -> Self {
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
    pub(super) fn update(&mut self, v: &[f64], curr: RotationStep, prev: RotationStep) {
        // Ratios consumed by the recurrence (Algorithm 2.8, "Update h̄, x, h").
        // Absolute f64::EPSILON guard rationale (Fong & Saunders use exact-zero):
        // the denominators here are products of Givens-rotation diagonals (ρ, ρ̄)
        // or single rotation diagonals — quantities that are O(1) in well-scaled
        // problems. A magnitude below f64::EPSILON for an O(1) quantity indicates
        // complete loss of significance, so the absolute guard is equivalent to a
        // relative guard against the rotation magnitudes. The `EPSILON` widening
        // is defensive; Fong & Saunders' reference implementation uses an exact-
        // zero check.
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

    pub(super) fn into_x(self) -> Vec<f64> {
        self.x
    }
}

/// Outcome of a single convergence test: continue iterating or stop.
pub(super) enum Stop {
    /// LSMR has not yet met the user-supplied tolerance.
    Continue,
    /// `‖Aᵀ r_k‖` (estimated via `‖ζ̄‖`) fell below tolerance.
    Converged,
}

/// Fong & Saunders' two stop criteria for LSMR plus the running `‖A‖_F²`
/// accumulator that the second criterion needs.
pub(super) struct ConvergenceState {
    /// `tol · ‖b‖`.
    abs_tol: f64,
    /// `tol`.
    rel_tol: f64,
    /// Running `‖A‖_F²` from the bidiagonal entries.
    a_norm_sq: f64,
}

impl ConvergenceState {
    pub(super) fn new(b_norm: f64, tol: f64, alpha1: f64) -> Self {
        Self {
            abs_tol: tol * b_norm,
            rel_tol: tol,
            a_norm_sq: alpha1 * alpha1,
        }
    }

    /// Fold a fresh bidiagonal step into the `‖A‖_F²` estimate.
    pub(super) fn observe(&mut self, s: BidiagStep) {
        self.a_norm_sq += s.alpha * s.alpha + s.beta * s.beta;
    }

    /// Check both stop criteria against the current scalar state.
    pub(super) fn check(&self, r: &LsmrRecurrenceState) -> Stop {
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
