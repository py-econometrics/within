//! LSMR scalar/vector recurrence consuming the bidiagonalization stream.
//!
//! Given `(α, β)` pairs from a [`super::bidiag::Bidiagonalization`], this
//! module builds the two interleaved Givens rotation chains (P̂_k, P̄_k)
//! that yield Algorithm 2.8 of Fong & Saunders, advances the `(x, h, h̄)`
//! solution recurrence, and tracks the dual stopping criterion.

use super::bidiag::{BidiagStep, LSMR_PAR_THRESHOLD, LSMR_UPDATE_CHUNK};
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::prelude::{ParallelSlice, ParallelSliceMut};

/// Givens rotation built from `(a, b)` so applying it yields `(r, 0)`, `r = hypot(a, b)`.
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

/// Natural outputs of one rotation step, feeding straight into the `(x, h, h̄)` recurrence.
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
    /// First-iteration seed; `theta_bar = 0` vanishes the `t_hbar` ratio, matching `h̄₀ = 0`.
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

/// LSMR scalar state: `α̅`/`φ̄` for the LSQR-side chain, `c̅, s̅, ζ̄` for the LSMR-side one.
pub(super) struct LsmrRecurrenceState {
    alpha_bar: f64,
    phi_bar: f64,
    c_bar: f64,
    s_bar: f64,
    zeta_bar: f64,
    zeta0: f64,
}

impl LsmrRecurrenceState {
    pub(super) fn init(s1: BidiagStep) -> Self {
        let zeta_bar = s1.alpha * s1.beta;
        Self {
            alpha_bar: s1.alpha,
            phi_bar: s1.beta,
            c_bar: 1.0,
            s_bar: 0.0,
            zeta_bar,
            zeta0: zeta_bar.abs().max(f64::MIN_POSITIVE),
        }
    }

    /// Construct and apply both rotations for the current step.
    pub(super) fn step(&mut self, s: BidiagStep) -> RotationStep {
        let p_hat = Givens::new(self.alpha_bar, s.beta);
        let theta_new = p_hat.s * s.alpha;
        let alpha_bar_new = -p_hat.c * s.alpha;
        let phi_bar_new = p_hat.s * self.phi_bar;

        // `theta_bar` MUST be read before `p_bar.s` is committed, or s̄_k mixes into θ̄_k.
        let theta_bar = self.s_bar * p_hat.r;
        let p_bar = Givens::new(self.c_bar * p_hat.r, theta_new);
        let zeta = p_bar.c * self.zeta_bar;
        // The minus comes from `[[c̄, s̄], [−s̄, c̄]]` acting on `(ζ̄, 0)`.
        let zeta_bar_new = -p_bar.s * self.zeta_bar;

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

    /// `|φ̄|` — conservative `‖r_k‖` estimate; LSMR's residual is bounded by LSQR's.
    pub(super) fn residual_estimate(&self) -> f64 {
        self.phi_bar.abs()
    }

    /// `|ζ̄|` — running estimate of `‖Aᵀ r_k‖` (Fong & Saunders).
    fn normal_eq_residual_estimate(&self) -> f64 {
        self.zeta_bar.abs()
    }

    /// `|ζ̄ₖ| / |ζ̄₀|` — normal-equation residual relative to `‖Aᵀb‖`; the `ζ̄₀` clamp guards it.
    pub(super) fn relative_normal_eq_residual(&self) -> f64 {
        self.normal_eq_residual_estimate() / self.zeta0
    }
}

/// Vectors carried by the recurrence; `(h, h̄)` let `x` be built without the full `V_k` basis.
pub(super) struct SolutionState {
    x: Vec<f64>,
    h: Vec<f64>,
    h_bar: Vec<f64>,
}

impl SolutionState {
    /// Initialize from the first normalized basis vector: `h₁ = v₁`, `x = 0`, `h̄₀ = 0`.
    pub(super) fn init(v1: &[f64]) -> Self {
        Self {
            x: vec![0.0; v1.len()],
            h: v1.to_vec(),
            h_bar: vec![0.0; v1.len()],
        }
    }

    /// One `(x, h, h̄)` step; `v` must be normalized `v_{k+1}` and `prev` carries `(ρ, ρ̄)_{k-1}`.
    pub(super) fn update(&mut self, v: &[f64], curr: RotationStep, prev: RotationStep) {
        // Denominators are O(1) Givens diagonals, so an absolute `f64::EPSILON` guard suffices.
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
    /// `‖r_k‖` estimate fell below absolute tolerance.
    ResidualTolerance,
    /// `‖Aᵀ r_k‖` estimate fell below relative tolerance.
    NormalEquationTolerance,
}

/// Immutable stopping criteria for one LSMR run.
#[derive(Clone, Copy)]
pub(super) struct ConvergenceCriteria {
    abs_tol: f64,
    rel_tol: f64,
}

impl ConvergenceCriteria {
    pub(super) fn new(reference_norm: f64, tol: f64) -> Self {
        Self {
            abs_tol: tol * reference_norm,
            rel_tol: tol,
        }
    }

    pub(super) fn start(self, alpha1: f64) -> ConvergenceState {
        ConvergenceState {
            criteria: self,
            a_norm_sq: alpha1 * alpha1,
        }
    }
}

/// Mutable convergence observations for one LSMR run.
pub(super) struct ConvergenceState {
    criteria: ConvergenceCriteria,
    a_norm_sq: f64,
}

impl ConvergenceState {
    /// Fold a fresh bidiagonal step into the `‖A‖_F²` estimate.
    pub(super) fn observe(&mut self, s: BidiagStep) {
        self.a_norm_sq += s.alpha * s.alpha + s.beta * s.beta;
    }

    /// Check both stop criteria against the current scalar state.
    pub(super) fn check(&self, r: &LsmrRecurrenceState) -> Stop {
        let residual = r.residual_estimate();
        if residual <= self.criteria.abs_tol {
            return Stop::ResidualTolerance;
        }
        let a_norm = self.a_norm_sq.sqrt().max(f64::MIN_POSITIVE);
        let normar = r.normal_eq_residual_estimate();
        if normar / (a_norm * residual.max(f64::MIN_POSITIVE)) <= self.criteria.rel_tol {
            return Stop::NormalEquationTolerance;
        }
        Stop::Continue
    }
}
