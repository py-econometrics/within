//! LSMR scalar/vector recurrence consuming the bidiagonalization stream.
//!
//! Given `(α, β)` pairs from a [`super::bidiag::Bidiagonalization`], this
//! module builds the three Givens rotation chains (P̂_k, P̄_k, Q̃_k)
//! that yield Algorithm 2.8 of Fong & Saunders, advances the `(x, h, h̄)`
//! solution recurrence, and tracks the dual stopping criterion.

use super::bidiag::{BidiagStep, LSMR_PAR_THRESHOLD, LSMR_UPDATE_CHUNK};
use super::magnitude::Magnitude;
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

/// LSMR scalar state: the three rotation sequences of Fong & Saunders, run on `u₁ = b / β₁`.
pub(super) struct LsmrRecurrenceState {
    /// `β₁`; on `b` itself `ζ̄₁ = α₁β₁` leaves the double range while `α₁` and `β₁` do not.
    rhs_norm: f64,
    bidiag_qr: BidiagQr,
    normal_eq_qr: NormalEqQr,
    residual: ResidualChain,
}

impl LsmrRecurrenceState {
    pub(super) fn init(s1: BidiagStep) -> Self {
        Self {
            rhs_norm: s1.beta,
            bidiag_qr: BidiagQr {
                alpha_bar: s1.alpha,
                beta_dd: 1.0,
            },
            normal_eq_qr: NormalEqQr {
                c_bar: 1.0,
                s_bar: 0.0,
                zeta_bar: s1.alpha,
                zeta0: s1.alpha,
            },
            residual: ResidualChain {
                beta_d: 0.0,
                rho_d: 1.0,
                tau_tilde: 0.0,
                theta_tilde: 0.0,
                zeta: 0.0,
            },
        }
    }

    /// Advance all three rotation sequences by one bidiagonal step.
    pub(super) fn step(&mut self, s: BidiagStep) -> RotationStep {
        let qr = self.bidiag_qr.advance(s);
        let rot = self.normal_eq_qr.advance(qr);
        self.residual.advance(qr.beta_hat, rot);
        rot
    }

    /// Running estimate of LSMR's own `‖r_k‖`.
    pub(super) fn residual_estimate(&self) -> f64 {
        self.rhs_norm * self.unit_residual()
    }

    /// `‖r_k‖` for `u₁`, where `‖r_k‖ ≤ 1`.
    fn unit_residual(&self) -> f64 {
        self.residual.normr(self.bidiag_qr.beta_dd)
    }

    /// Running estimate of `‖Âᵀ r_k‖` for `b`, a product that may leave the double range.
    pub(super) fn normal_eq_residual_estimate(&self) -> Magnitude {
        Magnitude::product(self.rhs_norm, self.normal_eq_qr.normar())
    }

    pub(super) fn relative_normal_eq_residual(&self) -> f64 {
        self.normal_eq_qr.normar() / self.normal_eq_qr.zeta0
    }
}

/// `P̂_k`: QR of the lower-bidiagonal `B_k` into `R_k`, carrying `ᾱ_k` and the rotated RHS `β̈_k`.
struct BidiagQr {
    alpha_bar: f64,
    /// `β̈_k`: LSQR's `φ̄_k`, the part of `β₁e₁` that `R_k` leaves unexplained.
    beta_dd: f64,
}

/// One `P̂_k`: the `R_k` column `(ρ_k, θ_{k+1})` and the rotated RHS entry `β̂_k`.
#[derive(Clone, Copy)]
struct BidiagQrStep {
    rho: f64,
    theta_new: f64,
    beta_hat: f64,
}

impl BidiagQr {
    fn advance(&mut self, s: BidiagStep) -> BidiagQrStep {
        let p_hat = Givens::new(self.alpha_bar, s.beta);
        // Fong & Saunders' sign for `ᾱ`; LSQR's `−c α` would flip `β̂` on every other step.
        self.alpha_bar = p_hat.c * s.alpha;
        let beta_hat = p_hat.c * self.beta_dd;
        self.beta_dd *= -p_hat.s;
        BidiagQrStep {
            rho: p_hat.r,
            theta_new: p_hat.s * s.alpha,
            beta_hat,
        }
    }
}

/// `P̄_k`: QR of `R_kᵀ`, carrying `(c̄, s̄)` and the rotated RHS `ζ̄_k`, `|ζ̄_k| ≈ ‖Aᵀr_k‖`.
struct NormalEqQr {
    c_bar: f64,
    s_bar: f64,
    zeta_bar: f64,
    /// `|ζ̄₀| = α₁ = ‖Âᵀu₁‖`, positive since a zero `α₁` never reaches the recurrence.
    zeta0: f64,
}

impl NormalEqQr {
    fn advance(&mut self, qr: BidiagQrStep) -> RotationStep {
        let BidiagQrStep { rho, theta_new, .. } = qr;
        // `theta_bar` MUST be read before `s̄` is committed, or s̄_k mixes into θ̄_k.
        let theta_bar = self.s_bar * rho;
        let p_bar = Givens::new(self.c_bar * rho, theta_new);
        let zeta = p_bar.c * self.zeta_bar;
        // The minus comes from `[[c̄, s̄], [−s̄, c̄]]` acting on `(ζ̄, 0)`.
        self.zeta_bar *= -p_bar.s;
        self.c_bar = p_bar.c;
        self.s_bar = p_bar.s;
        RotationStep {
            rho,
            rho_bar: p_bar.r,
            theta_new,
            theta_bar,
            zeta,
        }
    }

    /// `|ζ̄ₖ|` — running estimate of `‖Aᵀ r_k‖` for `u₁` (Fong & Saunders).
    fn normar(&self) -> f64 {
        self.zeta_bar.abs()
    }
}

/// Fong & Saunders' third rotation chain `Q̃` (§3.4, as in SciPy's `lsmr`), tracking `‖r_k‖`.
struct ResidualChain {
    beta_d: f64,
    rho_d: f64,
    tau_tilde: f64,
    theta_tilde: f64,
    zeta: f64,
}

impl ResidualChain {
    /// `Q̃` triangularizes the second column of `R̄ₖ`; `τ̃` solves the new triangle forward.
    fn advance(&mut self, beta_hat: f64, rot: RotationStep) {
        let RotationStep {
            rho_bar,
            theta_bar,
            zeta,
            ..
        } = rot;

        let theta_tilde_prev = self.theta_tilde;
        let q_tilde = Givens::new(self.rho_d, theta_bar);
        self.theta_tilde = q_tilde.s * rho_bar;
        self.rho_d = q_tilde.c * rho_bar;
        self.beta_d = -q_tilde.s * self.beta_d + q_tilde.c * beta_hat;

        self.tau_tilde = ratio(self.zeta - theta_tilde_prev * self.tau_tilde, q_tilde.r);
        self.zeta = zeta;
    }

    /// `‖r_k‖ = ‖(β̈ₖ, β̇ₖ − τ̇ₖ)‖`.
    fn normr(&self, beta_dd: f64) -> f64 {
        let tau_d = ratio(self.zeta - self.theta_tilde * self.tau_tilde, self.rho_d);
        f64::hypot(self.beta_d - tau_d, beta_dd)
    }
}

/// `a / b`, reading an exactly vanished denominator as an empty chain rather than a NaN one.
fn ratio(a: f64, b: f64) -> f64 {
    if b == 0.0 {
        0.0
    } else {
        a / b
    }
}

/// Vectors carried by the recurrence; `(h, h̄)` let `x` be built without the full `V_k` basis.
pub(super) struct SolutionState {
    /// `β₁`, restoring to `x` the scale the recurrence divided out of `b`.
    rhs_norm: f64,
    x: Vec<f64>,
    h: Vec<f64>,
    h_bar: Vec<f64>,
}

impl SolutionState {
    /// Initialize from the first normalized basis vector: `h₁ = v₁`, `x = 0`, `h̄₀ = 0`.
    pub(super) fn init(v1: &[f64], rhs_norm: f64) -> Self {
        Self {
            rhs_norm,
            x: vec![0.0; v1.len()],
            h: v1.to_vec(),
            h_bar: vec![0.0; v1.len()],
        }
    }

    /// One `(x, h, h̄)` step; `v` must be normalized `v_{k+1}` and `prev` carries `(ρ, ρ̄)_{k-1}`.
    pub(super) fn update(&mut self, v: &[f64], curr: RotationStep, prev: RotationStep) {
        // `ρρ̄` scales with `‖A‖²`, and `ζ/(ρρ̄)` can underflow where `β₁ζ/(ρρ̄)` does not.
        let t_x = if curr.rho == 0.0 || curr.rho_bar == 0.0 {
            0.0
        } else {
            let unsigned = Magnitude::product(self.rhs_norm, curr.zeta.abs())
                / Magnitude::product(curr.rho, curr.rho_bar);
            curr.zeta.signum() * unsigned.to_f64()
        };
        let t_hbar = ratio(curr.theta_bar, prev.rho) * ratio(curr.rho, prev.rho_bar);
        let t_h = ratio(curr.theta_new, curr.rho);

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
            a_norm: alpha1,
        }
    }

    /// Honest estimate drift is O(ε); a collapsed one misses by orders (van der Vorst & Ye 2000).
    fn residual_gap(&self) -> f64 {
        CERTIFICATION_SLACK * self.abs_tol
    }

    /// Whether the recomputed residual corroborates the recurrence's own estimate.
    pub(super) fn corroborates_residual(&self, recomputed: f64, estimate: f64) -> bool {
        (recomputed - estimate).abs() <= self.residual_gap()
    }

    /// An annihilated `α₁` certifies via the residual, or via a lower `‖A‖` bound that overstates.
    pub(super) fn corroborates(&self, normr: f64, normar: Magnitude, a_norm_below: f64) -> bool {
        normr <= self.residual_gap()
            || backward_error(normar, a_norm_below, normr) <= CERTIFICATION_SLACK * self.rel_tol
    }
}

/// Mutable convergence observations for one LSMR run.
pub(super) struct ConvergenceState {
    criteria: ConvergenceCriteria,
    /// `‖A‖_F` estimate, never squared: `‖A‖²` leaves the double range from `‖A‖ ≈ 1e±154`.
    a_norm: f64,
}

impl ConvergenceState {
    pub(super) fn observe(&mut self, s: BidiagStep) {
        self.a_norm = self.a_norm.hypot(s.alpha).hypot(s.beta);
    }

    /// Check both stop criteria against the current scalar state.
    pub(super) fn check(&self, r: &LsmrRecurrenceState) -> Stop {
        let unit_residual = r.unit_residual();
        if r.rhs_norm * unit_residual <= self.criteria.abs_tol {
            return Stop::ResidualTolerance;
        }
        // The stream's own backward error; `β₁` cancels, so the `u₁` form stays in range.
        let normar = Magnitude::from(r.normal_eq_qr.normar());
        let ratio = backward_error(normar, self.a_norm, unit_residual);
        if ratio <= self.criteria.rel_tol {
            return Stop::NormalEquationTolerance;
        }
        Stop::Continue
    }
}

/// `‖Aᵀr‖ / (‖A‖‖r‖)`; an uninformative denominator refuses, since clamping it would certify.
pub(super) fn backward_error(normar: Magnitude, a_norm: f64, residual: f64) -> f64 {
    if !(a_norm > 0.0 && a_norm.is_finite() && residual > 0.0 && residual.is_finite()) {
        return f64::INFINITY;
    }
    (normar / Magnitude::product(a_norm, residual)).to_f64()
}

/// Collapsed recurrences miss by orders of magnitude; the slack absorbs ordinary estimate drift.
const CERTIFICATION_SLACK: f64 = 100.0;
