//! LSMR scalar/vector recurrence consuming the bidiagonalization stream.
//!
//! Given `(α, β)` pairs from a [`super::bidiag::Bidiagonalization`], this
//! module builds the three Givens rotation chains (P̂_k, P̄_k, Q̃_k)
//! that yield Algorithm 2.8 of Fong & Saunders, advances the `(x, h, h̄)`
//! solution recurrence, and tracks the dual stopping criterion.

use super::bidiag::{update_threads, BidiagStep, LSMR_UPDATE_CHUNK};
use super::magnitude::Magnitude;
use super::LsmrStopReason;
use crate::SolveError;

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

        let all = (&mut self.h_bar[..], &mut self.h[..], &mut self.x[..], v);
        update_threads(n).for_each_chunk(all, LSMR_UPDATE_CHUNK, |_, (hb_c, h_c, x_c, v_c)| {
            chunk(hb_c, h_c, x_c, v_c)
        });
    }

    /// PROTOTYPE #607: the correction so far, without ending the pass.
    pub(super) fn x(&self) -> &[f64] {
        &self.x
    }

    pub(super) fn into_x(self) -> Vec<f64> {
        self.x
    }
}

/// What an audit's measurements decide.
#[derive(Debug)]
pub(super) enum Audit {
    /// The measured `r`, or `r` and `Aᵀr`, meet the tolerance.
    Certified(LsmrStopReason),
    /// Neither leg holds; `backward_error` is the measured `‖Aᵀr‖ / (L‖r‖)`.
    Refuted { backward_error: f64 },
}

/// Stopping criteria for one LSMR run: the certificate's tolerances, and the backward-error
/// trigger that asks for an audit.
#[derive(Clone, Copy)]
pub(super) struct ConvergenceCriteria {
    abs_tol: f64,
    rel_tol: f64,
    trigger_rel: f64,
}

impl ConvergenceCriteria {
    pub(super) fn new(reference_norm: f64, tol: f64) -> Self {
        Self {
            abs_tol: tol * reference_norm,
            rel_tol: tol,
            trigger_rel: tol,
        }
    }

    /// The trigger for the pass after a refuted audit: the tolerance over the measured ratio of
    /// the plain backward error to the metric one, which the metric estimate must undercut for
    /// the plain one to meet it. Never looser than the tolerance.
    pub(super) fn tightened(self, gap: f64) -> Self {
        Self {
            trigger_rel: self.rel_tol / gap.clamp(1.0, f64::MAX),
            ..self
        }
    }

    /// PROTOTYPE #607: the trigger a measured `metric / plain` backward-error ratio calls for, looser
    /// or tighter than the tolerance: the metric estimate the plain error meets the tolerance at.
    pub(super) fn recalibrated(self, metric_over_plain: f64) -> Self {
        Self {
            trigger_rel: self.rel_tol * metric_over_plain,
            ..self
        }
    }

    /// PROTOTYPE #607: the metric estimate at which a pass measures its plain backward error once,
    /// `PROBE_CAL`× the trigger (default 100; at most 1 disables it).
    pub(super) fn calibration_level(&self) -> Option<f64> {
        let k: f64 = std::env::var("PROBE_CAL")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(100.0);
        (k > 1.0).then_some(self.trigger_rel * k)
    }

    pub(super) fn rel_tol(&self) -> f64 {
        self.rel_tol
    }

    pub(super) fn abs_tol(&self) -> f64 {
        self.abs_tol
    }

    pub(super) fn start(self, alpha1: f64) -> ConvergenceState {
        ConvergenceState {
            criteria: self,
            alpha: alpha1,
            a_norm: alpha1,
        }
    }

    /// The stop a measured `‖r‖` certifies, asking `gradient` for the measured `‖Aᵀr‖` and a
    /// `floor ≤ ‖A‖₂` only when `‖r‖` alone misses: their ratio bounds the true backward error.
    pub(super) fn certify(
        &self,
        residual: f64,
        gradient: impl FnOnce() -> Result<(Magnitude, f64), SolveError>,
    ) -> Result<Audit, SolveError> {
        if residual <= self.abs_tol {
            return Ok(Audit::Certified(LsmrStopReason::ResidualTolerance));
        }
        let (normar, floor) = gradient()?;
        let backward_error = backward_error(normar, floor, residual);
        if std::env::var_os("PROBE_TRACE").is_some() {
            eprintln!(
                "[certify] backward_error={backward_error:.3e} rel_tol={:.1e} trigger_rel={:.3e}",
                self.rel_tol, self.trigger_rel
            );
        }
        Ok(if backward_error <= self.rel_tol {
            Audit::Certified(LsmrStopReason::NormalEquationTolerance)
        } else {
            Audit::Refuted { backward_error }
        })
    }
}

/// Mutable convergence observations for one LSMR run.
pub(super) struct ConvergenceState {
    criteria: ConvergenceCriteria,
    /// `α_k`, the diagonal of `B_k`'s column that the next `β_{k+1}` completes.
    alpha: f64,
    /// The largest column norm of `B_k` seen so far: a floor under `‖Â‖₂`, so the trigger cannot
    /// read a backward error smaller than the stream's own. Never squared, which would leave the
    /// double range from `‖Â‖ ≈ 1e±154`.
    a_norm: f64,
}

impl ConvergenceState {
    pub(super) fn observe(&mut self, s: BidiagStep) {
        self.a_norm = self.a_norm.max(self.alpha.hypot(s.beta)).max(s.alpha);
        self.alpha = s.alpha;
    }

    /// Whether the stream's own `‖r_k‖` or backward error meets its trigger; only measuring `x` can
    /// certify a stop.
    pub(super) fn triggers_audit(&self, r: &LsmrRecurrenceState) -> bool {
        let unit_residual = r.unit_residual();
        // `β₁` cancels from the backward error, so the `u₁` form stays in range.
        let normar = Magnitude::from(r.normal_eq_qr.normar());
        r.rhs_norm * unit_residual <= self.criteria.abs_tol
            || backward_error(normar, self.a_norm, unit_residual) <= self.criteria.trigger_rel
    }

    /// PROTOTYPE #607: the stream's metric backward-error estimate, as the trigger reads it.
    pub(super) fn backward_error_estimate(&self, r: &LsmrRecurrenceState) -> f64 {
        let normar = Magnitude::from(r.normal_eq_qr.normar());
        backward_error(normar, self.a_norm, r.unit_residual())
    }

    /// PROTOTYPE #607: swap in recalibrated criteria mid-pass.
    pub(super) fn set_criteria(&mut self, criteria: ConvergenceCriteria) {
        self.criteria = criteria;
    }

    /// The floor under `‖Â‖₂` this pass has seen.
    pub(super) fn a_norm(&self) -> f64 {
        self.a_norm
    }
}

/// `‖Aᵀr‖ / (‖A‖‖r‖)`; an uninformative denominator refuses, since clamping it would certify.
pub(super) fn backward_error(normar: Magnitude, a_norm: f64, residual: f64) -> f64 {
    if !(a_norm > 0.0 && a_norm.is_finite() && residual > 0.0 && residual.is_finite()) {
        return f64::INFINITY;
    }
    (normar / Magnitude::product(a_norm, residual)).to_f64()
}
