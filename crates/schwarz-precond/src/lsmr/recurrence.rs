//! LSMR scalar/vector recurrence consuming the bidiagonalization stream.
//!
//! Given `(α, β)` pairs from a [`super::bidiag::Bidiagonalization`], this
//! module builds the two interleaved Givens rotation chains (P̂_k, P̄_k)
//! that yield Algorithm 2.8 of Fong & Saunders, advances the `(x, h, h̄)`
//! solution recurrence, and tracks the dual stopping criterion.

use crate::SolveError;

use super::bidiag::{
    BidiagStep, Certificate, NormalEquationResidual, LSMR_PAR_THRESHOLD, LSMR_UPDATE_CHUNK,
};
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

/// LSMR scalar state: the three rotation sequences of Fong & Saunders.
pub(super) struct LsmrRecurrenceState {
    bidiag_qr: BidiagQr,
    normal_eq_qr: NormalEqQr,
    residual: ResidualChain,
}

impl LsmrRecurrenceState {
    pub(super) fn init(s1: BidiagStep) -> Self {
        Self {
            bidiag_qr: BidiagQr {
                alpha_bar: s1.alpha,
                beta_dd: s1.beta,
            },
            normal_eq_qr: NormalEqQr {
                c_bar: 1.0,
                s_bar: 0.0,
                zeta_bar: s1.alpha * s1.beta,
                zeta0: super::bidiag::reference_norm(s1.alpha, s1.beta),
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
        self.residual.normr(self.bidiag_qr.beta_dd)
    }

    fn normal_eq_residual_estimate(&self) -> f64 {
        self.normal_eq_qr.normar()
    }

    pub(super) fn relative_normal_eq_residual(&self) -> f64 {
        self.normal_eq_qr.relative_normar()
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
    /// `|ζ̄₀| = ‖Âᵀb‖`, clamped positive; the reference for relative NE residuals.
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

    /// `|ζ̄ₖ|` — running estimate of `‖Aᵀ r_k‖` (Fong & Saunders).
    fn normar(&self) -> f64 {
        self.zeta_bar.abs()
    }

    /// `|ζ̄ₖ| / |ζ̄₀|` — normal-equation residual relative to `‖Aᵀb‖`; the `ζ̄₀` clamp guards it.
    fn relative_normar(&self) -> f64 {
        self.normar() / self.zeta0
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
        // One diagonal at a time: `ρρ̄` scales with `‖A‖²` and leaves the double range first.
        let t_x = ratio(ratio(curr.zeta, curr.rho), curr.rho_bar);
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
            a_norm_sq: alpha1 * alpha1,
        }
    }

    /// The audit left once a stream reports no gradient at all, where the only drop measurable is
    /// between the two metrics: the reference within each is the numerator itself, or a clamp.
    /// `a_norm_below` replaces the `‖A‖` the stream never got to estimate; being a lower bound it
    /// can only overstate the backward error, so it never certifies one the true norm would not.
    pub(super) fn corroborated(
        &self,
        cert: &Certificate,
        a_norm_below: impl FnOnce() -> Result<f64, SolveError>,
    ) -> Result<bool, SolveError> {
        if self.solved(cert) || drops_agree(cert) {
            return Ok(true);
        }
        // Only a metric can hide a direction, and only then is the bound worth an apply.
        let Some(raw) = cert.normar_raw else {
            return Ok(false);
        };
        Ok(backward_error(raw.norm, a_norm_below()?, cert.normr)
            <= CERTIFICATION_SLACK * self.rel_tol)
    }

    /// The residual alone meets the tolerance, whatever the normal-equation legs report.
    fn solved(&self, cert: &Certificate) -> bool {
        cert.normr <= CERTIFICATION_SLACK * self.abs_tol
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

    /// The stream's own backward error, from its accumulated `‖A‖_F` estimate.
    fn ne_ratio(&self, residual: f64, normar: f64) -> f64 {
        backward_error(normar, self.a_norm_sq.sqrt(), residual)
    }

    /// Check both stop criteria against the current scalar state.
    pub(super) fn check(&self, r: &LsmrRecurrenceState) -> Stop {
        let residual = r.residual_estimate();
        if residual <= self.criteria.abs_tol {
            return Stop::ResidualTolerance;
        }
        if self.ne_ratio(residual, r.normal_eq_residual_estimate()) <= self.criteria.rel_tol {
            return Stop::NormalEquationTolerance;
        }
        Stop::Continue
    }

    /// True-residual audit of a tolerance stop (cf. van der Vorst & Ye, SISC 22(3), 2000).
    pub(super) fn certified(&self, cert: &Certificate) -> bool {
        if self.criteria.solved(cert) {
            return true;
        }
        let rel = CERTIFICATION_SLACK * self.criteria.rel_tol;
        let dropped = |ne: &NormalEquationResidual| ne.norm <= rel * ne.reference;
        // `normr → 0` degenerates the ratio test; the drop of `‖Âᵀr‖` vs its start certifies.
        let metric = dropped(&cert.normar) || self.ne_ratio(cert.normr, cert.normar.norm) <= rel;
        // A metric that annihilates a direction cannot audit it, so the plain norm must also pass.
        // `ne_ratio` cannot audit it: its `‖A‖` is preconditioned, so `M⁻¹`'s scale deflates it.
        metric && (cert.normar_raw.is_none_or(|raw| dropped(&raw)) || drops_agree(cert))
    }
}

/// `‖Aᵀr‖ / (‖A‖‖r‖)`, refusing outright on a denominator carrying no information: clamping one
/// up would flatter the ratio into certifying an unsolved stop. The product is the accurate form,
/// rounding once; only where it leaves the float range does dividing in turn beat it.
pub(super) fn backward_error(normar: f64, a_norm: f64, residual: f64) -> f64 {
    if !(a_norm > 0.0 && a_norm.is_finite() && residual > 0.0 && residual.is_finite()) {
        return f64::INFINITY;
    }
    let denominator = a_norm * residual;
    if denominator > 0.0 && denominator.is_finite() {
        return normar / denominator;
    }
    normar / a_norm / residual
}

/// The drop outside the stream's metric against the drop inside it, carrying neither the `A` nor
/// the `M` scale; vacuous for a stream with no metric to corroborate.
fn drops_agree(cert: &Certificate) -> bool {
    cert.normar_raw
        .is_none_or(|raw| raw.relative() <= CERTIFICATION_SLACK * cert.normar.relative())
}

/// Collapsed recurrences miss by orders of magnitude; the slack absorbs ordinary estimate drift.
const CERTIFICATION_SLACK: f64 = 100.0;
