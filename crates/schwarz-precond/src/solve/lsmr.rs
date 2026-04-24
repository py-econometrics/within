//! Modified LSMR for rectangular least-squares.
//!
//! Solves `min ‖b − A x‖₂` with an optional preconditioner `M ≈ A^T A`.
//! Uses the Modified Golub-Kahan Bidiagonalization (MGK) which requires
//! only **one** `M⁻¹` application per iteration — no square-root
//! factorization of `M` is needed.
//!
//! # Modified Golub-Kahan Bidiagonalization
//!
//! The standard preconditioned Golub-Kahan process requires two triangular
//! solves per iteration (`L⁻¹` and `L⁻ᵀ` where `M = LᵀL`). The modified
//! version introduces an auxiliary vector `p̃ = M ṽ` and exploits the
//! identity `‖ṽ‖_M = √⟨ṽ, p̃⟩` to merge both solves into a single
//! `M⁻¹` application.
//!
//! Per iteration:
//! 1. Forward matvec: `A ṽ_k`
//! 2. Adjoint matvec: `Aᵀ u_{k+1}`
//! 3. Preconditioner solve: `M⁻¹ p̃` (the only place `M` appears)
//! 4. M-norm via dot product: `α = √⟨ṽ, p̃⟩`
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
use crate::{IdentityOperator, Operator, SolveError};

/// Below this count the per-iteration vector kernels run sequentially —
/// rayon wake/steal overhead would dominate otherwise. Matches the threshold
/// used by `sparse_matrix::CsrMatrix::matvec_add`.
const LSMR_PAR_THRESHOLD: usize = 10_000;
/// Per-worker chunk size for the parallel vector kernels. Tuned to keep each
/// chunk's work above rayon dispatch overhead while staying L1-resident —
/// sizing chunks to `n / n_threads` instead regresses at 5M+ DOFs because
/// per-thread chunks blow L1/L2 and workers stream at DRAM bandwidth.
const LSMR_UPDATE_CHUNK: usize = 4096;

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

/// Result of an MLSMR solve.
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
// Pre-allocated buffers for the Modified Golub-Kahan Bidiagonalization
// ---------------------------------------------------------------------------

/// Pre-allocated buffers for the Modified Golub-Kahan Bidiagonalization.
struct MgkBuffers {
    /// Current u vector in observation space (length m).
    u: Vec<f64>,
    /// Current ṽ vector in DOF space (length n).
    v: Vec<f64>,
    /// Recurrence vector p̃ in DOF space (length n).
    /// Invariant: p̃ = M ṽ (up to normalization).
    p_tilde: Vec<f64>,
    /// Scratch for A·v (length m).
    av: Vec<f64>,
    /// Scratch for A^T·u (length n).
    atu: Vec<f64>,
}

impl MgkBuffers {
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

// ---------------------------------------------------------------------------
// Modified Golub-Kahan initialization
// ---------------------------------------------------------------------------

/// Initialize the Modified Golub-Kahan Bidiagonalization.
///
/// Computes β₁, u₁, p̃₁, ṽ₁, α₁. Returns (α₁, β₁).
fn mgk_init<A: Operator + ?Sized, M: Operator + ?Sized>(
    operator: &A,
    preconditioner: &M,
    b: &[f64],
    bufs: &mut MgkBuffers,
) -> Result<(f64, f64), SolveError> {
    // β₁ = ‖b‖, u₁ = b / β₁
    let beta = vec_norm(b);
    if beta > 0.0 {
        let inv = 1.0 / beta;
        for (ui, &bi) in bufs.u.iter_mut().zip(b) {
            *ui = bi * inv;
        }
    }

    // p̃ = A^T u₁
    operator.try_apply_adjoint(&bufs.u, &mut bufs.p_tilde)?;

    // ṽ₁ = M⁻¹ p̃
    preconditioner.try_apply(&bufs.p_tilde, &mut bufs.v)?;

    // α₁ = √⟨ṽ₁, p̃⟩ (M-norm via dot product trick)
    let vp = dot(&bufs.v, &bufs.p_tilde);
    let alpha = if vp > 0.0 { vp.sqrt() } else { 0.0 };

    // Normalize v (solvers clone it for recurrence vectors), but leave
    // p_tilde unnormalized (= α₁·p̃₁). The first mgk_step compensates
    // via the `alpha` parameter.
    if alpha > 0.0 {
        let inv = 1.0 / alpha;
        for vi in bufs.v.iter_mut() {
            *vi *= inv;
        }
    }

    Ok((alpha, beta))
}

// ---------------------------------------------------------------------------
// Modified Golub-Kahan step
// ---------------------------------------------------------------------------

fn mgk_step<A: Operator + ?Sized, M: Operator + ?Sized>(
    operator: &A,
    preconditioner: &M,
    bufs: &mut MgkBuffers,
    alpha: f64,
    beta_prev_inv: f64,
) -> Result<(f64, f64), SolveError> {
    // ũ_{k+1} = A ṽ_k − (α_k / β_k) ũ_k, with β_{k+1} = ‖ũ_{k+1}‖.
    let scale = -(alpha * beta_prev_inv);
    operator.try_apply(&bufs.v, &mut bufs.av)?;
    let beta_sq = axpy_with_sq_norm(&mut bufs.u, &bufs.av, scale);
    let beta = beta_sq.sqrt();
    let beta_inv = if beta > 0.0 { 1.0 / beta } else { 0.0 };

    // A^T on unnormalized ũ: result is β × the normalized adjoint.
    // p̃_stored = α_prev · p̃_normalized, so β/α_prev cancels the α_prev
    // factor in p̃_stored.
    operator.try_apply_adjoint(&bufs.u, &mut bufs.atu)?;
    let p_coeff = beta / alpha;
    axpby(&mut bufs.p_tilde, &bufs.atu, beta_inv, -p_coeff);

    // ṽ_{k+1} = M⁻¹ p̃
    preconditioner.try_apply(&bufs.p_tilde, &mut bufs.v)?;

    // α_{k+1} = √⟨ṽ_{k+1}, p̃⟩
    let vp = par_dot(&bufs.v, &bufs.p_tilde);
    let alpha_new = if vp > 0.0 { vp.sqrt() } else { 0.0 };

    // v and p_tilde are left unnormalized (= α_{k+1} · normalized). The
    // solver normalizes v in its update loop; p_tilde is compensated in
    // the next mgk_step via the `alpha` parameter.

    Ok((alpha_new, beta))
}

// ---------------------------------------------------------------------------
// MLSMR
// ---------------------------------------------------------------------------

/// Modified LSMR with optional preconditioner `M ≈ A^T A`.
///
/// Solves `min ‖b − A x‖₂` using the Modified Golub-Kahan
/// Bidiagonalization. Minimizes the normal-equation residual `‖Aᵀ r‖`,
/// giving smoother convergence than LSQR.
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
    match preconditioner {
        None => {
            let id = IdentityOperator::new(operator.ncols());
            mlsmr_solve(operator, &id, b, tol, maxiter)
        }
        Some(m) => mlsmr_solve(operator, m, b, tol, maxiter),
    }
}

/// Core MLSMR implementation.
///
/// LSMR applies two sets of Givens rotations to the bidiagonal B_k:
/// 1. First rotation eliminates the sub-diagonal (same as LSQR)
/// 2. Second rotation ("bar" rotation) constructs the LSMR solution
///
/// The solution is built incrementally via h̄-vector recurrence.
fn mlsmr_solve<A: Operator + ?Sized, M: Operator + ?Sized>(
    operator: &A,
    preconditioner: &M,
    b: &[f64],
    tol: f64,
    maxiter: usize,
) -> Result<LsmrResult, SolveError> {
    let m = operator.nrows();
    let n = operator.ncols();
    debug_assert_eq!(b.len(), m);

    let b_norm = vec_norm(b);
    if b_norm < f64::EPSILON {
        return Ok(LsmrResult {
            x: vec![0.0; n],
            converged: true,
            iterations: 0,
            residual_norm: 0.0,
        });
    }

    let mut bufs = MgkBuffers::new(m, n);
    let (mut alpha, beta) = mgk_init(operator, preconditioner, b, &mut bufs)?;

    if alpha == 0.0 {
        return Ok(LsmrResult {
            x: vec![0.0; n],
            converged: true,
            iterations: 0,
            residual_norm: b_norm,
        });
    }

    let abs_tol = tol * b_norm;

    // Solution and recurrence vectors
    let mut x = vec![0.0; n];
    let mut h = bufs.v.clone(); // h₁ = ṽ₁
    let mut h_bar = vec![0.0; n]; // h̄₀ = 0

    // First rotation state
    let mut alpha_bar = alpha; // ᾱ₁ = α₁
    let mut phi_bar = beta; // φ̄₁ = β₁ (tracks ‖r‖ from LSQR QR)

    // Second ("bar") rotation state — for LSMR solution path
    let mut c_bar: f64 = 1.0; // c̄₀ = 1
    let mut s_bar: f64 = 0.0; // s̄₀ = 0
    let mut zeta_bar = alpha * beta; // ζ̄₁ = α₁ β₁

    // Previous ρ and ρ̄ for h̄ recurrence
    let mut rho_prev: f64 = 1.0; // ρ_{k-1} (dummy for first iteration)
    let mut rho_bar_prev: f64 = 1.0; // ρ̄_{k-1} (dummy for first iteration)

    // ‖A‖_F estimate from bidiagonal entries
    let mut a_norm_sq = alpha * alpha;

    // Lazy normalization: u starts normalized after mgk_init.
    let mut beta_prev_inv = 1.0;

    for itn in 1..=maxiter {
        let (alpha_new, beta_new) =
            mgk_step(operator, preconditioner, &mut bufs, alpha, beta_prev_inv)?;
        beta_prev_inv = if beta_new > 0.0 { 1.0 / beta_new } else { 0.0 };

        // Update ‖A‖_F estimate
        a_norm_sq += alpha_new * alpha_new + beta_new * beta_new;

        // --- First Givens rotation: eliminate β_{k+1} (same as LSQR) ---
        let rho = f64::hypot(alpha_bar, beta_new);
        let c1 = if rho > 0.0 { alpha_bar / rho } else { 1.0 };
        let s1 = if rho > 0.0 { beta_new / rho } else { 0.0 };
        let theta_new = s1 * alpha_new;
        let alpha_bar_new = -c1 * alpha_new;
        let phi_bar_new = s1 * phi_bar;

        // --- Second Givens rotation: LSMR "bar" rotation ---
        let theta_bar = s_bar * rho;
        let rho_bar = f64::hypot(c_bar * rho, theta_new);
        let c_bar_new = if rho_bar > 0.0 {
            c_bar * rho / rho_bar
        } else {
            1.0
        };
        let s_bar_new = if rho_bar > 0.0 {
            theta_new / rho_bar
        } else {
            0.0
        };

        // ζ update
        let zeta = c_bar_new * zeta_bar;
        let zeta_bar_new = -s_bar_new * zeta_bar;

        // --- Fused solution update: h̄, x, h, and v-normalization in one pass ---
        // mgk_step leaves v unnormalized (= α·ṽ); we normalize on the fly.
        // The itn==1 special case is unnecessary: θ̄₁ = s̄₀·ρ = 0·ρ = 0
        // and ρ_prev=ρ̄_prev=1, so t_hbar=0 and the general formula
        // reduces to h̄ = h, matching the first-iteration semantics.
        {
            let x_denom = rho * rho_bar;
            let t_x = if x_denom.abs() > f64::EPSILON {
                zeta / x_denom
            } else {
                0.0
            };
            let hbar_denom = rho_prev * rho_bar_prev;
            let t_hbar = if hbar_denom.abs() > f64::EPSILON {
                theta_bar * rho / hbar_denom
            } else {
                0.0
            };
            let t_h = if rho.abs() > f64::EPSILON {
                theta_new / rho
            } else {
                0.0
            };
            let alpha_new_inv = if alpha_new > 0.0 {
                1.0 / alpha_new
            } else {
                0.0
            };

            let update_chunk =
                |hb_c: &mut [f64], h_c: &mut [f64], x_c: &mut [f64], v_c: &mut [f64]| {
                    for (((hbi, hi), xi), vi) in hb_c
                        .iter_mut()
                        .zip(h_c.iter_mut())
                        .zip(x_c.iter_mut())
                        .zip(v_c.iter_mut())
                    {
                        *vi *= alpha_new_inv; // normalize v in place
                        let h_old = *hi;
                        let hb = h_old - t_hbar * *hbi;
                        *hbi = hb;
                        *xi += t_x * hb;
                        *hi = *vi - t_h * h_old;
                    }
                };

            if n >= LSMR_PAR_THRESHOLD {
                h_bar
                    .par_chunks_mut(LSMR_UPDATE_CHUNK)
                    .zip(h.par_chunks_mut(LSMR_UPDATE_CHUNK))
                    .zip(x.par_chunks_mut(LSMR_UPDATE_CHUNK))
                    .zip(bufs.v.par_chunks_mut(LSMR_UPDATE_CHUNK))
                    .for_each(|(((hb_c, h_c), x_c), v_c)| update_chunk(hb_c, h_c, x_c, v_c));
            } else {
                update_chunk(&mut h_bar, &mut h, &mut x, &mut bufs.v);
            }
        }

        // --- Update state for next iteration ---
        alpha = alpha_new;
        alpha_bar = alpha_bar_new;
        phi_bar = phi_bar_new;
        c_bar = c_bar_new;
        s_bar = s_bar_new;
        zeta_bar = zeta_bar_new;
        rho_prev = rho;
        rho_bar_prev = rho_bar;

        // --- Convergence check ---
        // Use |φ̄| as a conservative estimate of ‖r_k‖.
        // The LSMR solution has ‖r_LSMR‖ ≤ ‖r_LSQR‖ = |φ̄|.
        let residual_norm = phi_bar.abs();
        if residual_norm <= abs_tol {
            return Ok(LsmrResult {
                x,
                converged: true,
                iterations: itn,
                residual_norm,
            });
        }

        // Normal-equation residual: ‖A^T r_k‖ ≈ |ζ̄_{k+1}| (Fong & Saunders).
        let a_norm = a_norm_sq.sqrt().max(f64::MIN_POSITIVE);
        let normar = zeta_bar.abs();
        if normar / (a_norm * residual_norm.max(f64::MIN_POSITIVE)) <= tol {
            return Ok(LsmrResult {
                x,
                converged: true,
                iterations: itn,
                residual_norm,
            });
        }

        // Breakdown: α → 0
        if alpha == 0.0 {
            return Ok(LsmrResult {
                x,
                converged: true,
                iterations: itn,
                residual_norm,
            });
        }
    }

    Ok(LsmrResult {
        x,
        converged: false,
        iterations: maxiter,
        residual_norm: phi_bar.abs(),
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

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
    /// Approximates (A^T A)⁻¹ = diag(2, 2, 1)⁻¹
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
}
