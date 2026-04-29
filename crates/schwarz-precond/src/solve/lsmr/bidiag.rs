//! Bidiagonalization stream feeding the LSMR recurrence.
//!
//! Produces a sequence of `(α, β)` scalars and the matching normalized basis
//! vector `v_k` from the operator (and, optionally, an `M ≈ AᵀA`
//! preconditioner). All vector kernels and the windowed reorthogonalization
//! buffers used to maintain Golub-Kahan basis quality live here as private
//! helpers — they are implementation detail of this stream, not an
//! independent subsystem.

use crate::solve::dot;
use crate::solve::lsmr::{LSMR_PAR_THRESHOLD, LSMR_UPDATE_CHUNK};
use crate::{Operator, SolveError};
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::prelude::{ParallelSlice, ParallelSliceMut};

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

/// Ring buffer of past `v` vectors for windowed modified Gram-Schmidt in the
/// standard Euclidean inner product. Used by [`GolubKahan`].
///
/// The disabled state is encoded as `Option<LocalReorth> = None` on the
/// owning buffer; this type is only ever constructed with a positive
/// capacity.
struct LocalReorth {
    slots: Vec<f64>,
    n: usize,
    next: usize,
    count: usize,
}

impl LocalReorth {
    /// Returns `None` when no reorthogonalization is requested (the effective
    /// capacity collapses to zero). The bidiagonalization can produce at most
    /// `min(m, n)` linearly independent basis vectors before α or β hits zero,
    /// so the ring is capped at `min(m, n)`.
    fn new(m: usize, n: usize, local_size: usize) -> Option<Self> {
        let cap = local_size.min(m.min(n));
        if cap == 0 {
            return None;
        }
        Some(Self {
            slots: vec![0.0; cap * n],
            n,
            next: 0,
            count: 0,
        })
    }

    /// Modified Gram-Schmidt sweep over stored slots in chronological order
    /// (oldest first). Subtracts the projection of `y` onto each stored
    /// vector.
    fn reorthogonalize(&self, y: &mut [f64]) {
        let cap = self.capacity();
        let start = if self.count < cap { 0 } else { self.next };
        for i in 0..self.count {
            let v_j = self.slot((start + i) % cap);
            let c = par_dot(y, v_j);
            axpby(y, v_j, -c, 1.0);
        }
    }

    /// Copy `v` into the next slot, advancing the ring index and saturating
    /// the count at capacity.
    fn push(&mut self, v: &[f64]) {
        let cap = self.capacity();
        let next = self.next;
        self.slot_mut(next).copy_from_slice(v);
        self.next = (self.next + 1) % cap;
        if self.count < cap {
            self.count += 1;
        }
    }

    #[inline]
    fn capacity(&self) -> usize {
        self.slots.len() / self.n
    }

    #[inline]
    fn slot(&self, slot: usize) -> &[f64] {
        let start = slot * self.n;
        &self.slots[start..start + self.n]
    }

    #[inline]
    fn slot_mut(&mut self, slot: usize) -> &mut [f64] {
        let start = slot * self.n;
        &mut self.slots[start..start + self.n]
    }
}

/// Ring buffer of past `(v, p̃ = M v)` pairs for windowed modified
/// Gram-Schmidt in the M-weighted inner product. Used by
/// [`ModifiedGolubKahan`]: `v` is M-orthogonal, not Euclidean, so the MGS
/// coefficient is `⟨v_new, M v_j⟩ = ⟨v_new, p̃_j⟩`. Subtracting the same
/// coefficient from both `v` and `p̃` in lockstep preserves the
/// `p̃ = M v` invariant on the recurrence vectors.
///
/// As with [`LocalReorth`], the disabled state is encoded as
/// `Option<ModifiedLocalReorth> = None` on the owning buffer.
struct ModifiedLocalReorth {
    v: Vec<f64>,
    p_tilde: Vec<f64>,
    n: usize,
    next: usize,
    count: usize,
}

impl ModifiedLocalReorth {
    /// Returns `None` when no reorthogonalization is requested. See
    /// [`LocalReorth::new`] for the capacity-clamping rationale.
    fn new(m: usize, n: usize, local_size: usize) -> Option<Self> {
        let cap = local_size.min(m.min(n));
        if cap == 0 {
            return None;
        }
        Some(Self {
            v: vec![0.0; cap * n],
            p_tilde: vec![0.0; cap * n],
            n,
            next: 0,
            count: 0,
        })
    }

    /// M-weighted MGS over stored `(v_j, p̃_j)` pairs. The coefficient
    /// `c = ⟨v, p̃_j⟩` is subtracted from `v` (against `v_j`) and from
    /// `p̃` (against `p̃_j`), keeping `p̃ = M v` consistent.
    fn reorthogonalize(&self, v: &mut [f64], p_tilde: &mut [f64]) {
        let cap = self.capacity();
        let start = if self.count < cap { 0 } else { self.next };
        for i in 0..self.count {
            let idx = (start + i) % cap;
            let v_j = self.v_slot(idx);
            let p_j = self.p_slot(idx);
            let c = par_dot(v, p_j);
            axpby(v, v_j, -c, 1.0);
            axpby(p_tilde, p_j, -c, 1.0);
        }
    }

    /// Copy the normalized `v` and `p_tilde * inv_alpha` (= `M · v_norm`)
    /// into the paired next slots, advancing the ring index.
    fn push(&mut self, v: &[f64], p_tilde_unscaled: &[f64], inv_alpha: f64) {
        let cap = self.capacity();
        let next = self.next;
        self.v_slot_mut(next).copy_from_slice(v);
        for (dst, &src) in self
            .p_slot_mut(next)
            .iter_mut()
            .zip(p_tilde_unscaled.iter())
        {
            *dst = src * inv_alpha;
        }
        self.next = (self.next + 1) % cap;
        if self.count < cap {
            self.count += 1;
        }
    }

    #[inline]
    fn capacity(&self) -> usize {
        self.v.len() / self.n
    }

    #[inline]
    fn v_slot(&self, slot: usize) -> &[f64] {
        let start = slot * self.n;
        &self.v[start..start + self.n]
    }

    #[inline]
    fn p_slot(&self, slot: usize) -> &[f64] {
        let start = slot * self.n;
        &self.p_tilde[start..start + self.n]
    }

    #[inline]
    fn v_slot_mut(&mut self, slot: usize) -> &mut [f64] {
        let start = slot * self.n;
        &mut self.v[start..start + self.n]
    }

    #[inline]
    fn p_slot_mut(&mut self, slot: usize) -> &mut [f64] {
        let start = slot * self.n;
        &mut self.p_tilde[start..start + self.n]
    }
}

/// One step of the bidiagonal sequence: the freshly computed `(α_{k+1},
/// β_{k+1})` scalars emitted by the bidiagonalization.
#[derive(Clone, Copy)]
pub(super) struct BidiagStep {
    pub(super) alpha: f64,
    pub(super) beta: f64,
}

/// A bidiagonalization stream feeding LSMR with `(α, β)` pairs and the
/// matching normalized basis vector `v_k`.
pub(super) trait Bidiagonalization {
    /// Advance one step. After the call, `v()` is the normalized `v_{k+1}`.
    fn step(&mut self) -> Result<BidiagStep, SolveError>;
    /// Most recent normalized basis vector.
    fn v(&self) -> &[f64];
}

impl<A: Operator + ?Sized> Bidiagonalization for GolubKahan<'_, A> {
    fn step(&mut self) -> Result<BidiagStep, SolveError> {
        // β_{k+1} u_{k+1} = A v_k − α_k u_k
        self.operator.try_apply(&self.bufs.v, &mut self.bufs.av)?;
        let beta_sq = axpy_with_sq_norm(&mut self.bufs.u, &self.bufs.av, -self.alpha);
        let beta = beta_sq.sqrt();
        if beta == 0.0 {
            // Lucky breakdown: the Krylov space is exhausted. Zero v so the
            // caller's `solution.update` produces no contribution this step;
            // alpha = 0 in the rotation makes the recurrence well-defined.
            self.bufs.v.fill(0.0);
            self.alpha = 0.0;
            return Ok(BidiagStep { alpha: 0.0, beta });
        }
        if beta > 0.0 {
            scale_in_place(&mut self.bufs.u, 1.0 / beta);
        }

        // α_{k+1} v_{k+1} = Aᵀ u_{k+1} − β_{k+1} v_k
        self.operator
            .try_apply_adjoint(&self.bufs.u, &mut self.bufs.atu)?;
        let mut alpha_sq = axpy_with_sq_norm(&mut self.bufs.v, &self.bufs.atu, -beta);

        // Windowed MGS in the standard inner product, before normalization.
        // After the correction, α must be re-derived from the updated v.
        if let Some(reorth) = &self.bufs.local_reorth {
            reorth.reorthogonalize(&mut self.bufs.v);
            alpha_sq = par_dot(&self.bufs.v, &self.bufs.v);
        }
        let alpha = alpha_sq.sqrt();
        if alpha > 0.0 {
            scale_in_place(&mut self.bufs.v, 1.0 / alpha);
        }

        // Push the normalized v_{k+1} into the ring.
        if let Some(reorth) = &mut self.bufs.local_reorth {
            reorth.push(&self.bufs.v);
        }

        self.alpha = alpha;
        Ok(BidiagStep { alpha, beta })
    }

    fn v(&self) -> &[f64] {
        &self.bufs.v
    }
}

impl<A: Operator + ?Sized, M: Operator + ?Sized> Bidiagonalization
    for ModifiedGolubKahan<'_, A, M>
{
    fn step(&mut self) -> Result<BidiagStep, SolveError> {
        // Phase 1 — ũ_{k+1} unnormalized: scale = −α_k / β_k.
        // Compute ũ_{k+1} = A ṽ_k − (α_k / β_k) ũ_k, then β_{k+1} = ‖ũ_{k+1}‖.
        let scale = -(self.alpha * self.beta_prev_inv);
        self.operator.try_apply(&self.bufs.v, &mut self.bufs.av)?;
        let beta_sq = axpy_with_sq_norm(&mut self.bufs.u, &self.bufs.av, scale);
        let beta = beta_sq.sqrt();
        if beta == 0.0 {
            // Lucky breakdown (preconditioned): same invariant as the
            // unpreconditioned path — zero v (and the paired p̃) so
            // `solution.update` is a no-op this step.
            self.bufs.v.fill(0.0);
            self.bufs.p_tilde.fill(0.0);
            self.alpha = 0.0;
            self.beta_prev_inv = 0.0;
            return Ok(BidiagStep { alpha: 0.0, beta });
        }
        let beta_inv = if beta > 0.0 { 1.0 / beta } else { 0.0 };

        // Phase 2 — Aᵀ on unnormalized u: result is β · Aᵀ u_norm.
        // Maintain the p̃ = α · M v invariant by scaling the running p̃ with
        // β / α_k: the α_k in the denominator cancels the α_k factor stored
        // in p̃ from the previous step, so the updated p̃ ends up as
        // α_new · M v_new (up to MGS) on completion of this iteration.
        // Precondition: α_k > 0 (the outer loop guard in lsmr_from_bidiag
        // never calls step() once α has collapsed to zero).
        self.operator
            .try_apply_adjoint(&self.bufs.u, &mut self.bufs.atu)?;
        debug_assert!(
            self.alpha > 0.0,
            "self.alpha must be > 0; lsmr_from_bidiag's loop guard prevents step() after alpha=0",
        );
        let p_coeff = beta / self.alpha;
        axpby(&mut self.bufs.p_tilde, &self.bufs.atu, beta_inv, -p_coeff);

        // Phase 3 — M-weighted MGS, then normalization, then push to ring.
        // First recover v from p̃ via the preconditioner: ṽ_{k+1} = M⁻¹ p̃
        // (equals α_new · v_norm before normalization). Then run modified
        // Gram-Schmidt against past (v, p̃ = M v) pairs, mutating v and
        // p_tilde in lockstep so the p̃ = M v invariant holds on the corrected
        // pair. After MGS, α_{k+1} = √⟨ṽ, p̃⟩ is the corrected norm; we
        // normalize v in place (p_tilde stays at α_new · p̃_norm for the next
        // step), and push the normalized v_{k+1} and p̃_{k+1,norm} = p_tilde /
        // α_new into the ring for the next step's MGS.
        self.preconditioner
            .try_apply(&self.bufs.p_tilde, &mut self.bufs.v)?;

        if let Some(reorth) = &self.bufs.local_reorth {
            reorth.reorthogonalize(&mut self.bufs.v, &mut self.bufs.p_tilde);
        }

        let vp = par_dot(&self.bufs.v, &self.bufs.p_tilde);
        let alpha_new = if vp > 0.0 { vp.sqrt() } else { 0.0 };

        if alpha_new > 0.0 {
            scale_in_place(&mut self.bufs.v, 1.0 / alpha_new);
        }

        if let Some(reorth) = &mut self.bufs.local_reorth {
            let inv_alpha = if alpha_new > 0.0 {
                1.0 / alpha_new
            } else {
                0.0
            };
            reorth.push(&self.bufs.v, &self.bufs.p_tilde, inv_alpha);
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
    /// Windowed reorthogonalization buffer; `None` disables it.
    local_reorth: Option<LocalReorth>,
}

impl GolubKahanBuffers {
    fn new(m: usize, n: usize, local_size: usize) -> Self {
        Self {
            u: vec![0.0; m],
            v: vec![0.0; n],
            av: vec![0.0; m],
            atu: vec![0.0; n],
            local_reorth: LocalReorth::new(m, n, local_size),
        }
    }
}

/// Standard Golub-Kahan bidiagonalization. No preconditioner, no `p̃`
/// buffer, two normalizations per step (`u` and `v`).
pub(super) struct GolubKahan<'a, A: Operator + ?Sized> {
    operator: &'a A,
    bufs: GolubKahanBuffers,
    /// Last `α` emitted; needed by the next step's u-update.
    alpha: f64,
}

impl<'a, A: Operator + ?Sized> GolubKahan<'a, A> {
    /// Initialize the bidiagonalization. Returns `Self` and the first step
    /// `(α₁, β₁)`.
    pub(super) fn init(
        operator: &'a A,
        b: &[f64],
        local_size: usize,
    ) -> Result<(Self, BidiagStep), SolveError> {
        let m = operator.nrows();
        let n = operator.ncols();
        let mut bufs = GolubKahanBuffers::new(m, n, local_size);

        // β₁ = ‖b‖, u₁ = b / β₁
        let beta = par_dot(b, b).sqrt();
        if beta > 0.0 {
            let inv = 1.0 / beta;
            for (ui, &bi) in bufs.u.iter_mut().zip(b) {
                *ui = bi * inv;
            }
        }

        // α₁ v₁ = Aᵀ u₁
        operator.try_apply_adjoint(&bufs.u, &mut bufs.v)?;
        let alpha = par_dot(&bufs.v, &bufs.v).sqrt();
        if alpha > 0.0 {
            scale_in_place(&mut bufs.v, 1.0 / alpha);
        }

        // Seed the reorth buffer with v₁ so the next step's MGS sees it.
        if let Some(reorth) = &mut bufs.local_reorth {
            reorth.push(&bufs.v);
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
    /// Windowed M-weighted reorthogonalization buffer; `None` disables it.
    local_reorth: Option<ModifiedLocalReorth>,
}

impl ModifiedGolubKahanBuffers {
    fn new(m: usize, n: usize, local_size: usize) -> Self {
        Self {
            u: vec![0.0; m],
            v: vec![0.0; n],
            p_tilde: vec![0.0; n],
            av: vec![0.0; m],
            atu: vec![0.0; n],
            local_reorth: ModifiedLocalReorth::new(m, n, local_size),
        }
    }
}

/// Modified Golub-Kahan bidiagonalization with `M ≈ AᵀA`. Stores `p̃`
/// scaled by `α` between steps so the preconditioner solve is a single
/// `M⁻¹` application per iteration.
pub(super) struct ModifiedGolubKahan<'a, A: Operator + ?Sized, M: Operator + ?Sized> {
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
    pub(super) fn init(
        operator: &'a A,
        preconditioner: &'a M,
        b: &[f64],
        local_size: usize,
    ) -> Result<(Self, BidiagStep), SolveError> {
        let m = operator.nrows();
        let n = operator.ncols();
        let mut bufs = ModifiedGolubKahanBuffers::new(m, n, local_size);

        // β₁ = ‖b‖, u₁ = b / β₁
        let beta = par_dot(b, b).sqrt();
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
        let vp = par_dot(&bufs.v, &bufs.p_tilde);
        let alpha = if vp > 0.0 { vp.sqrt() } else { 0.0 };

        // Normalize v; leave p_tilde at α₁·p̃_norm.
        if alpha > 0.0 {
            scale_in_place(&mut bufs.v, 1.0 / alpha);
        }

        // Seed the reorth buffer with the (v₁, p̃₁_norm = M v₁) pair.
        if let Some(reorth) = &mut bufs.local_reorth {
            let inv_alpha = if alpha > 0.0 { 1.0 / alpha } else { 0.0 };
            reorth.push(&bufs.v, &bufs.p_tilde, inv_alpha);
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
