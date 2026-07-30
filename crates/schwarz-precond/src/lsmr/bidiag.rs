//! Bidiagonalization stream feeding the LSMR recurrence.
//!
//! Produces a sequence of `(α, β)` scalars and the matching normalized basis
//! vector `v_k` from the operator (and, optionally, an `M ≈ AᵀA`
//! preconditioner). All vector kernels and the windowed reorthogonalization
//! buffers used to maintain Golub-Kahan basis quality live here as private
//! helpers — they are implementation detail of this stream, not an
//! independent subsystem.

use crate::{Operator, SolveError};
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::prelude::{ParallelSlice, ParallelSliceMut};

/// Below this count the vector kernels run sequentially; rayon wake/steal would dominate.
pub(super) const LSMR_PAR_THRESHOLD: usize = 10_000;
/// Per-worker chunk size: large enough to clear rayon dispatch, small enough to stay L1-resident.
pub(super) const LSMR_UPDATE_CHUNK: usize = 4096;

/// Fused `y = x + scale · y` returning `‖y_new‖²`; per-chunk partials avoid reduction traffic.
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

/// Inner product of two vectors.
#[inline]
pub(super) fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

/// Parallel dot product, falling back to the sequential `dot` below the threshold.
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

/// `α = √⟨v, p̃⟩`; a `vp` negative within `√ε·‖v‖‖p̃‖` clamps to 0, an indefinite `M` raises.
fn alpha_from_vp(v: &[f64], p_tilde: &[f64]) -> Result<f64, SolveError> {
    let vp = par_dot(v, p_tilde);
    if vp < 0.0 {
        let bound = f64::EPSILON.sqrt() * (par_dot(v, v) * par_dot(p_tilde, p_tilde)).sqrt();
        if vp < -bound {
            return Err(SolveError::InvalidInput {
                context: "mlsmr",
                message: "preconditioner not positive definite (⟨v, Mv⟩ < 0)".to_string(),
            });
        }
    }
    Ok(vp.max(0.0).sqrt())
}

/// Ring of recent basis vectors for windowed MGS; the disabled state is `None`, so `cap > 0`.
struct WindowRing<const L: usize> {
    /// `L` flat buffers; lane `l`, slot `s` is `[s*n .. s*n + n]` of `lanes[l]`.
    lanes: [Vec<f64>; L],
    n: usize,
    next: usize,
    count: usize,
}

impl<const L: usize> WindowRing<L> {
    /// `None` when no reorthogonalization is requested; capped at `min(m, n)`.
    fn new(m: usize, n: usize, local_size: usize) -> Option<Self> {
        let cap = local_size.min(m.min(n));
        if cap == 0 {
            return None;
        }
        Some(Self {
            lanes: std::array::from_fn(|_| vec![0.0; cap * n]),
            n,
            next: 0,
            count: 0,
        })
    }

    #[inline]
    fn capacity(&self) -> usize {
        self.lanes[0].len() / self.n
    }

    /// Ring slots currently filled, in chronological order (oldest first).
    fn chrono_slots(&self) -> impl Iterator<Item = usize> {
        let cap = self.capacity();
        let start = if self.count < cap { 0 } else { self.next };
        let count = self.count;
        (0..count).map(move |i| (start + i) % cap)
    }

    /// Reserve the next write slot, advancing the ring index and saturating the count at capacity.
    fn advance(&mut self) -> usize {
        let cap = self.capacity();
        let slot = self.next;
        self.next = (self.next + 1) % cap;
        if self.count < cap {
            self.count += 1;
        }
        slot
    }

    #[inline]
    fn lane(&self, l: usize, slot: usize) -> &[f64] {
        let start = slot * self.n;
        &self.lanes[l][start..start + self.n]
    }

    #[inline]
    fn lane_mut(&mut self, l: usize, slot: usize) -> &mut [f64] {
        let start = slot * self.n;
        &mut self.lanes[l][start..start + self.n]
    }
}

/// Euclidean windowed MGS over the single stored `v` lane, used by [`GolubKahan`].
impl WindowRing<1> {
    /// MGS sweep over stored slots oldest-first, subtracting each projection of `y`.
    fn reorthogonalize(&self, y: &mut [f64]) {
        for slot in self.chrono_slots() {
            let v_j = self.lane(0, slot);
            let c = par_dot(y, v_j);
            axpby(y, v_j, -c, 1.0);
        }
    }

    /// Copy `v` into the next slot, advancing the ring.
    fn push(&mut self, v: &[f64]) {
        let slot = self.advance();
        self.lane_mut(0, slot).copy_from_slice(v);
    }
}

/// M-weighted windowed MGS: `v` is M-orthogonal, so the coefficient is `⟨v_new, p̃_j⟩`.
impl WindowRing<2> {
    /// Subtracts `c = ⟨v, p̃_j⟩` from both `v` and `p̃`, keeping `p̃ = M v` consistent.
    fn reorthogonalize(&self, v: &mut [f64], p_tilde: &mut [f64]) {
        for slot in self.chrono_slots() {
            let v_j = self.lane(0, slot);
            let p_j = self.lane(1, slot);
            let c = par_dot(v, p_j);
            axpby(v, v_j, -c, 1.0);
            axpby(p_tilde, p_j, -c, 1.0);
        }
    }

    /// Copy normalized `v` and `p_tilde · inv_alpha` into the next slots, advancing the ring.
    fn push(&mut self, v: &[f64], p_tilde_unscaled: &[f64], inv_alpha: f64) {
        let slot = self.advance();
        self.lane_mut(0, slot).copy_from_slice(v);
        for (dst, &src) in self
            .lane_mut(1, slot)
            .iter_mut()
            .zip(p_tilde_unscaled.iter())
        {
            *dst = src * inv_alpha;
        }
    }
}

/// One step of the bidiagonal sequence: the freshly computed `(α_{k+1}, β_{k+1})` scalars.
#[derive(Clone, Copy)]
pub(super) struct BidiagStep {
    pub(super) alpha: f64,
    pub(super) beta: f64,
}

/// Stream feeding LSMR `(α, β)` pairs and the matching normalized `v_k`.
pub(super) trait Bidiagonalization {
    /// Advance one step. After the call, `v()` is the normalized `v_{k+1}`.
    fn step(&mut self) -> Result<BidiagStep, SolveError>;
    /// Most recent normalized basis vector.
    fn v(&self) -> &[f64];
}

impl<A: Operator + ?Sized> Bidiagonalization for GolubKahan<'_, A> {
    fn step(&mut self) -> Result<BidiagStep, SolveError> {
        self.operator.apply(&self.bufs.v, &mut self.bufs.av)?;
        let beta_sq = axpy_with_sq_norm(&mut self.bufs.u, &self.bufs.av, -self.alpha);
        let beta = beta_sq.sqrt();
        if beta == 0.0 {
            // Lucky breakdown: zero `v` so `solution.update` contributes nothing.
            self.bufs.v.fill(0.0);
            self.alpha = 0.0;
            return Ok(BidiagStep { alpha: 0.0, beta });
        }
        // beta > 0 here: the beta == 0 lucky breakdown returned above.
        scale_in_place(&mut self.bufs.u, 1.0 / beta);

        self.operator
            .apply_adjoint(&self.bufs.u, &mut self.bufs.atu)?;
        let mut alpha_sq = axpy_with_sq_norm(&mut self.bufs.v, &self.bufs.atu, -beta);

        // MGS runs before normalization, so α must be re-derived from the corrected `v`.
        if let Some(reorth) = &self.bufs.local_reorth {
            reorth.reorthogonalize(&mut self.bufs.v);
            alpha_sq = par_dot(&self.bufs.v, &self.bufs.v);
        }
        let alpha = alpha_sq.sqrt();
        if alpha > 0.0 {
            scale_in_place(&mut self.bufs.v, 1.0 / alpha);
        }

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
        let scale = -(self.alpha * self.beta_prev_inv);
        self.operator.apply(&self.bufs.v, &mut self.bufs.av)?;
        let beta_sq = axpy_with_sq_norm(&mut self.bufs.u, &self.bufs.av, scale);
        let beta = beta_sq.sqrt();
        if beta == 0.0 {
            // Lucky breakdown: zero `v` and its paired `p̃` so the update contributes nothing.
            self.bufs.v.fill(0.0);
            self.bufs.p_tilde.fill(0.0);
            self.alpha = 0.0;
            self.beta_prev_inv = 0.0;
            return Ok(BidiagStep { alpha: 0.0, beta });
        }
        // beta > 0 here: the beta == 0 lucky breakdown returned above.
        let beta_inv = 1.0 / beta;

        self.update_p_tilde(beta, beta_inv)?;
        let alpha_new = self.reorthonormalize_v()?;

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
    local_reorth: Option<WindowRing<1>>,
}

impl GolubKahanBuffers {
    fn new(m: usize, n: usize, local_size: usize) -> Self {
        Self {
            u: vec![0.0; m],
            v: vec![0.0; n],
            av: vec![0.0; m],
            atu: vec![0.0; n],
            local_reorth: WindowRing::<1>::new(m, n, local_size),
        }
    }
}

/// Standard Golub-Kahan bidiagonalization: no preconditioner, two normalizations per step.
pub(super) struct GolubKahan<'a, A: Operator + ?Sized> {
    operator: &'a A,
    bufs: GolubKahanBuffers,
    /// Last `α` emitted; needed by the next step's u-update.
    alpha: f64,
}

impl<'a, A: Operator + ?Sized> GolubKahan<'a, A> {
    /// Initialize the bidiagonalization, returning `Self` and the first step `(α₁, β₁)`.
    pub(super) fn init(
        operator: &'a A,
        b: &[f64],
        local_size: usize,
    ) -> Result<(Self, BidiagStep), SolveError> {
        let m = operator.nrows();
        let n = operator.ncols();
        let mut bufs = GolubKahanBuffers::new(m, n, local_size);

        // `par_dot(b, b).sqrt()` overflows to ∞ for large `b`, zeroing u₁ into a silent `x = 0`.
        let beta = super::vec_norm(b);
        if beta > 0.0 {
            let inv = 1.0 / beta;
            for (ui, &bi) in bufs.u.iter_mut().zip(b) {
                *ui = bi * inv;
            }
        }

        operator.apply_adjoint(&bufs.u, &mut bufs.v)?;
        let alpha = par_dot(&bufs.v, &bufs.v).sqrt();
        if alpha > 0.0 {
            scale_in_place(&mut bufs.v, 1.0 / alpha);
        }

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
    /// `u` left unnormalized between steps, so `‖u‖ = β_{k+1}`.
    u: Vec<f64>,
    /// `ṽ` in DOF space (length n). **Normalized** at the end of each step.
    v: Vec<f64>,
    /// `p̃` recurrence vector (length n); invariant `p_tilde_stored = α · M · v_normalized`.
    p_tilde: Vec<f64>,
    /// Scratch for `A · v` (length m).
    av: Vec<f64>,
    /// Scratch for `Aᵀ · u` (length n).
    atu: Vec<f64>,
    /// Windowed M-weighted reorthogonalization buffer; `None` disables it.
    local_reorth: Option<WindowRing<2>>,
}

impl ModifiedGolubKahanBuffers {
    fn new(m: usize, n: usize, local_size: usize) -> Self {
        Self {
            u: vec![0.0; m],
            v: vec![0.0; n],
            p_tilde: vec![0.0; n],
            av: vec![0.0; m],
            atu: vec![0.0; n],
            local_reorth: WindowRing::<2>::new(m, n, local_size),
        }
    }
}

/// Modified Golub-Kahan with `M ≈ AᵀA`, storing `p̃` scaled by `α` so a step costs one `M⁻¹`.
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
    /// Initialize the bidiagonalization, returning `Self` and the first step `(α₁, β₁)`.
    pub(super) fn init(
        operator: &'a A,
        preconditioner: &'a M,
        b: &[f64],
        local_size: usize,
    ) -> Result<(Self, BidiagStep), SolveError> {
        let m = operator.nrows();
        let n = operator.ncols();
        let mut bufs = ModifiedGolubKahanBuffers::new(m, n, local_size);

        // `par_dot(b, b).sqrt()` overflows to ∞ for large `b`, zeroing u₁ into a silent `x = 0`.
        let beta = super::vec_norm(b);
        if beta > 0.0 {
            let inv = 1.0 / beta;
            for (ui, &bi) in bufs.u.iter_mut().zip(b) {
                *ui = bi * inv;
            }
        }

        operator.apply_adjoint(&bufs.u, &mut bufs.p_tilde)?;

        preconditioner.apply(&bufs.p_tilde, &mut bufs.v)?;

        let alpha = alpha_from_vp(&bufs.v, &bufs.p_tilde)?;

        if alpha > 0.0 {
            scale_in_place(&mut bufs.v, 1.0 / alpha);
        }

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

    /// Scaling by `β / α_k` cancels the stored `α_k`; requires `α_k > 0`.
    fn update_p_tilde(&mut self, beta: f64, beta_inv: f64) -> Result<(), SolveError> {
        self.operator
            .apply_adjoint(&self.bufs.u, &mut self.bufs.atu)?;
        debug_assert!(
            self.alpha > 0.0,
            "self.alpha must be > 0; lsmr_from_bidiag's loop guard prevents step() after alpha=0",
        );
        let p_coeff = beta / self.alpha;
        axpby(&mut self.bufs.p_tilde, &self.bufs.atu, beta_inv, -p_coeff);
        Ok(())
    }

    /// Recover `ṽ = M⁻¹ p̃`, MGS in lockstep to hold `p̃ = M v`, normalize; returns `α_{k+1}`.
    fn reorthonormalize_v(&mut self) -> Result<f64, SolveError> {
        self.preconditioner
            .apply(&self.bufs.p_tilde, &mut self.bufs.v)?;

        if let Some(reorth) = &self.bufs.local_reorth {
            reorth.reorthogonalize(&mut self.bufs.v, &mut self.bufs.p_tilde);
        }

        let alpha_new = alpha_from_vp(&self.bufs.v, &self.bufs.p_tilde)?;

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

        Ok(alpha_new)
    }
}
