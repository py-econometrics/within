//! Bidiagonalization stream feeding the LSMR recurrence.
//!
//! Produces a sequence of `(α, β)` scalars and the matching normalized basis
//! vector `v_k` from the operator (and, optionally, an `M ≈ AᵀA`
//! preconditioner). All vector kernels and the windowed reorthogonalization
//! buffers used to maintain Golub-Kahan basis quality live here as private
//! helpers — they are implementation detail of this stream, not an
//! independent subsystem.

#[cfg(test)]
mod tests;

use super::finite;
use super::magnitude::{exponent, ldexp, Magnitude};
use crate::{Operator, SolveError};
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::prelude::{ParallelSlice, ParallelSliceMut};

/// Below this count the vector kernels run sequentially; rayon wake/steal would dominate.
pub(super) const LSMR_PAR_THRESHOLD: usize = 10_000;
/// Per-worker chunk size: large enough to clear rayon dispatch, small enough to stay L1-resident.
pub(super) const LSMR_UPDATE_CHUNK: usize = 4096;

/// Fused `y = x + scale · y` returning `‖y_new‖`; per-chunk partials avoid reduction traffic.
#[inline]
fn axpy_with_norm(y: &mut [f64], x: &[f64], scale: f64) -> f64 {
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
    let sq = if y.len() >= LSMR_PAR_THRESHOLD {
        y.par_chunks_mut(LSMR_UPDATE_CHUNK)
            .zip(x.par_chunks(LSMR_UPDATE_CHUNK))
            .map(|(y_c, x_c)| seq(y_c, x_c))
            .sum()
    } else {
        seq(y, x)
    };
    norm_from_sq(y, sq)
}

/// `y = alpha * x + beta * y`. Parallel above the threshold.
#[inline]
pub(super) fn axpby(y: &mut [f64], x: &[f64], alpha: f64, beta: f64) {
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

/// `y /= d` without over/underflow unless `y / d` does, to one rounding at `f64::MAX` (`drscl`).
#[inline]
fn normalize(y: &mut [f64], d: f64) {
    if d > 0.0 {
        let inv = 1.0 / d;
        let seq = |c: &mut [f64]| {
            if inv.is_normal() {
                c.iter_mut().for_each(|yi| *yi *= inv);
            } else {
                c.iter_mut().for_each(|yi| *yi /= d);
            }
        };
        if y.len() >= LSMR_PAR_THRESHOLD {
            y.par_chunks_mut(LSMR_UPDATE_CHUNK).for_each(seq);
        } else {
            seq(y);
        }
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

/// Why [`alpha_from_vp`] refused a pair.
#[derive(Debug)]
enum AlphaError {
    /// `⟨v, p̃⟩ < −√ε·‖v‖‖p̃‖`: an indefinite `M`, or a `v` that drifted from `M⁻¹ p̃`.
    NegativeMetric,
    Invalid(SolveError),
}

impl From<SolveError> for AlphaError {
    fn from(err: SolveError) -> Self {
        Self::Invalid(err)
    }
}

impl From<AlphaError> for SolveError {
    fn from(err: AlphaError) -> Self {
        match err {
            AlphaError::NegativeMetric => SolveError::InvalidInput {
                context: "mlsmr",
                message: "preconditioner not positive definite (⟨v, Mv⟩ < 0)".to_string(),
            },
            AlphaError::Invalid(err) => err,
        }
    }
}

/// `α = √⟨v, p̃⟩`; a `vp` negative within `√ε·‖v‖‖p̃‖` clamps to 0, an indefinite `M` raises.
fn alpha_from_vp(v: &[f64], p_tilde: &[f64]) -> Result<f64, AlphaError> {
    // A NaN `vp` may be `∞ − ∞` from finite vectors; the norms below reject a NaN entry.
    let vp = par_dot(v, p_tilde);
    if vp.is_normal() && vp > 0.0 {
        return Ok(vp.sqrt());
    }
    // `vp` is a product of two norms, so it vanishes and overflows at magnitudes α itself holds.
    let norm_v = finite(super::vec_norm(v), "‖v‖")?;
    let norm_p = finite(super::vec_norm(p_tilde), "‖p̃‖")?;
    if norm_v == 0.0 || norm_p == 0.0 {
        return Ok(0.0);
    }
    // Even shifts to near `2^500` keep the re-sum below `2^1003` and let `√` halve the total exactly.
    let shift = |norm: f64| (500 - exponent(norm)) & !1;
    let (kv, kp) = (shift(norm_v), shift(norm_p));
    let scaled: f64 = v
        .iter()
        .zip(p_tilde)
        .map(|(&x, &y)| ldexp(x, kv) * ldexp(y, kp))
        .sum();
    if scaled < -f64::EPSILON.sqrt() * ldexp(norm_v, kv) * ldexp(norm_p, kp) {
        return Err(AlphaError::NegativeMetric);
    }
    // A normal `vp` rounded fewer times than the re-sum; a subnormal or overflowed one did not.
    if vp < 0.0 && vp.is_normal() {
        return Ok(0.0);
    }
    Ok(ldexp(scaled.max(0.0).sqrt(), -(kv + kp) / 2))
}

/// `‖v‖` from an unscaled `‖v‖²`; an over/underflowed or subnormal sum pays the max-scaled pass.
fn norm_from_sq(v: &[f64], sq: f64) -> f64 {
    if sq.is_normal() {
        sq.sqrt()
    } else {
        super::vec_norm(v)
    }
}

fn par_norm(v: &[f64]) -> f64 {
    norm_from_sq(v, par_dot(v, v))
}

/// `u = rhs − A x`, returning `‖u‖`.
pub(super) fn residual_into<A: Operator + ?Sized>(
    operator: &A,
    x: &[f64],
    rhs: &[f64],
    u: &mut [f64],
) -> Result<f64, SolveError> {
    operator.apply(x, u)?;
    Ok(axpy_with_norm(u, rhs, -1.0))
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

    /// Forget every stored vector: a restart begins a new sequence.
    fn clear(&mut self) {
        self.next = 0;
        self.count = 0;
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
    /// Subtracts `c = ⟨v, p̃_j⟩` from both `v` and `p̃`.
    fn reorthogonalize(&self, v: &mut [f64], p_tilde: &mut [f64]) {
        for slot in self.chrono_slots() {
            let v_j = self.lane(0, slot);
            let p_j = self.lane(1, slot);
            let c = par_dot(v, p_j);
            axpby(v, v_j, -c, 1.0);
            axpby(p_tilde, p_j, -c, 1.0);
        }
    }

    /// Copy normalized `v` and `p_tilde / alpha` into the next slots, advancing the ring.
    fn push(&mut self, v: &[f64], p_tilde_unscaled: &[f64], alpha: f64) {
        let slot = self.advance();
        self.lane_mut(0, slot).copy_from_slice(v);
        let p = self.lane_mut(1, slot);
        if alpha > 0.0 {
            p.copy_from_slice(p_tilde_unscaled);
            normalize(p, alpha);
        } else {
            p.fill(0.0);
        }
    }
}

/// One step of the bidiagonal sequence: the freshly computed `(α_{k+1}, β_{k+1})` scalars.
#[derive(Clone, Copy)]
pub(super) struct BidiagStep {
    pub(super) alpha: f64,
    pub(super) beta: f64,
}

/// `‖Âᵀ rhs‖ = ‖Aᵀ rhs‖·√(ĝᵀM⁻¹ĝ)` for unit `ĝ ∥ Aᵀ rhs`; the raw product may not be a double.
pub(super) fn metric_gradient_norm<A: Operator + ?Sized, M: Operator + ?Sized>(
    operator: &A,
    preconditioner: &M,
    rhs: &[f64],
    rhs_norm: f64,
) -> Result<Magnitude, SolveError> {
    let (mut g, mut mv) = (vec![0.0; operator.ncols()], vec![0.0; operator.ncols()]);
    let image = adjoint_norm(operator, rhs, rhs_norm, &mut vec![0.0; rhs.len()], &mut g)?;
    let plain = par_norm(&g);
    if !plain.is_finite() {
        return Ok(Magnitude::from(plain));
    }
    normalize(&mut g, plain);
    preconditioner.apply(&g, &mut mv)?;
    Ok(image * Magnitude::from(alpha_from_vp(&mv, &g)?))
}

/// `‖Aᵀ rhs‖`, leaving the image in `g` up to scale; `rhs / ‖rhs‖` stands in past the normal range.
fn adjoint_norm<A: Operator + ?Sized>(
    operator: &A,
    rhs: &[f64],
    rhs_norm: f64,
    unit: &mut [f64],
    g: &mut [f64],
) -> Result<Magnitude, SolveError> {
    // Scaling first flushes the entries far below `‖rhs‖`, which `Aᵀ` may amplify back into range.
    operator.apply_adjoint(rhs, g)?;
    let raw = par_norm(g);
    if raw.is_normal() {
        return Ok(Magnitude::from(raw));
    }
    unit.copy_from_slice(rhs);
    normalize(unit, rhs_norm);
    operator.apply_adjoint(unit, g)?;
    Ok(Magnitude::product(rhs_norm, par_norm(g)))
}

/// Stream feeding LSMR `(α, β)` pairs and the matching normalized `v_k`.
pub(super) trait Bidiagonalization {
    /// Advance one step. After the call, `v()` is the normalized `v_{k+1}`.
    fn step(&mut self) -> Result<BidiagStep, SolveError>;
    /// Most recent normalized basis vector.
    fn v(&self) -> &[f64];
    /// `‖rhs − A x‖`, staging `rhs − A x` for [`restart`](Self::restart); clobbers the stream.
    fn residual_norm(&mut self, x: &[f64], rhs: &[f64]) -> Result<f64, SolveError>;
    /// Seed a fresh sequence from the staged residual and the `β₁` `residual_norm` returned.
    fn restart(&mut self, beta: f64) -> Result<BidiagStep, SolveError>;
    /// Spend the stream for the `rhs − A x` that [`residual_norm`](Self::residual_norm) staged.
    fn into_residual(self) -> Vec<f64>;
    /// After `α₁ = 0`, a gradient the metric hid: `‖Aᵀ rhs‖ / ‖rhs‖` and `‖Aᵀ b‖`.
    fn hidden_gradient(
        &mut self,
        _b: &[f64],
        _b_norm: f64,
    ) -> Result<Option<(f64, Magnitude)>, SolveError> {
        Ok(None)
    }
}

impl<A: Operator + ?Sized> Bidiagonalization for GolubKahan<'_, A> {
    fn step(&mut self) -> Result<BidiagStep, SolveError> {
        self.operator.apply(&self.bufs.v, &mut self.bufs.av)?;
        let beta = finite(
            axpy_with_norm(&mut self.bufs.u, &self.bufs.av, -self.alpha),
            "β",
        )?;
        if beta == 0.0 {
            // Lucky breakdown: zero `v` so `solution.update` contributes nothing.
            self.bufs.v.fill(0.0);
            self.alpha = 0.0;
            return Ok(BidiagStep { alpha: 0.0, beta });
        }
        normalize(&mut self.bufs.u, beta);

        self.operator
            .apply_adjoint(&self.bufs.u, &mut self.bufs.atu)?;
        let mut alpha = axpy_with_norm(&mut self.bufs.v, &self.bufs.atu, -beta);

        // MGS runs before normalization, so α must be re-derived from the corrected `v`.
        if let Some(reorth) = &self.bufs.local_reorth {
            reorth.reorthogonalize(&mut self.bufs.v);
            alpha = par_norm(&self.bufs.v);
        }
        let alpha = finite(alpha, "α")?;
        normalize(&mut self.bufs.v, alpha);

        if let Some(reorth) = &mut self.bufs.local_reorth {
            reorth.push(&self.bufs.v);
        }

        self.alpha = alpha;
        Ok(BidiagStep { alpha, beta })
    }

    fn v(&self) -> &[f64] {
        &self.bufs.v
    }

    fn residual_norm(&mut self, x: &[f64], rhs: &[f64]) -> Result<f64, SolveError> {
        residual_into(self.operator, x, rhs, &mut self.bufs.u)
    }

    fn into_residual(self) -> Vec<f64> {
        self.bufs.u
    }

    fn restart(&mut self, beta: f64) -> Result<BidiagStep, SolveError> {
        normalize(&mut self.bufs.u, beta);
        self.operator
            .apply_adjoint(&self.bufs.u, &mut self.bufs.v)?;
        let alpha = finite(par_norm(&self.bufs.v), "α")?;
        normalize(&mut self.bufs.v, alpha);
        if let Some(reorth) = &mut self.bufs.local_reorth {
            reorth.clear();
            reorth.push(&self.bufs.v);
        }
        self.alpha = alpha;
        Ok(BidiagStep { alpha, beta })
    }
}

impl<A: Operator + ?Sized, M: Operator + ?Sized> Bidiagonalization
    for ModifiedGolubKahan<'_, A, M>
{
    fn step(&mut self) -> Result<BidiagStep, SolveError> {
        let scale = -(self.alpha * self.u_norm_inv);
        self.operator.apply(&self.bufs.v, &mut self.bufs.av)?;
        let beta = finite(axpy_with_norm(&mut self.bufs.u, &self.bufs.av, scale), "β")?;
        if beta == 0.0 {
            // Lucky breakdown: zero `v` and its paired `p̃` so the update contributes nothing.
            self.bufs.v.fill(0.0);
            self.bufs.p_tilde.fill(0.0);
            self.alpha = 0.0;
            return Ok(BidiagStep { alpha: 0.0, beta });
        }
        // `Aᵀu` scales with `‖A‖β`; only a `β` far from 1 carries it off `p̃`'s own `‖A‖` scale.
        self.u_norm_inv = if RAW_U_NORMS.contains(&beta) {
            1.0 / beta
        } else {
            normalize(&mut self.bufs.u, beta);
            1.0
        };

        self.update_p_tilde(beta)?;
        let alpha_new = self.reorthonormalize_v()?;

        self.alpha = alpha_new;

        Ok(BidiagStep {
            alpha: alpha_new,
            beta,
        })
    }

    fn v(&self) -> &[f64] {
        &self.bufs.v
    }

    fn residual_norm(&mut self, x: &[f64], rhs: &[f64]) -> Result<f64, SolveError> {
        residual_into(self.operator, x, rhs, &mut self.bufs.u)
    }

    fn into_residual(self) -> Vec<f64> {
        self.bufs.u
    }

    fn restart(&mut self, beta: f64) -> Result<BidiagStep, SolveError> {
        normalize(&mut self.bufs.u, beta);
        self.operator
            .apply_adjoint(&self.bufs.u, &mut self.bufs.p_tilde)?;
        self.preconditioner
            .apply(&self.bufs.p_tilde, &mut self.bufs.v)?;
        let alpha = alpha_from_vp(&self.bufs.v, &self.bufs.p_tilde)?;
        normalize(&mut self.bufs.v, alpha);
        if let Some(reorth) = &mut self.bufs.local_reorth {
            reorth.clear();
            reorth.push(&self.bufs.v, &self.bufs.p_tilde, alpha);
        }
        self.alpha = alpha;
        self.u_norm_inv = 1.0; // u was normalized
        Ok(BidiagStep { alpha, beta })
    }

    fn hidden_gradient(
        &mut self,
        b: &[f64],
        b_norm: f64,
    ) -> Result<Option<(f64, Magnitude)>, SolveError> {
        // At `α₁ = 0` the stream holds `p̃ = Aᵀu` for the unit `u`, so `‖p̃‖ = ‖Aᵀ rhs‖ / ‖rhs‖`.
        let plain = par_norm(&self.bufs.p_tilde);
        if plain == 0.0 {
            return Ok(None);
        }
        let bufs = &mut self.bufs;
        let b_image = adjoint_norm(self.operator, b, b_norm, &mut bufs.u, &mut bufs.atu)?;
        Ok(Some((plain, b_image)))
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
        b_norm: f64,
        local_size: usize,
    ) -> Result<(Self, BidiagStep), SolveError> {
        let mut bufs = GolubKahanBuffers::new(operator.nrows(), operator.ncols(), local_size);
        bufs.u.copy_from_slice(b);
        let mut stream = Self {
            operator,
            bufs,
            alpha: 0.0,
        };
        let step1 = stream.restart(b_norm)?;
        Ok((stream, step1))
    }
}

/// Workspaces used by [`ModifiedGolubKahan`].
struct ModifiedGolubKahanBuffers {
    /// `u` left unnormalized between steps while `β_{k+1}` stays inside [`RAW_U_NORMS`].
    u: Vec<f64>,
    /// `ṽ` in DOF space (length n). **Normalized** at the end of each step.
    v: Vec<f64>,
    /// `p̃` recurrence vector (length n); `p_tilde_stored ≈ α · M · v_normalized`.
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

/// `‖u‖` left unnormalized to skip a pass over `m`; `‖Aᵀu‖ ≤ 1e64 ‖A‖` caps raw `‖A‖` at `1e244`.
const RAW_U_NORMS: std::ops::RangeInclusive<f64> = 1e-64..=1e64;

/// Modified Golub-Kahan with `M ≈ AᵀA`, storing `p̃` scaled by `α` so a step costs one `M⁻¹`.
pub(super) struct ModifiedGolubKahan<'a, A: Operator + ?Sized, M: Operator + ?Sized> {
    operator: &'a A,
    preconditioner: &'a M,
    bufs: ModifiedGolubKahanBuffers,
    /// Last `α` emitted; needed by the next step to scale `p_tilde`.
    alpha: f64,
    /// `1/‖u‖` as stored: `1/β_k`, or 1 once `u` was normalized in place.
    u_norm_inv: f64,
}

impl<'a, A: Operator + ?Sized, M: Operator + ?Sized> ModifiedGolubKahan<'a, A, M> {
    /// Initialize the bidiagonalization, returning `Self` and the first step `(α₁, β₁)`.
    pub(super) fn init(
        operator: &'a A,
        preconditioner: &'a M,
        b: &[f64],
        b_norm: f64,
        local_size: usize,
    ) -> Result<(Self, BidiagStep), SolveError> {
        let mut bufs =
            ModifiedGolubKahanBuffers::new(operator.nrows(), operator.ncols(), local_size);
        bufs.u.copy_from_slice(b);
        let mut stream = Self {
            operator,
            preconditioner,
            bufs,
            alpha: 0.0,
            u_norm_inv: 1.0,
        };
        let step1 = stream.restart(b_norm)?;
        Ok((stream, step1))
    }

    /// Scaling by `β / α_k` cancels the stored `α_k`; requires `α_k > 0`.
    fn update_p_tilde(&mut self, beta: f64) -> Result<(), SolveError> {
        self.operator
            .apply_adjoint(&self.bufs.u, &mut self.bufs.atu)?;
        debug_assert!(
            self.alpha > 0.0,
            "self.alpha must be > 0; lsmr_from_bidiag's loop guard prevents step() after alpha=0",
        );
        let mut p_coeff = beta / self.alpha;
        if !p_coeff.is_normal() {
            // `β/α` can leave the range while `p̃ / α = M v` does not, so divide first (`dlascl`).
            normalize(&mut self.bufs.p_tilde, self.alpha);
            p_coeff = beta;
        }
        axpby(
            &mut self.bufs.p_tilde,
            &self.bufs.atu,
            self.u_norm_inv,
            -p_coeff,
        );
        Ok(())
    }

    /// Recover `ṽ = M⁻¹ p̃`, MGS both in lockstep, normalize; returns `α_{k+1}`.
    fn reorthonormalize_v(&mut self) -> Result<f64, SolveError> {
        self.preconditioner
            .apply(&self.bufs.p_tilde, &mut self.bufs.v)?;

        if let Some(reorth) = &self.bufs.local_reorth {
            reorth.reorthogonalize(&mut self.bufs.v, &mut self.bufs.p_tilde);
        }

        let alpha_new = match alpha_from_vp(&self.bufs.v, &self.bufs.p_tilde) {
            // MGS updates `v` and `p̃` apart, so near breakdown `v` drifts from `M⁻¹ p̃`.
            Err(AlphaError::NegativeMetric) if self.bufs.local_reorth.is_some() => {
                self.preconditioner
                    .apply(&self.bufs.p_tilde, &mut self.bufs.v)?;
                alpha_from_vp(&self.bufs.v, &self.bufs.p_tilde)?
            }
            alpha => alpha?,
        };

        normalize(&mut self.bufs.v, alpha_new);

        if let Some(reorth) = &mut self.bufs.local_reorth {
            reorth.push(&self.bufs.v, &self.bufs.p_tilde, alpha_new);
        }

        Ok(alpha_new)
    }
}
