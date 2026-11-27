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
use crate::{MemBuffer, MemStack, Operator, SolveError, StackReq};
use inflight::parallel::{Blocks, Split, Threads};

/// Below this count the elementwise kernels run sequentially; rayon wake/steal would dominate.
const LSMR_PAR_THRESHOLD: usize = 10_000;
/// Per-worker chunk size: large enough to clear rayon dispatch, small enough to stay L1-resident.
pub(super) const LSMR_UPDATE_CHUNK: usize = 4096;

/// Every thread for an elementwise kernel over `len` entries past [`LSMR_PAR_THRESHOLD`], else the
/// calling thread.
pub(super) fn update_threads(len: usize) -> Threads {
    match len >= LSMR_PAR_THRESHOLD {
        true => Threads::All,
        false => Threads::ONE,
    }
}

/// Fused `y = x + scale · y` returning `‖y_new‖`; per-chunk partials avoid reduction traffic.
#[inline]
fn axpy_with_norm(y: &mut [f64], x: &[f64], scale: f64) -> f64 {
    debug_assert_eq!(x.len(), y.len());
    let sq = chunked_sum((&mut *y, x), |(y_c, x_c): (&mut [f64], &[f64])| {
        let step = |yi: &mut f64, xi: f64| {
            let val = xi + scale * *yi;
            *yi = val;
            val * val
        };
        let mut acc = [0.0; LANES];
        let mut y_lanes = y_c.chunks_exact_mut(LANES);
        let mut x_lanes = x_c.chunks_exact(LANES);
        for (y, x) in (&mut y_lanes).zip(&mut x_lanes) {
            for ((s, yi), &xi) in acc.iter_mut().zip(y).zip(x) {
                *s += step(yi, xi);
            }
        }
        let tail: f64 = (y_lanes.into_remainder().iter_mut())
            .zip(x_lanes.remainder())
            .map(|(yi, &xi)| step(yi, xi))
            .sum();
        acc.iter().sum::<f64>() + tail
    });
    norm_from_sq(y, sq)
}

/// Independent accumulators per leaf: a single add chain is latency-bound and never vectorizes.
const LANES: usize = 8;

/// Below this count a sum runs on the calling thread: a fork costs more than a sequential sum.
const LSMR_SUM_PAR_THRESHOLD: usize = 1 << 16;

/// `Σ leaf` over `LSMR_UPDATE_CHUNK` pieces in a tree fixed by the length.
#[inline]
fn chunked_sum<P: Split>(input: P, leaf: impl Fn(P) -> f64 + Sync) -> f64 {
    let threads = match input.n_items() > LSMR_SUM_PAR_THRESHOLD {
        true => Threads::All,
        false => Threads::ONE,
    };
    let blocks = Blocks::new(input, LSMR_UPDATE_CHUNK);
    threads.fold_tree(blocks, |(_, piece)| leaf(piece), |a, b| a + b)
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
    update_threads(y.len())
        .for_each_chunk((y, x), LSMR_UPDATE_CHUNK, |_, (y_c, x_c)| seq(y_c, x_c));
}

/// `y /= d` without over/underflow unless `y / d` does, to one rounding at `f64::MAX` (`drscl`).
#[inline]
pub(super) fn normalize(y: &mut [f64], d: f64) {
    if d > 0.0 {
        let inv = 1.0 / d;
        let seq = |c: &mut [f64]| {
            if inv.is_normal() {
                c.iter_mut().for_each(|yi| *yi *= inv);
            } else {
                c.iter_mut().for_each(|yi| *yi /= d);
            }
        };
        update_threads(y.len()).for_each_chunk(y, LSMR_UPDATE_CHUNK, |_, c| seq(c));
    }
}

/// Inner product of two vectors.
#[inline]
pub(super) fn dot(a: &[f64], b: &[f64]) -> f64 {
    let (a_lanes, b_lanes) = (a.chunks_exact(LANES), b.chunks_exact(LANES));
    let tail: f64 = (a_lanes.remainder().iter())
        .zip(b_lanes.remainder())
        .map(|(x, y)| x * y)
        .sum();
    let mut acc = [0.0; LANES];
    for (x, y) in a_lanes.zip(b_lanes) {
        for ((s, xi), yi) in acc.iter_mut().zip(x).zip(y) {
            *s += xi * yi;
        }
    }
    acc.iter().sum::<f64>() + tail
}

/// Dot product added in a tree fixed by the length alone.
#[inline]
fn par_dot(a: &[f64], b: &[f64]) -> f64 {
    chunked_sum((a, b), |(ac, bc)| dot(ac, bc))
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

pub(super) fn par_norm(v: &[f64]) -> f64 {
    norm_from_sq(v, par_dot(v, v))
}

/// `u = rhs − A x`, returning `‖u‖`.
fn residual_into<A: Operator + ?Sized>(
    operator: &A,
    x: &[f64],
    rhs: &[f64],
    u: &mut [f64],
    stack: &mut MemStack,
) -> Result<f64, SolveError> {
    operator.apply_with(x, u, stack)?;
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

/// `Aᵀ rhs = 2ᵏ·g` for the `g` that [`adjoint_image`] leaves.
#[derive(Clone, Copy)]
struct AdjointImage {
    k: i32,
    g_norm: f64,
}

impl AdjointImage {
    /// `‖Aᵀ rhs‖`, which may not be a double.
    fn norm(self) -> Magnitude {
        Magnitude::product(self.g_norm, ldexp(1.0, self.k))
    }
}

/// `Aᵀ` of `rhs`, or of `rhs·2⁻ᵏ` in `unit` where `‖rhs‖` or the raw image is not normal.
fn adjoint_image<A: Operator + ?Sized>(
    operator: &A,
    rhs: &[f64],
    rhs_norm: f64,
    unit: &mut [f64],
    g: &mut [f64],
    stack: &mut MemStack,
) -> Result<AdjointImage, SolveError> {
    // Scaling first flushes the entries far below `‖rhs‖`, which `Aᵀ` may amplify back into range.
    let k = exponent(rhs_norm);
    if rhs_norm.is_normal() {
        operator.apply_adjoint_with(rhs, g, stack)?;
        let g_norm = par_norm(g);
        if k == 0 || g_norm.is_normal() {
            return Ok(AdjointImage { k: 0, g_norm });
        }
    }
    // A power of two scales exactly.
    unit.iter_mut()
        .zip(rhs)
        .for_each(|(u, &r)| *u = ldexp(r, -k));
    operator.apply_adjoint_with(unit, g, stack)?;
    Ok(AdjointImage {
        k,
        g_norm: par_norm(g),
    })
}

/// Stream feeding LSMR `(α, β)` pairs and the matching normalized `v_k`.
pub(super) trait Bidiagonalization {
    /// Advance one step. After the call, `v()` is the normalized `v_{k+1}`.
    fn step(&mut self) -> Result<BidiagStep, SolveError>;
    /// Most recent normalized basis vector.
    fn v(&self) -> &[f64];
    /// `‖rhs − A x‖`, staging `rhs − A x` for [`restart`](Self::restart); clobbers the stream.
    fn residual_norm(&mut self, x: &[f64], rhs: &[f64]) -> Result<f64, SolveError>;
    /// `‖Aᵀr‖` for the residual staged by [`residual_norm`](Self::residual_norm).
    fn plain_gradient(&mut self, residual_norm: f64) -> Result<Magnitude, SolveError>;
    /// The operator's [`norm_floor`](crate::Operator::norm_floor), `L ≤ ‖A‖₂`, asked for once.
    fn norm_floor(&mut self) -> Result<f64, SolveError>;
    /// Seed a fresh sequence from the staged residual and the `β₁` `residual_norm` returned.
    fn restart(&mut self, beta: f64) -> Result<BidiagStep, SolveError>;
    /// Spend the stream for the `rhs − A x` that [`residual_norm`](Self::residual_norm) staged.
    fn into_residual(self) -> Vec<f64>;
    /// PROTOTYPE #607: `‖rhs − A x‖` and `‖Aᵀ(rhs − A x)‖` on scratch the stream does not carry,
    /// so the pass continues; `None` where either is out of range.
    fn measure(
        &mut self,
        _x: &[f64],
        _rhs: &[f64],
    ) -> Result<Option<(f64, Magnitude)>, SolveError> {
        Ok(None)
    }
    /// After `α₁ = 0`, a gradient the metric hid: `‖Aᵀ rhs‖ / ‖rhs‖` and `‖Aᵀ b‖`.
    fn hidden_gradient(
        &mut self,
        _b: &[f64],
        _b_norm: f64,
    ) -> Result<Option<(f64, Magnitude)>, SolveError> {
        Ok(None)
    }
}

impl<A: Operator + ?Sized, M: Operator + ?Sized> Bidiagonalization
    for ModifiedGolubKahan<'_, A, M>
{
    fn step(&mut self) -> Result<BidiagStep, SolveError> {
        let scale = -(self.alpha * self.u_norm_inv);
        let bufs = &mut self.bufs;
        self.operator
            .apply_with(&bufs.v, &mut bufs.av, MemStack::new(&mut bufs.stack))?;
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
        let bufs = &mut self.bufs;
        residual_into(
            self.operator,
            x,
            rhs,
            &mut bufs.u,
            MemStack::new(&mut bufs.stack),
        )
    }

    fn plain_gradient(&mut self, residual_norm: f64) -> Result<Magnitude, SolveError> {
        let bufs = &mut self.bufs;
        let image = adjoint_image(
            self.operator,
            &bufs.u,
            residual_norm,
            &mut bufs.av,
            &mut bufs.atu,
            MemStack::new(&mut bufs.stack),
        )?;
        Ok(image.norm())
    }

    fn norm_floor(&mut self) -> Result<f64, SolveError> {
        match self.floor {
            Some(floor) => Ok(floor),
            None => Ok(*self.floor.insert(self.operator.norm_floor()?)),
        }
    }

    fn into_residual(self) -> Vec<f64> {
        self.bufs.u
    }

    fn measure(&mut self, x: &[f64], rhs: &[f64]) -> Result<Option<(f64, Magnitude)>, SolveError> {
        let bufs = &mut self.bufs;
        // `av` and `atu` are rewritten by the next step before they are read.
        let residual = residual_into(
            self.operator,
            x,
            rhs,
            &mut bufs.av,
            MemStack::new(&mut bufs.stack),
        )?;
        if !residual.is_normal() {
            return Ok(None);
        }
        self.operator.apply_adjoint_with(
            &bufs.av,
            &mut bufs.atu,
            MemStack::new(&mut bufs.stack),
        )?;
        let g = par_norm(&bufs.atu);
        Ok(g.is_normal().then(|| (residual, Magnitude::from(g))))
    }

    fn restart(&mut self, beta: f64) -> Result<BidiagStep, SolveError> {
        normalize(&mut self.bufs.u, beta);
        let bufs = &mut self.bufs;
        self.operator.apply_adjoint_with(
            &bufs.u,
            &mut bufs.p_tilde,
            MemStack::new(&mut bufs.stack),
        )?;
        self.preconditioner.apply_with(
            &bufs.p_tilde,
            &mut bufs.v,
            MemStack::new(&mut bufs.stack),
        )?;
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
        let stack = MemStack::new(&mut bufs.stack);
        let b_image = adjoint_image(self.operator, b, b_norm, &mut bufs.u, &mut bufs.atu, stack)?;
        Ok(Some((plain, b_image.norm())))
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
    /// What one apply of the operator or the preconditioner takes; the run applies one at a time.
    stack: MemBuffer,
}

impl ModifiedGolubKahanBuffers {
    fn new(m: usize, n: usize, local_size: usize, stack: StackReq) -> Self {
        Self {
            stack: MemBuffer::new(stack),
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
    /// The operator's norm floor, kept across restarts once an audit asked for it.
    floor: Option<f64>,
}

/// A start vector `rhs − A x₀` staged in a [`ModifiedGolubKahan`]'s `u`, before its first `restart`.
pub(super) struct StagedStart<'a, A: Operator + ?Sized, M: Operator + ?Sized>(
    ModifiedGolubKahan<'a, A, M>,
);

impl<'a, A: Operator + ?Sized, M: Operator + ?Sized> StagedStart<'a, A, M> {
    /// `x₀ = 0`: `u = b`.
    pub(super) fn cold(
        operator: &'a A,
        preconditioner: &'a M,
        b: &[f64],
        local_size: usize,
    ) -> Self {
        let mut stream = ModifiedGolubKahan::new(operator, preconditioner, local_size);
        stream.bufs.u.copy_from_slice(b);
        Self(stream)
    }

    /// `u = b − A x₀`, the way a refuted stop restarts; also returns `‖b − A x₀‖`.
    pub(super) fn warm(
        operator: &'a A,
        preconditioner: &'a M,
        x0: &[f64],
        b: &[f64],
        local_size: usize,
    ) -> Result<(Self, f64), SolveError> {
        let mut stream = ModifiedGolubKahan::new(operator, preconditioner, local_size);
        let norm = stream.residual_norm(x0, b)?;
        Ok((Self(stream), norm))
    }

    /// `‖Âᵀ rhs‖ = ‖Aᵀ rhs‖·√(ĝᵀM⁻¹ĝ)` for unit `ĝ ∥ Aᵀ rhs`, on scratch that spares the staged `u`.
    pub(super) fn metric_reference(
        &mut self,
        rhs: &[f64],
        rhs_norm: f64,
    ) -> Result<Magnitude, SolveError> {
        let stream = &mut self.0;
        let bufs = &mut stream.bufs;
        let (g, mv) = (&mut bufs.atu, &mut bufs.v);
        let stack = MemStack::new(&mut bufs.stack);
        let image = adjoint_image(stream.operator, rhs, rhs_norm, &mut bufs.av, g, stack)?;
        if !image.g_norm.is_finite() {
            return Ok(Magnitude::from(image.g_norm));
        }
        // `gᵀM⁻¹g` overflows past `‖g‖ ≈ 1e154`, but `M⁻¹` of a subnormal `g` made unit may too.
        let scale = if image.g_norm.is_normal() {
            image.g_norm
        } else {
            1.0
        };
        normalize(g, scale);
        stream
            .preconditioner
            .apply_with(g, mv, MemStack::new(&mut bufs.stack))?;
        let alpha = alpha_from_vp(mv, g)?;
        Ok(Magnitude::product(ldexp(1.0, image.k), scale) * Magnitude::from(alpha))
    }

    /// Seed the sequence from the staged `u` with `β₁ = ‖u‖`, yielding the stream and `(α₁, β₁)`.
    pub(super) fn restart(
        self,
        beta: f64,
    ) -> Result<(ModifiedGolubKahan<'a, A, M>, BidiagStep), SolveError> {
        let mut stream = self.0;
        let step1 = stream.restart(beta)?;
        Ok((stream, step1))
    }
}

impl<'a, A: Operator + ?Sized, M: Operator + ?Sized> ModifiedGolubKahan<'a, A, M> {
    fn new(operator: &'a A, preconditioner: &'a M, local_size: usize) -> Self {
        let (m, n) = (operator.nrows(), operator.ncols());
        Self {
            operator,
            preconditioner,
            bufs: ModifiedGolubKahanBuffers::new(
                m,
                n,
                local_size,
                operator.scratch().or(preconditioner.scratch()),
            ),
            alpha: 0.0,
            u_norm_inv: 1.0,
            floor: None,
        }
    }

    /// Scaling by `β / α_k` cancels the stored `α_k`; requires `α_k > 0`.
    fn update_p_tilde(&mut self, beta: f64) -> Result<(), SolveError> {
        let bufs = &mut self.bufs;
        self.operator
            .apply_adjoint_with(&bufs.u, &mut bufs.atu, MemStack::new(&mut bufs.stack))?;
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
        let bufs = &mut self.bufs;
        self.preconditioner.apply_with(
            &bufs.p_tilde,
            &mut bufs.v,
            MemStack::new(&mut bufs.stack),
        )?;

        if let Some(reorth) = &self.bufs.local_reorth {
            reorth.reorthogonalize(&mut self.bufs.v, &mut self.bufs.p_tilde);
        }

        let alpha_new = match alpha_from_vp(&self.bufs.v, &self.bufs.p_tilde) {
            // MGS updates `v` and `p̃` apart, so near breakdown `v` drifts from `M⁻¹ p̃`.
            Err(AlphaError::NegativeMetric) if self.bufs.local_reorth.is_some() => {
                let bufs = &mut self.bufs;
                self.preconditioner.apply_with(
                    &bufs.p_tilde,
                    &mut bufs.v,
                    MemStack::new(&mut bufs.stack),
                )?;
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
