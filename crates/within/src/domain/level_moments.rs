//! Per-level weighted moments of a term's raw slopes, the input to slope whitening.

use super::{row_weight, Design};

/// One-pass weighted within-level moments (multivariate Welford); structural
/// zeros stay exact, so rank drops survive a zero tolerance.
pub(crate) struct LevelMoments {
    v: usize,
    intercept: bool,
    w_sum: Vec<DoubleWord>,
    /// Running Welford mean while observing, the faithful `Σwz / Σw` once built.
    mean: Vec<f64>,
    wz_sum: Vec<DoubleWord>,
    /// Per level, `Σ w (z−μ)(z−μ)ᵀ` packed as a row-major lower triangle.
    comoment: Vec<f64>,
}

/// Index of `(j, k)`, `k ≤ j`, in a packed row-major lower triangle.
fn tri_index(j: usize, k: usize) -> usize {
    j * (j + 1) / 2 + k
}

fn tri_len(v: usize) -> usize {
    v * (v + 1) / 2
}

fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let b_virtual = s - a;
    (s, (a - (s - b_virtual)) + (b - b_virtual))
}

/// Ogita–Rump–Oishi Dot2 accumulator: `hi` is the plain running sum, `lo` its rounding errors.
#[derive(Clone, Copy, Default)]
struct DoubleWord {
    hi: f64,
    lo: f64,
}

impl DoubleWord {
    fn add_product(&mut self, a: f64, b: f64) {
        let p = a * b;
        let (hi, err) = two_sum(self.hi, p);
        self.hi = hi;
        self.lo += err + a.mul_add(b, -p);
    }

    /// Faithful when `n²u(κ+1) < ¼`, `κ = Σ|wz|/|Σwz|`, so a representable quotient is exact.
    fn quotient(self, divisor: Self) -> f64 {
        let (s, s_lo) = two_sum(self.hi, self.lo);
        let (d, d_lo) = two_sum(divisor.hi, divisor.lo);
        let q = s / d;
        // The remainder of a rounded quotient is representable, so this fma is exact.
        let r = (-q).mul_add(d, s) + s_lo - q * d_lo;
        q + r / d
    }
}

impl LevelMoments {
    pub(crate) fn build(design: &Design<'_>, term: usize, sqrt_weights: Option<&[f64]>) -> Self {
        let t = &design.terms[term];
        let levels = t.levels();
        let zs: Vec<&[f64]> = t.raw_slopes().collect();
        let v = zs.len();
        let mut moments = Self {
            v,
            intercept: t.intercept,
            w_sum: vec![DoubleWord::default(); t.n_levels()],
            mean: vec![0.0; t.n_levels() * v],
            wz_sum: vec![DoubleWord::default(); t.n_levels() * v],
            comoment: vec![0.0; t.n_levels() * tri_len(v)],
        };
        let mut z_row = vec![0.0; v];
        let mut delta = vec![0.0; v];
        for (obs, &level) in levels.iter().enumerate() {
            for (zr, col) in z_row.iter_mut().zip(&zs) {
                *zr = col[obs];
            }
            let w = row_weight(sqrt_weights, obs);
            moments.observe(level as usize, &z_row, w, &mut delta);
        }
        // Welford's mean drifts by ulps, so a row at the exact mean would centre to noise, not 0.
        for (level, w_sum) in moments.w_sum.iter().enumerate() {
            if w_sum.hi > 0.0 {
                for (m, wz) in moments.mean[level * v..][..v]
                    .iter_mut()
                    .zip(&moments.wz_sum[level * v..])
                {
                    *m = wz.quotient(*w_sum);
                }
            }
        }
        moments
    }

    fn observe(&mut self, level: usize, z: &[f64], w: f64, delta: &mut [f64]) {
        if w <= 0.0 {
            return;
        }
        let v = self.v;
        self.w_sum[level].add_product(w, 1.0);
        let ratio = w / self.w_sum[level].hi;
        for (wz, &zj) in self.wz_sum[level * v..][..v].iter_mut().zip(z) {
            wz.add_product(w, zj);
        }
        let mean = &mut self.mean[level * v..][..v];
        for (dj, (zj, mj)) in delta.iter_mut().zip(z.iter().zip(mean.iter_mut())) {
            *dj = zj - *mj;
            *mj += ratio * *dj;
        }
        let com = &mut self.comoment[level * tri_len(v)..][..tri_len(v)];
        for j in 0..v {
            let dev = w * (z[j] - mean[j]);
            for k in 0..=j {
                com[tri_index(j, k)] += delta[k] * dev;
            }
        }
    }

    pub(crate) fn w_sum(&self, level: usize) -> f64 {
        self.w_sum[level].hi
    }

    pub(crate) fn mean(&self, level: usize) -> &[f64] {
        let v = self.v;
        &self.mean[level * v..][..v]
    }

    /// The level's row-major `v×v` Gramian: centered with an intercept, `M2 + w·μμᵀ` without one.
    pub(crate) fn fill_gram(&self, level: usize, gram: &mut [f64]) {
        let v = self.v;
        let com = &self.comoment[level * tri_len(v)..][..tri_len(v)];
        let mean = self.mean(level);
        let w = self.w_sum[level].hi;
        for j in 0..v {
            for k in 0..=j {
                let mut g = com[tri_index(j, k)];
                if !self.intercept {
                    g += w * mean[j] * mean[k];
                }
                gram[j * v + k] = g;
                gram[k * v + j] = g;
            }
        }
    }
}
