//! Per-level weighted moments of a term's raw slopes, the input to slope whitening.

use super::{row_weight, Design};

/// Weighted within-level mean and Gram, two-pass so structural zeros stay exact.
pub(crate) struct LevelMoments {
    v: usize,
    w_sum: Vec<f64>,
    /// Per level, the weighted mean with an intercept, `0` without one.
    center: Vec<f64>,
    /// Per level, `Σ w (z−c)(z−c)ᵀ` about the center, packed as a row-major lower triangle.
    gram: Vec<f64>,
}

/// Index of `(j, k)`, `k ≤ j`, in a packed row-major lower triangle.
fn tri_index(j: usize, k: usize) -> usize {
    j * (j + 1) / 2 + k
}

fn tri_len(v: usize) -> usize {
    v * (v + 1) / 2
}

/// A term's positive-weight rows, folded per level.
struct WeightedRows<'a> {
    levels: &'a [u32],
    n_levels: usize,
    sqrt_weights: Option<&'a [f64]>,
}

impl WeightedRows<'_> {
    /// Per level, `init` folded by `step(state, level, obs, w)` over the level's rows in row order.
    fn fold<S: Copy>(&self, init: S, step: impl Fn(S, usize, usize, f64) -> S) -> Vec<S> {
        let mut states = vec![init; self.n_levels];
        let mut start = 0;
        // A run of equal levels keeps the state in registers instead of round-tripping memory.
        for run in self.levels.chunk_by(|a, b| a == b) {
            let level = run[0] as usize;
            let rows = start..start + run.len();
            start = rows.end;
            let state = &mut states[level];
            *state = rows.fold(*state, |s, obs| {
                let w = row_weight(self.sqrt_weights, obs);
                if w > 0.0 {
                    step(s, level, obs, w)
                } else {
                    s
                }
            });
        }
        states
    }
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
    fn add(&mut self, x: f64) {
        let (hi, err) = two_sum(self.hi, x);
        self.hi = hi;
        self.lo += err;
    }

    fn add_product(&mut self, a: f64, b: f64) {
        let p = a * b;
        self.add(p);
        self.lo += a.mul_add(b, -p);
    }

    fn quotient(self, divisor: Self) -> Self {
        let (s, s_lo) = two_sum(self.hi, self.lo);
        let (d, d_lo) = two_sum(divisor.hi, divisor.lo);
        let q = s / d;
        // The remainder of a rounded quotient is representable, so this fma is exact.
        let r = (-q).mul_add(d, s) + s_lo - q * d_lo;
        Self { hi: q, lo: r / d }
    }

    fn plus(self, x: f64) -> f64 {
        let (hi, lo) = two_sum(x, self.hi);
        hi + (lo + self.lo)
    }

    fn value(self) -> f64 {
        self.hi + self.lo
    }
}

/// Weighted mean `s + Σw(z−s)/Σw`, `s` the first row, so the sums scale with the spread.
#[derive(Clone, Copy, Default)]
struct ShiftedMean {
    shift: f64,
    w_sum: DoubleWord,
    sum: DoubleWord,
}

impl ShiftedMean {
    fn add(mut self, w: f64, z: f64) -> Self {
        if self.w_sum.hi == 0.0 {
            self.shift = z;
        }
        let (d, e) = two_sum(z, -self.shift);
        self.sum.add_product(w, d);
        self.sum.lo += w * e;
        self.w_sum.add(w);
        self
    }

    /// `0` for a level without weight.
    fn mean(self) -> f64 {
        if self.w_sum.hi == 0.0 {
            return 0.0;
        }
        self.sum.quotient(self.w_sum).plus(self.shift)
    }
}

impl LevelMoments {
    pub(crate) fn build(design: &Design<'_>, term: usize, sqrt_weights: Option<&[f64]>) -> Self {
        let t = &design.terms[term];
        let zs: Vec<&[f64]> = t.raw_slopes().collect();
        let v = zs.len();
        let n_levels = t.n_levels();
        let rows = WeightedRows {
            levels: t.levels(),
            n_levels,
            sqrt_weights,
        };

        let mut center = vec![0.0; n_levels * v];
        let w_sum: Vec<f64> = if t.intercept {
            let means: Vec<Vec<ShiftedMean>> = zs
                .iter()
                .map(|z| rows.fold(ShiftedMean::default(), |m, _, obs, w| m.add(w, z[obs])))
                .collect();
            for (j, column) in means.iter().enumerate() {
                for (level, m) in column.iter().enumerate() {
                    center[level * v + j] = m.mean();
                }
            }
            means[0].iter().map(|m| m.w_sum.value()).collect()
        } else {
            let w_sums = rows.fold(DoubleWord::default(), |mut s, _, _, w| {
                s.add(w);
                s
            });
            w_sums.into_iter().map(DoubleWord::value).collect()
        };

        let mut gram = vec![0.0; n_levels * tri_len(v)];
        for j in 0..v {
            for k in 0..=j {
                let (zj, zk) = (zs[j], zs[k]);
                let entries = rows.fold(0.0, |g, level, obs, w| {
                    g + w * (zj[obs] - center[level * v + j]) * (zk[obs] - center[level * v + k])
                });
                for (level, g) in entries.into_iter().enumerate() {
                    gram[level * tri_len(v) + tri_index(j, k)] = g;
                }
            }
        }

        Self {
            v,
            w_sum,
            center,
            gram,
        }
    }

    pub(crate) fn w_sum(&self, level: usize) -> f64 {
        self.w_sum[level]
    }

    pub(crate) fn center(&self, level: usize) -> &[f64] {
        let v = self.v;
        &self.center[level * v..][..v]
    }

    /// The level's Gram unpacked into a row-major `v×v` matrix.
    pub(crate) fn fill_gram(&self, level: usize, gram: &mut [f64]) {
        let v = self.v;
        let packed = &self.gram[level * tri_len(v)..][..tri_len(v)];
        for j in 0..v {
            for k in 0..=j {
                let g = packed[tri_index(j, k)];
                gram[j * v + k] = g;
                gram[k * v + j] = g;
            }
        }
    }
}
