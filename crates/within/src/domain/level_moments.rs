//! Per-level weighted moments of a term's raw slopes, the input to slope whitening.

use std::ops::Range;

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

/// Each maximal run of equal levels as `(level, rows)`.
fn level_runs(levels: &[u32]) -> impl Iterator<Item = (usize, Range<usize>)> + '_ {
    let mut start = 0;
    levels.chunk_by(|a, b| a == b).map(move |run| {
        let rows = start..start + run.len();
        start = rows.end;
        (run[0] as usize, rows)
    })
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
}

/// One level's `Σw(z−s)` for one slope, shifted by its first row to scale with the spread.
#[derive(Clone, Copy, Default)]
struct ShiftedSum {
    shift: f64,
    sum: DoubleWord,
}

impl ShiftedSum {
    fn add(&mut self, w: f64, z: f64) {
        let (d, e) = two_sum(z, -self.shift);
        self.sum.add_product(w, d);
        self.sum.lo += w * e;
    }

    fn mean(self, w_sum: DoubleWord) -> f64 {
        self.sum.quotient(w_sum).plus(self.shift)
    }
}

impl LevelMoments {
    pub(crate) fn build(design: &Design<'_>, term: usize, sqrt_weights: Option<&[f64]>) -> Self {
        let t = &design.terms[term];
        let zs: Vec<&[f64]> = t.raw_slopes().collect();
        let v = zs.len();
        let n_levels = t.n_levels();
        let weight = |obs: usize| row_weight(sqrt_weights, obs);

        // Runs sum in registers, off the per-row store chain; random orders are all singletons.
        let mut w_sum = vec![DoubleWord::default(); n_levels];
        let n_shifted = if t.intercept { v } else { 0 };
        let mut shifted = vec![ShiftedSum::default(); n_levels * n_shifted];
        for (level, rows) in level_runs(t.levels()) {
            let level_w = &mut w_sum[level];
            let accs = &mut shifted[level * n_shifted..][..n_shifted];
            if rows.len() == 1 {
                let obs = rows.start;
                let w = weight(obs);
                if w > 0.0 {
                    let first = level_w.hi == 0.0;
                    for (acc, col) in accs.iter_mut().zip(&zs) {
                        if first {
                            acc.shift = col[obs];
                        }
                        acc.add(w, col[obs]);
                    }
                    level_w.add(w);
                }
                continue;
            }
            let first = if level_w.hi == 0.0 {
                rows.clone().find(|&obs| weight(obs) > 0.0)
            } else {
                None
            };
            let mut run_w = *level_w;
            for obs in rows.clone() {
                let w = weight(obs);
                if w > 0.0 {
                    run_w.add(w);
                }
            }
            *level_w = run_w;
            for (acc, col) in accs.iter_mut().zip(&zs) {
                let mut run_acc = *acc;
                if let Some(obs) = first {
                    run_acc.shift = col[obs];
                }
                for obs in rows.clone() {
                    let w = weight(obs);
                    if w > 0.0 {
                        run_acc.add(w, col[obs]);
                    }
                }
                *acc = run_acc;
            }
        }
        let mut center = vec![0.0; n_levels * v];
        if t.intercept {
            for (i, c) in center.iter_mut().enumerate() {
                let w = w_sum[i / v];
                if w.hi > 0.0 {
                    *c = shifted[i].mean(w);
                }
            }
        }

        let mut gram = vec![0.0; n_levels * tri_len(v)];
        let mut dev = vec![0.0; v];
        for (level, rows) in level_runs(t.levels()) {
            let c = &center[level * v..][..v];
            let g = &mut gram[level * tri_len(v)..][..tri_len(v)];
            if rows.len() == 1 {
                let obs = rows.start;
                let w = weight(obs);
                if w > 0.0 {
                    for ((d, col), cj) in dev.iter_mut().zip(&zs).zip(c) {
                        *d = col[obs] - cj;
                    }
                    for j in 0..v {
                        let wd = w * dev[j];
                        for k in 0..=j {
                            g[tri_index(j, k)] += wd * dev[k];
                        }
                    }
                }
                continue;
            }
            for j in 0..v {
                for k in 0..=j {
                    let mut acc = g[tri_index(j, k)];
                    for obs in rows.clone() {
                        let w = weight(obs);
                        if w > 0.0 {
                            acc += w * (zs[j][obs] - c[j]) * (zs[k][obs] - c[k]);
                        }
                    }
                    g[tri_index(j, k)] = acc;
                }
            }
        }

        Self {
            v,
            w_sum: w_sum.iter().map(|w| w.hi + w.lo).collect(),
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
