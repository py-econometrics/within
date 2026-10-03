//! Per-level weighted moments of a term's raw slopes, the input to slope whitening.

use super::{row_weight, Design, Term};

/// Weighted within-level mean and Gram, two-pass so structural zeros stay exact.
pub(crate) struct LevelMoments {
    /// `None` without an intercept.
    means: Option<LevelMeans>,
    gram: LevelGram,
}

/// Index of `(j, k)`, `k ≤ j`, in a packed row-major lower triangle.
fn tri_index(j: usize, k: usize) -> usize {
    j * (j + 1) / 2 + k
}

/// A term's positive-weight rows, folded per level.
struct WeightedRows<'a> {
    levels: &'a [u32],
    n_levels: usize,
    sqrt_weights: Option<&'a [f64]>,
}

impl<'a> WeightedRows<'a> {
    fn new(term: &'a Term<'_>, sqrt_weights: Option<&'a [f64]>) -> Self {
        Self {
            levels: term.levels(),
            n_levels: term.n_levels(),
            sqrt_weights,
        }
    }

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

/// Each level's weighted slope means, with the level's weight total.
struct LevelMeans {
    v: usize,
    w_sum: Vec<f64>,
    means: Vec<f64>,
}

impl LevelMeans {
    fn new(rows: &WeightedRows<'_>, zs: &[&[f64]]) -> Self {
        let v = zs.len();
        let columns: Vec<Vec<ShiftedMean>> = zs
            .iter()
            .map(|z| rows.fold(ShiftedMean::default(), |m, _, obs, w| m.add(w, z[obs])))
            .collect();
        let mut means = vec![0.0; rows.n_levels * v];
        for (j, column) in columns.iter().enumerate() {
            for (level, m) in column.iter().enumerate() {
                means[level * v + j] = m.mean();
            }
        }
        Self {
            v,
            w_sum: columns[0].iter().map(|m| m.w_sum.value()).collect(),
            means,
        }
    }

    fn of(&self, level: usize) -> &[f64] {
        &self.means[level * self.v..][..self.v]
    }
}

/// Per packed entry `(j, k)`, each level's `Σ w (z_j−c_j)(z_k−c_k)`, `c` its means or `0`.
struct LevelGram {
    v: usize,
    entries: Vec<Vec<f64>>,
}

impl LevelGram {
    fn new(rows: &WeightedRows<'_>, zs: &[&[f64]], means: Option<&LevelMeans>) -> Self {
        let entries = (0..zs.len())
            .flat_map(|j| (0..=j).map(move |k| (j, k)))
            .map(|(j, k)| {
                let (zj, zk) = (zs[j], zs[k]);
                match means {
                    Some(m) => rows.fold(0.0, |g, level, obs, w| {
                        let c = m.of(level);
                        g + w * (zj[obs] - c[j]) * (zk[obs] - c[k])
                    }),
                    None => rows.fold(0.0, |g, _, obs, w| g + w * zj[obs] * zk[obs]),
                }
            })
            .collect();
        Self {
            v: zs.len(),
            entries,
        }
    }

    /// The level's Gram unpacked into a row-major `v×v` matrix.
    fn fill(&self, level: usize, gram: &mut [f64]) {
        let v = self.v;
        for j in 0..v {
            for k in 0..=j {
                let g = self.entries[tri_index(j, k)][level];
                gram[j * v + k] = g;
                gram[k * v + j] = g;
            }
        }
    }
}

impl LevelMoments {
    pub(crate) fn build(design: &Design<'_>, term: usize, sqrt_weights: Option<&[f64]>) -> Self {
        let t = &design.terms[term];
        let zs: Vec<&[f64]> = t.raw_slopes().collect();
        let rows = WeightedRows::new(t, sqrt_weights);
        let means = t.intercept.then(|| LevelMeans::new(&rows, &zs));
        let gram = LevelGram::new(&rows, &zs, means.as_ref());
        Self { means, gram }
    }

    /// The level's weight total, tracked only with an intercept.
    pub(crate) fn w_sum(&self, level: usize) -> Option<f64> {
        self.means.as_ref().map(|m| m.w_sum[level])
    }

    /// The level's weighted slope means, tracked only with an intercept.
    pub(crate) fn mean(&self, level: usize) -> Option<&[f64]> {
        self.means.as_ref().map(|m| m.of(level))
    }

    pub(crate) fn fill_gram(&self, level: usize, gram: &mut [f64]) {
        self.gram.fill(level, gram);
    }
}
