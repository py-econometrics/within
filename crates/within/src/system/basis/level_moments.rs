//! Per-level weighted moments of a term's slopes, the input to one orthogonalization pass.

use super::super::RowWeights;
use crate::design::TermIncidence;

/// Slope columns one row pass folds into their means.
const MEANS_PER_PASS: usize = 4;
/// Gram entries one row pass folds.
const GRAMS_PER_PASS: usize = 6;

/// Weighted within-level mean and Gram, two-pass so structural zeros stay exact.
pub(super) struct LevelMoments {
    /// Slope columns per level: the stride of the means and the side of each level's Gram.
    n_slopes: usize,
    /// Each level's `v` weighted slope means, row-major; `None` without an intercept.
    means: Option<Vec<LevelMean>>,
    /// Per level, the packed lower triangle of `Σ w (z_j−c_j)(z_k−c_k)`, `c` its means or `0`.
    gram: Vec<f64>,
}

/// Index of `(j, k)`, `k ≤ j`, in a packed row-major lower triangle.
fn tri_index(j: usize, k: usize) -> usize {
    j * (j + 1) / 2 + k
}

/// A term's positive-weight rows, folded per level in the pieces its reduction fixes.
struct WeightedRows<'a> {
    term: TermIncidence<'a>,
    n_levels: usize,
    weights: &'a RowWeights,
}

impl WeightedRows<'_> {
    /// Per level, `init` folded by `step(state, level, obs, w)`, pieces combined by `merge`.
    fn fold<S: Copy + Send + Sync>(
        &self,
        init: S,
        step: impl Fn(&mut S, usize, usize, f64) + Sync,
        merge: impl Fn(&mut S, &S) + Sync,
    ) -> Vec<S> {
        let mut states = vec![init; self.n_levels];
        self.term.fold(
            init,
            &|s: &mut S, level, obs| {
                let w = self.weights.weight(obs);
                if w > 0.0 {
                    step(s, level, obs, w);
                }
            },
            &merge,
            &mut states[..],
        );
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

    fn plus(self, x: f64) -> LevelMean {
        let (hi, lo) = two_sum(x, self.hi);
        let (hi, lo) = two_sum(hi, lo + self.lo);
        LevelMean { hi, lo }
    }
}

/// A weighted level mean kept as `hi + lo`, so centring carries no eps·|mean| error.
#[derive(Clone, Copy, Default)]
pub(super) struct LevelMean {
    hi: f64,
    lo: f64,
}

impl LevelMean {
    pub(super) fn value(self) -> f64 {
        self.hi + self.lo
    }

    pub(super) fn centre(self, z: f64) -> f64 {
        (z - self.hi) - self.lo
    }
}

/// Weighted means `s + Σw(z−s)/Σw` of `V` columns over the same rows, `s` the first row, so the
/// sums scale with the spread.
#[derive(Clone, Copy)]
struct ShiftedMeans<const V: usize> {
    w_sum: DoubleWord,
    shift: [f64; V],
    sum: [DoubleWord; V],
}

impl<const V: usize> ShiftedMeans<V> {
    const EMPTY: Self = Self {
        w_sum: DoubleWord { hi: 0.0, lo: 0.0 },
        shift: [0.0; V],
        sum: [DoubleWord { hi: 0.0, lo: 0.0 }; V],
    };

    fn add(&mut self, w: f64, z: [f64; V]) {
        if self.w_sum.hi == 0.0 {
            self.shift = z;
        }
        for ((sum, &zj), &shift) in self.sum.iter_mut().zip(&z).zip(&self.shift) {
            let (d, e) = two_sum(zj, -shift);
            sum.add_product(w, d);
            sum.lo += w * e;
        }
        self.w_sum.add(w);
    }

    /// Adds `other`'s rows: `Σw(z − a) = Σw(z − b) + 2W·(b/2 − a/2)`; halved, the shift gap
    /// stays finite across the whole float range.
    fn merge(&mut self, other: &Self) {
        if other.w_sum.hi == 0.0 {
            return;
        }
        if self.w_sum.hi == 0.0 {
            *self = *other;
            return;
        }
        let w = other.w_sum;
        for j in 0..V {
            let (d, e) = two_sum(0.5 * other.shift[j], -0.5 * self.shift[j]);
            let sum = &mut self.sum[j];
            sum.add(other.sum[j].hi);
            sum.lo += other.sum[j].lo;
            sum.add_product(w.hi, d);
            sum.add_product(w.hi, d);
            sum.lo += 2.0 * (w.hi * e + w.lo * d);
        }
        self.w_sum.add(w.hi);
        self.w_sum.lo += w.lo;
    }

    /// Column `j`'s mean; `0` for a level without weight.
    fn mean(&self, j: usize) -> LevelMean {
        if self.w_sum.hi == 0.0 {
            return LevelMean::default();
        }
        self.sum[j].quotient(self.w_sum).plus(self.shift[j])
    }

    fn overflowed(&self) -> bool {
        self.sum.iter().any(|s| !s.hi.is_finite())
    }
}

/// Columns `first..first + V` of each level's means, in one row pass.
fn fold_means<const V: usize>(
    rows: &WeightedRows<'_>,
    zs: [&[f64]; V],
    first: usize,
    means: &mut [LevelMean],
) {
    let mut levels = rows.fold(
        ShiftedMeans::<V>::EMPTY,
        |m, _, obs, w| m.add(w, zs.map(|z| z[obs])),
        ShiftedMeans::merge,
    );
    // A spread near the float range overflows Σw(z−s); a power-of-two weight scale is exact.
    if levels.iter().any(ShiftedMeans::overflowed) {
        let scale = 2f64.powi(-64);
        levels = rows.fold(
            ShiftedMeans::<V>::EMPTY,
            |m, _, obs, w| m.add(w * scale, zs.map(|z| z[obs])),
            ShiftedMeans::merge,
        );
    }
    let v = means.len() / rows.n_levels;
    for (level, m) in levels.iter().enumerate() {
        for j in 0..V {
            means[level * v + first + j] = m.mean(j);
        }
    }
}

/// Each level's `v` weighted slope means, row-major, `MEANS_PER_PASS` columns per row pass.
fn level_means(rows: &WeightedRows<'_>, zs: &[&[f64]]) -> Vec<LevelMean> {
    let mut means = vec![LevelMean::default(); rows.n_levels * zs.len()];
    for (pass, group) in zs.chunks(MEANS_PER_PASS).enumerate() {
        let first = pass * MEANS_PER_PASS;
        match *group {
            [a] => fold_means(rows, [a], first, &mut means),
            [a, b] => fold_means(rows, [a, b], first, &mut means),
            [a, b, c] => fold_means(rows, [a, b, c], first, &mut means),
            [a, b, c, d] => fold_means(rows, [a, b, c, d], first, &mut means),
            _ => unreachable!("chunks hold at most MEANS_PER_PASS columns"),
        }
    }
    means
}

/// Packed Gram entries `first..first + P`, the pairs `pairs`, in one row pass.
fn fold_gram<const P: usize>(
    rows: &WeightedRows<'_>,
    zs: &[&[f64]],
    means: Option<&[LevelMean]>,
    pairs: &[(usize, usize)],
    first: usize,
    gram: &mut [f64],
) {
    let pairs: [(usize, usize); P] = pairs.try_into().expect("a chunk of P pairs");
    let v = zs.len();
    let columns = pairs.map(|(j, k)| (zs[j], zs[k]));
    let add = |l: &mut [f64; P], r: &[f64; P]| {
        for (l, r) in l.iter_mut().zip(r) {
            *l += r;
        }
    };
    let levels = match means {
        Some(m) => rows.fold(
            [0.0; P],
            |g, level, obs, w| {
                let c = &m[level * v..];
                for ((g, &(j, k)), &(zj, zk)) in g.iter_mut().zip(&pairs).zip(&columns) {
                    *g += w * c[j].centre(zj[obs]) * c[k].centre(zk[obs]);
                }
            },
            add,
        ),
        None => rows.fold(
            [0.0; P],
            |g, _, obs, w| {
                for (g, &(zj, zk)) in g.iter_mut().zip(&columns) {
                    *g += w * zj[obs] * zk[obs];
                }
            },
            add,
        ),
    };
    let n_pairs = gram.len() / rows.n_levels;
    for (level, g) in levels.iter().enumerate() {
        gram[level * n_pairs + first..][..P].copy_from_slice(g);
    }
}

impl LevelMoments {
    pub(super) fn build(term: TermIncidence<'_>, zs: &[&[f64]], weights: &RowWeights) -> Self {
        let layout = term.layout();
        let rows = WeightedRows {
            term,
            n_levels: layout.n_levels(),
            weights,
        };
        let v = zs.len();
        let means = layout.intercept().then(|| level_means(&rows, zs));
        let m = means.as_deref();
        let pairs: Vec<(usize, usize)> =
            (0..v).flat_map(|j| (0..=j).map(move |k| (j, k))).collect();
        let mut gram = vec![0.0; rows.n_levels * pairs.len()];
        for (pass, group) in pairs.chunks(GRAMS_PER_PASS).enumerate() {
            let first = pass * GRAMS_PER_PASS;
            match group.len() {
                1 => fold_gram::<1>(&rows, zs, m, group, first, &mut gram),
                2 => fold_gram::<2>(&rows, zs, m, group, first, &mut gram),
                3 => fold_gram::<3>(&rows, zs, m, group, first, &mut gram),
                4 => fold_gram::<4>(&rows, zs, m, group, first, &mut gram),
                5 => fold_gram::<5>(&rows, zs, m, group, first, &mut gram),
                6 => fold_gram::<6>(&rows, zs, m, group, first, &mut gram),
                _ => unreachable!("chunks hold at most GRAMS_PER_PASS entries"),
            }
        }
        Self {
            n_slopes: v,
            means,
            gram,
        }
    }

    /// The level's weighted slope means, tracked only with an intercept.
    pub(super) fn mean(&self, level: usize) -> Option<&[LevelMean]> {
        self.means
            .as_ref()
            .map(|m| &m[level * self.n_slopes..][..self.n_slopes])
    }

    /// The level's Gram unpacked into a row-major `v×v` matrix.
    pub(super) fn fill_gram(&self, level: usize, gram: &mut [f64]) {
        let v = self.n_slopes;
        let packed = &self.gram[level * v * (v + 1) / 2..];
        for j in 0..v {
            for k in 0..=j {
                let g = packed[tri_index(j, k)];
                gram[j * v + k] = g;
                gram[k * v + j] = g;
            }
        }
    }
}
