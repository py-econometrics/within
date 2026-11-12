//! Within-level reparametrization of a design's varying-slope terms.

use super::level_moments::{LevelMean, LevelMoments};
use super::{row_weight, Design, Term};
use crate::channel::{Channel, CoefficientPosition};
use crate::linalg::{add_compensated, dot, GramBasisWorkspace};

#[cfg(test)]
mod tests;

/// A slope term's change of basis to an identity within-level Gram.
pub(crate) struct TermReparam {
    /// The term's slopes in the solve basis, in coefficient-column order.
    pub(super) slopes: Vec<Vec<f64>>,
    /// Per level `u = W·z − shift`, laid out like [`Sweeps`].
    w: Vec<f64>,
    shift: Vec<f64>,
    /// Directions the data cannot identify, ascending in `(level, column)`.
    pub(super) unidentified: Vec<CoefficientPosition>,
}

/// Per level `u = W·(z − center)`, `W` in a row-major `V×V` slot whose rows past the rank are
/// zero, so an unidentified direction maps to `0`.
struct Sweeps {
    v: usize,
    w: Vec<f64>,
    /// Within-level weighted means; all-zero for slope-only terms.
    center: Vec<LevelMean>,
}

impl Sweeps {
    fn new(n_levels: usize, v: usize) -> Self {
        Self {
            v,
            w: vec![0.0; n_levels * v * v],
            center: vec![LevelMean::default(); n_levels * v],
        }
    }

    fn level(&self, level: usize) -> (&[f64], &[LevelMean]) {
        let v = self.v;
        (
            &self.w[level * v * v..][..v * v],
            &self.center[level * v..][..v],
        )
    }

    /// Rewrites each row of `slopes` (one `Vec` per column) by its level's sweep, handing the
    /// new row to `written`.
    fn apply(
        &self,
        levels: &[u32],
        slopes: &mut [Vec<f64>],
        mut written: impl FnMut(usize, usize, &[f64]),
    ) {
        let v = self.v;
        let (mut z_row, mut u_row) = (vec![0.0; v], vec![0.0; v]);
        for (i, &level) in levels.iter().enumerate() {
            let (w, center) = self.level(level as usize);
            for ((zr, col), c) in z_row.iter_mut().zip(&*slopes).zip(center) {
                *zr = c.centre(col[i]);
            }
            for ((ur, out), w_row) in u_row
                .iter_mut()
                .zip(slopes.iter_mut())
                .zip(w.chunks_exact(v))
            {
                *ur = dot(w_row, &z_row);
                out[i] = *ur;
            }
            written(level as usize, i, &u_row);
        }
    }

    /// `u = W₂·(W₁·(z − c₁) − c₂)` as `W = W₂·W₁`, `shift = W₂·(W₁·c₁ + c₂)`, `W` reusing
    /// the first sweep's storage.
    fn compose(self, second: &Self) -> (Vec<f64>, Vec<f64>) {
        let Self { v, mut w, center } = self;
        let mut shift = vec![0.0; center.len()];
        let (mut centre, mut composed) = (vec![0.0; v], vec![0.0; v * v]);
        for (level, (w1, s)) in w
            .chunks_exact_mut(v * v)
            .zip(shift.chunks_exact_mut(v))
            .enumerate()
        {
            let c1 = &center[level * v..][..v];
            let (w2, c2) = second.level(level);
            for ((cm, w1_row), c2m) in centre.iter_mut().zip(w1.chunks_exact(v)).zip(c2) {
                *cm = w1_row
                    .iter()
                    .zip(c1)
                    .map(|(a, c)| a * c.value())
                    .sum::<f64>()
                    + c2m.value();
            }
            composed.fill(0.0);
            for ((row, w2_row), sk) in composed
                .chunks_exact_mut(v)
                .zip(w2.chunks_exact(v))
                .zip(&mut *s)
            {
                for (&a, w1_row) in w2_row.iter().zip(w1.chunks_exact(v)) {
                    for (rj, &b) in row.iter_mut().zip(w1_row) {
                        *rj += a * b;
                    }
                }
                *sk = dot(w2_row, &centre);
            }
            w1.copy_from_slice(&composed);
        }
        (w, shift)
    }
}

/// Per level, `Σw`, `Σw·u` and `Σw·u·uᵀ` of whitened rows `u`.
struct RowSums {
    v: usize,
    w: Vec<f64>,
    u: Vec<f64>,
    /// Compensation of `u`: a mean-valued row must load exactly `0` after re-centring.
    u_err: Vec<f64>,
    uu: Vec<f64>,
}

impl RowSums {
    fn new(n_levels: usize, v: usize) -> Self {
        Self {
            v,
            w: vec![0.0; n_levels],
            u: vec![0.0; n_levels * v],
            u_err: vec![0.0; n_levels * v],
            uu: vec![0.0; n_levels * v * v],
        }
    }

    fn add(&mut self, level: usize, w: f64, row: &[f64]) {
        let v = self.v;
        self.w[level] += w;
        let u = &mut self.u[level * v..][..v];
        let u_err = &mut self.u_err[level * v..][..v];
        let uu = &mut self.uu[level * v * v..][..v * v];
        for (j, &rj) in row.iter().enumerate() {
            add_compensated(&mut u[j], &mut u_err[j], w * rj);
            for (k, &rk) in row[..=j].iter().enumerate() {
                uu[j * v + k] += w * rj * rk;
            }
        }
    }

    /// The level's weighted mean into `out`; left at `0` without centring.
    fn mean_into(&self, level: usize, out: &mut [LevelMean]) {
        let w = self.w[level];
        if w > 0.0 {
            let range = level * self.v..(level + 1) * self.v;
            for ((m, &s), &e) in out
                .iter_mut()
                .zip(&self.u[range.clone()])
                .zip(&self.u_err[range])
            {
                *m = LevelMean::new((s + e) / w);
            }
        }
    }

    /// `Σw·(u − m)(u − m)ᵀ = Σw·uuᵀ − s·mmᵀ`, exact enough since `m` is rounding-small.
    fn fill_gram(&self, level: usize, mean: &[LevelMean], gram: &mut [f64]) {
        let v = self.v;
        let (w, uu) = (self.w[level], &self.uu[level * v * v..][..v * v]);
        for j in 0..v {
            for k in 0..=j {
                let g = uu[j * v + k] - w * mean[j].value() * mean[k].value();
                gram[j * v + k] = g;
                gram[k * v + j] = g;
            }
        }
    }
}

impl TermReparam {
    /// `None` without slopes; unidentified directions become zero columns, so they solve to `0`.
    /// Whitening twice (CholeskyQR2) leaves an orthonormality error of eps, not eps·κ(G).
    pub(crate) fn build(
        design: &Design<'_>,
        term: usize,
        sqrt_weights: Option<&[f64]>,
    ) -> Option<Self> {
        let t = &design.terms[term];
        if !t.has_slopes() {
            return None;
        }
        let levels = t.levels();
        let n_levels = t.n_levels();
        let mut slopes: Vec<Vec<f64>> = t.raw_slopes().map(<[f64]>::to_vec).collect();
        let v = slopes.len();
        let mut workspace = GramBasisWorkspace::new(v);

        let mut unidentified = Vec::new();
        let mut first = Sweeps::new(n_levels, v);
        let raw: Vec<&[f64]> = slopes.iter().map(Vec::as_slice).collect();
        let moments = LevelMoments::build(t, &raw, sqrt_weights);
        for level in 0..n_levels {
            let basis = workspace.orthonormalize(|gram| moments.fill_gram(level, gram));
            first.w[level * v * v..][..basis.rows.len()].copy_from_slice(basis.rows);
            if let Some(mean) = moments.mean(level) {
                first.center[level * v..][..v].copy_from_slice(mean);
            }
            if moments.w_sum(level) == Some(0.0) {
                unidentified.push(CoefficientPosition {
                    channel: Channel { term, column: 0 },
                    level,
                });
            }
            for j in (0..v).filter(|&j| !basis.kept[j]) {
                unidentified.push(CoefficientPosition {
                    channel: Channel {
                        term,
                        column: t.slope_column(j),
                    },
                    level,
                });
            }
        }
        drop(moments);
        let mut sums = RowSums::new(n_levels, v);
        first.apply(levels, &mut slopes, |level, i, u| {
            let w = row_weight(sqrt_weights, i);
            if w > 0.0 {
                sums.add(level, w, u);
            }
        });

        // The rank tolerance keeps the first sweep well-conditioned, so this one drops nothing.
        let mut second = Sweeps::new(n_levels, v);
        for level in 0..n_levels {
            let center = &mut second.center[level * v..][..v];
            if t.intercept {
                sums.mean_into(level, center);
            }
            let basis = workspace.orthonormalize(|gram| sums.fill_gram(level, center, gram));
            second.w[level * v * v..][..basis.rows.len()].copy_from_slice(basis.rows);
        }
        drop(sums);
        second.apply(levels, &mut slopes, |_, _, _| {});

        let (w, shift) = first.compose(&second);
        Some(Self {
            slopes,
            w,
            shift,
            unidentified,
        })
    }

    /// Map this term's solve-basis coefficients back; slots outside its block are untouched.
    pub(crate) fn back_transform(&self, term: &Term<'_>, x: &mut [f64]) {
        let slope_slot =
            |j: usize, level: usize| term.column_dofs(term.slope_column(j)).start + level;
        let v = self.slopes.len();
        let mut b = vec![0.0; v];
        for (l, (w, shift)) in self
            .w
            .chunks_exact(v * v)
            .zip(self.shift.chunks_exact(v))
            .enumerate()
        {
            b.fill(0.0);
            let mut offset = 0.0;
            for ((k, w_row), s) in w.chunks_exact(v).enumerate().zip(shift) {
                let bk = x[slope_slot(k, l)];
                offset += bk * s;
                for (bj, wj) in b.iter_mut().zip(w_row) {
                    *bj += wj * bk;
                }
            }
            for (j, &bj) in b.iter().enumerate() {
                x[slope_slot(j, l)] = bj;
            }
            if term.intercept {
                x[term.column_dofs(0).start + l] -= offset;
            }
        }
    }
}
