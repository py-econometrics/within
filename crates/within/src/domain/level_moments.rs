//! Per-level weighted moments of a term's loading columns, the input to slope whitening.

use super::{row_weight, Design};

/// One-pass weighted within-level moments (multivariate Welford); structural
/// zeros stay exact, so rank drops survive a zero tolerance.
pub(crate) struct LevelMoments {
    v: usize,
    intercept: bool,
    w_sum: Vec<f64>,
    mean: Vec<f64>,
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

impl LevelMoments {
    pub(crate) fn build(design: &Design<'_>, term: usize, sqrt_weights: Option<&[f64]>) -> Self {
        let t = &design.terms[term];
        let (layout, levels) = (&t.layout, t.levels());
        let zs: Vec<&[f64]> = layout
            .covariates()
            .map(|c| design.raw_loading_column(c as usize))
            .collect();
        let v = zs.len();
        let mut moments = Self {
            v,
            intercept: layout.has_intercept(),
            w_sum: vec![0.0; layout.n_levels()],
            mean: vec![0.0; layout.n_levels() * v],
            comoment: vec![0.0; layout.n_levels() * tri_len(v)],
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
        moments
    }

    fn observe(&mut self, level: usize, z: &[f64], w: f64, delta: &mut [f64]) {
        if w <= 0.0 {
            return;
        }
        let v = self.v;
        self.w_sum[level] += w;
        let ratio = w / self.w_sum[level];
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
        self.w_sum[level]
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
        let w = self.w_sum[level];
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
