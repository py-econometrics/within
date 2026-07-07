//! Within-level reparametrization of a design's sole varying-slope term.

use crate::domain::Design;

use super::UnidentifiedDirection;

#[cfg(test)]
mod tests;

/// Relative rank tolerance: a slope direction is dropped once its remaining
/// within-level variance is no longer above `RANK_TOL` × its own initial
/// variance. Zero keeps every direction whose remaining variance stays
/// positive, so only exact structural zeros drop.
const RANK_TOL: f64 = 1e-10;

/// Per-level change of basis for the design's sole slope term: the solve sees
/// weighted-orthonormal slope directions that are uncorrelated with the
/// level's intercept, leaving fitted values invariant while the per-level
/// Gram becomes diagonal. [`Self::back_transform`] returns coefficients to
/// the user's parametrization.
pub(crate) struct SlopeReparam {
    offset: usize,
    n_levels: usize,
    intercept: bool,
    n_slopes: usize,
    transforms: Vec<LevelTransform>,
    /// Directions the data cannot identify, ascending in `(level, column)`.
    pub(crate) unidentified: Vec<UnidentifiedDirection>,
}

/// One level's whitening map `u = W·(z − center)` with `W` of shape
/// `rank × V` (row-major). `Wᵀ` maps solved coefficients back to the user's
/// basis; a dropped slope is a zero column of `W`, so its coefficient comes
/// back as an exact `0`. An empty `w` marks a level whose every slope
/// direction dropped (or an empty level).
struct LevelTransform {
    w: Box<[f64]>,
    /// Within-level weighted means; all-zero for slope-only terms.
    center: Box<[f64]>,
}

impl SlopeReparam {
    /// Whiten the sole slope term's loading columns in place and record how
    /// to undo it. Returns `None` for slope-free designs.
    ///
    /// Identification is judged over positive-weight rows: an empty level
    /// drops the term's every column; within-level rank deficiency (constant
    /// or collinear slopes, per [`RANK_TOL`]) drops the directions the pivot
    /// order rejects, keeping the largest-remaining-variance direction at
    /// each step. Dropped directions are materialized as exact zeros, so the
    /// minimal-norm LSMR solution leaves an exact `0` in the coefficient
    /// slot and the layout stays data-independent.
    pub(crate) fn build(design: &mut Design<'_>, weights: Option<&[f64]>) -> Option<Self> {
        let term = design.terms.iter().position(|t| !t.slopes.is_empty())?;
        let meta = &design.terms[term];
        let (offset, n_levels, intercept) = (meta.offset, meta.n_levels, meta.intercept);
        let z_cols = meta.slopes.clone();
        let v = z_cols.len();
        let levels = design.frame.level_column(term);
        let zs: Vec<&[f64]> = z_cols
            .iter()
            .map(|&c| design.frame.loading_column(c))
            .collect();

        let mut moments = LevelMoments::new(n_levels, v);
        let mut z_row = vec![0.0; v];
        for (i, &level) in levels.iter().enumerate() {
            for (zr, col) in z_row.iter_mut().zip(&zs) {
                *zr = col[i];
            }
            moments.observe(level as usize, &z_row, weights.map_or(1.0, |ws| ws[i]));
        }

        let mut transforms = Vec::with_capacity(n_levels);
        let mut unidentified = Vec::new();
        let mut gram = vec![0.0; v * v];
        for level in 0..n_levels {
            moments.gram(level, intercept, &mut gram);
            let (w, kept) = whitening_rows(&gram, v, RANK_TOL);
            // An empty level loses its intercept column too; its every slope
            // direction already drops through the all-zero Gram.
            if intercept && moments.w_sum[level] == 0.0 {
                unidentified.push(UnidentifiedDirection {
                    term,
                    level,
                    column: 0,
                });
            }
            for (j, &kept) in kept.iter().enumerate() {
                if !kept {
                    unidentified.push(UnidentifiedDirection {
                        term,
                        level,
                        column: usize::from(intercept) + j,
                    });
                }
            }
            let center = if intercept {
                moments.mean(level).into()
            } else {
                vec![0.0; v].into()
            };
            transforms.push(LevelTransform {
                w: w.into(),
                center,
            });
        }

        // The whitened columns mix all raw columns, so materialize every
        // output before writing any back. Whitened direction `k` lands in
        // column slot `k`; slots past the level's rank stay exactly zero.
        let mut outs = vec![vec![0.0; levels.len()]; v];
        for (i, &level) in levels.iter().enumerate() {
            let t = &transforms[level as usize];
            for ((zr, col), cj) in z_row.iter_mut().zip(&zs).zip(&*t.center) {
                *zr = col[i] - cj;
            }
            for (k, out) in outs.iter_mut().enumerate().take(t.w.len() / v) {
                let row = &t.w[k * v..(k + 1) * v];
                out[i] = row.iter().zip(&z_row).map(|(wj, zj)| wj * zj).sum();
            }
        }
        for (out, &c) in outs.into_iter().zip(&z_cols) {
            design.frame.set_loading_column(c, out);
        }

        Some(Self {
            offset,
            n_levels,
            intercept,
            n_slopes: v,
            transforms,
            unidentified,
        })
    }

    /// Map solve-basis coefficients back to the user's parametrization.
    pub(crate) fn back_transform(&self, x: &mut [f64]) {
        let v = self.n_slopes;
        let slope_base = self.offset + if self.intercept { self.n_levels } else { 0 };
        let mut b = vec![0.0; v];
        for (l, t) in self.transforms.iter().enumerate() {
            let rank = t.w.len() / v;
            // Fully dropped levels were never transformed; their slots are 0.
            if rank == 0 {
                continue;
            }
            b.fill(0.0);
            for k in 0..rank {
                let bw = x[slope_base + k * self.n_levels + l];
                for (bj, wj) in b.iter_mut().zip(&t.w[k * v..(k + 1) * v]) {
                    *bj += wj * bw;
                }
            }
            for (j, &bj) in b.iter().enumerate() {
                x[slope_base + j * self.n_levels + l] = bj;
            }
            if self.intercept {
                let shift: f64 = b.iter().zip(&*t.center).map(|(bj, cj)| bj * cj).sum();
                x[self.offset + l] -= shift;
            }
        }
    }
}

/// Weighted within-level first and second moments of the slope loadings,
/// accumulated in one pass (multivariate Welford) over positive-weight rows.
/// A structurally constant column keeps an exactly-zero variance row, so
/// exact rank drops survive a zero tolerance.
struct LevelMoments {
    n_slopes: usize,
    w_sum: Vec<f64>,
    mean: Vec<f64>,
    /// Per level, `Σ w (z−μ)(z−μ)ᵀ` packed as a row-major lower triangle.
    comoment: Vec<f64>,
    delta: Vec<f64>,
}

impl LevelMoments {
    fn new(n_levels: usize, n_slopes: usize) -> Self {
        Self {
            n_slopes,
            w_sum: vec![0.0; n_levels],
            mean: vec![0.0; n_levels * n_slopes],
            comoment: vec![0.0; n_levels * n_slopes * (n_slopes + 1) / 2],
            delta: vec![0.0; n_slopes],
        }
    }

    fn observe(&mut self, level: usize, z: &[f64], w: f64) {
        // Zero-weight rows may carry arbitrary values; they must not affect
        // identification or the whitening.
        if w <= 0.0 {
            return;
        }
        let v = self.n_slopes;
        self.w_sum[level] += w;
        let ratio = w / self.w_sum[level];
        let mean = &mut self.mean[level * v..(level + 1) * v];
        for (dj, (zj, mj)) in self.delta.iter_mut().zip(z.iter().zip(mean.iter_mut())) {
            *dj = zj - *mj;
            *mj += ratio * *dj;
        }
        let tri = v * (v + 1) / 2;
        let com = &mut self.comoment[level * tri..(level + 1) * tri];
        let mut idx = 0;
        for j in 0..v {
            let wpost = w * (z[j] - mean[j]);
            for k in 0..=j {
                com[idx] += self.delta[k] * wpost;
                idx += 1;
            }
        }
    }

    fn mean(&self, level: usize) -> &[f64] {
        &self.mean[level * self.n_slopes..(level + 1) * self.n_slopes]
    }

    /// The level's dense V×V slope Gram: centered against a pinned intercept,
    /// raw (`Σwzzᵀ = M2 + w·μμᵀ`) for a slope-only term.
    fn gram(&self, level: usize, intercept: bool, out: &mut [f64]) {
        let v = self.n_slopes;
        let tri = v * (v + 1) / 2;
        let com = &self.comoment[level * tri..(level + 1) * tri];
        let mean = self.mean(level);
        let w = self.w_sum[level];
        let mut idx = 0;
        for j in 0..v {
            for k in 0..=j {
                let mut g = com[idx];
                if !intercept {
                    g += w * mean[j] * mean[k];
                }
                out[j * v + k] = g;
                out[k * v + j] = g;
                idx += 1;
            }
        }
    }
}

/// Pivoted Cholesky whitening of one level's PSD slope Gram `g` (dense V×V,
/// row-major): picks the largest-remaining-variance direction at each step
/// and stops a column once its remaining variance is no longer above
/// `tol` × its own initial variance (relative, hence scale-invariant per
/// column). Returns the `rank × V` whitening rows `w` — `w·g·wᵀ = I` — and
/// which columns were kept. Pivots are tracked through original column
/// indices; nothing is swapped in place.
fn whitening_rows(g: &[f64], v: usize, tol: f64) -> (Vec<f64>, Vec<bool>) {
    let mut d: Vec<f64> = (0..v).map(|j| g[j * v + j]).collect();
    let thresholds: Vec<f64> = d.iter().map(|&dj| tol * dj).collect();
    // lfac[j][t]: weighted inner product of column j with whitened row t.
    let mut lfac = vec![0.0; v * v];
    let mut w: Vec<f64> = Vec::new();
    let mut kept = vec![false; v];
    for step in 0..v {
        let pivot = (0..v)
            .filter(|&j| !kept[j] && d[j].is_finite() && d[j] > thresholds[j])
            .max_by(|&a, &b| d[a].total_cmp(&d[b]));
        let Some(p) = pivot else { break };
        kept[p] = true;
        let root = d[p].sqrt();

        // Whitened row `step`: (e_p − Σ_t lfac[p][t]·w[t]) / root.
        let row_base = step * v;
        w.resize(row_base + v, 0.0);
        w[row_base + p] = 1.0;
        for t in 0..step {
            let c = lfac[p * v + t];
            for j in 0..v {
                w[row_base + j] -= c * w[t * v + j];
            }
        }
        for wj in &mut w[row_base..row_base + v] {
            *wj /= root;
        }

        for j in (0..v).filter(|&j| !kept[j]) {
            let mut val = g[j * v + p];
            for t in 0..step {
                val -= lfac[j * v + t] * lfac[p * v + t];
            }
            val /= root;
            lfac[j * v + step] = val;
            d[j] -= val * val;
        }
    }
    (w, kept)
}
