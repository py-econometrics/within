//! Within-level reparametrization of a design's varying-slope terms.

use crate::domain::Design;

use super::CoefficientAddress;
use crate::channel::Channel;

#[cfg(test)]
mod tests;

/// Relative rank tolerance: a slope direction drops once its remaining
/// within-level variance falls to `RANK_TOL` × its own initial variance.
const RANK_TOL: f64 = 1e-10;

/// Per-level change of basis making each slope-bearing term's within-level
/// Gram the identity; [`Self::back_transform`] restores the user's
/// parametrization.
pub(crate) struct SlopeReparam {
    terms: Vec<TermReparam>,
    /// Directions the data cannot identify, ascending in `(term, level, column)`.
    pub(crate) unidentified: Vec<CoefficientAddress>,
}

/// One slope-bearing term's whitening state.
struct TermReparam {
    offset: usize,
    n_levels: usize,
    /// Coefficient column of the term's constant, if it has one.
    intercept_column: Option<usize>,
    /// Coefficient column of each covariate, in order.
    slope_columns: Box<[usize]>,
    transforms: Vec<LevelTransform>,
}

/// One level's `u = W·(z − center)`, `W` row-major `rank × V`; an empty `w`
/// marks a fully dropped level, a dropped slope is a zero column of `W`.
struct LevelTransform {
    w: Box<[f64]>,
    /// Within-level weighted means; all-zero for slope-only terms.
    center: Box<[f64]>,
}

impl SlopeReparam {
    /// Orthonormalize every slope-bearing term's loading columns in place;
    /// `None` for slope-free designs. Unidentified directions become
    /// exact-zero columns, so the minimal-norm solve leaves exact-`0`
    /// coefficients.
    pub(crate) fn build(design: &mut Design<'_>, weights: Option<&[f64]>) -> Option<Self> {
        let mut terms = Vec::new();
        let mut unidentified = Vec::new();
        for term in 0..design.terms.len() {
            if !design.terms[term]
                .columns
                .iter()
                .any(|c| c.covariate().is_some())
            {
                continue;
            }
            terms.push(TermReparam::build(design, term, weights, &mut unidentified));
        }
        (!terms.is_empty()).then_some(Self {
            terms,
            unidentified,
        })
    }

    /// Map solve-basis coefficients back to the user's parametrization.
    pub(crate) fn back_transform(&self, x: &mut [f64]) {
        for term in &self.terms {
            term.back_transform(x);
        }
    }
}

impl TermReparam {
    /// Whiten one term's loading columns in place, appending its unidentified
    /// directions ascending in `(level, column)`.
    fn build(
        design: &mut Design<'_>,
        term: usize,
        weights: Option<&[f64]>,
        unidentified: &mut Vec<CoefficientAddress>,
    ) -> Self {
        let meta = &design.terms[term];
        let (offset, n_levels) = (meta.offset, meta.n_levels);
        let mut intercept_column = None;
        let mut slope_columns = Vec::new();
        let mut z_cols = Vec::new();
        for (column, loading) in meta.columns.iter().enumerate() {
            match loading.covariate() {
                Some(&k) => {
                    slope_columns.push(column);
                    z_cols.push(k as usize);
                }
                None => intercept_column = Some(column),
            }
        }
        let intercept = intercept_column.is_some();
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
        let mut gram = vec![0.0; v * v];
        for level in 0..n_levels {
            moments.gram(level, intercept, &mut gram);
            let (w, kept) = pivoted_gram_schmidt(&gram, v, RANK_TOL);
            if intercept && moments.w_sum[level] == 0.0 {
                unidentified.push(CoefficientAddress {
                    channel: Channel { term, column: 0 },
                    level,
                });
            }
            for (j, &kept_j) in kept.iter().enumerate() {
                if !kept_j {
                    unidentified.push(CoefficientAddress {
                        channel: Channel {
                            term,
                            column: slope_columns[j],
                        },
                        level,
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

        let mut u_cols = vec![vec![0.0; levels.len()]; v];
        for (i, &level) in levels.iter().enumerate() {
            let t = &transforms[level as usize];
            for ((zr, col), cj) in z_row.iter_mut().zip(&zs).zip(&*t.center) {
                *zr = col[i] - cj;
            }
            for (w_row, out) in t.w.chunks_exact(v).zip(&mut u_cols) {
                out[i] = dot(w_row, &z_row);
            }
        }
        for (out, &c) in u_cols.into_iter().zip(&z_cols) {
            design.frame.set_loading_column(c, out);
        }

        Self {
            offset,
            n_levels,
            intercept_column,
            slope_columns: slope_columns.into(),
            transforms,
        }
    }

    /// Map this term's solve-basis coefficients back to the user's
    /// parametrization; slots outside the term's block are untouched.
    fn back_transform(&self, x: &mut [f64]) {
        let v = self.slope_columns.len();
        let mut b = vec![0.0; v];
        for (l, t) in self.transforms.iter().enumerate() {
            if t.w.is_empty() {
                continue;
            }
            b.fill(0.0);
            for (k, w_row) in t.w.chunks_exact(v).enumerate() {
                let bk = x[self.slope_slot(k, l)];
                for (bj, wj) in b.iter_mut().zip(w_row) {
                    *bj += wj * bk;
                }
            }
            for (j, &bj) in b.iter().enumerate() {
                x[self.slope_slot(j, l)] = bj;
            }
            if let Some(c) = self.intercept_column {
                x[self.offset + c * self.n_levels + l] -= dot(&b, &t.center);
            }
        }
    }

    fn slope_slot(&self, j: usize, level: usize) -> usize {
        self.offset + self.slope_columns[j] * self.n_levels + level
    }
}

/// One-pass weighted within-level moments (multivariate Welford); structural
/// zeros stay exact, so rank drops survive a zero tolerance.
struct LevelMoments {
    n_slopes: usize,
    w_sum: Vec<f64>,
    mean: Vec<f64>,
    /// Per level, `Σ w (z−μ)(z−μ)ᵀ` packed as a row-major lower triangle.
    comoment: Vec<f64>,
    /// Scratch: deviations from the pre-update means.
    delta: Vec<f64>,
}

/// Index of `(j, k)`, `k ≤ j`, in a packed row-major lower triangle.
fn tri_index(j: usize, k: usize) -> usize {
    j * (j + 1) / 2 + k
}

fn tri_len(v: usize) -> usize {
    v * (v + 1) / 2
}

impl LevelMoments {
    fn new(n_levels: usize, n_slopes: usize) -> Self {
        Self {
            n_slopes,
            w_sum: vec![0.0; n_levels],
            mean: vec![0.0; n_levels * n_slopes],
            comoment: vec![0.0; n_levels * tri_len(n_slopes)],
            delta: vec![0.0; n_slopes],
        }
    }

    fn observe(&mut self, level: usize, z: &[f64], w: f64) {
        if w <= 0.0 {
            return;
        }
        let v = self.n_slopes;
        self.w_sum[level] += w;
        let ratio = w / self.w_sum[level];
        let mean = &mut self.mean[level * v..][..v];
        for (dj, (zj, mj)) in self.delta.iter_mut().zip(z.iter().zip(mean.iter_mut())) {
            *dj = zj - *mj;
            *mj += ratio * *dj;
        }
        let com = &mut self.comoment[level * tri_len(v)..][..tri_len(v)];
        for j in 0..v {
            let dev = w * (z[j] - mean[j]);
            for k in 0..=j {
                com[tri_index(j, k)] += self.delta[k] * dev;
            }
        }
    }

    fn mean(&self, level: usize) -> &[f64] {
        &self.mean[level * self.n_slopes..][..self.n_slopes]
    }

    /// The level's V×V slope Gram: centered against a pinned intercept, raw
    /// (`M2 + w·μμᵀ`) for a slope-only term.
    fn gram(&self, level: usize, intercept: bool, out: &mut [f64]) {
        let v = self.n_slopes;
        let com = &self.comoment[level * tri_len(v)..][..tri_len(v)];
        let mean = self.mean(level);
        let w = self.w_sum[level];
        for j in 0..v {
            for k in 0..=j {
                let mut g = com[tri_index(j, k)];
                if !intercept {
                    g += w * mean[j] * mean[k];
                }
                out[j * v + k] = g;
                out[k * v + j] = g;
            }
        }
    }
}

/// Pivoted Gram–Schmidt on the raw slope columns, run in coordinates:
/// column `j` enters as `e_j` and all geometry goes through the dense
/// row-major `v×v` level Gram, `⟨a, b⟩ = a·g·bᵀ`. Returns the `rank × v`
/// orthonormal rows `w` — `w·g·wᵀ = I` — plus the kept-column mask. A column
/// drops once its residual variance falls to `tol` × its initial variance;
/// pivots keep their original column indices — nothing is swapped.
fn pivoted_gram_schmidt(g: &[f64], v: usize, tol: f64) -> (Vec<f64>, Vec<bool>) {
    let mut residual: Vec<f64> = (0..v).map(|j| g[j * v + j]).collect();
    let mut kept = vec![false; v];
    let mut basis = Vec::new();
    while let Some(p) = (0..v)
        .filter(|&j| !kept[j] && residual[j].is_finite() && residual[j] > tol * g[j * v + j])
        .max_by(|&a, &b| residual[a].total_cmp(&residual[b]))
    {
        kept[p] = true;

        // q = e_p − Σₜ ⟨e_p, qₜ⟩·qₜ, whose norm is already √residual[p].
        let mut q = vec![0.0; v];
        q[p] = 1.0;
        for q_t in basis.chunks_exact(v) {
            let c = dot(&g[p * v..][..v], q_t);
            for (qj, &qtj) in q.iter_mut().zip(q_t) {
                *qj -= c * qtj;
            }
        }
        let norm = residual[p].sqrt();
        for qj in &mut q {
            *qj /= norm;
        }

        // Pythagoras: projecting out q costs every column ⟨e_j, q⟩² of variance.
        for (r, g_row) in residual.iter_mut().zip(g.chunks_exact(v)) {
            let c = dot(g_row, &q);
            *r -= c * c;
        }
        basis.extend_from_slice(&q);
    }
    (basis, kept)
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
