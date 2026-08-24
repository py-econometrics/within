//! Per-level weighted moments of a term's loading columns, built once and read
//! by both the collinearity screen and the slope whitening.

use rayon::prelude::*;

use super::Design;

/// Relative rank tolerance: a slope direction drops once its remaining
/// within-level variance falls to `RANK_TOL` × its own initial variance.
pub(crate) const RANK_TOL: f64 = 1e-10;

/// Every term's [`LevelMoments`], indexed by term.
pub(crate) struct TermMoments(Vec<LevelMoments>);

impl TermMoments {
    /// `None` when no term carries a covariate, so nothing downstream has work.
    pub(crate) fn build(design: &Design<'_>, weights: Option<&[f64]>) -> Option<Self> {
        let has_slopes = design
            .terms
            .iter()
            .any(|t| t.columns.iter().any(|c| c.covariate().is_some()));
        has_slopes.then(|| {
            Self(
                (0..design.terms.len())
                    .into_par_iter()
                    .map(|term| LevelMoments::build(design, term, weights))
                    .collect(),
            )
        })
    }
}

impl std::ops::Index<usize> for TermMoments {
    type Output = LevelMoments;

    fn index(&self, term: usize) -> &LevelMoments {
        &self.0[term]
    }
}

/// One-pass weighted within-level moments (multivariate Welford); structural
/// zeros stay exact, so rank drops survive a zero tolerance.
pub(crate) struct LevelMoments {
    /// Frame columns of the term's covariates, in coefficient-column order.
    covariates: Box<[u32]>,
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
    fn build(design: &Design<'_>, term: usize, weights: Option<&[f64]>) -> Self {
        let meta = &design.terms[term];
        let covariates: Box<[u32]> = meta
            .columns
            .iter()
            .filter_map(|c| c.covariate().copied())
            .collect();
        let v = covariates.len();
        let mut moments = Self {
            intercept: v < meta.columns.len(),
            w_sum: vec![0.0; meta.n_levels],
            mean: vec![0.0; meta.n_levels * v],
            comoment: vec![0.0; meta.n_levels * tri_len(v)],
            covariates,
        };
        let zs: Vec<&[f64]> = moments
            .covariates
            .iter()
            .map(|&c| design.loading_column(c as usize))
            .collect();
        let mut z_row = vec![0.0; v];
        let mut delta = vec![0.0; v];
        for (obs, &level) in design.level_column(term).iter().enumerate() {
            for (zr, col) in z_row.iter_mut().zip(&zs) {
                *zr = col[obs];
            }
            let w = weights.map_or(1.0, |w| w[obs]);
            moments.observe(level as usize, &z_row, w, &mut delta);
        }
        moments
    }

    fn observe(&mut self, level: usize, z: &[f64], w: f64, delta: &mut [f64]) {
        if w <= 0.0 {
            return;
        }
        let v = self.n_slopes();
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

    pub(crate) fn n_slopes(&self) -> usize {
        self.covariates.len()
    }

    /// Frame columns of the term's covariates, in coefficient-column order.
    pub(crate) fn covariates(&self) -> &[u32] {
        &self.covariates
    }

    pub(crate) fn intercept(&self) -> bool {
        self.intercept
    }

    pub(crate) fn n_levels(&self) -> usize {
        self.w_sum.len()
    }

    pub(crate) fn w_sum(&self, level: usize) -> f64 {
        self.w_sum[level]
    }

    pub(crate) fn mean(&self, level: usize) -> &[f64] {
        let v = self.n_slopes();
        &self.mean[level * v..][..v]
    }

    /// The level's orthonormal rows `w` (`w·G·wᵀ = I`) and kept-column mask,
    /// left in `scratch` so a sweep over levels allocates nothing. `G` is
    /// centered against a pinned intercept, raw (`M2 + w·μμᵀ`) without one.
    pub(crate) fn basis(&self, level: usize, scratch: &mut BasisScratch) {
        let v = self.n_slopes();
        let com = &self.comoment[level * tri_len(v)..][..tri_len(v)];
        let mean = self.mean(level);
        let w = self.w_sum[level];
        for j in 0..v {
            for k in 0..=j {
                let mut g = com[tri_index(j, k)];
                if !self.intercept {
                    g += w * mean[j] * mean[k];
                }
                scratch.gram[j * v + k] = g;
                scratch.gram[k * v + j] = g;
            }
        }
        scratch.orthonormalize(v, RANK_TOL);
    }
}

/// Reusable buffers for a sweep of [`LevelMoments::basis`] over many levels.
pub(crate) struct BasisScratch {
    gram: Vec<f64>,
    residual: Vec<f64>,
    q: Vec<f64>,
    /// The last level's `rank × v` orthonormal rows.
    pub(crate) basis: Vec<f64>,
    /// The last level's kept-column mask.
    pub(crate) kept: Vec<bool>,
}

impl BasisScratch {
    pub(crate) fn new(v: usize) -> Self {
        Self {
            gram: vec![0.0; v * v],
            residual: vec![0.0; v],
            q: vec![0.0; v],
            basis: Vec::with_capacity(v * v),
            kept: vec![false; v],
        }
    }
}

impl BasisScratch {
    /// Pivoted Gram–Schmidt on the raw slope columns, run in coordinates:
    /// column `j` enters as `e_j` and all geometry goes through the dense
    /// row-major `v×v` level Gram in `self.gram`, `⟨a, b⟩ = a·g·bᵀ`. Leaves the
    /// `rank × v` orthonormal rows in `self.basis` — `w·g·wᵀ = I` — plus the
    /// kept-column mask. A column drops once its residual variance falls to
    /// `tol` × its initial variance; pivots keep their original column indices
    /// — nothing is swapped.
    fn orthonormalize(&mut self, v: usize, tol: f64) {
        let Self {
            gram: g,
            residual,
            q,
            basis,
            kept,
        } = self;
        for (r, j) in residual.iter_mut().zip(0..v) {
            *r = g[j * v + j];
        }
        kept.fill(false);
        basis.clear();
        while let Some(p) = (0..v)
            .filter(|&j| !kept[j] && residual[j].is_finite() && residual[j] > tol * g[j * v + j])
            .max_by(|&a, &b| residual[a].total_cmp(&residual[b]))
        {
            kept[p] = true;

            // q = e_p − Σₜ ⟨e_p, qₜ⟩·qₜ, whose norm is already √residual[p].
            q.fill(0.0);
            q[p] = 1.0;
            for q_t in basis.chunks_exact(v) {
                let c = crate::linalg::dot(&g[p * v..][..v], q_t);
                for (qj, &qtj) in q.iter_mut().zip(q_t) {
                    *qj -= c * qtj;
                }
            }
            let norm = residual[p].sqrt();
            for qj in q.iter_mut() {
                *qj /= norm;
            }

            // Pythagoras: projecting out q costs every column ⟨e_j, q⟩² of variance.
            for (r, g_row) in residual.iter_mut().zip(g.chunks_exact(v)) {
                let c = crate::linalg::dot(g_row, q);
                *r -= c * c;
            }
            basis.extend_from_slice(q);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pivoted_gram_schmidt(g: &[f64], v: usize, tol: f64) -> (Vec<f64>, Vec<bool>) {
        let mut scratch = BasisScratch::new(v);
        scratch.gram.copy_from_slice(g);
        scratch.orthonormalize(v, tol);
        (scratch.basis, scratch.kept)
    }

    #[test]
    fn gram_schmidt_orthonormalizes_under_a_non_monotonic_pivot_order() {
        // Diagonals [2, 5, 3] force the pivot sequence 1 → 2 → 0, breaking order assumptions.
        let g = [2.0, 1.0, 0.5, 1.0, 5.0, 2.0, 0.5, 2.0, 3.0];
        let (w, kept) = pivoted_gram_schmidt(&g, 3, RANK_TOL);
        assert_eq!(kept, [true; 3]);
        assert_eq!(w.len(), 9);

        for r in 0..3 {
            for s in 0..3 {
                let wgw: f64 = (0..3)
                    .flat_map(|j| (0..3).map(move |k| (j, k)))
                    .map(|(j, k)| w[r * 3 + j] * g[j * 3 + k] * w[s * 3 + k])
                    .sum();
                let expected = if r == s { 1.0 } else { 0.0 };
                assert!(
                    (wgw - expected).abs() < 1e-12,
                    "(W·G·Wᵀ)[{r}][{s}] = {wgw}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn zero_tolerance_keeps_a_near_degenerate_direction_the_default_drops() {
        let eps = 1e-12;
        let g = [1.0, 1.0 - eps, 1.0 - eps, 1.0];
        let (_, kept) = pivoted_gram_schmidt(&g, 2, RANK_TOL);
        assert_eq!(kept.iter().filter(|&&k| k).count(), 1);
        let (_, kept) = pivoted_gram_schmidt(&g, 2, 0.0);
        assert_eq!(kept, [true, true]);
    }
}
