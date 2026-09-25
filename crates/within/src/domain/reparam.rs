//! Within-level reparametrization of a design's varying-slope terms.

use super::level_moments::{BasisScratch, LevelMoments};
use super::{Design, TermMeta};
use crate::channel::{Channel, CoefficientPosition};
use crate::linalg::dot;

#[cfg(test)]
mod tests;

/// One slope-bearing term in the solve basis: a per-level change of basis making its
/// within-level Gram the identity; [`Self::back_transform`] restores the user's parametrization.
pub(crate) struct WhitenedTerm {
    transforms: Vec<LevelTransform>,
    /// Solve columns in slope-column order; column `k` is the `k`-th basis row, not covariate.
    pub(crate) loadings: Vec<Vec<f64>>,
    /// Directions the data cannot identify, ascending in `(level, column)`.
    pub(crate) unidentified: Vec<CoefficientPosition>,
}

/// One level's `u = W·(z − center)`, `W` row-major `rank × V`; an empty `w`
/// marks a fully dropped level, a dropped slope is a zero column of `W`.
struct LevelTransform {
    w: Box<[f64]>,
    /// Within-level weighted means; all-zero for slope-only terms.
    center: Box<[f64]>,
}

impl WhitenedTerm {
    /// Unidentified directions become exact-zero columns, so the minimal-norm solve leaves `0`.
    pub(crate) fn build(design: &Design<'_>, term: usize, sqrt_weights: Option<&[f64]>) -> Self {
        let moments = LevelMoments::build(design, term, sqrt_weights);
        let meta = &design.terms[term];
        let n_levels = meta.n_levels();
        let intercept = meta.has_intercept();
        let zs: Vec<&[f64]> = meta
            .covariates()
            .map(|c| design.frame.loading_column(c as usize))
            .collect();
        let v = zs.len();
        let levels = design.frame.level_column(term);

        let mut z_row = vec![0.0; v];
        let mut transforms = Vec::with_capacity(n_levels);
        let mut unidentified = Vec::new();
        let mut scratch = BasisScratch::new(v);
        for level in 0..n_levels {
            moments.basis(level, &mut scratch);
            let (w, kept) = (&scratch.basis, &scratch.kept);
            if intercept && moments.w_sum(level) == 0.0 {
                unidentified.push(CoefficientPosition {
                    channel: Channel { term, column: 0 },
                    level,
                });
            }
            for (j, &kept_j) in kept.iter().enumerate() {
                if !kept_j {
                    unidentified.push(CoefficientPosition {
                        channel: Channel {
                            term,
                            column: j + intercept as usize,
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
                w: w.clone().into(),
                center,
            });
        }

        let mut loadings = vec![vec![0.0; levels.len()]; v];
        for (i, &level) in levels.iter().enumerate() {
            let t = &transforms[level as usize];
            for ((zr, col), cj) in z_row.iter_mut().zip(&zs).zip(&*t.center) {
                *zr = col[i] - cj;
            }
            for (w_row, out) in t.w.chunks_exact(v).zip(&mut loadings) {
                out[i] = dot(w_row, &z_row);
            }
        }

        Self {
            transforms,
            loadings,
            unidentified,
        }
    }

    /// Map this term's solve-basis coefficients back to the user's
    /// parametrization; slots outside the term's block are untouched.
    pub(crate) fn back_transform(&self, meta: &TermMeta, x: &mut [f64]) {
        let (offset, n_levels) = (meta.offset, meta.n_levels());
        let intercept = meta.has_intercept();
        let slope_slot =
            |j: usize, level: usize| offset + (j + intercept as usize) * n_levels + level;
        let v = self.transforms.first().map_or(0, |t| t.center.len());
        let mut b = vec![0.0; v];
        for (l, t) in self.transforms.iter().enumerate() {
            if t.w.is_empty() {
                continue;
            }
            b.fill(0.0);
            for (k, w_row) in t.w.chunks_exact(v).enumerate() {
                let bk = x[slope_slot(k, l)];
                for (bj, wj) in b.iter_mut().zip(w_row) {
                    *bj += wj * bk;
                }
            }
            for (j, &bj) in b.iter().enumerate() {
                x[slope_slot(j, l)] = bj;
            }
            if intercept {
                x[offset + l] -= dot(&b, &t.center);
            }
        }
    }
}
