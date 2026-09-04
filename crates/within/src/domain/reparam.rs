//! Within-level reparametrization of a design's varying-slope terms.

use rayon::prelude::*;

use super::level_moments::{BasisScratch, LevelMoments};
use super::Design;
use crate::channel::{Channel, CoefficientAddress};
use crate::linalg::dot;

#[cfg(test)]
mod tests;

/// Per-level change of basis making each slope-bearing term's within-level
/// Gram the identity; [`Self::back_transform`] restores the user's
/// parametrization.
pub(crate) struct SlopeReparam {
    terms: Vec<TermReparam>,
    /// The frame's loading columns in the solve basis, indexed like the frame's.
    loadings: Vec<Vec<f64>>,
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
    /// Orthonormalize every slope-bearing term's loading columns into `loadings`;
    /// `None` for slope-free designs. Unidentified directions become
    /// exact-zero columns, so the minimal-norm solve leaves exact-`0`
    /// coefficients.
    pub(crate) fn build(design: &Design<'_>, weights: Option<&[f64]>) -> Option<Self> {
        let mut loadings = vec![Vec::new(); design.frame.n_loading_columns()];
        let slope_terms: Vec<usize> = (0..design.terms.len())
            .filter(|&t| design.terms[t].has_slopes())
            .collect();
        let moments: Vec<LevelMoments> = slope_terms
            .par_iter()
            .map(|&term| LevelMoments::build(design, term, weights))
            .collect();
        let mut unidentified = Vec::new();
        let terms: Vec<TermReparam> = slope_terms
            .iter()
            .zip(&moments)
            .map(|(&term, moments)| {
                TermReparam::build(design, term, moments, &mut loadings, &mut unidentified)
            })
            .collect();
        (!terms.is_empty()).then_some(Self {
            terms,
            loadings,
            unidentified,
        })
    }

    /// Loading column `column` in the solve basis.
    pub(crate) fn loading_column(&self, column: usize) -> &[f64] {
        &self.loadings[column]
    }

    /// Map solve-basis coefficients back to the user's parametrization.
    pub(crate) fn back_transform(&self, x: &mut [f64]) {
        for term in &self.terms {
            term.back_transform(x);
        }
    }
}

impl TermReparam {
    /// Whitens one term into `loadings`; unidentified directions append in `(level, column)` order.
    fn build(
        design: &Design<'_>,
        term: usize,
        moments: &LevelMoments,
        loadings: &mut [Vec<f64>],
        unidentified: &mut Vec<CoefficientAddress>,
    ) -> Self {
        let meta = &design.terms[term];
        let (offset, n_levels) = (meta.offset, meta.n_levels);
        let intercept_column = meta.columns.iter().position(|l| l.covariate().is_none());
        let (slope_columns, z_cols): (Vec<usize>, Vec<usize>) = meta
            .columns
            .iter()
            .enumerate()
            .filter_map(|(col, l)| l.covariate().map(|&z| (col, z as usize)))
            .unzip();
        let intercept = intercept_column.is_some();
        let v = slope_columns.len();
        let levels = design.frame.level_column(term);
        let zs: Vec<&[f64]> = z_cols
            .iter()
            .map(|&c| design.frame.loading_column(c))
            .collect();

        let mut z_row = vec![0.0; v];
        let mut transforms = Vec::with_capacity(n_levels);
        let mut scratch = BasisScratch::new(v);
        for level in 0..n_levels {
            moments.basis(level, &mut scratch);
            let (w, kept) = (&scratch.basis, &scratch.kept);
            if intercept && moments.w_sum(level) == 0.0 {
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
                w: w.clone().into(),
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
            loadings[c] = out;
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
