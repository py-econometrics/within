//! Within-level orthonormal basis of a design's varying-slope terms.

use super::level_moments::{BasisScratch, TermMoments};
use super::Design;
use crate::channel::{Channel, CoefficientAddress};
use crate::linalg::dot;

#[cfg(test)]
mod tests;

/// Per-level basis making each slope term's within-level Gram the identity; empty without slopes.
pub(crate) struct SlopeBasis {
    terms: Vec<TermBasis>,
    /// The frame's loading columns in the solve basis, indexed like the frame's.
    loadings: Vec<Vec<f64>>,
    /// Directions the data cannot identify, ascending in `(term, level, column)`.
    pub(crate) unidentified: Vec<CoefficientAddress>,
}

/// One slope-bearing term's whitening state.
struct TermBasis {
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

impl SlopeBasis {
    pub(crate) fn build(design: &Design<'_>, weights: Option<&[f64]>) -> Self {
        let mut loadings = vec![Vec::new(); design.frame.n_loading_columns()];
        let mut terms = Vec::new();
        let mut unidentified = Vec::new();
        if let Some(moments) = TermMoments::build(design, weights) {
            for term in (0..design.terms.len()).filter(|&t| design.terms[t].has_slopes()) {
                terms.push(TermBasis::build(
                    design,
                    term,
                    &moments,
                    &mut loadings,
                    &mut unidentified,
                ));
            }
        }
        // `Design::new` claims every loading column, so whitening leaves none unwritten.
        debug_assert!(loadings.iter().all(|l| l.len() == design.n_obs));
        Self {
            terms,
            loadings,
            unidentified,
        }
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

impl TermBasis {
    /// Whiten one term into `loadings`; unidentified directions append ascending in `(level, column)`.
    fn build(
        design: &Design<'_>,
        term: usize,
        moments: &TermMoments,
        loadings: &mut [Vec<f64>],
        unidentified: &mut Vec<CoefficientAddress>,
    ) -> Self {
        let meta = &design.terms[term];
        let (offset, n_levels) = (meta.offset, meta.n_levels);
        let mut intercept_column = None;
        let mut slope_columns = Vec::new();
        for (column, loading) in meta.columns.iter().enumerate() {
            match loading.covariate() {
                Some(_) => slope_columns.push(column),
                None => intercept_column = Some(column),
            }
        }
        let intercept = intercept_column.is_some();
        let moments = &moments[term];
        let v = moments.n_slopes();
        let z_cols: Vec<usize> = moments.covariates().iter().map(|&c| c as usize).collect();
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
