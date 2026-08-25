//! Within-level reparametrization of a design's varying-slope terms.

use crate::domain::level_moments::{BasisScratch, TermMoments};
use crate::domain::Design;

use super::null_space::{AliasCandidate, NullSpace};
use super::CoefficientPosition;
use crate::channel::Channel;
use crate::linalg::dot;
use crate::BuildWarning;

#[cfg(test)]
mod tests;

/// Per-level change of basis making each slope-bearing term's within-level
/// Gram the identity; [`Self::back_transform`] restores the user's
/// parametrization.
pub(crate) struct SlopeReparam {
    terms: Vec<TermReparam>,
    /// Every direction the solve space excludes.
    pub(super) null: NullSpace,
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
    /// Orthonormalize every slope-bearing term's loading columns in place and certify the
    /// screen's aliasing proposals against the result; `None` for slope-free designs.
    /// Unidentified directions become exact-zero columns, so the minimal-norm solve leaves
    /// exact-`0` coefficients.
    pub(crate) fn build(
        design: &mut Design<'_>,
        moments: &TermMoments,
        weights: Option<&[f64]>,
        warnings: &mut [BuildWarning],
    ) -> Option<Self> {
        let candidates = AliasCandidate::capture(design, weights, warnings);
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
            terms.push(TermReparam::build(design, term, moments, &mut unidentified));
        }
        if terms.is_empty() {
            return None;
        }
        let mut null = NullSpace {
            dropped: unidentified,
            rows: Vec::new(),
            n_dofs: design.n_dofs,
        };
        null.constrain(candidates, design, weights, warnings);
        Some(Self { terms, null })
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
        moments: &TermMoments,
        unidentified: &mut Vec<CoefficientPosition>,
    ) -> Self {
        let meta = &design.terms[term];
        let (offset, n_levels) = (meta.offset, meta.n_levels());
        let intercept_column = meta.intercept_column();
        let slope_columns: Vec<usize> = meta
            .columns
            .iter()
            .enumerate()
            .filter_map(|(column, loading)| loading.covariate().map(|_| column))
            .collect();
        let intercept = intercept_column.is_some();
        let moments = &moments[term];
        let v = moments.n_slopes();
        let z_cols: Vec<usize> = moments.covariates().iter().map(|&c| c as usize).collect();
        let levels = design.level_column(term);
        let zs: Vec<&[f64]> = z_cols.iter().map(|&c| design.loading_column(c)).collect();

        let mut z_row = vec![0.0; v];
        let mut transforms = Vec::with_capacity(n_levels);
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
                            column: slope_columns[j],
                        },
                        level,
                    });
                }
            }
            transforms.push(LevelTransform {
                w: w.clone().into(),
                center: moments.center(level).into(),
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
            design.replace_loading_column(c, out);
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
