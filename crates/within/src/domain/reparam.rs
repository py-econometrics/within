//! Within-level reparametrization of a design's varying-slope terms.

use super::level_moments::LevelMoments;
use super::{Design, Term};
use crate::channel::{Channel, CoefficientPosition};
use crate::linalg::{dot, GramBasis, GramBasisWorkspace, RANK_TOL};

#[cfg(test)]
mod tests;

/// A slope term's change of basis to an identity within-level Gram.
pub(crate) struct TermReparam {
    /// The term's slopes in the solve basis, in coefficient-column order.
    pub(super) slopes: Vec<Vec<f64>>,
    transforms: Vec<LevelTransform>,
    /// Directions the data cannot identify, ascending in `(level, column)`.
    pub(super) unidentified: Vec<CoefficientPosition>,
}

/// One level's `u = W·(z − center)`, `W` row-major `rank × V`; an empty `w`
/// marks a fully dropped level, a dropped slope is a zero column of `W`.
struct LevelTransform {
    w: Box<[f64]>,
    /// Within-level weighted means; all-zero for slope-only terms.
    center: Box<[f64]>,
}

impl TermReparam {
    /// `None` without slopes; unidentified directions become zero columns, so they solve to `0`.
    pub(crate) fn build(
        design: &Design<'_>,
        term: usize,
        sqrt_weights: Option<&[f64]>,
    ) -> Option<Self> {
        let t = &design.terms[term];
        if !t.has_slopes() {
            return None;
        }
        let moments = LevelMoments::build(design, term, sqrt_weights);
        let levels = t.levels();
        let n_levels = t.n_levels();
        let intercept = t.intercept;
        let zs: Vec<&[f64]> = t.raw_slopes().collect();
        let v = zs.len();

        let mut z_row = vec![0.0; v];
        let mut unidentified = Vec::new();
        let mut transforms = Vec::with_capacity(n_levels);
        let mut workspace = GramBasisWorkspace::new(v, RANK_TOL);
        for level in 0..n_levels {
            let GramBasis { rows: w, kept } =
                workspace.orthonormalize(|gram| moments.fill_gram(level, gram));
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
                            column: t.slope_column(j),
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

        let mut slopes = vec![vec![0.0; levels.len()]; v];
        for (i, &level) in levels.iter().enumerate() {
            let t = &transforms[level as usize];
            for ((zr, col), cj) in z_row.iter_mut().zip(&zs).zip(&*t.center) {
                *zr = col[i] - cj;
            }
            for (w_row, out) in t.w.chunks_exact(v).zip(&mut slopes) {
                out[i] = dot(w_row, &z_row);
            }
        }

        Some(Self {
            slopes,
            transforms,
            unidentified,
        })
    }

    /// Map this term's solve-basis coefficients back; slots outside its block are untouched.
    pub(crate) fn back_transform(&self, term: &Term<'_>, x: &mut [f64]) {
        let slope_slot =
            |j: usize, level: usize| term.column_dofs(term.slope_column(j)).start + level;
        let v = self.slopes.len();
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
            if term.intercept {
                x[term.column_dofs(0).start + l] -= dot(&b, &t.center);
            }
        }
    }
}
