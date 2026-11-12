//! Within-level reparametrization of a design's varying-slope terms.

use rayon::prelude::*;

use super::level_moments::{LevelMean, LevelMoments};
use super::scatter_split::LEVEL_GROUP;
use super::{Design, ScatterSplit, Term};
use crate::channel::{Channel, CoefficientPosition};
use crate::linalg::{dot, GramBasisWorkspace};

#[cfg(test)]
mod tests;

/// Rows per parallel task of a sweep's row pass.
const ROW_CHUNK: usize = 1 << 14;

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
    /// Whitens each level's Gram in `moments`; `kept[l·V + j]` says whether slope `j` survived.
    fn whiten(moments: &LevelMoments, n_levels: usize, v: usize) -> (Self, Vec<bool>) {
        let mut w = vec![0.0; n_levels * v * v];
        let mut center = vec![LevelMean::default(); n_levels * v];
        let mut kept = vec![false; n_levels * v];
        w.par_chunks_exact_mut(v * v)
            .zip(center.par_chunks_exact_mut(v))
            .zip(kept.par_chunks_exact_mut(v))
            .enumerate()
            .with_min_len(LEVEL_GROUP)
            .for_each_init(
                || GramBasisWorkspace::new(v),
                |workspace, (level, ((w, center), kept))| {
                    let basis = workspace.orthonormalize(|gram| moments.fill_gram(level, gram));
                    w[..basis.rows.len()].copy_from_slice(basis.rows);
                    kept.copy_from_slice(basis.kept);
                    if let Some(mean) = moments.mean(level) {
                        center.copy_from_slice(mean);
                    }
                },
            );
        (Self { v, w, center }, kept)
    }

    fn level(&self, level: usize) -> (&[f64], &[LevelMean]) {
        let v = self.v;
        (
            &self.w[level * v * v..][..v * v],
            &self.center[level * v..][..v],
        )
    }

    /// Rewrites each row of `slopes` (one `Vec` per column) by its level's sweep.
    fn apply(&self, levels: &[u32], slopes: &mut [Vec<f64>]) {
        let v = self.v;
        let mut columns: Vec<_> = slopes.iter_mut().map(|c| c.chunks_mut(ROW_CHUNK)).collect();
        let chunks: Vec<Vec<&mut [f64]>> = (0..levels.len().div_ceil(ROW_CHUNK))
            .map(|_| {
                columns
                    .iter_mut()
                    .map(|c| c.next().expect("a chunk per column"))
                    .collect()
            })
            .collect();
        chunks
            .into_par_iter()
            .zip(levels.par_chunks(ROW_CHUNK))
            .for_each(|(mut columns, levels)| {
                let mut z_row = vec![0.0; v];
                for (r, &level) in levels.iter().enumerate() {
                    let (w, center) = self.level(level as usize);
                    for ((zr, col), c) in z_row.iter_mut().zip(&columns).zip(center) {
                        *zr = c.centre(col[r]);
                    }
                    for (col, w_row) in columns.iter_mut().zip(w.chunks_exact(v)) {
                        col[r] = dot(w_row, &z_row);
                    }
                }
            });
    }

    /// `u = W₂·(W₁·(z − c₁) − c₂)` as `W = W₂·W₁`, `shift = W₂·(W₁·c₁ + c₂)`, `W` reusing
    /// the first sweep's storage.
    fn compose(self, second: &Self) -> (Vec<f64>, Vec<f64>) {
        let Self { v, mut w, center } = self;
        let mut shift = vec![0.0; center.len()];
        w.par_chunks_exact_mut(v * v)
            .zip(shift.par_chunks_exact_mut(v))
            .enumerate()
            .with_min_len(LEVEL_GROUP)
            .for_each_init(
                || (vec![0.0; v], vec![0.0; v * v]),
                |(centre, composed), (level, (w1, s))| {
                    let c1 = &center[level * v..][..v];
                    let (w2, c2) = second.level(level);
                    for ((cm, w1_row), c2m) in centre.iter_mut().zip(w1.chunks_exact(v)).zip(c2) {
                        let w1c1: f64 = w1_row.iter().zip(c1).map(|(a, c)| a * c.value()).sum();
                        *cm = w1c1 + c2m.value();
                    }
                    composed.fill(0.0);
                    for ((row, w2_row), sk) in composed
                        .chunks_exact_mut(v)
                        .zip(w2.chunks_exact(v))
                        .zip(s.iter_mut())
                    {
                        for (&a, w1_row) in w2_row.iter().zip(w1.chunks_exact(v)) {
                            for (rj, &b) in row.iter_mut().zip(w1_row) {
                                *rj += a * b;
                            }
                        }
                        *sk = dot(w2_row, centre);
                    }
                    w1.copy_from_slice(composed);
                },
            );
        (w, shift)
    }
}

fn columns(slopes: &[Vec<f64>]) -> Vec<&[f64]> {
    slopes.iter().map(Vec::as_slice).collect()
}

impl TermReparam {
    /// `None` without slopes; unidentified directions become zero columns, so they solve to `0`.
    /// Whitening twice (CholeskyQR2) leaves an orthonormality error of eps, not eps·κ(G).
    pub(crate) fn build(
        design: &Design<'_>,
        term: usize,
        split: &ScatterSplit,
        sqrt_weights: Option<&[f64]>,
    ) -> Option<Self> {
        let t = &design.terms[term];
        if !t.has_slopes() {
            return None;
        }
        let levels = t.levels();
        let n_levels = t.n_levels();
        let mut slopes: Vec<Vec<f64>> = t
            .raw_slopes()
            .map(|z| z.par_iter().with_min_len(ROW_CHUNK).copied().collect())
            .collect();
        let v = slopes.len();

        let moments = LevelMoments::build(t, split, &columns(&slopes), sqrt_weights);
        let (first, kept) = Sweeps::whiten(&moments, n_levels, v);
        let mut unidentified = Vec::new();
        for (level, kept) in kept.chunks_exact(v).enumerate() {
            if moments.w_sum(level) == Some(0.0) {
                unidentified.push(CoefficientPosition {
                    channel: Channel { term, column: 0 },
                    level,
                });
            }
            for j in (0..v).filter(|&j| !kept[j]) {
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
        first.apply(levels, &mut slopes);

        // The rank tolerance keeps the first sweep well-conditioned, so this one drops nothing.
        let moments = LevelMoments::build(t, split, &columns(&slopes), sqrt_weights);
        let (second, _) = Sweeps::whiten(&moments, n_levels, v);
        drop(moments);
        second.apply(levels, &mut slopes);

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
