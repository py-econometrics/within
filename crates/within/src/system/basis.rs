//! `R`: each term's slopes orthogonalized by a weighted QR over the term's levels.

use std::borrow::Cow;

use rayon::prelude::*;

mod level_moments;

use super::RowWeights;
use crate::design::Slopes;
use crate::design::LEVEL_GROUP;
use crate::design::{TermIncidence, TermLayout};
use crate::linalg::{dot, GramBasisWorkspace};
use level_moments::{LevelMean, LevelMoments};

/// Rows per parallel task of a sweep's row pass.
const ROW_CHUNK: usize = 1 << 14;

/// `X_t = X̃_t·R_t⁻¹`: the term's slopes `W`-orthogonal within each level, with `R_t`.
pub(super) struct OrthogonalSlopes {
    /// `X̃_t`; empty for a slope-free term, whose `R_t` is the identity.
    columns: Vec<Vec<f64>>,
    /// Per level `u = factor·z − shift`, a row-major `V×V` block whose rows past the rank are zero.
    factor: Vec<f64>,
    shift: Vec<f64>,
    /// Per level × slope, whether the data identify the caller's slope `j`.
    kept: Vec<bool>,
}

impl Slopes<'_> {
    /// `X̃_t` and `R_t` by two weighted Cholesky QR passes over each level of `term`, in place.
    /// Unidentified directions become zero columns, so they solve to `0`.
    /// Two passes (CholeskyQR2) leave an orthonormality error of eps, not eps·κ(G).
    pub(super) fn orthogonalize(
        self,
        term: TermIncidence<'_>,
        weights: &RowWeights,
    ) -> OrthogonalSlopes {
        let mut columns: Vec<Vec<f64>> = self
            .columns
            .into_iter()
            .map(|z| match z {
                Cow::Owned(z) => z,
                Cow::Borrowed(z) => z.par_iter().with_min_len(ROW_CHUNK).copied().collect(),
            })
            .collect();
        let (layout, v) = (term.layout(), columns.len());
        let n_levels = layout.n_levels();
        if v == 0 {
            return OrthogonalSlopes {
                columns,
                factor: Vec::new(),
                shift: Vec::new(),
                kept: Vec::new(),
            };
        }
        let moments = LevelMoments::build(term, &views(&columns), weights);
        let (first, kept) = Sweeps::orthonormalize(&moments, n_levels, v);
        drop(moments);
        first.apply(term.levels(), &mut columns);

        // The rank tolerance keeps the first pass well-conditioned, so this one drops nothing.
        let moments = LevelMoments::build(term, &views(&columns), weights);
        let (second, _) = Sweeps::orthonormalize(&moments, n_levels, v);
        drop(moments);
        second.apply(term.levels(), &mut columns);

        let (factor, shift) = first.compose(&second);
        OrthogonalSlopes {
            columns,
            factor,
            shift,
            kept,
        }
    }
}

impl OrthogonalSlopes {
    pub(super) fn columns(&self) -> &[Vec<f64>] {
        &self.columns
    }

    /// The caller's slopes the data cannot identify in `level`.
    pub(super) fn dropped(&self, level: usize) -> impl Iterator<Item = usize> + '_ {
        let v = self.columns.len();
        let kept = &self.kept[level * v..][..v];
        (0..v).filter(|&j| !kept[j])
    }

    /// How many slopes the data identify in `level`.
    pub(super) fn rank(&self, level: usize) -> usize {
        let v = self.columns.len();
        self.kept[level * v..][..v].iter().filter(|&&k| k).count()
    }

    /// Map this term's block of solve-basis coefficients back: `block ← R_t·block`.
    pub(super) fn back_transform(&self, layout: &TermLayout, block: &mut [f64]) {
        let slope =
            |j: usize, level: usize| layout.block_column(layout.slope_column(j)).start + level;
        let v = layout.n_slopes();
        if v == 0 {
            return;
        }
        let mut b = vec![0.0; v];
        for (l, (w, shift)) in self
            .factor
            .chunks_exact(v * v)
            .zip(self.shift.chunks_exact(v))
            .enumerate()
        {
            b.fill(0.0);
            let mut offset = 0.0;
            for ((k, w_row), s) in w.chunks_exact(v).enumerate().zip(shift) {
                let bk = block[slope(k, l)];
                offset += bk * s;
                for (bj, wj) in b.iter_mut().zip(w_row) {
                    *bj += wj * bk;
                }
            }
            for (j, &bj) in b.iter().enumerate() {
                block[slope(j, l)] = bj;
            }
            if let Some(intercept) = layout.intercept_column() {
                block[layout.block_column(intercept)][l] -= offset;
            }
        }
    }
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
    /// One Cholesky QR pass over each level's Gram in `moments`; `kept[l·V + j]` says whether
    /// slope `j` survived.
    fn orthonormalize(moments: &LevelMoments, n_levels: usize, v: usize) -> (Self, Vec<bool>) {
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

fn views(slopes: &[Vec<f64>]) -> Vec<&[f64]> {
    slopes.iter().map(Vec::as_slice).collect()
}
