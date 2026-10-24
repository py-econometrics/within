use std::ops::Range;
use std::sync::Arc;

use rayon::prelude::*;

use super::{row_weight, PreparedDesign};
use crate::linalg::{dot, GramBasisWorkspace};

/// Where grouped row `p` came from: itself when the levels were already sorted.
pub(crate) enum RowOrder {
    Identity(usize),
    Permutation(Vec<u32>),
}

impl RowOrder {
    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Identity(n) => *n,
            Self::Permutation(rows) => rows.len(),
        }
    }

    pub(crate) fn rows(&self, range: Range<usize>) -> impl Iterator<Item = u32> + '_ {
        match self {
            Self::Identity(_) => rayon::iter::Either::Left(range.map(|i| i as u32)),
            Self::Permutation(rows) => rayon::iter::Either::Right(rows[range].iter().copied()),
        }
    }

    pub(crate) fn par_iter(&self) -> impl IndexedParallelIterator<Item = u32> + '_ {
        match self {
            Self::Identity(n) => {
                rayon::iter::Either::Left((0..*n).into_par_iter().map(|i| i as u32))
            }
            Self::Permutation(rows) => rayon::iter::Either::Right(rows.par_iter().copied()),
        }
    }
}

/// Observations grouped by eliminated level.
pub(crate) struct LevelGroups {
    /// Grouped row `p` is original row `order[p]`.
    pub(crate) order: RowOrder,
    /// Eliminated level `e` owns grouped rows `starts[e]..starts[e + 1]`.
    pub(crate) starts: Vec<usize>,
}

impl LevelGroups {
    fn build(levels: &[u32], n_levels: usize) -> Self {
        let mut starts = vec![0usize; n_levels + 1];
        for &l in levels {
            starts[l as usize + 1] += 1;
        }
        for l in 0..n_levels {
            starts[l + 1] += starts[l];
        }
        let order = if levels.is_sorted() {
            RowOrder::Identity(levels.len())
        } else {
            let mut order = vec![0u32; levels.len()];
            for (i, &l) in levels.iter().enumerate() {
                order[starts[l as usize]] = i as u32;
                starts[l as usize] += 1;
            }
            // The insertion cursors are now level ends; shift them back to starts.
            starts.rotate_right(1);
            starts[0] = 0;
            RowOrder::Permutation(order)
        };
        Self { order, starts }
    }

    /// Per original row, `loadings` at that row, laid out in grouped row order.
    pub(crate) fn gather(&self, loadings: &[&[f64]]) -> Vec<f64> {
        if loadings.is_empty() {
            return Vec::new();
        }
        let mut values = vec![0.0; self.order.len() * loadings.len()];
        values
            .par_chunks_mut(loadings.len())
            .zip(self.order.par_iter())
            .for_each(|(row, i)| {
                row.iter_mut()
                    .zip(loadings)
                    .for_each(|(v, z)| *v = z[i as usize]);
            });
        values
    }
}

/// The largest term eliminated per level: its rows grouped and its Gram blocks inverted.
pub(crate) struct Elimination<'a> {
    prepared: &'a PreparedDesign<'a>,
    term: usize,
    removed: Range<usize>,
    groups: Arc<LevelGroups>,
    /// The eliminated term's coefficient columns: each level's Gram block is `width × width`.
    width: usize,
    /// Per eliminated level, a generalized inverse of its Gram block, row-major.
    inverse: Arc<Vec<f64>>,
}

/// A kept coefficient column: its levels' first slot among the kept coefficients, and what it multiplies.
#[derive(Clone, Copy)]
pub(crate) struct KeptChannel<'p> {
    pub(crate) start: usize,
    pub(crate) levels: &'p [u32],
    /// `None` is an intercept.
    pub(crate) loading: Option<&'p [f64]>,
}

impl<'a> Elimination<'a> {
    pub(crate) fn new(prepared: &'a PreparedDesign<'a>) -> Self {
        let (term, largest) = prepared
            .design
            .terms
            .iter()
            .enumerate()
            .max_by_key(|(_, t)| t.n_dofs())
            .expect("validated nonempty design");
        let removed = largest.dofs();
        let mut e = Self {
            prepared,
            term,
            removed,
            groups: Arc::new(LevelGroups::build(largest.levels(), largest.n_levels())),
            width: largest.n_columns(),
            inverse: Arc::default(),
        };
        e.inverse = Arc::new(e.block_inverses());
        e
    }

    /// `Wᵀ W` from pivoted Gram–Schmidt is a generalized inverse, so `A G⁻ Aᵀ` is the exact projector.
    fn block_inverses(&self) -> Vec<f64> {
        let c = self.width;
        let loadings = self.eliminated_loadings();
        let sqrt_weights = self.prepared.sqrt_weights();
        let LevelGroups { order, starts } = &*self.groups;
        if loadings == [None] {
            return starts
                .par_windows(2)
                .map(|bounds| {
                    let weight = match sqrt_weights {
                        Some(sw) => order
                            .rows(bounds[0]..bounds[1])
                            .map(|i| row_weight(Some(sw), i as usize))
                            .sum(),
                        None => (bounds[1] - bounds[0]) as f64,
                    };
                    if weight > 0.0 && weight.is_finite() {
                        weight.recip()
                    } else {
                        0.0
                    }
                })
                .collect();
        }
        let mut inverse = vec![0.0; self.n_levels() * c * c];
        (inverse.par_chunks_mut(c * c).enumerate().with_min_len(256)).for_each_init(
            || GramBasisWorkspace::new(c),
            |workspace, (l, out)| {
                let basis = workspace.orthonormalize(|block| {
                    block.fill(0.0);
                    for i in order.rows(starts[l]..starts[l + 1]) {
                        let (i, w) = (i as usize, row_weight(sqrt_weights, i as usize));
                        for (j, aj) in loadings.iter().enumerate() {
                            let wa = aj.map_or(w, |z| w * z[i]);
                            for (k, ak) in loadings.iter().enumerate() {
                                block[j * c + k] += ak.map_or(wa, |z| wa * z[i]);
                            }
                        }
                    }
                });
                for row in basis.rows.chunks_exact(c) {
                    for (j, &rj) in row.iter().enumerate() {
                        for (k, &rk) in row.iter().enumerate() {
                            out[j * c + k] += rj * rk;
                        }
                    }
                }
            },
        );
        inverse
    }

    pub(crate) fn prepared(&self) -> &'a PreparedDesign<'a> {
        self.prepared
    }

    #[cfg(feature = "pair-schur-prototype")]
    pub(crate) fn term(&self) -> usize {
        self.term
    }

    /// The eliminated coefficients' slots in the full coefficient vector.
    pub(crate) fn removed(&self) -> Range<usize> {
        self.removed.clone()
    }

    pub(crate) fn groups(&self) -> &Arc<LevelGroups> {
        &self.groups
    }

    pub(crate) fn width(&self) -> usize {
        self.width
    }

    pub(crate) fn inverse(&self) -> &Arc<Vec<f64>> {
        &self.inverse
    }

    fn levels(&self) -> &'a [u32] {
        self.prepared.design.terms[self.term].levels()
    }

    /// Per eliminated coefficient column, its loading; `None` is the intercept.
    pub(crate) fn eliminated_loadings(&self) -> Vec<Option<&'a [f64]>> {
        let prepared: &'a PreparedDesign<'a> = self.prepared;
        let term = prepared.term(self.term);
        (0..self.width).map(|c| term.loading(c)).collect()
    }

    /// Every kept coefficient column, in kept-coefficient order.
    pub(crate) fn kept_channels(&self) -> Vec<KeptChannel<'a>> {
        let prepared: &'a PreparedDesign<'a> = self.prepared;
        let mut start = 0;
        let mut channels = Vec::new();
        for (index, term) in prepared.terms().enumerate() {
            if index == self.term {
                continue;
            }
            for c in 0..term.term.n_columns() {
                channels.push(KeptChannel {
                    start: start + c * term.term.n_levels(),
                    levels: term.term.levels(),
                    loading: term.loading(c),
                });
            }
            start += term.term.n_dofs();
        }
        channels
    }

    pub(crate) fn n_levels(&self) -> usize {
        self.removed.len() / self.width
    }

    pub(crate) fn n_kept(&self) -> usize {
        self.prepared.design.n_dofs - self.removed.len()
    }

    pub(crate) fn embed(&self, reduced: &[f64], full: &mut [f64]) {
        let start = self.removed.start;
        full[..start].copy_from_slice(&reduced[..start]);
        full[self.removed.clone()].fill(0.0);
        full[self.removed.end..].copy_from_slice(&reduced[start..]);
    }

    pub(crate) fn restrict(&self, full: &[f64], reduced: &mut [f64]) {
        let start = self.removed.start;
        reduced[..start].copy_from_slice(&full[..start]);
        reduced[start..].copy_from_slice(&full[self.removed.end..]);
    }

    pub(crate) fn scale(&self, row: usize) -> f64 {
        self.prepared.sqrt_weights().map_or(1.0, |w| w[row])
    }

    /// The eliminated coefficients fitting `values`, laid out like `removed`.
    pub(crate) fn coefficients(&self, values: &[f64], coefficients: &mut [f64]) {
        let (n, c) = (self.n_levels(), self.width);
        let loadings = self.eliminated_loadings();
        let mut sums = vec![0.0; n * c];
        for (i, &level) in self.levels().iter().enumerate() {
            let v = self.scale(i) * values[i];
            for (sum, a) in sums[level as usize * c..][..c].iter_mut().zip(&loadings) {
                *sum += a.map_or(v, |z| z[i] * v);
            }
        }
        for (l, sums) in sums.chunks_exact(c).enumerate() {
            let inverse = &self.inverse[l * c * c..][..c * c];
            for (j, row) in inverse.chunks_exact(c).enumerate() {
                coefficients[j * n + l] = dot(row, sums);
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn project(&self, values: &mut [f64], coefficients: &mut [f64]) {
        self.coefficients(values, coefficients);
        let n = self.n_levels();
        let loadings = self.eliminated_loadings();
        for (i, &level) in self.levels().iter().enumerate() {
            let fit: f64 = (loadings.iter().enumerate())
                .map(|(j, a)| {
                    let b = coefficients[j * n + level as usize];
                    a.map_or(b, |z| z[i] * b)
                })
                .sum();
            values[i] -= self.scale(i) * fit;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn level_groups_preserve_rows_with_empty_and_uneven_levels() {
        for (levels, n_levels) in [
            (&[][..], 0),
            (&[][..], 3),
            (&[0, 0, 0][..], 1),
            (&[1, 1, 3, 3][..], 6),
            (&[3, 1, 3, 0, 3, 1, 3][..], 6),
        ] {
            let groups = LevelGroups::build(levels, n_levels);
            assert_eq!(groups.starts[0], 0);
            assert_eq!(groups.starts[n_levels], levels.len());
            for l in 0..n_levels {
                let expected: Vec<u32> = levels
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &level)| (level as usize == l).then_some(i as u32))
                    .collect();
                assert_eq!(
                    groups
                        .order
                        .rows(groups.starts[l]..groups.starts[l + 1])
                        .collect::<Vec<_>>(),
                    expected
                );
            }
        }
    }
}
