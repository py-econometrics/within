use std::ops::Range;
use std::sync::Arc;

use rayon::prelude::*;

use approx_chol::Side;

use super::factor_pairs::{PairMatrix, PairSubdomains};
use super::{PreparedDesign, PreparedTerm};
use crate::channel::ChannelPair;
use crate::linalg::GramBasisWorkspace;

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

/// Each grouped row's eliminated loadings: an implicit leading `1` for an intercept, then slopes.
pub(crate) struct EliminatedRows {
    pub(crate) width: usize,
    pub(crate) intercept: bool,
    /// Row-major, `width - intercept` per row.
    slopes: Vec<f64>,
}

impl EliminatedRows {
    #[inline(always)]
    pub(crate) fn slopes(&self, p: usize) -> &[f64] {
        let n = self.width - self.intercept as usize;
        &self.slopes[p * n..][..n]
    }

    #[inline(always)]
    pub(crate) fn loading(&self, p: usize, column: usize) -> f64 {
        match (self.intercept, column) {
            (true, 0) => 1.0,
            (true, j) => self.slopes(p)[j - 1],
            (false, j) => self.slopes(p)[j],
        }
    }
}

/// The kept design's rows in grouped order: unit intercept columns, then valued slope columns.
pub(crate) struct KeptRows {
    /// Each row's reduced coefficient per kept channel, intercepts first, row-major.
    columns: Vec<u32>,
    /// Each row's loading per kept slope channel, row-major in `columns`' order.
    values: Vec<f64>,
    pub(crate) n_channels: usize,
    n_intercepts: usize,
}

impl KeptRows {
    #[inline(always)]
    fn columns<const WIDTH: usize>(&self, p: usize) -> &[u32] {
        let width = if WIDTH == 0 { self.n_channels } else { WIDTH };
        &self.columns[p * width..][..width]
    }

    /// `D_kept x` at grouped row `p`; `WIDTH` is the channel count, `0` when not specialized.
    #[inline(always)]
    pub(crate) fn row<const WIDTH: usize>(&self, p: usize, x: &[f64]) -> f64 {
        let columns = self.columns::<WIDTH>(p);
        if self.values.is_empty() {
            return columns.iter().map(|&c| x[c as usize]).sum();
        }
        let (intercepts, slopes) = columns.split_at(self.n_intercepts);
        let unit: f64 = intercepts.iter().map(|&c| x[c as usize]).sum();
        let values = &self.values[p * slopes.len()..][..slopes.len()];
        (slopes.iter().zip(values)).fold(unit, |acc, (&c, &z)| acc + z * x[c as usize])
    }

    /// Each `(column, loading)` of row `p`.
    #[inline(always)]
    pub(crate) fn for_each_entry(&self, p: usize, mut f: impl FnMut(u32, f64)) {
        let columns = self.columns::<0>(p);
        let (intercepts, slopes) = columns.split_at(self.n_intercepts);
        intercepts.iter().for_each(|&c| f(c, 1.0));
        let values = &self.values[p * slopes.len()..][..slopes.len()];
        (slopes.iter().zip(values)).for_each(|(&c, &z)| f(c, z));
    }

    /// `acc += v D_kept[p, :]`.
    #[inline(always)]
    pub(crate) fn add_row(&self, p: usize, v: f64, acc: &mut [f64]) {
        let columns = self.columns::<0>(p);
        if self.values.is_empty() {
            return columns.iter().for_each(|&c| acc[c as usize] += v);
        }
        let (intercepts, slopes) = columns.split_at(self.n_intercepts);
        intercepts.iter().for_each(|&c| acc[c as usize] += v);
        let values = &self.values[p * slopes.len()..][..slopes.len()];
        (slopes.iter().zip(values)).for_each(|(&c, &z)| acc[c as usize] += z * v);
    }
}

/// The design's rows grouped by eliminated level, so each level's rows are contiguous.
pub(crate) struct GroupedRows {
    /// Grouped row `p` is original row `order[p]`.
    pub(crate) order: RowOrder,
    /// Eliminated level `e` owns grouped rows `starts[e]..starts[e + 1]`.
    pub(crate) starts: Vec<usize>,
    pub(crate) sqrt_weights: Option<Vec<f64>>,
    pub(crate) eliminated: EliminatedRows,
    pub(crate) kept: KeptRows,
}

/// Where each level's rows go: the grouped order and each level's first grouped row.
fn group_by_level(levels: &[u32], n_levels: usize) -> (RowOrder, Vec<usize>) {
    let mut starts = vec![0usize; n_levels + 1];
    for &l in levels {
        starts[l as usize + 1] += 1;
    }
    for l in 0..n_levels {
        starts[l + 1] += starts[l];
    }
    if levels.is_sorted() {
        return (RowOrder::Identity(levels.len()), starts);
    }
    let mut order = vec![0u32; levels.len()];
    for (i, &l) in levels.iter().enumerate() {
        order[starts[l as usize]] = i as u32;
        starts[l as usize] += 1;
    }
    // The insertion cursors are now level ends; shift them back to starts.
    starts.rotate_right(1);
    starts[0] = 0;
    (RowOrder::Permutation(order), starts)
}

impl GroupedRows {
    fn build(
        prepared: &PreparedDesign<'_>,
        eliminated: &PreparedTerm<'_>,
        kept: &[KeptChannel<'_>],
    ) -> Self {
        let (order, starts) = group_by_level(eliminated.term.levels(), eliminated.term.n_levels());
        let gather = |loadings: &[&[f64]]| -> Vec<f64> {
            let mut values = vec![0.0; order.len() * loadings.len()];
            if !loadings.is_empty() {
                (values.par_chunks_mut(loadings.len()).zip(order.par_iter())).for_each(
                    |(row, i)| {
                        (row.iter_mut().zip(loadings)).for_each(|(v, z)| *v = z[i as usize]);
                    },
                );
            }
            values
        };
        let width = eliminated.term.n_columns();
        let slopes: Vec<&[f64]> = (0..width).filter_map(|c| eliminated.loading(c)).collect();
        let (intercepts, valued): (Vec<&KeptChannel<'_>>, Vec<_>) =
            kept.iter().partition(|channel| channel.loading.is_none());
        let channels: Vec<_> = intercepts.iter().chain(&valued).copied().collect();
        let mut columns = vec![0; order.len() * channels.len()];
        if !channels.is_empty() {
            (columns.par_chunks_mut(channels.len()).zip(order.par_iter())).for_each(|(row, i)| {
                row.iter_mut().zip(&channels).for_each(|(c, channel)| {
                    *c = channel.start as u32 + channel.levels[i as usize];
                });
            });
        }
        let values: Vec<&[f64]> = valued.iter().filter_map(|c| c.loading).collect();
        Self {
            sqrt_weights: (prepared.sqrt_weights())
                .map(|sw| order.par_iter().map(|i| sw[i as usize]).collect()),
            eliminated: EliminatedRows {
                width,
                intercept: width > slopes.len(),
                slopes: gather(&slopes),
            },
            kept: KeptRows {
                columns,
                values: gather(&values),
                n_channels: channels.len(),
                n_intercepts: intercepts.len(),
            },
            order,
            starts,
        }
    }

    pub(crate) fn n_levels(&self) -> usize {
        self.starts.len() - 1
    }

    /// Grouped row `p`'s weight `w = (√w)²`.
    #[inline(always)]
    pub(crate) fn weight(&self, p: usize) -> f64 {
        self.sqrt_weights.as_ref().map_or(1.0, |s| s[p] * s[p])
    }
}

/// The largest term eliminated per level: its rows grouped and its Gram blocks inverted.
pub(crate) struct Elimination<'a> {
    prepared: &'a PreparedDesign<'a>,
    term: usize,
    removed: Range<usize>,
    rows: Arc<GroupedRows>,
    /// Per eliminated level, a generalized inverse of its `width × width` Gram block, row-major.
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

/// `Wᵀ W` from pivoted Gram–Schmidt is a generalized inverse, so `A G⁻ Aᵀ` is the exact projector.
fn block_inverses(rows: &GroupedRows) -> Vec<f64> {
    let (starts, eliminated) = (&rows.starts, &rows.eliminated);
    let c = eliminated.width;
    if c == 1 && eliminated.intercept {
        return (starts.par_windows(2))
            .map(|bounds| {
                let weight: f64 = match rows.sqrt_weights {
                    Some(_) => (bounds[0]..bounds[1]).map(|p| rows.weight(p)).sum(),
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
    let mut inverse = vec![0.0; rows.n_levels() * c * c];
    (inverse.par_chunks_mut(c * c).enumerate().with_min_len(256)).for_each_init(
        || GramBasisWorkspace::new(c),
        |workspace, (l, out)| {
            let basis = workspace.orthonormalize(|block| {
                block.fill(0.0);
                for p in starts[l]..starts[l + 1] {
                    let w = rows.weight(p);
                    for j in 0..c {
                        let wa = w * eliminated.loading(p, j);
                        for k in 0..c {
                            block[j * c + k] += wa * eliminated.loading(p, k);
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

impl<'a> Elimination<'a> {
    pub(crate) fn new(prepared: &'a PreparedDesign<'a>) -> Self {
        let (term, largest) = (prepared.design.terms.iter().enumerate())
            .max_by_key(|(_, t)| t.n_dofs())
            .expect("validated nonempty design");
        let kept = kept_channels(prepared, term);
        let rows = GroupedRows::build(prepared, &prepared.term(term), &kept);
        Self {
            prepared,
            term,
            removed: largest.dofs(),
            inverse: Arc::new(block_inverses(&rows)),
            rows: Arc::new(rows),
        }
    }

    pub(crate) fn prepared(&self) -> &'a PreparedDesign<'a> {
        self.prepared
    }

    /// The eliminated coefficients' slots in the full coefficient vector.
    pub(crate) fn removed(&self) -> Range<usize> {
        self.removed.clone()
    }

    pub(crate) fn rows(&self) -> &Arc<GroupedRows> {
        &self.rows
    }

    pub(crate) fn term(&self) -> usize {
        self.term
    }

    pub(crate) fn width(&self) -> usize {
        self.rows.eliminated.width
    }

    pub(crate) fn inverse(&self) -> &Arc<Vec<f64>> {
        &self.inverse
    }

    /// Per eliminated coefficient column, its loading; `None` is the intercept.
    pub(crate) fn eliminated_loadings(&self) -> Vec<Option<&'a [f64]>> {
        let prepared: &'a PreparedDesign<'a> = self.prepared;
        let term = prepared.term(self.term);
        (0..self.width()).map(|c| term.loading(c)).collect()
    }

    /// Every kept coefficient column, in kept-coefficient order.
    pub(crate) fn kept_channels(&self) -> Vec<KeptChannel<'a>> {
        kept_channels(self.prepared, self.term)
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

    #[cfg(test)]
    pub(crate) fn restrict(&self, full: &[f64], reduced: &mut [f64]) {
        let start = self.removed.start;
        reduced[..start].copy_from_slice(&full[..start]);
        reduced[start..].copy_from_slice(&full[self.removed.end..]);
    }
}

fn kept_channels<'a>(prepared: &'a PreparedDesign<'a>, eliminated: usize) -> Vec<KeptChannel<'a>> {
    let mut start = 0;
    let mut channels = Vec::new();
    for (index, term) in prepared.terms().enumerate() {
        if index == eliminated {
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

/// Which factor pairs the reduced subdomains keep.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Pairs {
    /// Every pair: Schur complements plus the retained pairs' Grams, partition-weighted.
    All,
    /// Only the pairs split by the eliminated factor: block Jacobi on `S`.
    SchurOnly,
}

/// Subdomains over the kept coefficients: the eliminated term's pairs factor their complement.
pub(crate) struct ReducedPairs<'e> {
    pub(crate) elimination: &'e Elimination<'e>,
    pub(crate) pairs: Pairs,
    pub(crate) scales: Option<&'e PieceScales>,
}

/// Per kept coefficient, `(S_ll / piece_ll)^{-γ}`: one vector for the retained Grams, one per eliminated column.
pub(crate) struct PieceScales {
    pub(crate) gram: Arc<Vec<f64>>,
    pub(crate) complement: Vec<Arc<Vec<f64>>>,
}

impl PairSubdomains for ReducedPairs<'_> {
    fn matrix(&self, pair: ChannelPair) -> Option<PairMatrix> {
        let term = self.elimination.term;
        match (pair.rows.term == term, pair.cols.term == term) {
            (true, _) => Some(PairMatrix::Complement(Side::Right)),
            (_, true) => Some(PairMatrix::Complement(Side::Left)),
            _ if self.pairs == Pairs::SchurOnly => None,
            _ => Some(PairMatrix::Gram),
        }
    }

    fn dof(&self, dof: usize) -> usize {
        let removed = &self.elimination.removed;
        debug_assert!(!removed.contains(&dof), "a kept coefficient");
        if dof >= removed.end {
            dof - removed.len()
        } else {
            dof
        }
    }

    fn n_dofs(&self, _: &PreparedDesign<'_>) -> usize {
        self.elimination.n_kept()
    }

    /// Unlike the full design's, slope overlaps do not slow the reduced solve under `1/√c`.
    fn partition_weighted(&self, _: &PreparedDesign<'_>) -> bool {
        true
    }

    fn piece_scale(&self, pair: ChannelPair) -> Option<Arc<Vec<f64>>> {
        let (scales, term) = (self.scales?, self.elimination.term);
        Some(match (pair.rows.term == term, pair.cols.term == term) {
            (true, _) => scales.complement[pair.rows.column].clone(),
            (_, true) => scales.complement[pair.cols.column].clone(),
            _ => scales.gram.clone(),
        })
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
            let (order, starts) = group_by_level(levels, n_levels);
            assert_eq!(starts[0], 0);
            assert_eq!(starts[n_levels], levels.len());
            for l in 0..n_levels {
                let expected: Vec<u32> = levels
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &level)| (level as usize == l).then_some(i as u32))
                    .collect();
                assert_eq!(
                    order.rows(starts[l]..starts[l + 1]).collect::<Vec<_>>(),
                    expected
                );
            }
        }
    }
}
