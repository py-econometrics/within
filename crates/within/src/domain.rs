//! Domain layer: [`Design`] (design-matrix metadata) and factor-pair [`Subdomain`] construction.

pub(crate) mod collinearity;
pub(crate) mod cross_tab;
mod effect;
pub(crate) mod factor_pairs;
mod level_moments;
mod prepared;
mod reparam;

pub(crate) use cross_tab::{BlockDiagonals, CrossTab};
pub(crate) use prepared::{PreparedDesign, PreparedTerm};
use reparam::TermReparam;

pub use effect::Effect;

pub(crate) use factor_pairs::{
    build_local_domains, CoordinateMap, Grounding, LocalComponent, LocalDomain, MatrixForm,
    SddmMatrix,
};

use crate::channel::Channel;
use crate::BuildError;
use ndarray::{ArrayView2, Axis};
use rayon::prelude::*;
use rustc_hash::FxBuildHasher;
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::ops::Range;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize};
use std::sync::Arc;

/// What one coefficient column of a term multiplies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Column {
    Intercept,
    /// The term's slope `j`.
    Slope(usize),
}

/// Mapping between caller-visible factor labels and compact numerical positions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum FactorEncoding {
    /// Caller label `k` is internal position `k`.
    Identity { n_levels: usize },
    /// Arbitrary integer caller labels, ordered by internal position.
    Integer { labels: Arc<[u32]> },
}

impl FactorEncoding {
    fn identity(n_levels: usize) -> Self {
        Self::Identity { n_levels }
    }

    fn integer(labels: Vec<u32>) -> Self {
        debug_assert!(labels.windows(2).all(|pair| pair[0] < pair[1]));
        Self::Integer {
            labels: labels.into(),
        }
    }

    // Inlined into `build`, its label loops lose registers and reload pointers from the stack.
    #[inline(never)]
    fn encode_labels(labels: Cow<'_, [u32]>) -> EncodedFactor<'_> {
        let (Some(&first), Some(&last)) = (labels.first(), labels.last()) else {
            return EncodedFactor {
                encoding: Self::identity(0),
                levels: labels,
                sorted: true,
            };
        };
        let sorted = match is_non_decreasing_with_gaps(&labels) {
            // Non-decreasing in steps of 0 or 1, hence no gaps and no compaction needed.
            Some(false) => return EncodedFactor::contiguous(labels, first, last, true),
            // Non-decreasing with gaps
            Some(true) => true,
            // Not non-decreasing but potentially without gaps
            None => false,
        };
        // Get min and max label
        let (min, max) = if sorted {
            // Min and max of a sorted range are its first and last entry.
            (first, last)
        } else {
            labels.par_iter().map(|&label| (label, label)).reduce(
                || (u32::MAX, u32::MIN),
                |(lo, hi), (a, b)| (lo.min(a), hi.max(b)),
            )
        };
        // Go straight to sparse encoding if label range does not fit within labels' length
        if (max - min) as usize >= labels.len() {
            return EncodedFactor::sparse(&labels, sorted);
        }
        // Get distinct labels for compaction to distinguish the remaining three cases:
        // (1) sorted with gaps; (2) unsorted with gaps; (3) unsorted without gaps
        let distinct_labels: Vec<u32> = if sorted {
            // Sorted, so equal labels sit next to each other: taking the first label of each
            // group of equal labels lists every distinct label once, in ascending order.
            labels
                .par_chunk_by(|a, b| a == b)
                .map(|run| run[0])
                .collect()
        } else if let Some(is_gap) = find_gaps(&labels, min, max) {
            // Unsorted with gaps; need to find gaps
            is_gap
                .par_iter()
                .positions(|&gap| !gap)
                .map(|offset| min + offset as u32)
                .collect()
        } else {
            // No gaps, labels are unsorted but already contiguous
            return EncodedFactor::contiguous(labels, min, max, false);
        };

        EncodedFactor::dense(&labels, distinct_labels, sorted)
    }

    pub(crate) fn n_levels(&self) -> usize {
        match self {
            Self::Identity { n_levels } => *n_levels,
            Self::Integer { labels } => labels.len(),
        }
    }

    pub(crate) fn position(&self, label: u32) -> Option<usize> {
        match self {
            Self::Identity { n_levels } => {
                let position = label as usize;
                (position < *n_levels).then_some(position)
            }
            Self::Integer { labels } => labels.binary_search(&label).ok(),
        }
    }

    pub(crate) fn label(&self, position: usize) -> Option<u32> {
        match self {
            Self::Identity { n_levels } => {
                if position >= *n_levels {
                    return None;
                }

                u32::try_from(position).ok()
            }

            Self::Integer { labels } => labels.get(position).copied(),
        }
    }
}

/// A factor's level encoding and level column (the input itself if already positions).
struct EncodedFactor<'a> {
    encoding: FactorEncoding,
    levels: Cow<'a, [u32]>,
    /// `levels` is non-decreasing.
    sorted: bool,
}

impl<'a> EncodedFactor<'a> {
    /// Every value in `min..=max` occurs, so a label's internal position is `label - min`.
    fn contiguous(labels: Cow<'a, [u32]>, min: u32, max: u32, sorted: bool) -> Self {
        if min == 0 {
            return Self {
                encoding: FactorEncoding::identity(max as usize + 1),
                levels: labels,
                sorted,
            };
        }
        Self {
            encoding: FactorEncoding::integer((min..=max).into_par_iter().collect()),
            levels: Cow::Owned(labels.par_iter().map(|&label| label - min).collect()),
            sorted,
        }
    }
}

impl EncodedFactor<'static> {
    /// Each label's internal position is its index in `distinct_labels`,
    /// which must hold the column's distinct labels in ascending order.
    fn dense(labels: &[u32], distinct_labels: Vec<u32>, sorted: bool) -> Self {
        let min = distinct_labels[0];
        let range_width = (distinct_labels[distinct_labels.len() - 1] - min) as usize + 1;
        // Each slot is written by exactly one label, so relaxed atomics are plain stores.
        let position_by_offset: Vec<AtomicU32> = (0..range_width)
            .into_par_iter()
            .map(|_| AtomicU32::new(0))
            .collect();
        distinct_labels
            .par_iter()
            .enumerate()
            .for_each(|(position, &label)| {
                position_by_offset[(label - min) as usize].store(position as u32, Relaxed);
            });
        let positions = labels
            .par_iter()
            .map(|&label| position_by_offset[(label - min) as usize].load(Relaxed))
            .collect();

        EncodedFactor {
            encoding: FactorEncoding::integer(distinct_labels),
            levels: Cow::Owned(positions),
            sorted,
        }
    }

    /// Each rayon task numbers its labels in the order it first sees them,
    /// one hash per label; only distinct labels are sorted, and their ranks
    /// replace the task-local numbers through a small array per task.
    fn sparse(labels: &[u32], sorted: bool) -> Self {
        let mut positions = vec![0u32; labels.len()];
        let pieces: Vec<(&mut [u32], Vec<u32>)> =
            rayon::iter::split((labels, &mut positions[..]), |(labels, positions)| {
                if labels.len() < 2 {
                    return ((labels, positions), None);
                }
                let mid = labels.len() / 2;
                let (left, right) = labels.split_at(mid);
                let (left_positions, right_positions) = positions.split_at_mut(mid);
                ((left, left_positions), Some((right, right_positions)))
            })
            .map(|(labels, positions)| {
                let mut local_id = HashMap::<u32, u32, FxBuildHasher>::default();
                let mut first_seen = Vec::new();
                for (&label, position) in labels.iter().zip(positions.iter_mut()) {
                    *position = *local_id.entry(label).or_insert_with(|| {
                        first_seen.push(label);
                        (first_seen.len() - 1) as u32
                    });
                }
                (positions, first_seen)
            })
            .collect();

        // Internal positions follow ascending caller-label order.
        let mut caller_labels: Vec<u32> = pieces
            .iter()
            .flat_map(|(_, first_seen)| first_seen.iter().copied())
            .collect::<HashSet<u32, FxBuildHasher>>()
            .into_iter()
            .collect();
        caller_labels.par_sort_unstable();
        let rank: HashMap<u32, u32, FxBuildHasher> = caller_labels
            .iter()
            .enumerate()
            .map(|(position, &label)| {
                let position =
                    u32::try_from(position).expect("an internal label position fits in u32");
                (label, position)
            })
            .collect();
        pieces.into_par_iter().for_each(|(positions, first_seen)| {
            let rank_by_local_id: Vec<u32> = first_seen.iter().map(|label| rank[label]).collect();
            for position in positions {
                *position = rank_by_local_id[*position as usize];
            }
        });

        EncodedFactor {
            encoding: FactorEncoding::integer(caller_labels),
            levels: Cow::Owned(positions),
            sorted,
        }
    }
}

/// One term's coefficients and rows: column `c` of `level` lives at `offset + c · n_levels + level`.
#[derive(Debug, Clone)]
pub(crate) struct Term<'a> {
    pub(crate) encoding: FactorEncoding,
    pub(crate) offset: usize,
    /// Column 0 is an intercept; the slopes follow it.
    pub(crate) intercept: bool,
    /// Internal level position of every observation, in the design's row order.
    levels: Cow<'a, [u32]>,
    /// `levels` is non-decreasing.
    sorted: bool,
    /// Raw slopes in coefficient-column order, before whitening.
    slopes: Vec<Cow<'a, [f64]>>,
}

impl Term<'_> {
    pub(crate) fn n_levels(&self) -> usize {
        self.encoding.n_levels()
    }

    pub(crate) fn has_slopes(&self) -> bool {
        !self.slopes.is_empty()
    }

    pub(crate) fn n_columns(&self) -> usize {
        self.intercept as usize + self.slopes.len()
    }

    pub(crate) fn n_dofs(&self) -> usize {
        self.n_columns() * self.n_levels()
    }

    /// This term's coefficients in the global DOF vector.
    pub(crate) fn dofs(&self) -> Range<usize> {
        self.offset..self.offset + self.n_dofs()
    }

    /// Coefficient column `column`'s `n_levels` slots in the global DOF vector.
    pub(crate) fn column_dofs(&self, column: usize) -> Range<usize> {
        let start = self.offset + column * self.n_levels();
        start..start + self.n_levels()
    }

    /// Coefficient column of slope `j`; the intercept, when present, comes first.
    pub(crate) fn slope_column(&self, j: usize) -> usize {
        self.intercept as usize + j
    }

    /// Column `index` in layout order: the inverse of [`slope_column`](Self::slope_column).
    pub(crate) fn column(&self, index: usize) -> Column {
        debug_assert!(index < self.n_columns());
        match index.checked_sub(self.intercept as usize) {
            None => Column::Intercept,
            Some(j) => Column::Slope(j),
        }
    }

    pub(crate) fn levels(&self) -> &[u32] {
        &self.levels
    }

    pub(crate) fn sorted(&self) -> bool {
        self.sorted
    }

    /// Raw slopes in coefficient-column order.
    pub(crate) fn raw_slopes(&self) -> impl ExactSizeIterator<Item = &[f64]> {
        self.slopes.iter().map(|slope| &**slope)
    }

    fn into_owned(self) -> Term<'static> {
        Term {
            encoding: self.encoding,
            offset: self.offset,
            intercept: self.intercept,
            levels: Cow::Owned(self.levels.into_owned()),
            sorted: self.sorted,
            slopes: self
                .slopes
                .into_iter()
                .map(|slope| Cow::Owned(slope.into_owned()))
                .collect(),
        }
    }
}

/// The Gram weight of row `obs`: the operator applies `s`, so its normal matrix carries `s²`.
pub(crate) fn row_weight(sqrt_weights: Option<&[f64]>, obs: usize) -> f64 {
    sqrt_weights.map_or(1.0, |s| s[obs] * s[obs])
}

// Inspired by [within::operator::design::scatter::scatter_sorted_coalesced]
const LABEL_CHUNK: usize = 65_536;

/// Checks if `labels` are in non-decreasing order and whether there are gaps.
///
/// Returns `None` if not non-decreasing; `true` if non-decreasing without gaps; `false` otherwise.
/// Each chunk is scanned to the end with no early exit, so the comparisons vectorise;
/// once a chunk descends, rayon skips the chunks not yet started.
fn is_non_decreasing_with_gaps(labels: &[u32]) -> Option<bool> {
    labels
        .par_chunks(LABEL_CHUNK)
        .enumerate()
        .map(|(c_idx, chunk)| {
            let start = c_idx * LABEL_CHUNK;
            // Reach back one label so the pair across the chunk boundary is compared too.
            let pairs = &labels[start.saturating_sub(1)..start + chunk.len()];
            let (descends, skips) = pairs.windows(2).fold((false, false), |(d, s), pair| {
                (
                    d | (pair[1] < pair[0]),
                    s | (pair[1].wrapping_sub(pair[0]) > 1),
                )
            });
            (!descends).then_some(skips)
        })
        .try_reduce(|| false, |a, b| Some(a | b))
}

/// Finds the values in `min..=max` that do not occur in `labels`, indexed by `value - min`.
/// Stops reading labels as soon as every value has been seen, so a column without
/// gaps often costs only a prefix; with a gap it reads every label.
///
/// Every label must lie in `min..=max`, and the range must be no wider than the column.
fn find_gaps(labels: &[u32], min: u32, max: u32) -> Option<Vec<bool>> {
    let width = (max - min) as usize + 1;
    debug_assert!(
        width <= labels.len(),
        "a table over the range would outgrow the column"
    );
    let present: Vec<AtomicBool> = (0..width)
        .into_par_iter()
        .map(|_| AtomicBool::new(false))
        .collect();
    // Every slot below the watermark is known to be filled.
    let watermark = AtomicUsize::new(0);
    // Moves the watermark past the slots seen filled and returns where it stops.
    let advance = || {
        let start = watermark.load(Relaxed);
        let end = present[start..]
            .iter()
            .position(|slot| !slot.load(Relaxed))
            .map_or(width, |offset| start + offset);
        watermark.fetch_max(end, Relaxed);
        end
    };
    let filled_early = labels.par_chunks(LABEL_CHUNK).any(|chunk| {
        let mut filled_any = false;
        for &label in chunk {
            let slot = &present[(label - min) as usize];
            // Loading first keeps repeated labels from bouncing a cache line between threads.
            if !slot.load(Relaxed) {
                slot.store(true, Relaxed);
                filled_any = true;
            }
        }
        // Only a chunk that filled a slot can have completed the table.
        filled_any && advance() == width
    });
    // After the join every store is visible, so this last scan sees the final table.
    if filled_early || advance() == width {
        return None;
    }
    // A gap is not filled, hence `!slot.into_inner()`
    Some(
        present
            .into_par_iter()
            .map(|slot| !slot.into_inner())
            .collect(),
    )
}

/// Stable argsort of observations by a level column, ascending.
///
/// Compact internal positions guarantee `n_levels <= key.len()`, so counting sort
/// takes `O(n_obs + n_levels)` time and `O(n_levels)` temporary memory.
fn stable_argsort(key: &[u32], n_levels: usize) -> Vec<u32> {
    let n_obs = key.len();
    debug_assert!(
        u32::try_from(n_obs).is_ok(),
        "observation index must fit the u32 permutation"
    );
    debug_assert!(
        n_levels <= n_obs,
        "compact level count cannot exceed observation count"
    );
    let mut cursors = vec![0usize; n_levels + 1];
    for &k in key {
        debug_assert!(
            (k as usize) < n_levels,
            "counting sort key must be a level id (< n_levels)"
        );
        cursors[k as usize + 1] += 1;
    }
    for i in 1..cursors.len() {
        cursors[i] += cursors[i - 1];
    }
    let mut perm = vec![0u32; n_obs];
    for (i, &k) in key.iter().enumerate() {
        let cursor = &mut cursors[k as usize];
        perm[*cursor] = i as u32;
        *cursor += 1;
    }
    perm
}

// Inlined into `Design::build`, its loop measured +3–8% on slope designs.
#[inline(never)]
fn gather<T: Copy>(col: &[T], perm: &[u32]) -> Vec<T> {
    perm.iter().map(|&k| col[k as usize]).collect()
}

/// Fixed-effects design: its terms; clones share rows.
#[derive(Clone, Debug)]
pub struct Design<'a> {
    /// Rows in internal order (caller's, or an owned locality-sorted copy).
    pub(crate) terms: Arc<Vec<Term<'a>>>,
    pub(crate) n_obs: usize,
    pub(crate) n_dofs: usize,
    /// `obs_perm[k]` = caller's original index of the observation at internal position `k`.
    pub(crate) obs_perm: Option<Arc<[u32]>>,
}

impl<'a> Design<'a> {
    /// Lower effect terms into a design, laid out term-major (`offset[t] + c · L_t + level`).
    pub fn new(effects: impl IntoIterator<Item = Effect<'a>>) -> Result<Self, BuildError> {
        Self::build(effects.into_iter().collect(), true)
    }

    /// [`new`](Self::new) without the locality sort — profiling escape hatch.
    #[doc(hidden)]
    pub fn new_unsorted(effects: impl IntoIterator<Item = Effect<'a>>) -> Result<Self, BuildError> {
        Self::build(effects.into_iter().collect(), false)
    }

    /// Build an intercept-only design from an observation-major categories matrix.
    pub fn from_categories(categories: ArrayView2<'a, u32>) -> Result<Self, BuildError> {
        // Gather strided (C-order) columns once so every downstream read is contiguous.
        let effects = (0..categories.ncols())
            .map(|factor| {
                let column = categories.index_axis_move(Axis(1), factor);
                let levels = match column.to_slice() {
                    Some(values) => Cow::Borrowed(values),
                    None => Cow::Owned(column.to_vec()),
                };
                Effect {
                    levels,
                    intercept: true,
                    slopes: Vec::new(),
                }
            })
            .collect();
        Self::build(effects, true)
    }

    fn build(effects: Vec<Effect<'a>>, locality_sort: bool) -> Result<Self, BuildError> {
        let n_obs = effects.first().map_or(0, |e| e.levels.len());
        for (effect, e) in effects.iter().enumerate() {
            if e.levels.len() != n_obs {
                return Err(BuildError::ObservationCountMismatch {
                    effect,
                    expected: n_obs,
                    got: e.levels.len(),
                });
            }
        }
        if n_obs == 0 {
            return Err(BuildError::EmptyObservations);
        }

        let mut terms = Vec::with_capacity(effects.len());
        let mut offset = 0;
        for Effect {
            levels,
            intercept,
            slopes,
        } in effects
        {
            debug_assert!(
                slopes.iter().all(|s| s.len() == levels.len()) && (intercept || !slopes.is_empty()),
                "effect bypassed Effect::new validation"
            );
            let EncodedFactor {
                encoding,
                levels,
                sorted,
            } = FactorEncoding::encode_labels(levels);
            let term = Term {
                encoding,
                offset,
                intercept,
                levels,
                sorted,
                slopes,
            };
            offset += term.n_dofs();
            terms.push(term);
        }

        // Rejected here rather than left to panic in `to_u32`.
        if u32::try_from(offset).is_err() {
            return Err(BuildError::DofSpaceExceedsU32 { n_dofs: offset });
        }

        // Sort by the term contributing the most DOFs so its gather/scatter runs sequentially.
        let dominant = terms.iter().max_by_key(|t| t.n_dofs());
        let obs_perm = match dominant {
            Some(d) if locality_sort && !d.sorted && u32::try_from(n_obs).is_ok() => {
                let perm = stable_argsort(&d.levels, d.n_levels());
                // Factors nested in the dominant one come out sorted, keeping coalesced scatter.
                for t in terms.iter_mut() {
                    let levels = gather(&t.levels, &perm);
                    t.sorted = levels.is_sorted();
                    t.levels = Cow::Owned(levels);
                }
                // Level columns before any slope: interleaving them measured +2–4% here.
                for slope in terms.iter_mut().flat_map(|t| &mut t.slopes) {
                    *slope = Cow::Owned(gather(slope, &perm));
                }
                Some(perm.into())
            }
            _ => None,
        };

        Ok(Design {
            terms: Arc::new(terms),
            n_obs,
            n_dofs: offset,
            obs_perm,
        })
    }

    /// Convert every column to owned, dropping ties to caller buffers.
    pub fn into_owned(self) -> Design<'static> {
        let terms = Arc::unwrap_or_clone(self.terms);
        Design {
            terms: Arc::new(terms.into_iter().map(Term::into_owned).collect()),
            n_obs: self.n_obs,
            n_dofs: self.n_dofs,
            obs_perm: self.obs_perm,
        }
    }

    /// Caller order → internal order: `out[k] = v[obs_perm[k]]`; borrows when unpermuted.
    pub(crate) fn permute_obs_in<'v>(&self, v: &'v [f64]) -> Cow<'v, [f64]> {
        debug_assert_eq!(v.len(), self.n_obs);
        match &self.obs_perm {
            None => Cow::Borrowed(v),
            Some(perm) => Cow::Owned(gather(v, perm)),
        }
    }

    /// Internal order → caller order: `out[obs_perm[k]] = v[k]`.
    pub(crate) fn permute_obs_out(&self, v: Vec<f64>) -> Vec<f64> {
        debug_assert_eq!(v.len(), self.n_obs);
        match &self.obs_perm {
            None => v,
            Some(perm) => {
                let mut out = vec![0.0; v.len()];
                for (k, &orig) in perm.iter().enumerate() {
                    out[orig as usize] = v[k];
                }
                out
            }
        }
    }

    /// Number of categorical factors in the design.
    #[inline]
    pub fn n_factors(&self) -> usize {
        self.terms.len()
    }

    /// The term's coefficient columns in layout order.
    pub(crate) fn channels(&self, term: usize) -> impl Iterator<Item = Channel> + '_ {
        (0..self.terms[term].n_columns()).map(move |column| Channel { term, column })
    }

    /// What `channel` multiplies.
    pub(crate) fn column(&self, channel: Channel) -> Column {
        self.terms[channel.term].column(channel.column)
    }

    /// `channel`'s slope in internal row order, before whitening; `None` for an intercept.
    pub(crate) fn raw_slope(&self, channel: Channel) -> Option<&[f64]> {
        match self.column(channel) {
            Column::Intercept => None,
            Column::Slope(j) => Some(&self.terms[channel.term].slopes[j]),
        }
    }

    /// Number of observations (rows of D).
    #[inline]
    pub fn n_obs(&self) -> usize {
        self.n_obs
    }

    /// Total degrees of freedom (columns of D).
    #[inline]
    pub fn n_dofs(&self) -> usize {
        self.n_dofs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    impl Design<'static> {
        pub(crate) fn from_levels_for_test(columns: Vec<Vec<u32>>) -> Self {
            let effects = columns
                .iter()
                .map(|c| Effect::new(c, true, []).expect("intercept effect"));
            Design::new(effects).expect("valid design").into_owned()
        }
    }

    /// Counting sort must preserve caller order within each level so locality
    /// sorting does not change downstream summation order.
    #[test]
    fn stable_argsort_agrees_with_a_stable_reference() {
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        // `key_span` is decoupled so the counting sort also covers empty buckets.
        for (n_obs, n_levels, key_span) in [
            (0usize, 0usize, 1usize),
            (1, 1, 1),
            (997, 1, 1),
            (997, 16, 16),
            (997, 996, 996),
            (997, 997, 997),
            (4096, 4096, 8),
            (4096, 4096, 4096),
        ] {
            assert!(key_span <= n_levels.max(1), "keys must stay below n_levels");
            let key: Vec<u32> = (0..n_obs)
                .map(|_| (next() % key_span as u64) as u32)
                .collect();
            let mut expected: Vec<u32> = (0..n_obs as u32).collect();
            expected.sort_by_key(|&i| key[i as usize]);
            assert_eq!(
                stable_argsort(&key, n_levels),
                expected,
                "n_obs={n_obs} n_levels={n_levels}"
            );
        }
    }

    #[test]
    fn build_compacts_large_integer_label() {
        let design = Design::from_levels_for_test(vec![vec![u32::MAX]]);

        assert_eq!(design.n_dofs, 1);
        assert_eq!(design.terms[0].levels(), &[0]);
        assert_eq!(design.terms[0].encoding.label(0), Some(u32::MAX));
    }

    #[test]
    fn integer_factor_encoding_round_trips() {
        let encoding = FactorEncoding::integer(vec![10, 100, 500]);

        assert_eq!(encoding.n_levels(), 3);
        assert_eq!(encoding.position(10), Some(0));
        assert_eq!(encoding.position(100), Some(1));
        assert_eq!(encoding.position(500), Some(2));
        assert_eq!(encoding.position(99), None);

        assert_eq!(encoding.label(0), Some(10));
        assert_eq!(encoding.label(1), Some(100));
        assert_eq!(encoding.label(2), Some(500));
        assert_eq!(encoding.label(3), None);
    }

    #[test]
    fn encode_labels_preserves_identity_encoding() {
        let EncodedFactor {
            encoding,
            levels,
            sorted,
        } = FactorEncoding::encode_labels(Cow::Borrowed(&[2, 0, 1, 2]));

        assert_eq!(encoding, FactorEncoding::identity(3));
        assert!(matches!(levels, Cow::Borrowed([2, 0, 1, 2])));
        assert!(!sorted);
    }

    #[test]
    fn encode_labels_compacts_bounded_gappy_labels() {
        // Range width = 3 and n_obs = 3, so this exercises the presence-table path.
        let EncodedFactor {
            encoding,
            levels,
            sorted,
        } = FactorEncoding::encode_labels(Cow::Borrowed(&[2, 0, 2]));

        assert_eq!(encoding, FactorEncoding::integer(vec![0, 2]));
        assert_eq!(*levels, [1, 0, 1]);
        assert!(!sorted);
    }

    #[test]
    fn encode_labels_compacts_shifted_bounded_range() {
        let EncodedFactor {
            encoding,
            levels,
            sorted,
        } = FactorEncoding::encode_labels(Cow::Borrowed(&[1_000_000, 1_000_001, 1_000_002]));

        assert_eq!(
            encoding,
            FactorEncoding::integer(vec![1_000_000, 1_000_001, 1_000_002])
        );
        assert_eq!(*levels, [0, 1, 2]);
        assert!(sorted);
    }

    #[test]
    fn encode_labels_compacts_large_span_without_span_allocation() {
        let EncodedFactor {
            encoding,
            levels,
            sorted,
        } = FactorEncoding::encode_labels(Cow::Borrowed(&[u32::MAX, 7, u32::MAX]));

        assert_eq!(encoding, FactorEncoding::integer(vec![7, u32::MAX]));
        assert_eq!(*levels, [1, 0, 1]);
        assert!(!sorted);
    }

    #[test]
    fn new_sorts_unsorted_dominant() {
        // Factor 0 (3 levels) dominates and is unsorted; factor 1 starts sorted.
        let design = Design::from_levels_for_test(vec![vec![2, 0, 1, 0], vec![0, 0, 1, 1]]);

        // Stable argsort of [2,0,1,0] → original indices [1,3,2,0].
        assert_eq!(design.obs_perm.as_deref(), Some(&[1u32, 3, 2, 0][..]));
        assert!(design.terms[0].sorted());
        // Factor 1's permuted column [0,1,1,0] is no longer non-decreasing.
        assert!(!design.terms[1].sorted());

        assert_eq!(design.terms[0].levels(), [0, 0, 1, 2]);
        assert_eq!(design.terms[1].levels(), [0, 1, 1, 0]);
    }

    #[test]
    fn rescan_marks_nested_factor_sorted_after_permutation() {
        // Factor 1 is nested in dominant factor 0, so the rescan must detect it stays sorted.
        let col0 = vec![3u32, 0, 2, 1];
        let col1: Vec<u32> = col0.iter().map(|&v| v / 2).collect();
        let design = Design::from_levels_for_test(vec![col0, col1]);
        assert!(design.obs_perm.is_some());
        assert!(design.terms[0].sorted());
        assert!(design.terms[1].sorted());
    }

    #[test]
    fn new_keeps_sorted_input() {
        let design = Design::from_levels_for_test(vec![vec![0, 0, 1, 2], vec![1, 0, 1, 0]]);
        assert!(design.obs_perm.is_none());
        assert!(design.terms[0].sorted());
        assert!(!design.terms[1].sorted());
    }

    #[test]
    fn clone_shares_sorted_row_storage() {
        // The locality sort owns both levels and slopes; a solver's clone must not copy them.
        let (f, z) = ([2u32, 0, 1, 0], [1.0, 2.0, 3.0, 4.0]);
        let design = Design::new(vec![Effect::new(&f, true, [&z[..]]).unwrap()]).unwrap();
        assert!(design.obs_perm.is_some());
        let clone = design.clone();
        assert!(Arc::ptr_eq(&clone.terms, &design.terms));
    }

    #[test]
    fn new_lays_out_slope_terms_term_major() {
        // Sorted levels keep the locality sort a no-op, so columns stay in caller order.
        let f0 = [0u32, 0, 1, 1];
        let f1 = [0u32, 2, 1, 0];
        let z0 = [1.0, 2.0, 3.0, 4.0];
        let z1 = [5.0, 6.0, 7.0, 8.0];
        let effects = vec![
            Effect::new(&f0, true, [&z0[..], &z1[..]]).unwrap(),
            Effect::new(&f1, true, []).unwrap(),
            Effect::new(&f0, false, [&z1[..]]).unwrap(),
        ];
        let design = Design::new(effects).unwrap();

        // term 0: [intercept, z0, z1] over 2 levels; term 1: intercept over 3; term 2: slope.
        let layout = |t: &Term| (t.offset, t.intercept, t.n_columns(), t.n_dofs());
        assert_eq!(layout(&design.terms[0]), (0, true, 3, 6));
        assert_eq!(layout(&design.terms[1]), (6, true, 1, 3));
        assert_eq!(layout(&design.terms[2]), (9, false, 1, 2));
        assert_eq!(design.n_dofs, 11);

        // Each term's slopes are the effect's own, in effect order.
        let slope = |term, column| design.raw_slope(Channel { term, column });
        assert_eq!(slope(0, 1), Some(&z0[..]));
        assert_eq!(slope(0, 2), Some(&z1[..]));
        assert_eq!(slope(1, 0), None);
        assert_eq!(slope(2, 0), Some(&z1[..]));
    }

    #[test]
    fn locality_sort_keeps_levels_and_slopes_row_aligned() {
        // Dominant factor [2,0,1,0] argsorts to caller positions [1,3,2,0].
        let (f0, f1) = ([2u32, 0, 1, 0], [0u32, 1, 1, 0]);
        let (z0, z1) = ([10.0, 20.0, 30.0, 40.0], [1.0, 2.0, 3.0, 4.0]);
        let design = Design::new(vec![
            Effect::new(&f0, true, [&z0[..]]).unwrap(),
            Effect::new(&f1, false, [&z1[..]]).unwrap(),
        ])
        .unwrap();

        assert_eq!(design.obs_perm.as_deref(), Some(&[1, 3, 2, 0][..]));
        assert_eq!(design.terms[0].levels(), &[0, 0, 1, 2]);
        assert_eq!(design.terms[1].levels(), &[1, 0, 1, 0]);
        let slope = |term: usize| {
            let column = design.terms[term].slope_column(0);
            design.raw_slope(Channel { term, column }).unwrap()
        };
        assert_eq!(slope(0), &[20.0, 40.0, 30.0, 10.0]);
        assert_eq!(slope(1), &[2.0, 4.0, 3.0, 1.0]);
    }
}
