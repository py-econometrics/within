//! Domain layer: [`Design`] (design-matrix metadata) and factor-pair [`Subdomain`] construction.

pub(crate) mod collinearity;
pub(crate) mod cross_tab;
mod effect;
pub(crate) mod factor_pairs;
pub(crate) mod level_moments;

pub(crate) use cross_tab::{BlockDiagonals, CrossTab};

pub use effect::Effect;

pub(crate) use factor_pairs::{
    build_local_domains, CoordinateMap, Grounding, LocalComponent, LocalDomain, MatrixForm,
    SddmMatrix,
};

use crate::channel::Channel;
use crate::observation::ObservationFrame;
use crate::BuildError;
use ndarray::{ArrayView2, Axis};
use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

/// A slice that is guaranteed non-empty by construction.
#[repr(transparent)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NonEmpty<T>(Box<[T]>);

impl<T> NonEmpty<T> {
    /// `None` if `items` is empty.
    pub fn new(items: impl Into<Box<[T]>>) -> Option<Self> {
        let items = items.into();
        (!items.is_empty()).then(|| Self(items))
    }

    /// A single-element run.
    pub fn of(item: T) -> Self {
        Self(Box::new([item]))
    }

    /// Structure-preserving map; non-emptiness is carried over.
    pub fn map<U>(&self, f: impl FnMut(&T) -> U) -> NonEmpty<U> {
        NonEmpty(self.0.iter().map(f).collect())
    }
}

impl<T> std::ops::Deref for NonEmpty<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        &self.0
    }
}

/// A coefficient column's loading: the intercept's implicit `1.0`, or a covariate as `T`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Loading<T> {
    /// The intercept column; loading value `1.0` at every observation.
    Constant,
    /// A slope column.
    Covariate(T),
}

impl<T> Loading<T> {
    /// The covariate payload; `None` for the constant column.
    pub fn covariate(&self) -> Option<&T> {
        match self {
            Self::Constant => None,
            Self::Covariate(t) => Some(t),
        }
    }

    /// Replace the covariate payload, preserving which variant this is.
    pub fn map<U>(&self, f: impl FnOnce(&T) -> U) -> Loading<U> {
        match self {
            Self::Constant => Loading::Constant,
            Self::Covariate(t) => Loading::Covariate(f(t)),
        }
    }
}

/// Mapping between caller-visible factor labels and compact numerical positions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum FactorEncoding {
    /// Caller label `k` is internal position `k`.
    Identity { n_levels: usize },
    /// Arbitrary integer caller labels, ordered by internal position.
    Integer { labels: Arc<[u32]> },
}

struct EncodedFactor {
    encoding: FactorEncoding,
    positions: Option<Vec<u32>>,
    sorted: bool,
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

    fn encode_labels(labels: &[u32]) -> EncodedFactor {
        let Some((&first, remaining)) = labels.split_first() else {
            return EncodedFactor {
                encoding: Self::identity(0),
                positions: None,
                sorted: true,
            };
        };

        let mut min = first;
        let mut max = first;
        let mut previous = first;
        let mut sorted = true;
        for &label in remaining {
            min = min.min(label);
            max = max.max(label);
            sorted &= label >= previous;
            previous = label;
        }

        let range_width = u64::from(max) - u64::from(min) + 1;
        let presence_by_label = usize::try_from(range_width)
            .ok()
            .filter(|&width| width <= labels.len())
            .map(|width| {
                let mut present = vec![false; width];
                for &label in labels {
                    present[(label - min) as usize] = true;
                }
                present
            });

        match presence_by_label {
            // Path 1: labels already form the zero-based identity range.
            Some(present) if min == 0 && present.iter().all(|&is_present| is_present) => {
                EncodedFactor {
                    encoding: Self::identity(present.len()),
                    positions: None,
                    sorted,
                }
            }
            // Path 2: the observed label range is bounded by the observation count.
            Some(present) => {
                let range_width = present.len();
                let caller_labels: Vec<u32> = present
                    .iter()
                    .enumerate()
                    .filter_map(|(offset, &present)| present.then_some(min + offset as u32))
                    .collect();

                let mut position_by_label = vec![0u32; range_width];
                for (position, &label) in caller_labels.iter().enumerate() {
                    position_by_label[(label - min) as usize] = position as u32;
                }

                let positions = labels
                    .iter()
                    .map(|&label| position_by_label[(label - min) as usize])
                    .collect();
                EncodedFactor {
                    encoding: Self::integer(caller_labels),
                    positions: Some(positions),
                    sorted,
                }
            }
            // Path 3: the observed label range is too wide for an indexed table.
            None => {
                // Collect distinct caller labels
                let mut position_by_label = HashMap::<u32, u32>::new();
                for &label in labels {
                    position_by_label.entry(label).or_default();
                }
                // Internal positions follow ascending caller-label order
                let mut caller_labels: Vec<u32> = position_by_label.keys().copied().collect();
                caller_labels.sort_unstable();
                // Populate the caller-label to internal-position map
                for (position, &label) in caller_labels.iter().enumerate() {
                    let position =
                        u32::try_from(position).expect("an internal label position fits in u32");

                    *position_by_label
                        .get_mut(&label)
                        .expect("label was collected from this map") = position;
                }

                let positions = labels
                    .iter()
                    .map(|label| {
                        *position_by_label
                            .get(label)
                            .expect("every input label was inserted")
                    })
                    .collect();

                EncodedFactor {
                    encoding: Self::integer(caller_labels),
                    positions: Some(positions),
                    sorted,
                }
            }
        }
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

/// Per-term metadata; coefficient `c` of `level` lives at `offset + c · n_levels + level`.
#[derive(Debug, Clone)]
pub(crate) struct TermMeta {
    pub(crate) encoding: FactorEncoding,
    pub offset: usize,
    /// Non-decreasing in the design's internal row order (fixed at construction).
    pub sorted: bool,
    /// Coefficient columns in layout order; `Covariate` indexes the frame's continuous columns.
    pub columns: NonEmpty<Loading<u32>>,
}

impl TermMeta {
    pub fn n_levels(&self) -> usize {
        self.encoding.n_levels()
    }

    pub fn n_columns(&self) -> usize {
        self.columns.len()
    }

    pub fn n_dofs(&self) -> usize {
        self.n_columns() * self.n_levels()
    }

    /// Global DOF base of coefficient column `column`.
    pub fn column_base(&self, column: usize) -> usize {
        self.offset + column * self.n_levels()
    }
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

/// Fixed-effects design: observation columns plus coefficient-space layout.
#[derive(Clone, Debug)]
pub struct Design<'a> {
    /// Columns in internal row order (caller's, or an owned locality-sorted copy).
    frame: ObservationFrame<'a>,
    pub(crate) terms: Vec<TermMeta>,
    pub(crate) n_obs: usize,
    pub(crate) n_dofs: usize,
    /// `obs_perm[k]` = caller's original index of the observation at internal position `k`.
    pub(crate) obs_perm: Option<Vec<u32>>,
}

impl<'a> Design<'a> {
    /// Lower effect terms into a design, laid out term-major (`offset[t] + c · L_t + level`).
    pub fn new(effects: impl IntoIterator<Item = Effect<'a>>) -> Result<Self, BuildError> {
        let mut categorical: Vec<Cow<'a, [u32]>> = Vec::new();
        let mut continuous: Vec<Cow<'a, [f64]>> = Vec::new();
        let mut structure: Vec<NonEmpty<Loading<u32>>> = Vec::new();
        for effect in effects {
            structure.push(effect.columns().map(|column| {
                column.map(|&z| {
                    continuous.push(Cow::Borrowed(z));
                    (continuous.len() - 1) as u32
                })
            }));
            categorical.push(Cow::Borrowed(effect.levels()));
        }
        let frame = ObservationFrame::new(categorical, continuous)?;
        Self::build(frame, structure, true)
    }

    /// Intercept-only factors; compacts observed labels and locality-sorts an unsorted dominant factor.
    pub fn from_frame(frame: ObservationFrame<'a>) -> Result<Self, BuildError> {
        let structure = vec![NonEmpty::of(Loading::Constant); frame.n_factors()];
        Self::build(frame, structure, true)
    }

    /// Build an intercept-only design from an observation-major categories matrix.
    pub fn from_categories(categories: ArrayView2<'a, u32>) -> Result<Self, BuildError> {
        // Gather strided (C-order) columns once so every downstream read is contiguous.
        let categorical = (0..categories.ncols())
            .map(|factor| {
                let column = categories.index_axis_move(Axis(1), factor);
                match column.to_slice() {
                    Some(values) => Cow::Borrowed(values),
                    None => Cow::Owned(column.to_vec()),
                }
            })
            .collect();
        Self::from_frame(ObservationFrame::new(categorical, Vec::new())?)
    }

    /// [`from_frame`](Self::from_frame) without the locality sort — profiling escape hatch.
    #[doc(hidden)]
    pub fn from_frame_unsorted(frame: ObservationFrame<'a>) -> Result<Self, BuildError> {
        let structure = vec![NonEmpty::of(Loading::Constant); frame.n_factors()];
        Self::build(frame, structure, false)
    }

    /// `column_structure[term]` = that term's coefficient columns, aligned with the frame.
    fn build(
        mut frame: ObservationFrame<'a>,
        column_structure: Vec<NonEmpty<Loading<u32>>>,
        locality_sort: bool,
    ) -> Result<Self, BuildError> {
        if frame.n_obs() == 0 {
            return Err(BuildError::EmptyObservations);
        }
        debug_assert_eq!(column_structure.len(), frame.n_factors());

        let n_obs = frame.n_obs();
        let mut terms = Vec::with_capacity(frame.n_factors());
        let mut offset = 0;
        for (q, columns) in column_structure.into_iter().enumerate() {
            let EncodedFactor {
                encoding,
                positions,
                sorted,
            } = FactorEncoding::encode_labels(frame.level_column(q));
            if let Some(positions) = positions {
                frame.replace_level_column(q, positions);
            }
            let meta = TermMeta {
                encoding,
                offset,
                sorted,
                columns,
            };
            offset += meta.n_dofs();
            terms.push(meta);
        }

        // Rejected here rather than left to panic in `to_u32`.
        if u32::try_from(offset).is_err() {
            return Err(BuildError::DofSpaceExceedsU32 { n_dofs: offset });
        }

        // Sort by the term contributing the most DOFs so its gather/scatter runs sequentially.
        let dominant = (0..terms.len()).max_by_key(|&q| terms[q].n_dofs());
        let (frame, obs_perm) = match dominant {
            Some(d) if locality_sort && !terms[d].sorted && u32::try_from(n_obs).is_ok() => {
                let perm = stable_argsort(frame.level_column(d), terms[d].n_levels());
                let sorted_frame = frame.permuted(&perm);
                // Factors nested in the dominant one come out sorted, keeping coalesced scatter.
                for (q, meta) in terms.iter_mut().enumerate() {
                    meta.sorted = sorted_frame.level_column(q).is_sorted();
                }
                (sorted_frame, Some(perm))
            }
            _ => (frame, None),
        };

        Ok(Design {
            frame,
            terms,
            n_obs,
            n_dofs: offset,
            obs_perm,
        })
    }

    /// Convert the frame's columns to owned, dropping ties to caller buffers.
    pub fn into_owned(self) -> Design<'static> {
        Design {
            frame: self.frame.into_owned(),
            terms: self.terms,
            n_obs: self.n_obs,
            n_dofs: self.n_dofs,
            obs_perm: self.obs_perm,
        }
    }

    /// Validate that an optional weight slice matches this design's observation count.
    pub(crate) fn validate_weights(&self, weights: Option<&[f64]>) -> Result<(), BuildError> {
        if let Some(w) = weights {
            if w.len() != self.n_obs {
                return Err(BuildError::WeightCountMismatch {
                    expected: self.n_obs,
                    got: w.len(),
                });
            }
            // `wi >= 0.0` already rejects NaN; `is_finite` additionally rejects `+∞`.
            if let Some((index, &value)) = w
                .iter()
                .enumerate()
                .find(|&(_, &wi)| !(wi >= 0.0 && wi.is_finite()))
            {
                return Err(BuildError::InvalidWeight { index, value });
            }
        }
        Ok(())
    }

    /// Caller order → internal order: `out[k] = v[obs_perm[k]]`; borrows when unpermuted.
    pub(crate) fn permute_obs_in<'v>(&self, v: &'v [f64]) -> Cow<'v, [f64]> {
        debug_assert_eq!(v.len(), self.n_obs);
        match &self.obs_perm {
            None => Cow::Borrowed(v),
            Some(perm) => Cow::Owned(perm.iter().map(|&i| v[i as usize]).collect()),
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

    fn n_loading_columns(&self) -> usize {
        self.frame.n_loading_columns()
    }

    /// The term's coefficient columns in layout order.
    pub(crate) fn channels(&self, term: usize) -> impl Iterator<Item = Channel> + '_ {
        (0..self.terms[term].n_columns()).map(move |column| Channel { term, column })
    }

    /// How `channel` loads onto each observation.
    pub(crate) fn loading(&self, channel: Channel) -> Loading<u32> {
        self.terms[channel.term].columns[channel.column]
    }

    /// Level codes for one term, in internal observation order.
    pub(crate) fn level_column(&self, term: usize) -> &[u32] {
        self.frame.level_column(term)
    }

    /// Continuous loading column in internal observation order.
    pub(crate) fn loading_column(&self, column: usize) -> &[f64] {
        self.frame.loading_column(column)
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

pub(crate) struct LoadingOverrides(Vec<Option<Vec<f64>>>);

impl LoadingOverrides {
    pub(crate) fn new(design: &Design<'_>) -> Self {
        Self((0..design.n_loading_columns()).map(|_| None).collect())
    }

    pub(crate) fn replace(&mut self, design: &Design<'_>, column: usize, values: Vec<f64>) {
        debug_assert_eq!(values.len(), design.n_obs());
        self.0[column] = Some(values);
    }
}

/// Read-only solver-specific view of a design, including any transformed loadings.
pub(crate) struct SolverDesign<'a> {
    design: &'a Design<'a>,
    loading_overrides: Option<&'a LoadingOverrides>,
}

impl<'a> SolverDesign<'a> {
    #[cfg(test)]
    pub(crate) fn new(design: &'a Design<'a>) -> Self {
        Self {
            design,
            loading_overrides: None,
        }
    }

    pub(crate) fn with_loading_overrides(
        design: &'a Design<'a>,
        loading_overrides: &'a LoadingOverrides,
    ) -> Self {
        Self {
            design,
            loading_overrides: Some(loading_overrides),
        }
    }

    pub(crate) fn design(&self) -> &'a Design<'a> {
        self.design
    }

    pub(crate) fn loading_column(&self, column: usize) -> &[f64] {
        self.loading_overrides
            .and_then(|overrides| overrides.0[column].as_deref())
            .unwrap_or_else(|| self.design.loading_column(column))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observation::ObservationFrame;

    fn frame(categorical: Vec<Vec<u32>>, continuous: Vec<Vec<f64>>) -> ObservationFrame<'static> {
        ObservationFrame::new(
            categorical.into_iter().map(Into::into).collect(),
            continuous.into_iter().map(Into::into).collect(),
        )
        .unwrap()
    }

    impl Design<'static> {
        pub(crate) fn from_levels_for_test(columns: Vec<Vec<u32>>) -> Self {
            Design::from_frame(frame(columns, Vec::new())).expect("valid design")
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
        let design = Design::from_frame(frame(vec![vec![u32::MAX]], vec![])).unwrap();

        assert_eq!(design.n_dofs, 1);
        assert_eq!(design.level_column(0), &[0]);
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
            positions,
            sorted,
        } = FactorEncoding::encode_labels(&[2, 0, 1, 2]);

        assert_eq!(encoding, FactorEncoding::identity(3));
        assert!(positions.is_none());
        assert!(!sorted);
    }

    #[test]
    fn encode_labels_compacts_bounded_gappy_labels() {
        // Range width = 3 and n_obs = 3, so this exercises the presence-table path.
        let EncodedFactor {
            encoding,
            positions,
            sorted,
        } = FactorEncoding::encode_labels(&[2, 0, 2]);

        assert_eq!(encoding, FactorEncoding::integer(vec![0, 2]));
        assert_eq!(positions.as_deref(), Some(&[1u32, 0, 1][..]));
        assert!(!sorted);
    }

    #[test]
    fn encode_labels_compacts_shifted_bounded_range() {
        let EncodedFactor {
            encoding,
            positions,
            sorted,
        } = FactorEncoding::encode_labels(&[1_000_000, 1_000_001, 1_000_002]);

        assert_eq!(
            encoding,
            FactorEncoding::integer(vec![1_000_000, 1_000_001, 1_000_002])
        );
        assert_eq!(positions.as_deref(), Some(&[0u32, 1, 2][..]));
        assert!(sorted);
    }

    #[test]
    fn encode_labels_compacts_large_span_without_span_allocation() {
        let EncodedFactor {
            encoding,
            positions,
            sorted,
        } = FactorEncoding::encode_labels(&[u32::MAX, 7, u32::MAX]);

        assert_eq!(encoding, FactorEncoding::integer(vec![7, u32::MAX]));
        assert_eq!(positions.as_deref(), Some(&[1u32, 0, 1][..]));
        assert!(!sorted);
    }

    #[test]
    fn validate_weights_checks_count_and_finiteness() {
        let design = Design::from_frame(frame(vec![vec![0, 0, 0, 0, 0]], vec![])).unwrap();
        assert!(design.validate_weights(None).is_ok());
        assert!(design
            .validate_weights(Some(&[1.0, 2.0, 3.0, 4.0, 5.0]))
            .is_ok());
        // Zero weights are valid (an excluded observation).
        assert!(design
            .validate_weights(Some(&[0.0, 1.0, 2.0, 3.0, 4.0]))
            .is_ok());
        // Length mismatch.
        assert!(design.validate_weights(Some(&[1.0, 2.0])).is_err());
        // Negative / non-finite weights are rejected with the offending index.
        assert!(matches!(
            design.validate_weights(Some(&[1.0, -2.0, 3.0, 4.0, 5.0])),
            Err(BuildError::InvalidWeight { index: 1, .. })
        ));
        assert!(matches!(
            design.validate_weights(Some(&[1.0, 2.0, f64::NAN, 4.0, 5.0])),
            Err(BuildError::InvalidWeight { index: 2, .. })
        ));
        assert!(matches!(
            design.validate_weights(Some(&[1.0, 2.0, 3.0, f64::INFINITY, 5.0])),
            Err(BuildError::InvalidWeight { index: 3, .. })
        ));
    }

    #[test]
    fn from_frame_sorts_owned_unsorted_dominant() {
        // Factor 0 (3 levels) dominates and is unsorted; factor 1 starts sorted.
        let design =
            Design::from_frame(frame(vec![vec![2, 0, 1, 0], vec![0, 0, 1, 1]], vec![])).unwrap();

        // Stable argsort of [2,0,1,0] → original indices [1,3,2,0].
        assert_eq!(design.obs_perm.as_deref(), Some(&[1u32, 3, 2, 0][..]));
        assert!(design.terms[0].sorted);
        // Factor 1's permuted column [0,1,1,0] is no longer non-decreasing.
        assert!(!design.terms[1].sorted);

        assert_eq!(design.level_column(0), [0, 0, 1, 2]);
        assert_eq!(design.level_column(1), [0, 1, 1, 0]);
    }

    #[test]
    fn rescan_marks_nested_factor_sorted_after_permutation() {
        // Factor 1 is nested in dominant factor 0, so the rescan must detect it stays sorted.
        let col0 = vec![3u32, 0, 2, 1];
        let col1: Vec<u32> = col0.iter().map(|&v| v / 2).collect();
        let design = Design::from_frame(frame(vec![col0, col1], vec![])).unwrap();
        assert!(design.obs_perm.is_some());
        assert!(design.terms[0].sorted);
        assert!(design.terms[1].sorted);
    }

    #[test]
    fn from_frame_keeps_sorted_input() {
        let design =
            Design::from_frame(frame(vec![vec![0, 0, 1, 2], vec![1, 0, 1, 0]], vec![])).unwrap();
        assert!(design.obs_perm.is_none());
        assert!(design.terms[0].sorted);
        assert!(!design.terms[1].sorted);
    }

    #[test]
    fn continuous_column_stays_row_aligned_after_locality_sort() {
        let design = Design::from_frame(frame(
            vec![vec![2, 0, 1, 0]],
            vec![vec![10.0, 20.0, 30.0, 40.0]],
        ))
        .unwrap();

        let perm = design.obs_perm.as_ref().expect("permutation applied");
        assert_eq!(perm, &[1, 3, 2, 0]);
        assert_eq!(design.level_column(0), [0, 0, 1, 2]);
        assert_eq!(design.loading_column(0), [20.0, 40.0, 30.0, 10.0]);
    }

    #[test]
    fn new_lays_out_slope_terms_term_major() {
        // Sorted levels keep the locality sort a no-op, so frame columns stay in caller order.
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
        assert_eq!(design.terms[0].offset, 0);
        assert_eq!(design.terms[0].n_dofs(), 6);
        assert_eq!(design.terms[1].offset, 6);
        assert_eq!(design.terms[1].n_dofs(), 3);
        assert_eq!(design.terms[2].offset, 9);
        assert!(!matches!(design.terms[2].columns[0], Loading::Constant));
        assert_eq!(design.terms[2].n_dofs(), 2);
        assert_eq!(design.n_dofs, 11);

        // slope indices resolve to the effects' loading columns in the frame.
        assert_eq!(
            &*design.terms[0].columns,
            &[
                Loading::Constant,
                Loading::Covariate(0),
                Loading::Covariate(1)
            ]
        );
        assert_eq!(&*design.terms[2].columns, &[Loading::Covariate(2)]);
        assert_eq!(design.loading_column(0), &z0[..]);
        assert_eq!(design.loading_column(2), &z1[..]);
    }
}
