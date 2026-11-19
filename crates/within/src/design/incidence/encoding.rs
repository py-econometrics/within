//! Caller factor labels ⇄ compact internal level positions.

use rayon::prelude::*;
use rustc_hash::FxBuildHasher;
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::Ordering::Relaxed;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize};
use std::sync::Arc;

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
    pub(in crate::design) fn encode_labels(labels: Cow<'_, [u32]>) -> EncodedFactor<'_> {
        if labels.is_empty() {
            return EncodedFactor {
                encoding: Self::identity(0),
                levels: labels,
            };
        }
        match identify_label_layout(&labels) {
            LabelLayout::Contiguous { min, max } => EncodedFactor::contiguous(labels, min, max),
            LabelLayout::SortedWithGaps => EncodedFactor::sorted_gaps(&labels),
            LabelLayout::UnsortedDense { distinct_labels } => {
                EncodedFactor::dense(&labels, distinct_labels)
            }
            LabelLayout::UnsortedSparse => EncodedFactor::sparse(&labels),
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

/// A factor's level encoding and level column (the input itself if already positions).
pub(in crate::design) struct EncodedFactor<'a> {
    pub(in crate::design) encoding: FactorEncoding,
    pub(in crate::design) levels: Cow<'a, [u32]>,
}

impl<'a> EncodedFactor<'a> {
    /// Every value in `min..=max` occurs, so a label's internal position is `label - min`.
    fn contiguous(labels: Cow<'a, [u32]>, min: u32, max: u32) -> Self {
        if min == 0 {
            return Self {
                encoding: FactorEncoding::identity(max as usize + 1),
                levels: labels,
            };
        }
        Self {
            encoding: FactorEncoding::integer((min..=max).into_par_iter().collect()),
            levels: Cow::Owned(labels.par_iter().map(|&label| label - min).collect()),
        }
    }
}

impl EncodedFactor<'static> {
    /// Compact sorted labels with gaps by numbering runs, regardless of range width.
    fn sorted_gaps(labels: &[u32]) -> Self {
        let counts: Vec<usize> = labels
            .par_chunks(LABEL_CHUNK)
            .enumerate()
            .map(|(chunk_idx, chunk)| {
                let start = chunk_idx * LABEL_CHUNK;
                // A run crossing a chunk boundary is counted only where it starts.
                usize::from(start == 0 || chunk[0] != labels[start - 1])
                    + chunk.windows(2).filter(|pair| pair[0] != pair[1]).count()
            })
            .collect();
        let mut caller_labels = vec![0u32; counts.iter().sum()];
        let mut positions = vec![0u32; labels.len()];
        let mut remaining_labels = caller_labels.as_mut_slice();
        let mut next_level = 0;
        // Prefix the run counts and give each task disjoint output slices.
        let chunks: Vec<_> = positions
            .chunks_mut(LABEL_CHUNK)
            .zip(counts)
            .enumerate()
            .map(|(chunk_idx, (positions, count))| {
                let (distinct, rest) = std::mem::take(&mut remaining_labels).split_at_mut(count);
                remaining_labels = rest;
                let first_new_level = next_level;
                next_level += count;
                (chunk_idx, positions, distinct, first_new_level)
            })
            .collect();
        chunks
            .into_par_iter()
            .for_each(|(chunk_idx, positions, distinct, mut next_level)| {
                let start = chunk_idx * LABEL_CHUNK;
                let chunk = &labels[start..start + positions.len()];
                let mut previous = start.checked_sub(1).map(|i| labels[i]);
                let mut distinct = distinct.iter_mut();
                for (&label, position) in chunk.iter().zip(positions) {
                    if previous != Some(label) {
                        *distinct.next().expect("each run was counted") = label;
                        next_level += 1;
                        previous = Some(label);
                    }
                    // A continuing run retains the previous chunk's final ID.
                    *position = (next_level - 1) as u32;
                }
            });

        Self {
            encoding: FactorEncoding::integer(caller_labels),
            levels: Cow::Owned(positions),
        }
    }

    /// Encode unsorted labels using their indices in `distinct_labels`,
    /// which must hold the column's distinct labels in ascending order.
    fn dense(labels: &[u32], distinct_labels: Vec<u32>) -> Self {
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
        }
    }

    /// Encode unsorted labels: each rayon task numbers them in first-seen order,
    /// one hash per label; only distinct labels are sorted, and their ranks
    /// replace the task-local numbers through a small array per task.
    fn sparse(labels: &[u32]) -> Self {
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
        }
    }
}

const LABEL_CHUNK: usize = 65_536;

/// Label layout used to select an encoding strategy.
enum LabelLayout {
    /// Every value in `min..=max` occurs.
    Contiguous { min: u32, max: u32 },
    /// Non-decreasing labels with missing values between their minimum and maximum.
    SortedWithGaps,
    /// Unsorted labels with gaps and a range no wider than the observation count.
    /// `distinct_labels` contains the observed labels in non-decreasing order.
    UnsortedDense { distinct_labels: Vec<u32> },
    /// Unsorted labels whose range exceeds the observation count.
    UnsortedSparse,
}

/// Identifies the label layout in terms of ordering, contiguity and range width.
///
/// Requires nonempty `labels`.
fn identify_label_layout(labels: &[u32]) -> LabelLayout {
    match is_non_decreasing_with_gaps(labels) {
        // Non-decreasing and contiguous.
        Some(false) => LabelLayout::Contiguous {
            min: labels[0],
            max: labels[labels.len() - 1],
        },
        // Non-decreasing with gaps.
        Some(true) => LabelLayout::SortedWithGaps,
        // Not non-decreasing: distinguish sparse gaps, dense gaps, and contiguous.
        None => {
            let (min, max) = labels.par_iter().map(|&label| (label, label)).reduce(
                || (u32::MAX, u32::MIN),
                |(lo, hi), (a, b)| (lo.min(a), hi.max(b)),
            );
            if (max - min) as usize >= labels.len() {
                // Unsorted labels with sparse gaps
                LabelLayout::UnsortedSparse
            } else if let Some(distinct_labels) = get_distinct_labels(labels, min, max) {
                // Unsorted labels with dense gaps
                LabelLayout::UnsortedDense { distinct_labels }
            } else {
                // Unsorted contiguous labels
                LabelLayout::Contiguous { min, max }
            }
        }
    }
}

/// Checks if `labels` are in non-decreasing order and whether there are gaps.
///
/// Returns `None` if not non-decreasing, `Some(true)` if non-decreasing with gaps,
/// and `Some(false)` if non-decreasing without gaps.
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

/// Finds the distinct values in `labels` and stops as soon as every value in `min..=max`
/// has been seen. Returns `None` if all values in `min..=max` occur and otherwise the
/// distinct values (`Some`).
///
/// Every label must lie in `min..=max`, and the range must be no wider than the column.
fn get_distinct_labels(labels: &[u32], min: u32, max: u32) -> Option<Vec<u32>> {
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
    Some(
        present
            .into_par_iter()
            .enumerate()
            .filter_map(|(offset, slot)| slot.into_inner().then_some(min + offset as u32))
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let EncodedFactor { encoding, levels } =
            FactorEncoding::encode_labels(Cow::Borrowed(&[2, 0, 1, 2]));

        assert_eq!(encoding, FactorEncoding::identity(3));
        assert!(matches!(levels, Cow::Borrowed([2, 0, 1, 2])));
    }

    #[test]
    fn encode_labels_compacts_bounded_gappy_labels() {
        // Range width = 3 and n_obs = 3, so this exercises the presence-table path.
        let EncodedFactor { encoding, levels } =
            FactorEncoding::encode_labels(Cow::Borrowed(&[2, 0, 2]));

        assert_eq!(encoding, FactorEncoding::integer(vec![0, 2]));
        assert_eq!(*levels, [1, 0, 1]);
    }

    #[test]
    fn encode_labels_compacts_shifted_bounded_range() {
        let EncodedFactor { encoding, levels } =
            FactorEncoding::encode_labels(Cow::Borrowed(&[1_000_000, 1_000_001, 1_000_002]));

        assert_eq!(
            encoding,
            FactorEncoding::integer(vec![1_000_000, 1_000_001, 1_000_002])
        );
        assert_eq!(*levels, [0, 1, 2]);
    }

    #[test]
    fn encode_labels_compacts_large_span_without_span_allocation() {
        let EncodedFactor { encoding, levels } =
            FactorEncoding::encode_labels(Cow::Borrowed(&[u32::MAX, 7, u32::MAX]));

        assert_eq!(encoding, FactorEncoding::integer(vec![7, u32::MAX]));
        assert_eq!(*levels, [1, 0, 1]);
    }
}
