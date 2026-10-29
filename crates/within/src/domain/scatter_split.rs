//! How a term's scatter-add splits into fixed parallel pieces, chosen from its rows alone.

use std::ops::Range;

use super::Term;

/// Rows per level from which splitting by row beats splitting by level.
const FOLD_ROWS_PER_LEVEL: usize = 12;

/// The split of `z[l] = Σ_{level(i) = l} vᵢ` into pieces; the thread count only schedules them.
pub(crate) enum ScatterSplit {
    /// Each level sums its own rows: fixed row blocks in row order, added by a fixed binary tree.
    ByLevel(LevelRows),
    /// Fixed row blocks scatter into partial blocks that a fixed binary tree adds up.
    ByRow,
}

impl ScatterSplit {
    pub(crate) fn new(term: &Term<'_>) -> Self {
        let (levels, n_levels) = (term.levels(), term.n_levels());
        if !term.sorted() {
            // An unsorted term's row index stores u32 rows.
            if levels.len() >= FOLD_ROWS_PER_LEVEL * n_levels
                || u32::try_from(levels.len()).is_err()
            {
                return Self::ByRow;
            }
            let (offsets, rows) = super::stable_argsort(levels, n_levels);
            let rows = Some(rows.into_boxed_slice());
            return Self::ByLevel(LevelRows { offsets, rows });
        }
        let mut offsets = vec![0; n_levels + 1];
        // One store per run: a per-row increment chains each run through one counter.
        for run in levels.chunk_by(|a, b| a == b) {
            offsets[run[0] as usize + 1] = run.len();
        }
        for level in 0..n_levels {
            offsets[level + 1] += offsets[level];
        }
        Self::ByLevel(LevelRows {
            offsets,
            rows: None,
        })
    }
}

/// The rows of each level, ascending: the inverse of the row → level map.
pub(crate) struct LevelRows {
    /// Level `l` owns positions `offsets[l]..offsets[l + 1]`.
    offsets: Vec<usize>,
    /// Row at each position; `None` when the levels are sorted, so positions are rows.
    rows: Option<Box<[u32]>>,
}

impl LevelRows {
    /// Positions of `level`'s rows, for [`for_each_row`](Self::for_each_row).
    pub(crate) fn positions(&self, level: usize) -> Range<usize> {
        self.offsets[level]..self.offsets[level + 1]
    }

    /// Calls `f` on the row at each of `positions`, in order.
    #[inline(always)]
    pub(crate) fn for_each_row(&self, positions: Range<usize>, f: impl FnMut(usize)) {
        match &self.rows {
            None => positions.for_each(f),
            Some(rows) => rows[positions].iter().map(|&r| r as usize).for_each(f),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{Design, Effect};

    #[test]
    fn level_rows_list_each_levels_rows_ascending() {
        let ScatterSplit::ByLevel(rows) = split(&[2, 0, 2, 1, 0]) else {
            panic!()
        };
        assert_eq!(
            (rows.offsets, rows.rows.as_deref()),
            (vec![0, 2, 3, 5], Some(&[1, 4, 3, 0, 2][..]))
        );
        let ScatterSplit::ByLevel(sorted) = split(&[0, 0, 1, 1, 1]) else {
            panic!()
        };
        assert_eq!((sorted.offsets, sorted.rows), (vec![0, 2, 5], None));
    }

    fn split(levels: &[u32]) -> ScatterSplit {
        let slope = vec![1.0; levels.len()];
        let effect = Effect::new(levels, true, vec![&slope[..]; 4]).expect("valid effect");
        ScatterSplit::new(&Design::new_unsorted([effect]).expect("valid design").terms[0])
    }

    #[test]
    fn by_row_needs_unsorted_rows_and_enough_rows_per_level() {
        let levels: Vec<u32> = (0..48).map(|i| i % 2).collect();
        assert!(matches!(split(&levels), ScatterSplit::ByRow));
        let sorted: Vec<u32> = (0..48).map(|i| i / 24).collect();
        assert!(matches!(split(&sorted), ScatterSplit::ByLevel(_)));
        assert!(matches!(split(&levels[..23]), ScatterSplit::ByLevel(_)));
        assert!(matches!(split(&levels[..24]), ScatterSplit::ByRow));
    }
}
