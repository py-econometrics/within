//! Stable per-level observation membership, shared by every RHS operator.

use std::ops::Range;

#[derive(Clone, Debug)]
pub(crate) struct LevelMembership {
    offsets: Vec<usize>,
    // A sorted level column already has contiguous membership and needs no index bank.
    rows: Option<Vec<usize>>,
}

impl LevelMembership {
    pub(crate) fn new(levels: &[u32], n_levels: usize, sorted: bool) -> Self {
        let mut offsets = vec![0; n_levels + 1];
        for &level in levels {
            offsets[level as usize + 1] += 1;
        }
        for level in 0..n_levels {
            offsets[level + 1] += offsets[level];
        }
        let rows = if sorted {
            None
        } else {
            let mut rows = vec![0; levels.len()];
            let mut cursors = offsets[..n_levels].to_vec();
            for (row, &level) in levels.iter().enumerate() {
                let cursor = &mut cursors[level as usize];
                rows[*cursor] = row;
                *cursor += 1;
            }
            Some(rows)
        };
        Self { offsets, rows }
    }

    pub(crate) fn level_range(&self, level: usize) -> Range<usize> {
        self.offsets[level]..self.offsets[level + 1]
    }

    pub(crate) fn row(&self, position: usize) -> usize {
        self.rows.as_ref().map_or(position, |rows| rows[position])
    }
}
