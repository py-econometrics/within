use std::collections::HashMap;

/// Max entries in a dense pair table (~32 MB at 8 bytes each).
pub(super) const DENSE_TABLE_MAX_ENTRIES: usize = 5_000_000;

enum PairStorage {
    Dense(Vec<f64>),
    Sparse(HashMap<usize, f64>),
}

/// Accumulates weighted counts for a factor-pair cross table.
pub(super) struct PairAccumulator {
    n_rows: usize,
    n_cols: usize,
    storage: PairStorage,
}

impl PairAccumulator {
    pub(super) fn new(n_rows: usize, n_cols: usize, n_unique: usize) -> Self {
        let table_size = n_rows * n_cols;
        let storage = if table_size <= DENSE_TABLE_MAX_ENTRIES {
            PairStorage::Dense(vec![0.0; table_size])
        } else {
            PairStorage::Sparse(HashMap::with_capacity(n_unique.min(table_size)))
        };
        Self {
            n_rows,
            n_cols,
            storage,
        }
    }

    #[inline]
    pub(super) fn add(&mut self, row: usize, col: usize, weight: f64) {
        let key = row * self.n_cols + col;
        match &mut self.storage {
            PairStorage::Dense(table) => {
                table[key] += weight;
            }
            PairStorage::Sparse(table) => {
                *table.entry(key).or_insert(0.0) += weight;
            }
        }
    }

    pub(super) fn for_each_nonzero(&self, mut f: impl FnMut(usize, usize, f64)) {
        match &self.storage {
            PairStorage::Dense(table) => {
                for row in 0..self.n_rows {
                    for col in 0..self.n_cols {
                        let v = table[row * self.n_cols + col];
                        if v > 0.0 {
                            f(row, col, v);
                        }
                    }
                }
            }
            PairStorage::Sparse(table) => {
                for (&key, &v) in table {
                    if v > 0.0 {
                        f(key / self.n_cols, key % self.n_cols, v);
                    }
                }
            }
        }
    }
}
