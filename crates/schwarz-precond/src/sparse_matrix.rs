/// Square sparse matrix in Compressed Sparse Row (CSR) format.
///
/// Used internally for explicit Gramian submatrices, Laplacian representations,
/// and BFS adjacency graphs. All methods are pure Rust (`Send + Sync`).
///
/// # CSR Invariants
///
/// A well-formed `SparseMatrix` satisfies:
/// - `indptr.len() == n + 1`
/// - `indptr[0] == 0` and `indptr` is non-decreasing
/// - `indices.len() == data.len() == indptr[n] as usize` (the number of non-zeros)
/// - All column indices in `indices` are in `0..n`
/// - Within each row, column indices are sorted ascending with no duplicates
#[derive(Clone)]
pub struct SparseMatrix {
    indptr: Vec<u32>,
    indices: Vec<u32>,
    data: Vec<f64>,
    n: usize,
}

impl SparseMatrix {
    /// Create a new `SparseMatrix` from raw CSR components.
    ///
    /// # Arguments
    ///
    /// * `indptr` — Row pointer array (length `n + 1`), non-decreasing with `indptr[0] == 0`.
    /// * `indices` — Column indices (length `nnz`), sorted ascending per row.
    /// * `data` — Non-zero values (length `nnz`), parallel to `indices`.
    /// * `n` — Matrix dimension (square: `n` rows x `n` columns).
    pub fn new(indptr: Vec<u32>, indices: Vec<u32>, data: Vec<f64>, n: usize) -> Self {
        Self {
            indptr,
            indices,
            data,
            n,
        }
    }

    /// Row pointer array (length `n + 1`).
    #[inline]
    pub fn indptr(&self) -> &[u32] {
        &self.indptr
    }

    /// Column indices (length `nnz`).
    #[inline]
    pub fn indices(&self) -> &[u32] {
        &self.indices
    }

    /// Non-zero values (length `nnz`), parallel to `indices`.
    #[inline]
    pub fn data(&self) -> &[f64] {
        &self.data
    }

    /// Matrix dimension (the matrix is always square).
    #[inline]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Sparse matrix-vector multiply: y += A @ x.
    ///
    /// Does NOT zero y first -- caller controls initialization.
    pub fn matvec_add(&self, x: &[f64], y: &mut [f64]) {
        debug_assert_eq!(x.len(), self.n);
        debug_assert_eq!(y.len(), self.n);
        for (row, yi) in y[..self.n].iter_mut().enumerate() {
            let start = self.indptr[row] as usize;
            let end = self.indptr[row + 1] as usize;
            let row_data = &self.data[start..end];
            let row_idx = &self.indices[start..end];
            let mut acc = 0.0;
            for (&val, &col) in row_data.iter().zip(row_idx) {
                acc += val * x[col as usize];
            }
            *yi += acc;
        }
    }

    /// Sparse matrix-vector multiply: y = A @ x (zeroes y first).
    pub fn matvec(&self, x: &[f64], y: &mut [f64]) {
        y.fill(0.0);
        self.matvec_add(x, y);
    }

    /// Extract diagonal of the matrix. Returns zero for rows with no diagonal entry.
    pub fn diagonal(&self) -> Vec<f64> {
        let mut diag = vec![0.0; self.n];
        for (row, di) in diag.iter_mut().enumerate() {
            let start = self.indptr[row] as usize;
            let end = self.indptr[row + 1] as usize;
            let row_idx = &self.indices[start..end];
            let row_data = &self.data[start..end];
            for (&col, &val) in row_idx.iter().zip(row_data) {
                if col as usize == row {
                    *di = val;
                    break;
                }
            }
        }
        diag
    }

    /// Number of rows (and columns -- the matrix is square).
    pub fn n_rows(&self) -> usize {
        self.n
    }

    /// Number of stored non-zero entries.
    pub fn nnz(&self) -> usize {
        self.data.len()
    }

    /// Validate CSR invariants in debug builds. No-op in release.
    ///
    /// Checks: `indptr` length and monotonicity, `indices`/`data` length consistency,
    /// column bounds, and per-row sorted-unique column indices.
    pub fn debug_assert_valid(&self) {
        if !cfg!(debug_assertions) {
            return;
        }
        assert_eq!(
            self.indptr.len(),
            self.n + 1,
            "indptr length {} != n+1 = {}",
            self.indptr.len(),
            self.n + 1
        );
        assert_eq!(self.indptr[0], 0, "indptr[0] must be 0");
        for i in 0..self.n {
            assert!(
                self.indptr[i] <= self.indptr[i + 1],
                "indptr not non-decreasing at row {i}"
            );
        }
        let nnz = self.indptr[self.n] as usize;
        assert_eq!(self.indices.len(), nnz, "indices length mismatch");
        assert_eq!(self.data.len(), nnz, "data length mismatch");
        for row in 0..self.n {
            let start = self.indptr[row] as usize;
            let end = self.indptr[row + 1] as usize;
            for idx in start..end {
                let col = self.indices[idx];
                assert!(
                    (col as usize) < self.n,
                    "column index {col} out of bounds for n={}",
                    self.n
                );
                if idx > start {
                    assert!(
                        self.indices[idx - 1] < col,
                        "column indices not sorted/unique in row {row}"
                    );
                }
            }
        }
    }

    /// Extract principal submatrix G[subset, subset] with global->local index remapping.
    ///
    /// Returns a new SparseMatrix of size `subset.len()` x `subset.len()`.
    /// `subset` must contain valid row/column indices into `self`.
    ///
    /// Adaptively picks a dense `Vec` or `HashMap` for the global-to-local map
    /// depending on the ratio of `max(subset)` to `subset.len()`.
    pub fn extract_submatrix(&self, subset: &[usize]) -> SparseMatrix {
        let m = subset.len();
        if m == 0 {
            return SparseMatrix {
                indptr: vec![0],
                indices: Vec::new(),
                data: Vec::new(),
                n: 0,
            };
        }

        let max_idx = subset.iter().copied().max().unwrap_or(0);

        let mut new_indptr = Vec::with_capacity(m + 1);
        let mut new_indices = Vec::new();
        let mut new_data = Vec::new();
        new_indptr.push(0u32);

        if max_idx + 1 > 4 * m {
            // Sparse subset relative to index range: use HashMap.
            let global_to_local: std::collections::HashMap<usize, usize> = subset
                .iter()
                .enumerate()
                .map(|(local, &global)| (global, local))
                .collect();

            let collect_row =
                |global_row: usize, new_indices: &mut Vec<u32>, new_data: &mut Vec<f64>| {
                    let start = self.indptr[global_row] as usize;
                    let end = self.indptr[global_row + 1] as usize;
                    for idx in start..end {
                        let global_col = self.indices[idx] as usize;
                        if let Some(&local_col) = global_to_local.get(&global_col) {
                            new_indices.push(local_col as u32);
                            new_data.push(self.data[idx]);
                        }
                    }
                };

            for &global_row in subset {
                collect_row(global_row, &mut new_indices, &mut new_data);
                new_indptr.push(new_indices.len() as u32);
            }
        } else {
            // Dense subset relative to index range: use Vec for O(1) lookup.
            let mut global_to_local = vec![usize::MAX; max_idx + 1];
            for (local, &global) in subset.iter().enumerate() {
                global_to_local[global] = local;
            }

            let collect_row =
                |global_row: usize, new_indices: &mut Vec<u32>, new_data: &mut Vec<f64>| {
                    let start = self.indptr[global_row] as usize;
                    let end = self.indptr[global_row + 1] as usize;
                    for idx in start..end {
                        let global_col = self.indices[idx] as usize;
                        if global_col <= max_idx {
                            let local_col = global_to_local[global_col];
                            if local_col != usize::MAX {
                                new_indices.push(local_col as u32);
                                new_data.push(self.data[idx]);
                            }
                        }
                    }
                };

            for &global_row in subset {
                collect_row(global_row, &mut new_indices, &mut new_data);
                new_indptr.push(new_indices.len() as u32);
            }
        }

        SparseMatrix {
            indptr: new_indptr,
            indices: new_indices,
            data: new_data,
            n: m,
        }
    }
}

// ---------------------------------------------------------------------------
// Sparse interop: faer
// ---------------------------------------------------------------------------

impl From<faer::sparse::SparseRowMatRef<'_, u32, f64>> for SparseMatrix {
    /// Convert a `faer` sparse row-major matrix view into an owned `SparseMatrix`.
    ///
    /// # Panics
    ///
    /// Panics if the matrix is not square.
    fn from(mat: faer::sparse::SparseRowMatRef<'_, u32, f64>) -> Self {
        assert_eq!(
            mat.nrows(),
            mat.ncols(),
            "SparseMatrix::from(faer): matrix must be square"
        );
        let symbolic = mat.symbolic();
        Self {
            indptr: symbolic.row_ptr().to_vec(),
            indices: symbolic.col_idx().to_vec(),
            data: mat.val().to_vec(),
            n: mat.nrows(),
        }
    }
}

// ---------------------------------------------------------------------------
// Connected components (Union-Find)
// ---------------------------------------------------------------------------

impl SparseMatrix {
    /// Detect connected components using Union-Find on CSR adjacency.
    ///
    /// Returns a list of components, each containing the node indices in that component.
    pub fn connected_components(&self) -> Vec<Vec<usize>> {
        let n = self.n;
        if n == 0 {
            return Vec::new();
        }

        let mut parent: Vec<usize> = (0..n).collect();
        let mut rank = vec![0u8; n];

        fn find(parent: &mut [usize], mut x: usize) -> usize {
            while parent[x] != x {
                parent[x] = parent[parent[x]];
                x = parent[x];
            }
            x
        }

        fn union(parent: &mut [usize], rank: &mut [u8], a: usize, b: usize) {
            let ra = find(parent, a);
            let rb = find(parent, b);
            if ra == rb {
                return;
            }
            if rank[ra] < rank[rb] {
                parent[ra] = rb;
            } else if rank[ra] > rank[rb] {
                parent[rb] = ra;
            } else {
                parent[rb] = ra;
                rank[ra] += 1;
            }
        }

        for row in 0..n {
            let start = self.indptr[row] as usize;
            let end = self.indptr[row + 1] as usize;
            for idx in start..end {
                let col = self.indices[idx] as usize;
                if col != row && self.data[idx].abs() > 0.0 {
                    union(&mut parent, &mut rank, row, col);
                }
            }
        }

        let mut component_map: Vec<Vec<usize>> = Vec::new();
        let mut root_to_idx = vec![usize::MAX; n];

        for i in 0..n {
            let root = find(&mut parent, i);
            if root_to_idx[root] == usize::MAX {
                root_to_idx[root] = component_map.len();
                component_map.push(Vec::new());
            }
            component_map[root_to_idx[root]].push(i);
        }

        component_map
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_3x3_laplacian() -> SparseMatrix {
        SparseMatrix {
            indptr: vec![0, 3, 5, 7],
            indices: vec![0, 1, 2, 0, 1, 0, 2],
            data: vec![2.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0],
            n: 3,
        }
    }

    // --- SparseMatrix basics ---

    #[test]
    fn test_extract_submatrix_full() {
        let m = make_3x3_laplacian();
        let sub = m.extract_submatrix(&[0, 1, 2]);
        assert_eq!(sub.n, 3);
        assert_eq!(sub.data.len(), m.data.len());
    }

    #[test]
    fn test_extract_submatrix_partial() {
        let m = make_3x3_laplacian();
        let sub = m.extract_submatrix(&[0, 2]);
        assert_eq!(sub.n, 2);
        assert_eq!(sub.indptr, vec![0, 2, 4]);
    }

    #[test]
    fn test_extract_submatrix_empty() {
        let m = make_3x3_laplacian();
        let sub = m.extract_submatrix(&[]);
        assert_eq!(sub.n, 0);
        assert_eq!(sub.indptr, vec![0]);
    }

    // --- Connected components ---

    #[test]
    fn test_connected_components_single() {
        let m = SparseMatrix {
            indptr: vec![0, 3, 5, 7],
            indices: vec![0, 1, 2, 0, 1, 0, 2],
            data: vec![2.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0],
            n: 3,
        };
        let comps = m.connected_components();
        assert_eq!(comps.len(), 1);
        assert_eq!(comps[0].len(), 3);
    }

    #[test]
    fn test_connected_components_disconnected() {
        let m = SparseMatrix {
            indptr: vec![0, 1, 2],
            indices: vec![0, 1],
            data: vec![1.0, 1.0],
            n: 2,
        };
        let comps = m.connected_components();
        assert_eq!(comps.len(), 2);
    }
}
