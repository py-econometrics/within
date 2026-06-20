//! Schur complement (a graph Laplacian on the kept factor levels) of an
//! eliminated factor block in a bipartite SDDM system.
//!
//! Two [`SchurComplement`] impls share block-selection ([`Elimination`]) and
//! emit a [`SchurLaplacian`]:
//! - [`ExactSchurComplement`] — row-workspace accumulation.
//! - [`ApproxSchurComplement`] — clique-tree sampling, keeping `S` sparse for
//!   high-degree eliminated levels.

use super::csr_matrix::CsrMatrix;
use rayon::prelude::*;

use super::elimination::{Edge, Elimination, SampledCliqueEmitter};
use crate::config::ApproxSchurConfig;
use crate::csr_block::CsrBlock;

/// Checked `usize -> u32` for CSR indptr/index values. A silent `as u32`
/// truncation above `u32::MAX` would corrupt the factorization with no
/// diagnostic; these are build-path invariants, so panic loudly instead.
#[inline]
fn to_u32(x: usize) -> u32 {
    u32::try_from(x).expect("CSR index exceeds u32::MAX")
}

pub(crate) struct SchurLaplacian;

impl SchurLaplacian {
    /// Build the Schur complement via row-workspace accumulation (exact path).
    ///
    /// Computes `S = D_keep − keep_to_elim · diag(inv_diag_elim) · elim_to_keep`
    /// directly, without materializing intermediate edges. Each keep-block row
    /// scatters into a dense workspace, then extracts non-zeros.
    fn from_elimination(elim: &Elimination) -> CsrMatrix {
        let n_keep = elim.n_keep;
        let inv_diag_elim = &elim.inv_diag_elim;
        let diag_keep = elim.diag_keep;
        let keep_to_elim = elim.keep_to_elim;
        let elim_to_keep = elim.elim_to_keep;

        // Per-row Schur complement accumulation, parallelized via map_init.
        // The (work, touched) pair is allocated once per rayon task and reused
        // across rows assigned to that task.
        let rows: Vec<(Vec<u32>, Vec<f64>)> = (0..n_keep)
            .into_par_iter()
            .map_init(
                || (vec![0.0f64; n_keep], Vec::new()),
                |(work, touched), i| {
                    Self::compute_schur_row_dense(
                        i,
                        diag_keep,
                        keep_to_elim,
                        elim_to_keep,
                        inv_diag_elim,
                        work,
                        touched,
                    );
                    let result = Self::extract_sparse_row(i, work, touched);
                    touched.clear();
                    result
                },
            )
            .collect();

        Self::assemble_schur_csr(rows, n_keep)
    }

    /// Build the anchored top-left Schur minor `(n_keep-1) x (n_keep-1)` in row-major.
    ///
    /// This is the matrix actually factored by dense anchored Cholesky, so building
    /// it directly avoids allocating a full `n_keep x n_keep` dense Schur matrix.
    pub(crate) fn anchored_minor_from_elimination(elim: &Elimination) -> Vec<f64> {
        let n_keep = elim.n_keep;
        if n_keep <= 1 {
            return Vec::new();
        }

        let m = n_keep - 1;
        let mut dense_minor = vec![0.0; m * m];

        // Start with the kept diagonal block on anchored rows/cols.
        for i in 0..m {
            dense_minor[i * m + i] = elim.diag_keep[i];
        }

        let inv_diag_elim = &elim.inv_diag_elim;
        let keep_to_elim = elim.keep_to_elim;
        let elim_to_keep = elim.elim_to_keep;

        // S_minor = D_keep_minor - keep_to_elim_minor * inv(D_elim) * elim_to_keep_minor
        for i in 0..m {
            let fwd_start = keep_to_elim.indptr[i] as usize;
            let fwd_end = keep_to_elim.indptr[i + 1] as usize;
            for fwd_idx in fwd_start..fwd_end {
                let k = keep_to_elim.indices[fwd_idx] as usize;
                let scale = keep_to_elim.data[fwd_idx] * inv_diag_elim[k];
                let bwd_start = elim_to_keep.indptr[k] as usize;
                let bwd_end = elim_to_keep.indptr[k + 1] as usize;
                for bwd_idx in bwd_start..bwd_end {
                    let j = elim_to_keep.indices[bwd_idx] as usize;
                    if j < m {
                        dense_minor[i * m + j] -= scale * elim_to_keep.data[bwd_idx];
                    }
                }
            }
        }

        dense_minor
    }

    /// Scatter the Schur row `i` into a dense workspace.
    ///
    /// Computes `work[j] = D_keep[i] δ_{ij} - Σ_k (keep_to_elim[i,k] / D_elim[k]) * elim_to_keep[k,j]`
    /// and records touched column indices.
    fn compute_schur_row_dense(
        i: usize,
        diag_keep: &[f64],
        keep_to_elim: &CsrBlock,
        elim_to_keep: &CsrBlock,
        inv_diag_elim: &[f64],
        work: &mut [f64],
        touched: &mut Vec<usize>,
    ) {
        work[i] = diag_keep[i];
        touched.push(i);

        let fwd_start = keep_to_elim.indptr[i] as usize;
        let fwd_end = keep_to_elim.indptr[i + 1] as usize;
        for fwd_idx in fwd_start..fwd_end {
            let k = keep_to_elim.indices[fwd_idx] as usize;
            let scale = keep_to_elim.data[fwd_idx] * inv_diag_elim[k];
            let bwd_start = elim_to_keep.indptr[k] as usize;
            let bwd_end = elim_to_keep.indptr[k + 1] as usize;
            for bwd_idx in bwd_start..bwd_end {
                let j = elim_to_keep.indices[bwd_idx] as usize;
                if work[j] == 0.0 && j != i {
                    touched.push(j);
                }
                work[j] -= scale * elim_to_keep.data[bwd_idx];
            }
        }
    }

    /// Extract non-zero entries from the dense workspace into sparse row arrays.
    ///
    /// Sorts touched columns, emits non-zero values (preserving the diagonal even
    /// if numerically zero for SDDM structure), and clears the workspace.
    fn extract_sparse_row(
        i: usize,
        work: &mut [f64],
        touched: &mut [usize],
    ) -> (Vec<u32>, Vec<f64>) {
        touched.sort_unstable();
        // `touched.len()` is a tight upper bound on the emitted non-zeros
        // (the diagonal plus every fill column), so size both buffers once.
        let mut row_indices = Vec::with_capacity(touched.len());
        let mut row_data = Vec::with_capacity(touched.len());
        for &j in touched.iter() {
            let v = work[j];
            if v != 0.0 || j == i {
                row_indices.push(to_u32(j));
                row_data.push(v);
            }
            work[j] = 0.0;
        }
        (row_indices, row_data)
    }

    /// Assemble a CSR matrix from per-row sparse results.
    fn assemble_schur_csr(rows: Vec<(Vec<u32>, Vec<f64>)>, n_keep: usize) -> CsrMatrix {
        let mut s_indptr = vec![0u32; n_keep + 1];
        // Pre-count total NNZ so the value/index buffers allocate exactly once.
        let total_nnz: usize = rows.iter().map(|(ri, _)| ri.len()).sum();
        let mut s_indices = Vec::with_capacity(total_nnz);
        let mut s_data = Vec::with_capacity(total_nnz);
        for (i, (ri, rd)) in rows.into_iter().enumerate() {
            s_indices.extend_from_slice(&ri);
            s_data.extend_from_slice(&rd);
            s_indptr[i + 1] = to_u32(s_indices.len());
        }
        CsrMatrix::new(s_indptr, s_indices, s_data, n_keep)
    }

    /// Build symmetric CSR Laplacian from sorted upper-triangular edges.
    ///
    /// Edges must be sorted by (lo, hi) with lo < hi. This lets us place
    /// lower-triangle, diagonal, and upper-triangle entries in correct column
    /// order without any per-row sorting.
    fn build_laplacian_csr(edges: &[Edge], n_keep: usize) -> CsrMatrix {
        debug_assert!(edges.iter().all(|&(lo, hi, _)| lo < hi));

        // Count lower/upper entries per row and accumulate diagonal weights.
        let mut lower_count = vec![0u32; n_keep];
        let mut upper_count = vec![0u32; n_keep];
        let mut diag = vec![0.0f64; n_keep];
        for &(lo, hi, w) in edges {
            upper_count[lo as usize] += 1; // row lo gets col hi (upper)
            lower_count[hi as usize] += 1; // row hi gets col lo (lower)
            diag[lo as usize] += w;
            diag[hi as usize] += w;
        }

        // Row layout: [lower entries | diagonal | upper entries]
        let mut offsets = vec![0u32; n_keep + 1];
        for i in 0..n_keep {
            offsets[i + 1] = offsets[i] + lower_count[i] + 1 + upper_count[i];
        }
        let total_nnz = offsets[n_keep] as usize;
        let mut indices = vec![0u32; total_nnz];
        let mut data = vec![0.0f64; total_nnz];

        // Place diagonals and initialize cursors.
        let mut lower_cursor: Vec<u32> = (0..n_keep).map(|i| offsets[i]).collect();
        let mut upper_cursor: Vec<u32> = (0..n_keep)
            .map(|i| offsets[i] + lower_count[i] + 1)
            .collect();
        for i in 0..n_keep {
            let pos = (offsets[i] + lower_count[i]) as usize;
            indices[pos] = to_u32(i);
            data[pos] = diag[i];
        }

        // Single pass: edges sorted by (lo, hi) guarantees both lower and
        // upper entries arrive in column-sorted order per row.
        for &(lo, hi, w) in edges {
            let lo_idx = lo as usize;
            let hi_idx = hi as usize;
            // Upper triangle: row lo, column hi
            let pos = upper_cursor[lo_idx] as usize;
            indices[pos] = hi;
            data[pos] = -w;
            upper_cursor[lo_idx] += 1;
            // Lower triangle: row hi, column lo
            let pos = lower_cursor[hi_idx] as usize;
            indices[pos] = lo;
            data[pos] = -w;
            lower_cursor[hi_idx] += 1;
        }

        CsrMatrix::new(offsets, indices, data, n_keep)
    }
}

// ===========================================================================
// Trait + implementations
// ===========================================================================

/// Strategy for computing the Schur complement from a prepared [`Elimination`].
pub(crate) trait SchurComplement {
    fn compute(&self, elim: &Elimination) -> CsrMatrix;
}

/// Exact Schur complement via block elimination.
pub(crate) struct ExactSchurComplement;

/// Approximate Schur complement via clique-tree sampling.
pub(crate) struct ApproxSchurComplement {
    config: ApproxSchurConfig,
}

impl ApproxSchurComplement {
    pub(crate) fn new(config: ApproxSchurConfig) -> Self {
        Self { config }
    }
}

impl SchurComplement for ExactSchurComplement {
    /// Compute the exact Schur complement using row-workspace accumulation.
    ///
    /// For the bipartite SDDM `[D_q, -C; -C^T, D_r]`, eliminates the larger
    /// block (exact since it's diagonal) to get a reduced Laplacian on the
    /// smaller block.
    fn compute(&self, elim: &Elimination) -> CsrMatrix {
        SchurLaplacian::from_elimination(elim)
    }
}

impl SchurComplement for ApproxSchurComplement {
    /// Compute an approximate Schur complement by sampling clique-trees.
    ///
    /// Each eliminated vertex produces at most deg-1 fill edges via the
    /// GKS 2023 Algorithm 5 clique-tree approximation.
    fn compute(&self, elim: &Elimination) -> CsrMatrix {
        let emitter = SampledCliqueEmitter::new(&self.config);
        let edges = elim.par_emit(&emitter);
        SchurLaplacian::build_laplacian_csr(&edges, elim.n_keep)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csr_block::CsrBlock;
    use crate::domain::{BlockDiagonals, CrossTab};

    fn make_cross_tab(
        c_dense: &[f64],
        n_q: usize,
        n_r: usize,
        diag_q: Vec<f64>,
        diag_r: Vec<f64>,
    ) -> (CrossTab, BlockDiagonals) {
        let c = CsrBlock::from_dense_table(c_dense, n_q, n_r);
        let ct = c.transpose();
        (
            CrossTab { c, ct },
            BlockDiagonals {
                q: diag_q,
                r: diag_r,
            },
        )
    }

    fn sparse_to_dense(matrix: &CsrMatrix) -> Vec<Vec<f64>> {
        let n = matrix.n();
        let mut dense = vec![vec![0.0; n]; n];
        for (i, row) in dense.iter_mut().enumerate().take(n) {
            let start = matrix.indptr()[i] as usize;
            let end = matrix.indptr()[i + 1] as usize;
            for idx in start..end {
                let j = matrix.indices()[idx] as usize;
                row[j] = matrix.data()[idx];
            }
        }
        dense
    }

    fn dense_exact_schur(
        c_dense: &[f64],
        n_q: usize,
        n_r: usize,
        diag_q: &[f64],
        diag_r: &[f64],
        eliminate_q: bool,
    ) -> Vec<Vec<f64>> {
        if eliminate_q {
            let mut s = vec![vec![0.0; n_r]; n_r];
            for i in 0..n_r {
                s[i][i] = diag_r[i];
            }
            for k in 0..n_q {
                let inv = if diag_q[k] > 0.0 {
                    1.0 / diag_q[k]
                } else {
                    0.0
                };
                for i in 0..n_r {
                    let cki = c_dense[k * n_r + i];
                    for j in 0..n_r {
                        let ckj = c_dense[k * n_r + j];
                        s[i][j] -= cki * inv * ckj;
                    }
                }
            }
            s
        } else {
            let mut s = vec![vec![0.0; n_q]; n_q];
            for i in 0..n_q {
                s[i][i] = diag_q[i];
            }
            for k in 0..n_r {
                let inv = if diag_r[k] > 0.0 {
                    1.0 / diag_r[k]
                } else {
                    0.0
                };
                for i in 0..n_q {
                    let cik = c_dense[i * n_r + k];
                    for j in 0..n_q {
                        let cjk = c_dense[j * n_r + k];
                        s[i][j] -= cik * inv * cjk;
                    }
                }
            }
            s
        }
    }

    fn assert_dense_close(lhs: &[Vec<f64>], rhs: &[Vec<f64>], tol: f64) {
        assert_eq!(lhs.len(), rhs.len(), "row count mismatch");
        for i in 0..lhs.len() {
            assert_eq!(lhs[i].len(), rhs[i].len(), "col count mismatch on row {i}");
            for j in 0..lhs[i].len() {
                assert!(
                    (lhs[i][j] - rhs[i][j]).abs() <= tol,
                    "mismatch at ({i}, {j}): lhs={}, rhs={}",
                    lhs[i][j],
                    rhs[i][j]
                );
            }
        }
    }

    #[test]
    fn exact_schur_matches_dense_reference_when_eliminating_q() {
        // C is 3x2, so q-block is eliminated (n_q >= n_r).
        let c_dense = vec![1.0, 2.0, 3.0, 0.0, 0.0, 4.0];
        let diag_q = vec![5.0, 6.0, 8.0];
        let diag_r = vec![7.0, 9.0];
        let (cross_tab, diagonals) = make_cross_tab(&c_dense, 3, 2, diag_q.clone(), diag_r.clone());
        let elim = Elimination::new(&cross_tab, &diagonals).unwrap();

        assert!(elim.eliminate_q);
        assert_eq!(elim.inv_diag_elim.len(), 3);
        for (&got, &expected) in elim
            .inv_diag_elim
            .iter()
            .zip([1.0 / 5.0, 1.0 / 6.0, 1.0 / 8.0].iter())
        {
            assert!((got - expected).abs() < 1e-12);
        }

        let matrix = ExactSchurComplement.compute(&elim);
        let expected = dense_exact_schur(&c_dense, 3, 2, &diag_q, &diag_r, true);
        let got = sparse_to_dense(&matrix);
        assert_dense_close(&got, &expected, 1e-12);
    }

    #[test]
    fn exact_schur_rejects_zero_eliminated_diagonal() {
        // C is 2x3, so r-block is eliminated (n_q < n_r). Last eliminated
        // diagonal is zero — Elimination::new should return SingularDiagonal.
        let c_dense = vec![2.0, 0.0, 1.0, 0.0, 3.0, 4.0];
        let diag_q = vec![8.0, 9.0];
        let diag_r = vec![5.0, 6.0, 0.0];
        let (cross_tab, diagonals) = make_cross_tab(&c_dense, 2, 3, diag_q, diag_r);

        let result = Elimination::new(&cross_tab, &diagonals);
        match result {
            Err(crate::BuildError::SingularDiagonal { index: 2, .. }) => {}
            Err(e) => panic!("expected SingularDiagonal at index 2, got: {e}"),
            Ok(_) => panic!("expected SingularDiagonal error, got Ok"),
        }
    }

    #[test]
    fn approximate_schur_is_seed_deterministic_and_laplacian_like() {
        // Degree-3 star in eliminated block gives nontrivial sampled edges.
        let c_dense = vec![1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let (cross_tab, diagonals) =
            make_cross_tab(&c_dense, 3, 3, vec![10.0, 4.0, 5.0], vec![2.0, 3.0, 4.0]);
        let elim_a = Elimination::new(&cross_tab, &diagonals).unwrap();
        let elim_b = Elimination::new(&cross_tab, &diagonals).unwrap();
        let approx = ApproxSchurComplement::new(crate::config::ApproxSchurConfig {
            seed: 12345,
            ..Default::default()
        });

        let a = approx.compute(&elim_a);
        let b = approx.compute(&elim_b);

        assert_eq!(elim_a.eliminate_q, elim_b.eliminate_q);
        assert_eq!(elim_a.inv_diag_elim, elim_b.inv_diag_elim);
        assert_eq!(a.indptr(), b.indptr());
        assert_eq!(a.indices(), b.indices());
        assert_eq!(a.data(), b.data());

        let dense = sparse_to_dense(&a);
        for (i, row) in dense.iter().enumerate() {
            let mut row_sum = 0.0;
            for (j, &value) in row.iter().enumerate() {
                row_sum += value;
                assert!(
                    (value - dense[j][i]).abs() <= 1e-12,
                    "matrix not symmetric at ({i}, {j})"
                );
                if i != j {
                    assert!(value <= 1e-12, "off-diagonal should be non-positive");
                }
            }
            assert!(row_sum.abs() <= 1e-10, "row {i} sum is not near zero");
            assert!(row[i] >= -1e-12, "diagonal should be non-negative");
        }
    }
}
