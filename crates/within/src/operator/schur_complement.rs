//! Schur complement computation for bipartite SDDM systems.
//!
//! Provides a [`SchurComplement`] trait with two implementations:
//! - [`ExactSchurComplement`]: exact block elimination via row-workspace accumulation
//! - [`ApproxSchurComplement`]: clique-tree sampling approximation (GKS 2023)
//!
//! # Internal pipeline
//!
//! Both implementations share block-selection logic ([`elimination::Elimination`])
//! and produce a [`schur_laplacian::SchurLaplacian`], but differ in assembly strategy:
//!
//! - **Exact**: row-workspace accumulation ([`SchurLaplacian::from_elimination`]) —
//!   avoids materializing intermediate edges
//! - **Approximate**: star-based edge emission via [`CliqueEmitter`](elimination::CliqueEmitter),
//!   then sort-merge assembly ([`SchurLaplacian::from_edges`])

mod elimination;
mod schur_laplacian;

use schwarz_precond::SparseMatrix;

pub(crate) use elimination::EliminationInfo;
use elimination::{Elimination, SampledCliqueEmitter};
use schur_laplacian::SchurLaplacian;

use super::gramian::CrossTab;
use crate::config::ApproxSchurConfig;

/// Undirected fill edge: `(lo_col, hi_col, weight)` with `lo_col < hi_col`.
type Edge = (u32, u32, f64);

/// Result of Schur complement computation on a bipartite SDDM.
///
/// Pure data bundle: the Schur complement matrix + elimination metadata.
pub(crate) struct SchurResult {
    /// The Schur complement as a sparse matrix.
    pub matrix: SparseMatrix,
    /// Elimination metadata for the back-substitution step.
    pub elimination: EliminationInfo,
}

/// Dense Schur complement result (row-major matrix + elimination metadata).
#[cfg(test)]
pub(crate) struct DenseSchurResult {
    /// Row-major dense Schur matrix (size `n * n`).
    pub matrix: Vec<f64>,
    /// Matrix dimension.
    pub n: usize,
    /// Elimination metadata for the back-substitution step.
    pub elimination: EliminationInfo,
}

/// Anchored dense Schur result: top-left principal minor + elimination metadata.
pub(crate) struct AnchoredDenseSchurResult {
    /// Row-major anchored minor of size `(n-1) x (n-1)`.
    pub anchored_minor: Vec<f64>,
    /// Full Schur dimension before anchoring.
    pub n: usize,
    /// Elimination metadata for the back-substitution step.
    pub elimination: EliminationInfo,
}

// ---------------------------------------------------------------------------
// Trait + implementations
// ---------------------------------------------------------------------------

/// Strategy for computing the Schur complement of a [`CrossTab`].
pub(crate) trait SchurComplement {
    fn compute(&self, cross_tab: &CrossTab) -> SchurResult;
}

/// Exact Schur complement via block elimination.
pub(crate) struct ExactSchurComplement;

/// Approximate Schur complement via clique-tree sampling.
pub(crate) struct ApproxSchurComplement {
    config: ApproxSchurConfig,
}

impl ApproxSchurComplement {
    pub fn new(config: ApproxSchurConfig) -> Self {
        Self { config }
    }
}

impl SchurComplement for ExactSchurComplement {
    /// Compute the exact Schur complement using row-workspace accumulation.
    ///
    /// For the bipartite SDDM `[D_q, -C; -C^T, D_r]`, eliminates the larger
    /// block (exact since it's diagonal) to get a reduced Laplacian on the
    /// smaller block.
    fn compute(&self, cross_tab: &CrossTab) -> SchurResult {
        let elim = Elimination::new(cross_tab);
        let laplacian = SchurLaplacian::from_elimination(&elim);
        SchurResult {
            matrix: laplacian.matrix,
            elimination: elim.into_info(),
        }
    }
}

impl ExactSchurComplement {
    /// Compute the exact Schur complement as a dense row-major matrix.
    ///
    /// Used by the tiny-system fast path to avoid sparse Schur assembly and
    /// sparse ApproxChol builder overhead.
    #[cfg(test)]
    pub(crate) fn compute_dense(&self, cross_tab: &CrossTab) -> DenseSchurResult {
        let elim = Elimination::new(cross_tab);
        let matrix = SchurLaplacian::dense_from_elimination(&elim);
        DenseSchurResult {
            matrix,
            n: elim.n_keep,
            elimination: elim.into_info(),
        }
    }

    /// Compute the exact Schur anchored dense minor directly.
    ///
    /// The anchored top-left principal minor is what dense Cholesky factors, so
    /// this avoids allocating the full dense Schur matrix.
    pub(crate) fn compute_dense_anchored(&self, cross_tab: &CrossTab) -> AnchoredDenseSchurResult {
        let elim = Elimination::new(cross_tab);
        let anchored_minor = SchurLaplacian::anchored_minor_from_elimination(&elim);
        AnchoredDenseSchurResult {
            anchored_minor,
            n: elim.n_keep,
            elimination: elim.into_info(),
        }
    }
}

impl SchurComplement for ApproxSchurComplement {
    /// Compute an approximate Schur complement by sampling clique-trees.
    ///
    /// Each eliminated vertex produces at most deg-1 fill edges via the
    /// GKS 2023 Algorithm 5 clique-tree approximation.
    fn compute(&self, cross_tab: &CrossTab) -> SchurResult {
        let elim = Elimination::new(cross_tab);
        let emitter = SampledCliqueEmitter::new(self.config.seed);
        let edges = elim.par_emit(&emitter);
        let laplacian = SchurLaplacian::from_edges(edges, elim.n_keep);
        SchurResult {
            matrix: laplacian.matrix,
            elimination: elim.into_info(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operator::csr_block::CsrBlock;

    fn make_cross_tab(
        c_dense: &[f64],
        n_q: usize,
        n_r: usize,
        diag_q: Vec<f64>,
        diag_r: Vec<f64>,
    ) -> CrossTab {
        let c = CsrBlock::from_dense_table(c_dense, n_q, n_r);
        let ct = c.transpose();
        CrossTab {
            c,
            ct,
            diag_q,
            diag_r,
        }
    }

    fn sparse_to_dense(matrix: &SparseMatrix) -> Vec<Vec<f64>> {
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
        let cross_tab = make_cross_tab(&c_dense, 3, 2, diag_q.clone(), diag_r.clone());

        let result = ExactSchurComplement.compute(&cross_tab);

        assert!(result.elimination.eliminate_q);
        assert_eq!(result.elimination.inv_diag_elim.len(), 3);
        for (&got, &expected) in result
            .elimination
            .inv_diag_elim
            .iter()
            .zip([1.0 / 5.0, 1.0 / 6.0, 1.0 / 8.0].iter())
        {
            assert!((got - expected).abs() < 1e-12);
        }

        let expected = dense_exact_schur(&c_dense, 3, 2, &diag_q, &diag_r, true);
        let got = sparse_to_dense(&result.matrix);
        assert_dense_close(&got, &expected, 1e-12);
    }

    #[test]
    fn exact_schur_handles_zero_eliminated_diagonal_when_eliminating_r() {
        // C is 2x3, so r-block is eliminated (n_q < n_r). Last eliminated
        // diagonal is zero, so its inverse contribution should be 0.
        let c_dense = vec![2.0, 0.0, 1.0, 0.0, 3.0, 4.0];
        let diag_q = vec![8.0, 9.0];
        let diag_r = vec![5.0, 6.0, 0.0];
        let cross_tab = make_cross_tab(&c_dense, 2, 3, diag_q.clone(), diag_r.clone());

        let result = ExactSchurComplement.compute(&cross_tab);

        assert!(!result.elimination.eliminate_q);
        assert_eq!(result.elimination.inv_diag_elim.len(), 3);
        assert!((result.elimination.inv_diag_elim[0] - 1.0 / 5.0).abs() < 1e-12);
        assert!((result.elimination.inv_diag_elim[1] - 1.0 / 6.0).abs() < 1e-12);
        assert_eq!(result.elimination.inv_diag_elim[2], 0.0);

        let expected = dense_exact_schur(&c_dense, 2, 3, &diag_q, &diag_r, false);
        let got = sparse_to_dense(&result.matrix);
        assert_dense_close(&got, &expected, 1e-12);
    }

    #[test]
    fn exact_dense_schur_matches_sparse_exact() {
        let c_dense = vec![1.0, 2.0, 3.0, 0.0, 0.0, 4.0];
        let cross_tab = make_cross_tab(&c_dense, 3, 2, vec![5.0, 6.0, 8.0], vec![7.0, 9.0]);

        let sparse = ExactSchurComplement.compute(&cross_tab);
        let dense = ExactSchurComplement.compute_dense(&cross_tab);

        assert_eq!(dense.n, sparse.matrix.n());
        assert_eq!(
            dense.elimination.eliminate_q,
            sparse.elimination.eliminate_q
        );
        assert_eq!(
            dense.elimination.inv_diag_elim.len(),
            sparse.elimination.inv_diag_elim.len()
        );
        for (&a, &b) in dense
            .elimination
            .inv_diag_elim
            .iter()
            .zip(sparse.elimination.inv_diag_elim.iter())
        {
            assert!((a - b).abs() < 1e-15);
        }

        let got_dense: Vec<Vec<f64>> = (0..dense.n)
            .map(|i| dense.matrix[i * dense.n..(i + 1) * dense.n].to_vec())
            .collect();
        let got_sparse = sparse_to_dense(&sparse.matrix);
        assert_dense_close(&got_dense, &got_sparse, 1e-12);
    }

    #[test]
    fn exact_dense_anchored_matches_full_dense_minor() {
        let c_dense = vec![1.0, 2.0, 3.0, 0.0, 0.0, 4.0];
        let cross_tab = make_cross_tab(&c_dense, 3, 2, vec![5.0, 6.0, 8.0], vec![7.0, 9.0]);

        let full = ExactSchurComplement.compute_dense(&cross_tab);
        let anchored = ExactSchurComplement.compute_dense_anchored(&cross_tab);
        assert_eq!(full.n, anchored.n);

        let m = full.n.saturating_sub(1);
        let mut full_minor = vec![0.0; m * m];
        for i in 0..m {
            for j in 0..m {
                full_minor[i * m + j] = full.matrix[i * full.n + j];
            }
        }

        assert_eq!(anchored.anchored_minor.len(), full_minor.len());
        for (&a, &b) in anchored.anchored_minor.iter().zip(full_minor.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn approximate_schur_is_seed_deterministic_and_laplacian_like() {
        // Degree-3 star in eliminated block gives nontrivial sampled edges.
        let c_dense = vec![1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let cross_tab = make_cross_tab(&c_dense, 3, 3, vec![10.0, 4.0, 5.0], vec![2.0, 3.0, 4.0]);
        let approx = ApproxSchurComplement::new(crate::config::ApproxSchurConfig { seed: 12345 });

        let a = approx.compute(&cross_tab);
        let b = approx.compute(&cross_tab);

        assert_eq!(a.elimination.eliminate_q, b.elimination.eliminate_q);
        assert_eq!(a.elimination.inv_diag_elim, b.elimination.inv_diag_elim);
        assert_eq!(a.matrix.indptr(), b.matrix.indptr());
        assert_eq!(a.matrix.indices(), b.matrix.indices());
        assert_eq!(a.matrix.data(), b.matrix.data());

        let dense = sparse_to_dense(&a.matrix);
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
