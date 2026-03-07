//! Schur complement computation for bipartite SDDM systems.
//!
//! Provides a [`SchurComplement`] trait with two implementations:
//! - [`ExactSchurComplement`]: exact block elimination via row-workspace accumulation
//! - [`ApproxSchurComplement`]: clique-tree sampling approximation (GKS 2023)
//!
//! # Internal pipeline
//!
//! Both implementations share block-selection logic ([`Elimination`])
//! and produce a [`SchurLaplacian`], but differ in assembly strategy:
//!
//! - **Exact**: row-workspace accumulation ([`SchurLaplacian::from_elimination`]) —
//!   avoids materializing intermediate edges
//! - **Approximate**: star-based edge emission via [`CliqueEmitter`],
//!   then sort-merge assembly ([`SchurLaplacian::from_edges`])

use approx_chol::low_level::clique_tree_sample;
use rayon::prelude::*;
use schwarz_precond::SparseMatrix;

use super::csr_block::CsrBlock;
use super::gramian::CrossTab;
use crate::config::ApproxSchurConfig;

/// Undirected fill edge: `(lo_col, hi_col, weight)` with `lo_col < hi_col`.
type Edge = (u32, u32, f64);

// ===========================================================================
// EliminationInfo — public output
// ===========================================================================

/// Metadata from the block-elimination step needed by `BlockElimSolver`.
pub(crate) struct EliminationInfo {
    /// 1 / D_elim[k] for the eliminated diagonal block.
    pub inv_diag_elim: Vec<f64>,
    /// True if the q-block was eliminated (n_q >= n_r).
    pub eliminate_q: bool,
}

// ===========================================================================
// Star — zero-copy neighborhood view
// ===========================================================================

/// One eliminated vertex's neighbors in the keep-block.
///
/// References into [`CsrBlock`]'s arrays for zero-copy access.
struct Star<'a> {
    /// Eliminated vertex index (used for deterministic seeding).
    index: usize,
    /// Neighbor columns in the keep-block.
    col_indices: &'a [u32],
    /// Edge weights to each neighbor.
    weights: &'a [f64],
    /// `D_elim[k]` (needed by `clique_tree_sample`).
    diag: f64,
}

impl Star<'_> {
    fn degree(&self) -> usize {
        self.col_indices.len()
    }
}

// ===========================================================================
// CliqueEmitter — strategy for edge emission
// ===========================================================================

/// Strategy for producing fill edges from a star neighborhood.
trait CliqueEmitter {
    /// Per-thread reusable scratch state.
    type Scratch: Default + Send;

    /// Emit fill edges from `star` into `edges`, using `scratch` for temporaries.
    fn emit(&self, star: &Star, edges: &mut Vec<Edge>, scratch: &mut Self::Scratch);
}

/// Emits sampled clique-tree fill edges for every star.
struct SampledCliqueEmitter {
    seed: u64,
}

impl SampledCliqueEmitter {
    fn new(seed: u64) -> Self {
        Self { seed }
    }
}

/// Thread-local scratch for [`SampledCliqueEmitter`].
#[derive(Default)]
struct SampledScratch {
    /// AoS neighbor copy for `clique_tree_sample`.
    buf: Vec<(u32, f64)>,
}

impl CliqueEmitter for SampledCliqueEmitter {
    type Scratch = SampledScratch;

    fn emit(&self, star: &Star, edges: &mut Vec<Edge>, scratch: &mut SampledScratch) {
        scratch.buf.clear();
        for (&col, &w) in star.col_indices.iter().zip(star.weights) {
            scratch.buf.push((col, w));
        }
        let seed = self.seed.wrapping_add(star.index as u64);
        clique_tree_sample(&mut scratch.buf, star.diag, seed, edges);
    }
}

// ===========================================================================
// Elimination — block selection + star iteration
// ===========================================================================

/// Block-selection decision and star iteration for Schur elimination.
///
/// Encapsulates which block to eliminate, precomputed inverse-diagonals,
/// and provides zero-copy [`Star`] views for each eliminated vertex.
struct Elimination<'a> {
    eliminate_q: bool,
    n_keep: usize,
    n_elim: usize,
    inv_diag_elim: Vec<f64>,
    diag_elim: &'a [f64],
    diag_keep: &'a [f64],
    keep_to_elim: &'a CsrBlock,
    elim_to_keep: &'a CsrBlock,
}

impl<'a> Elimination<'a> {
    /// Select which block to eliminate and precompute inverse-diagonals.
    fn new(cross_tab: &'a CrossTab) -> Self {
        let n_q = cross_tab.n_q();
        let n_r = cross_tab.n_r();
        // Eliminate the larger block to minimize the reduced system size.
        let eliminate_q = n_q >= n_r;
        let (n_keep, n_elim) = if eliminate_q { (n_r, n_q) } else { (n_q, n_r) };

        let diag_elim = if eliminate_q {
            &cross_tab.diag_q
        } else {
            &cross_tab.diag_r
        };
        let inv_diag_elim: Vec<f64> = diag_elim
            .iter()
            .map(|&d| if d > 0.0 { 1.0 / d } else { 0.0 })
            .collect();

        let diag_keep = if eliminate_q {
            &cross_tab.diag_r
        } else {
            &cross_tab.diag_q
        };

        let (keep_to_elim, elim_to_keep) = if eliminate_q {
            (&cross_tab.ct, &cross_tab.c)
        } else {
            (&cross_tab.c, &cross_tab.ct)
        };

        Self {
            eliminate_q,
            n_keep,
            n_elim,
            inv_diag_elim,
            diag_elim,
            diag_keep,
            keep_to_elim,
            elim_to_keep,
        }
    }

    /// Create a zero-copy [`Star`] view for eliminated vertex `k`.
    fn star(&self, k: usize) -> Star<'_> {
        let start = self.elim_to_keep.indptr[k] as usize;
        let end = self.elim_to_keep.indptr[k + 1] as usize;
        Star {
            index: k,
            col_indices: &self.elim_to_keep.indices[start..end],
            weights: &self.elim_to_keep.data[start..end],
            diag: self.diag_elim[k],
        }
    }

    /// Parallel edge emission over all stars using the given [`CliqueEmitter`].
    fn par_emit<E: CliqueEmitter + Sync>(&self, emitter: &E) -> Vec<Edge> {
        (0..self.n_elim)
            .into_par_iter()
            .fold(
                || (Vec::new(), E::Scratch::default()),
                |(mut edges, mut scratch), k| {
                    let star = self.star(k);
                    if star.degree() > 1 {
                        emitter.emit(&star, &mut edges, &mut scratch);
                    }
                    (edges, scratch)
                },
            )
            .map(|(edges, _)| edges)
            .reduce(Vec::new, |mut a, b| {
                a.extend(b);
                a
            })
    }

    /// Package elimination metadata into [`EliminationInfo`] for the solver.
    fn into_info(self) -> EliminationInfo {
        EliminationInfo {
            inv_diag_elim: self.inv_diag_elim,
            eliminate_q: self.eliminate_q,
        }
    }
}

// ===========================================================================
// SchurLaplacian — Laplacian assembly
// ===========================================================================

/// Assembled Schur complement Laplacian matrix.
struct SchurLaplacian {
    matrix: SparseMatrix,
}

impl SchurLaplacian {
    /// Build a symmetric CSR Laplacian from fill edges (sort-merge pipeline).
    ///
    /// Used by the approximate path: edges are par-sorted, duplicates merged,
    /// negligible entries dropped, then assembled into CSR with row-sum diagonal.
    fn from_edges(mut edges: Vec<Edge>, n_keep: usize) -> Self {
        // Sort by (lo, hi), merge duplicates.
        edges.par_sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

        let mut merged: Vec<Edge> = Vec::with_capacity(edges.len());
        for &(lo, hi, w) in &edges {
            if let Some(last) = merged.last_mut() {
                if last.0 == lo && last.1 == hi {
                    last.2 += w;
                    continue;
                }
            }
            merged.push((lo, hi, w));
        }
        merged.retain(|&(_, _, w)| w > f64::EPSILON);

        Self {
            matrix: Self::edges_to_laplacian_csr(&merged, n_keep),
        }
    }

    /// Build the Schur complement via row-workspace accumulation (exact path).
    ///
    /// Computes `S = D_keep − keep_to_elim · diag(inv_diag_elim) · elim_to_keep`
    /// directly, without materializing intermediate edges. Each keep-block row
    /// scatters into a dense workspace, then extracts non-zeros.
    fn from_elimination(elim: &Elimination) -> Self {
        let n_keep = elim.n_keep;
        let inv_diag_elim = &elim.inv_diag_elim;
        let diag_keep = elim.diag_keep;
        let keep_to_elim = elim.keep_to_elim;
        let elim_to_keep = elim.elim_to_keep;

        let rows: Vec<(Vec<u32>, Vec<f64>)> = (0..n_keep)
            .into_par_iter()
            .map_init(
                || (vec![0.0f64; n_keep], Vec::new()),
                |(work, touched), i| {
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

                    touched.sort_unstable();
                    let mut row_indices = Vec::new();
                    let mut row_data = Vec::new();
                    for &j in touched.iter() {
                        let v = work[j];
                        if v != 0.0 || j == i {
                            row_indices.push(j as u32);
                            row_data.push(v);
                        }
                        work[j] = 0.0;
                    }
                    touched.clear();

                    (row_indices, row_data)
                },
            )
            .collect();

        let mut s_indptr = vec![0u32; n_keep + 1];
        let mut s_indices = Vec::new();
        let mut s_data = Vec::new();
        for (i, (ri, rd)) in rows.into_iter().enumerate() {
            s_indices.extend_from_slice(&ri);
            s_data.extend_from_slice(&rd);
            s_indptr[i + 1] = s_indices.len() as u32;
        }

        Self {
            matrix: SparseMatrix::new(s_indptr, s_indices, s_data, n_keep),
        }
    }

    /// Build a dense row-major Schur matrix from elimination data.
    ///
    /// Intended for tiny reduced systems where dense factorization is cheaper
    /// than sparse setup.
    #[cfg(test)]
    fn dense_from_elimination(elim: &Elimination) -> Vec<f64> {
        let n_keep = elim.n_keep;
        let mut dense = vec![0.0; n_keep * n_keep];

        // Start with the keep-block diagonal.
        for i in 0..n_keep {
            dense[i * n_keep + i] = elim.diag_keep[i];
        }

        let inv_diag_elim = &elim.inv_diag_elim;
        let keep_to_elim = elim.keep_to_elim;
        let elim_to_keep = elim.elim_to_keep;

        // S = D_keep - keep_to_elim * diag(inv_diag_elim) * elim_to_keep
        for i in 0..n_keep {
            let fwd_start = keep_to_elim.indptr[i] as usize;
            let fwd_end = keep_to_elim.indptr[i + 1] as usize;
            for fwd_idx in fwd_start..fwd_end {
                let k = keep_to_elim.indices[fwd_idx] as usize;
                let scale = keep_to_elim.data[fwd_idx] * inv_diag_elim[k];
                let bwd_start = elim_to_keep.indptr[k] as usize;
                let bwd_end = elim_to_keep.indptr[k + 1] as usize;
                for bwd_idx in bwd_start..bwd_end {
                    let j = elim_to_keep.indices[bwd_idx] as usize;
                    dense[i * n_keep + j] -= scale * elim_to_keep.data[bwd_idx];
                }
            }
        }

        dense
    }

    /// Build the anchored top-left Schur minor `(n_keep-1) x (n_keep-1)` in row-major.
    ///
    /// This is the matrix actually factored by dense anchored Cholesky, so building
    /// it directly avoids allocating a full `n_keep x n_keep` dense Schur matrix.
    fn anchored_minor_from_elimination(elim: &Elimination) -> Vec<f64> {
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

    /// Convert merged upper-triangular edge list to symmetric CSR Laplacian.
    ///
    /// Uses a single flat buffer instead of per-row `Vec`s. Each row gets its
    /// diagonal at slot 0, off-diagonal entries via cursors, then a
    /// sort-and-rotate pass places the diagonal in its sorted position.
    fn edges_to_laplacian_csr(edges: &[Edge], n_keep: usize) -> SparseMatrix {
        // Count off-diagonal entries per row, accumulate diagonal weights.
        let mut offdiag_count = vec![0u32; n_keep];
        let mut diag = vec![0.0f64; n_keep];
        for &(lo, hi, w) in edges {
            offdiag_count[lo as usize] += 1;
            offdiag_count[hi as usize] += 1;
            diag[lo as usize] += w;
            diag[hi as usize] += w;
        }

        // Build row offsets (each row = 1 diagonal + off-diag entries).
        let mut offsets = vec![0u32; n_keep + 1];
        for i in 0..n_keep {
            offsets[i + 1] = offsets[i] + offdiag_count[i] + 1;
        }
        let total_nnz = offsets[n_keep] as usize;
        let mut buf: Vec<(u32, f64)> = vec![(0, 0.0); total_nnz];

        // Place diagonal at slot 0 of each row, fill off-diagonal via cursors.
        let mut cursors: Vec<u32> = offsets[..n_keep].to_vec();
        for (i, cur) in cursors.iter_mut().enumerate() {
            buf[*cur as usize] = (i as u32, diag[i]);
            *cur += 1;
        }
        for &(lo, hi, w) in edges {
            let lo = lo as usize;
            let hi = hi as usize;
            buf[cursors[lo] as usize] = (hi as u32, -w);
            cursors[lo] += 1;
            buf[cursors[hi] as usize] = (lo as u32, -w);
            cursors[hi] += 1;
        }

        // Sort each row's off-diagonal portion, merge duplicates, place diagonal.
        for i in 0..n_keep {
            let start = offsets[i] as usize;
            let end = offsets[i + 1] as usize;
            if end - start <= 1 {
                continue;
            }
            buf[start + 1..end].sort_unstable_by(|a, b| a.0.cmp(&b.0));
            let diag_col = i as u32;
            let mut write = start + 1;
            let mut read = start + 1;
            while read < end {
                let col = buf[read].0;
                let mut w = buf[read].1;
                read += 1;
                while read < end && buf[read].0 == col {
                    w += buf[read].1;
                    read += 1;
                }
                if w.abs() > f64::EPSILON {
                    buf[write] = (col, w);
                    write += 1;
                }
            }
            let offdiag_end = write;
            let diag_pos =
                buf[start + 1..offdiag_end].partition_point(|e| e.0 < diag_col) + start + 1;
            buf[start..offdiag_end].rotate_left(1);
            let target = diag_pos - 1;
            if target < offdiag_end - 1 {
                buf[target..offdiag_end].rotate_right(1);
            }
            offsets[i + 1] = offdiag_end as u32;
        }
        for i in 1..=n_keep {
            offsets[i] = offsets[i].max(offsets[i - 1]);
        }

        let final_nnz = offsets[n_keep] as usize;
        let mut indices = Vec::with_capacity(final_nnz);
        let mut data = Vec::with_capacity(final_nnz);
        for &(col, val) in &buf[..final_nnz] {
            indices.push(col);
            data.push(val);
        }

        SparseMatrix::new(offsets, indices, data, n_keep)
    }
}

// ===========================================================================
// Result types
// ===========================================================================

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

// ===========================================================================
// Trait + implementations
// ===========================================================================

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
