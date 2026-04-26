//! Schur complement computation for bipartite SDDM systems.
//!
//! When using block elimination to solve the local bipartite Gramian
//! (see [`local_solver`](super::local_solver)), we need the Schur complement
//! of the eliminated factor block. For the 2x2 block system:
//!
//! ```text
//!     [ D_elim    C_e ]       [ z_elim ]   [ r_elim ]
//!     [                ]  *   [         ] = [         ]
//!     [ C_e^T    D_keep]       [ z_keep ]   [ r_keep ]
//! ```
//!
//! eliminating `z_elim` (trivial since `D_elim` is diagonal) yields the
//! reduced system `S z_keep = r_keep - C_e^T D_elim^{-1} r_elim` where the
//! **Schur complement** is:
//!
//! ```text
//! S = D_keep - C_e^T D_elim^{-1} C_e
//! ```
//!
//! This `S` is a graph Laplacian on the kept factor levels, encoding how
//! they co-occur through the eliminated factor.
//!
//! # Exact vs approximate
//!
//! This module provides a [`SchurComplement`] trait with two implementations:
//!
//! - [`ExactSchurComplement`]: row-workspace accumulation computes `S` exactly.
//!   Each kept-block row scatters fill contributions into a dense workspace,
//!   then extracts non-zeros. Cost is O(nnz(S)), but `S` can be dense when
//!   the eliminated factor has high-degree levels (many keep-block DOFs
//!   share an eliminated level, creating fill edges between all of them).
//!
//! - [`ApproxSchurComplement`]: clique-tree sampling approximation (GKS 2023).
//!   Each eliminated vertex contributes a "star" (clique) in the fill graph.
//!   Instead of materializing all O(deg^2) fill edges per star, the
//!   clique-tree sampler produces only O(deg) edges that spectrally
//!   approximate the exact clique. This keeps `S` sparse even when the
//!   exact Schur complement would be dense — critical for high-cardinality
//!   factor structures.
//!
//! # Internal pipeline
//!
//! Both implementations share block-selection logic ([`Elimination`])
//! and produce a [`SchurLaplacian`], but differ in assembly strategy:
//!
//! - **Exact**: row-workspace accumulation ([`SchurLaplacian::from_elimination`]) —
//!   avoids materializing intermediate edges
//! - **Approximate**: star-based edge emission via [`SampledCliqueEmitter`],
//!   then sort-merge assembly ([`SchurLaplacian::from_edges`])

use approx_chol::low_level::{clique_tree_sample, clique_tree_sample_multi};
use rayon::prelude::*;
use schwarz_precond::SparseMatrix;

use super::csr_block::CsrBlock;
use super::gramian::CrossTab;
use crate::config::ApproxSchurConfig;
use crate::{WithinError, WithinResult};

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

// The Schur complement S = D_keep - C_keep^T * D_elim^{-1} * C_keep arises from
// block-eliminating the diagonal block of the larger partition in the bipartite
// SDDM system [D_q, -C; -C^T, D_r]. Since D_elim is diagonal, the elimination
// is exact and each eliminated vertex k contributes a rank-1 clique (star) to
// the fill graph: all pairs of k's neighbors in the keep-block get a fill edge.
//
// Two strategies materialize these fill edges:
// - `ExactCliqueEmitter` (not shown here; the exact path uses row-workspace
//   accumulation in `SchurLaplacian::from_elimination` instead)
// - `SampledCliqueEmitter`: uses GKS 2023 clique-tree sampling to approximate
//   high-degree cliques with O(deg) edges instead of O(deg^2), keeping the
//   Schur complement spectrally close to the exact one.

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

/// Emits sampled clique-tree fill edges for every star.
struct SampledCliqueEmitter {
    seed: u64,
    split: u32,
}

impl SampledCliqueEmitter {
    fn new(config: &ApproxSchurConfig) -> Self {
        Self {
            seed: config.seed,
            split: config.split,
        }
    }

    /// Emit fill edges from `star` into `edges`, using `scratch` for temporaries.
    fn emit(&self, star: &Star, edges: &mut Vec<Edge>, scratch: &mut SampledScratch) {
        scratch.buf.clear();
        for (&col, &w) in star.col_indices.iter().zip(star.weights) {
            scratch.buf.push((col, w));
        }
        let seed = self.seed.wrapping_add(star.index as u64);
        if self.split <= 1 {
            clique_tree_sample(&mut scratch.buf, star.diag, seed, edges);
        } else {
            clique_tree_sample_multi(&mut scratch.buf, self.split, seed, edges);
        }
    }
}

/// Thread-local scratch for [`SampledCliqueEmitter`].
#[derive(Default)]
struct SampledScratch {
    /// AoS neighbor copy for `clique_tree_sample`.
    buf: Vec<(u32, f64)>,
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
    fn new(cross_tab: &'a CrossTab) -> WithinResult<Self> {
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
        let mut inv_diag_elim = Vec::with_capacity(diag_elim.len());
        for (i, &d) in diag_elim.iter().enumerate() {
            if d > 0.0 {
                inv_diag_elim.push(1.0 / d);
            } else {
                return Err(WithinError::SingularDiagonal {
                    block: if eliminate_q { "q (elim)" } else { "r (elim)" },
                    index: i,
                });
            }
        }

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

        Ok(Self {
            eliminate_q,
            n_keep,
            n_elim,
            inv_diag_elim,
            diag_elim,
            diag_keep,
            keep_to_elim,
            elim_to_keep,
        })
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

    /// Parallel edge emission over all stars using the given emitter.
    ///
    /// Each rayon task sorts and deduplicates its local edges, then the reduce
    /// tree merges sorted chunks — avoiding a single O(E log E) global sort.
    fn par_emit(&self, emitter: &SampledCliqueEmitter) -> Vec<Edge> {
        (0..self.n_elim)
            .into_par_iter()
            .fold(
                || (Vec::new(), SampledScratch::default()),
                |(mut edges, mut scratch), k| {
                    let star = self.star(k);
                    if star.degree() > 1 {
                        emitter.emit(&star, &mut edges, &mut scratch);
                    }
                    (edges, scratch)
                },
            )
            .map(|(mut edges, _)| {
                sort_and_dedup(&mut edges);
                edges
            })
            .reduce(Vec::new, merge_dedup)
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
// Edge sort-merge helpers
// ===========================================================================

/// Sort edges by (lo, hi) and merge duplicates by summing weights.
fn sort_and_dedup(edges: &mut Vec<Edge>) {
    edges.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    if edges.len() <= 1 {
        return;
    }
    let mut write = 0;
    for read in 1..edges.len() {
        if edges[write].0 == edges[read].0 && edges[write].1 == edges[read].1 {
            edges[write].2 += edges[read].2;
        } else {
            write += 1;
            edges[write] = edges[read];
        }
    }
    edges.truncate(write + 1);
}

/// Merge two sorted, deduplicated edge lists, summing weights for duplicates.
fn merge_dedup(a: Vec<Edge>, b: Vec<Edge>) -> Vec<Edge> {
    if a.is_empty() {
        return b;
    }
    if b.is_empty() {
        return a;
    }
    let mut result = Vec::with_capacity(a.len() + b.len());
    let (mut ia, mut ib) = (0, 0);
    while ia < a.len() && ib < b.len() {
        let ka = (a[ia].0, a[ia].1);
        let kb = (b[ib].0, b[ib].1);
        match ka.cmp(&kb) {
            std::cmp::Ordering::Less => {
                result.push(a[ia]);
                ia += 1;
            }
            std::cmp::Ordering::Greater => {
                result.push(b[ib]);
                ib += 1;
            }
            std::cmp::Ordering::Equal => {
                result.push((a[ia].0, a[ia].1, a[ia].2 + b[ib].2));
                ia += 1;
                ib += 1;
            }
        }
    }
    if ia < a.len() {
        result.extend_from_slice(&a[ia..]);
    }
    if ib < b.len() {
        result.extend_from_slice(&b[ib..]);
    }
    result
}

// ===========================================================================
// Schur row-workspace helpers
// ===========================================================================

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
fn extract_sparse_row(i: usize, work: &mut [f64], touched: &mut [usize]) -> (Vec<u32>, Vec<f64>) {
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
    (row_indices, row_data)
}

/// Assemble a CSR matrix from per-row sparse results.
fn assemble_schur_csr(rows: Vec<(Vec<u32>, Vec<f64>)>, n_keep: usize) -> SparseMatrix {
    let mut s_indptr = vec![0u32; n_keep + 1];
    let mut s_indices = Vec::new();
    let mut s_data = Vec::new();
    for (i, (ri, rd)) in rows.into_iter().enumerate() {
        s_indices.extend_from_slice(&ri);
        s_data.extend_from_slice(&rd);
        s_indptr[i + 1] = s_indices.len() as u32;
    }
    SparseMatrix::new(s_indptr, s_indices, s_data, n_keep)
}

// ===========================================================================
// SchurLaplacian — Laplacian assembly
// ===========================================================================

/// Assembled Schur complement Laplacian matrix.
struct SchurLaplacian {
    matrix: SparseMatrix,
}

impl SchurLaplacian {
    /// Build a symmetric CSR Laplacian from pre-sorted, deduplicated fill edges.
    fn from_edges(edges: Vec<Edge>, n_keep: usize) -> Self {
        Self {
            matrix: Self::build_laplacian_csr(&edges, n_keep),
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

        // Per-row Schur complement accumulation, parallelized via map_init.
        // The (work, touched) pair is allocated once per rayon task and reused
        // across rows assigned to that task.
        let rows: Vec<(Vec<u32>, Vec<f64>)> = (0..n_keep)
            .into_par_iter()
            .map_init(
                || (vec![0.0f64; n_keep], Vec::new()),
                |(work, touched), i| {
                    compute_schur_row_dense(
                        i,
                        diag_keep,
                        keep_to_elim,
                        elim_to_keep,
                        inv_diag_elim,
                        work,
                        touched,
                    );
                    let result = extract_sparse_row(i, work, touched);
                    touched.clear();
                    result
                },
            )
            .collect();

        let matrix = assemble_schur_csr(rows, n_keep);
        Self { matrix }
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

    /// Build symmetric CSR Laplacian from sorted upper-triangular edges.
    ///
    /// Edges must be sorted by (lo, hi) with lo < hi. This lets us place
    /// lower-triangle, diagonal, and upper-triangle entries in correct column
    /// order without any per-row sorting.
    fn build_laplacian_csr(edges: &[Edge], n_keep: usize) -> SparseMatrix {
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
            indices[pos] = i as u32;
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
    fn compute(&self, cross_tab: &CrossTab) -> WithinResult<SchurResult>;
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
    fn compute(&self, cross_tab: &CrossTab) -> WithinResult<SchurResult> {
        let elim = Elimination::new(cross_tab)?;
        let laplacian = SchurLaplacian::from_elimination(&elim);
        Ok(SchurResult {
            matrix: laplacian.matrix,
            elimination: elim.into_info(),
        })
    }
}

impl ExactSchurComplement {
    /// Compute the exact Schur complement as a dense row-major matrix.
    ///
    /// Used by the tiny-system fast path to avoid sparse Schur assembly and
    /// sparse ApproxChol builder overhead.
    #[cfg(test)]
    pub(crate) fn compute_dense(&self, cross_tab: &CrossTab) -> WithinResult<DenseSchurResult> {
        let elim = Elimination::new(cross_tab)?;
        let matrix = SchurLaplacian::dense_from_elimination(&elim);
        Ok(DenseSchurResult {
            matrix,
            n: elim.n_keep,
            elimination: elim.into_info(),
        })
    }

    /// Compute the exact Schur anchored dense minor directly.
    ///
    /// The anchored top-left principal minor is what dense Cholesky factors, so
    /// this avoids allocating the full dense Schur matrix.
    pub(crate) fn compute_dense_anchored(
        &self,
        cross_tab: &CrossTab,
    ) -> WithinResult<AnchoredDenseSchurResult> {
        let elim = Elimination::new(cross_tab)?;
        let anchored_minor = SchurLaplacian::anchored_minor_from_elimination(&elim);
        Ok(AnchoredDenseSchurResult {
            anchored_minor,
            n: elim.n_keep,
            elimination: elim.into_info(),
        })
    }
}

impl SchurComplement for ApproxSchurComplement {
    /// Compute an approximate Schur complement by sampling clique-trees.
    ///
    /// Each eliminated vertex produces at most deg-1 fill edges via the
    /// GKS 2023 Algorithm 5 clique-tree approximation.
    fn compute(&self, cross_tab: &CrossTab) -> WithinResult<SchurResult> {
        let elim = Elimination::new(cross_tab)?;
        let emitter = SampledCliqueEmitter::new(&self.config);
        let edges = elim.par_emit(&emitter);
        let laplacian = SchurLaplacian::from_edges(edges, elim.n_keep);
        Ok(SchurResult {
            matrix: laplacian.matrix,
            elimination: elim.into_info(),
        })
    }
}
