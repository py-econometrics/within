//! Schur complement of the eliminated diagonal block in a bipartite SDDM
//! system — SDDM in, SDDM on the kept levels out.
//!
//! [`exact`] accumulates the true reduced rows; [`sampled`] approximates each
//! eliminated star's clique on the explicit augmented graph; [`dense_minor`]
//! materializes the exact top-left `m × m` minor for the dense direct path.

use super::compensated_sum;
use super::csr_matrix::CsrMatrix;
use rayon::prelude::*;

use super::elimination::{Edge, Elimination};
use crate::config::ApproxSchurConfig;
use crate::csr_block::to_u32;
use crate::domain::SolveSpace;

/// Build the exact Schur complement via row-workspace accumulation.
///
/// Computes `S = D_keep − keep_to_elim · diag(inv_diag_elim) · elim_to_keep`
/// directly, without materializing intermediate edges. Each keep-block row
/// scatters into a dense workspace, then extracts non-zeros.
pub(crate) fn exact(elim: &Elimination) -> CsrMatrix {
    let n_keep = elim.n_keep;

    // Per-row Schur complement accumulation, parallelized via map_init.
    // The (work, touched) pair is allocated once per rayon task and reused
    // across rows assigned to that task.
    let rows: Vec<(Vec<u32>, Vec<f64>)> = (0..n_keep)
        .into_par_iter()
        .map_init(
            || (vec![0.0f64; n_keep], Vec::new()),
            |(work, touched), i| {
                compute_schur_row_dense(elim, i, work, touched);
                let result = extract_sparse_row(i, work, touched);
                touched.clear();
                result
            },
        )
        .collect();

    assemble_schur_csr(rows, n_keep)
}

/// Build the sampled Schur complement as a Laplacian. Grounded systems retain
/// the ground vertex as an ordinary final vertex.
pub(crate) fn sampled(elim: &Elimination, config: &ApproxSchurConfig) -> CsrMatrix {
    let edges = elim.par_emit(config);
    let n = elim.n_keep + usize::from(elim.solve_space == SolveSpace::Grounded);
    build_laplacian_csr(&edges, n)
}

pub(crate) fn exact_for_factor(elim: &Elimination) -> CsrMatrix {
    let principal = exact(elim);
    let surplus = reduced_surplus(elim);
    build_explicit_laplacian(&principal, &surplus, elim.solve_space)
}

/// Build the exact top-left `m × m` Schur minor in row-major order.
///
/// `m == n_keep` is the full grounded complement; `m == n_keep − 1` anchors the
/// last node of a floating Laplacian.
pub(crate) fn dense_minor(elim: &Elimination, m: usize) -> Vec<f64> {
    if m == 0 {
        return Vec::new();
    }

    let mut minor = vec![0.0; m * m];
    for i in 0..m {
        minor[i * m + i] = elim.diag_keep[i];
    }

    // S_minor = D_keep_minor - keep_to_elim_minor * inv(D_elim) * elim_to_keep_minor
    for i in 0..m {
        for (k, w) in elim.keep_to_elim.row(i) {
            let scale = w * elim.inv_diag_elim[k];
            for (j, v) in elim.elim_to_keep.row(k) {
                if j < m {
                    minor[i * m + j] -= scale * v;
                }
            }
        }
    }

    minor
}

/// Scatter the Schur row `i` into a dense workspace.
///
/// Computes `work[j] = D_keep[i] δ_{ij} - Σ_k (keep_to_elim[i,k] / D_elim[k]) * elim_to_keep[k,j]`
/// and records touched column indices.
fn compute_schur_row_dense(
    elim: &Elimination,
    i: usize,
    work: &mut [f64],
    touched: &mut Vec<usize>,
) {
    work[i] = elim.diag_keep[i];
    touched.push(i);

    for (k, w) in elim.keep_to_elim.row(i) {
        let scale = w * elim.inv_diag_elim[k];
        for (j, v) in elim.elim_to_keep.row(k) {
            if work[j] == 0.0 && j != i {
                touched.push(j);
            }
            work[j] -= scale * v;
        }
    }
}

/// Extract non-zero entries from the dense workspace into sparse row arrays.
///
/// Sorts touched columns, emits non-zero values (preserving the diagonal even
/// if numerically zero for SDDM structure), and clears the workspace.
fn extract_sparse_row(i: usize, work: &mut [f64], touched: &mut [usize]) -> (Vec<u32>, Vec<f64>) {
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

/// Build a symmetric Laplacian CSR from sorted upper-triangular edges.
///
/// Edges must be sorted by (lo, hi) with lo < hi, which lets lower-triangle,
/// diagonal, and upper-triangle entries land in column order without per-row
/// sorting.
fn build_laplacian_csr(edges: &[Edge], n: usize) -> CsrMatrix {
    debug_assert!(edges.iter().all(|&(lo, hi, _)| lo < hi));

    // Count lower/upper entries per row and accumulate diagonal weights.
    let mut lower_count = vec![0u32; n];
    let mut upper_count = vec![0u32; n];
    let mut diag = vec![0.0; n];
    for &(lo, hi, w) in edges {
        diag[lo as usize] += w;
        upper_count[lo as usize] += 1; // row lo gets col hi (upper)
        lower_count[hi as usize] += 1; // row hi gets col lo (lower)
        diag[hi as usize] += w;
    }

    // Row layout: [lower entries | diagonal | upper entries]
    let mut offsets = vec![0u32; n + 1];
    for i in 0..n {
        offsets[i + 1] = offsets[i] + lower_count[i] + 1 + upper_count[i];
    }
    let total_nnz = offsets[n] as usize;
    let mut indices = vec![0u32; total_nnz];
    let mut data = vec![0.0f64; total_nnz];

    // Place diagonals and initialize cursors.
    let mut lower_cursor: Vec<u32> = (0..n).map(|i| offsets[i]).collect();
    let mut upper_cursor: Vec<u32> = (0..n).map(|i| offsets[i] + lower_count[i] + 1).collect();
    for i in 0..n {
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

    CsrMatrix::new(offsets, indices, data, n)
}

fn reduced_surplus(elim: &Elimination) -> Vec<f64> {
    if elim.solve_space == SolveSpace::Floating {
        return vec![0.0; elim.n_keep];
    }
    let scaled: Vec<f64> = elim
        .inv_diag_elim
        .iter()
        .zip(elim.surplus_elim)
        .map(|(&inv_diag, &surplus)| inv_diag * surplus)
        .collect();
    let mut surplus = vec![0.0; elim.n_keep];
    elim.keep_to_elim
        .spmv_assign_add(&scaled, elim.surplus_keep, &mut surplus, false);
    surplus
}

pub(super) fn build_explicit_laplacian(
    principal: &CsrMatrix,
    surplus: &[f64],
    solve_space: SolveSpace,
) -> CsrMatrix {
    let n_keep = principal.n();
    let ground = to_u32(n_keep);
    let grounded = solve_space == SolveSpace::Grounded;
    let n = n_keep + usize::from(grounded);
    let mut indptr = Vec::with_capacity(n + 1);
    let mut indices = Vec::new();
    let mut data = Vec::new();
    indptr.push(0);

    for (i, &row_surplus) in surplus.iter().enumerate().take(n_keep) {
        let start = principal.indptr()[i] as usize;
        let end = principal.indptr()[i + 1] as usize;
        let mut adjacency = 0.0;
        let mut diagonal_position = None;
        for (&j, &value) in principal.indices()[start..end]
            .iter()
            .zip(&principal.data()[start..end])
        {
            if j as usize == i {
                diagonal_position = Some(indices.len());
                indices.push(j);
                data.push(0.0);
            } else {
                adjacency -= value;
                indices.push(j);
                data.push(value);
            }
        }
        let diagonal_position = diagonal_position.expect("exact Schur row must contain a diagonal");
        data[diagonal_position] = adjacency + row_surplus;
        if grounded && row_surplus > 0.0 {
            indices.push(ground);
            data.push(-row_surplus);
        }
        indptr.push(to_u32(indices.len()));
    }

    if grounded {
        for (i, &value) in surplus.iter().enumerate() {
            if value > 0.0 {
                indices.push(to_u32(i));
                data.push(-value);
            }
        }
        indices.push(ground);
        data.push(compensated_sum(surplus));
        indptr.push(to_u32(indices.len()));
    }
    CsrMatrix::new(indptr, indices, data, n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csr_block::CsrBlock;
    use crate::domain::{BlockDiagonals, CrossTab, GroundEdges};

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

    fn surplus_of(cross_tab: &CrossTab, diagonals: &BlockDiagonals) -> GroundEdges {
        let row_surplus = |block: &CsrBlock, diagonal: &[f64]| {
            (0..block.nrows)
                .map(|i| (diagonal[i] - block.row(i).map(|(_, v)| v).sum::<f64>()).max(0.0))
                .collect()
        };
        GroundEdges {
            q: row_surplus(&cross_tab.c, &diagonals.q),
            r: row_surplus(&cross_tab.ct, &diagonals.r),
        }
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
        let surplus = surplus_of(&cross_tab, &diagonals);
        let elim =
            Elimination::new(&cross_tab, &diagonals, &surplus, SolveSpace::Grounded).unwrap();

        assert!(elim.eliminate_q);
        assert_eq!(elim.inv_diag_elim.len(), 3);
        for (&got, &expected) in elim
            .inv_diag_elim
            .iter()
            .zip([1.0 / 5.0, 1.0 / 6.0, 1.0 / 8.0].iter())
        {
            assert!((got - expected).abs() < 1e-12);
        }

        let matrix = exact(&elim);
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

        let surplus = surplus_of(&cross_tab, &diagonals);
        let result = Elimination::new(&cross_tab, &diagonals, &surplus, SolveSpace::Grounded);
        match result {
            Err(crate::BuildError::SingularDiagonal { index: 2, .. }) => {}
            Err(e) => panic!("expected SingularDiagonal at index 2, got: {e}"),
            Ok(_) => panic!("expected SingularDiagonal error, got Ok"),
        }
    }

    #[test]
    fn approximate_schur_is_seed_deterministic_and_laplacian_like() {
        // Degree-3 star in eliminated block gives nontrivial sampled edges.
        // Diagonals equal the adjacency row/column sums exactly, so the
        // reduced system is a pure (zero-row-sum) Laplacian.
        let c_dense = vec![1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let (cross_tab, diagonals) =
            make_cross_tab(&c_dense, 3, 3, vec![6.0, 1.0, 1.0], vec![2.0, 3.0, 3.0]);
        let surplus = surplus_of(&cross_tab, &diagonals);
        let elim_a =
            Elimination::new(&cross_tab, &diagonals, &surplus, SolveSpace::Floating).unwrap();
        let elim_b =
            Elimination::new(&cross_tab, &diagonals, &surplus, SolveSpace::Floating).unwrap();
        let config = crate::config::ApproxSchurConfig {
            seed: 12345,
            ..Default::default()
        };

        let a = sampled(&elim_a, &config);
        let b = sampled(&elim_b, &config);

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

    #[test]
    fn sampled_schur_carries_surplus_exactly_on_low_degree_stars() {
        // Surplus on both blocks, geometry chosen so every eliminated star has
        // at most two entries *including* its ground entry — clique-tree
        // sampling of a 2-entry star is deterministic and exact, so the
        // sampled reduction must equal the exact Schur complement:
        //   q0: adjacency {r0: 1}, diag 3   -> star {(r0,1), (g,2)}
        //   q1: adjacency {r0: 2, r1: 3}, diag 5 (no surplus)
        //   q2: adjacency {r1: 4}, diag 6   -> star {(r1,4), (g,2)}
        // Kept rows carry their own surplus: diag_r = col sums + [0.5, 0].
        let c_dense = vec![1.0, 0.0, 2.0, 3.0, 0.0, 4.0];
        let diag_q = vec![3.0, 5.0, 6.0];
        let diag_r = vec![3.5, 7.0];
        let (cross_tab, diagonals) = make_cross_tab(&c_dense, 3, 2, diag_q.clone(), diag_r.clone());
        let surplus = surplus_of(&cross_tab, &diagonals);
        let elim =
            Elimination::new(&cross_tab, &diagonals, &surplus, SolveSpace::Grounded).unwrap();
        assert!(elim.eliminate_q);

        let sampled_dense = sparse_to_dense(&sampled(&elim, &Default::default()));
        let expected = dense_exact_schur(&c_dense, 3, 2, &diag_q, &diag_r, true);
        let sampled_principal: Vec<Vec<f64>> = sampled_dense[..expected.len()]
            .iter()
            .map(|row| row[..expected.len()].to_vec())
            .collect();
        assert_dense_close(&sampled_principal, &expected, 1e-12);

        // Surplus is represented by an explicit ground vertex, so the augmented
        // reduced matrix is an exact zero-row-sum Laplacian.
        for (i, row) in sampled_dense.iter().enumerate() {
            let row_sum: f64 = row.iter().sum();
            assert!(row_sum.abs() < 1e-12, "row {i} sum is {row_sum}");
        }
    }
}
