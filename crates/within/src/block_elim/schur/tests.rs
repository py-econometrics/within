use super::*;
use crate::csr_block::CsrBlock;
use crate::domain::CrossTab;

fn make_operator(
    c_dense: &[f64],
    n_rows: usize,
    n_cols: usize,
    row_diag: Vec<f64>,
    col_diag: Vec<f64>,
    grounding: Grounding,
) -> SddmMatrix {
    let c = CsrBlock::from_dense_table(c_dense, n_rows, n_cols);
    let ct = c.transpose();
    let cross_tab = CrossTab { c, ct };
    let diagonal: Vec<f64> = row_diag.into_iter().chain(col_diag).collect();
    let ground_edges = (0..cross_tab.n_local())
        .map(|i| (diagonal[i] - cross_tab.neighbors(i).map(|(_, v)| v).sum::<f64>()).max(0.0))
        .collect();
    SddmMatrix {
        cross_tab,
        diagonal,
        ground_edges,
        grounding,
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
    n_rows: usize,
    n_cols: usize,
    row_diag: &[f64],
    col_diag: &[f64],
    eliminate_rows: bool,
) -> Vec<Vec<f64>> {
    if eliminate_rows {
        let mut s = vec![vec![0.0; n_cols]; n_cols];
        for i in 0..n_cols {
            s[i][i] = col_diag[i];
        }
        for k in 0..n_rows {
            let inv = if row_diag[k] > 0.0 {
                1.0 / row_diag[k]
            } else {
                0.0
            };
            for i in 0..n_cols {
                let cki = c_dense[k * n_cols + i];
                for j in 0..n_cols {
                    let ckj = c_dense[k * n_cols + j];
                    s[i][j] -= cki * inv * ckj;
                }
            }
        }
        s
    } else {
        let mut s = vec![vec![0.0; n_rows]; n_rows];
        for i in 0..n_rows {
            s[i][i] = row_diag[i];
        }
        for k in 0..n_cols {
            let inv = if col_diag[k] > 0.0 {
                1.0 / col_diag[k]
            } else {
                0.0
            };
            for i in 0..n_rows {
                let cik = c_dense[i * n_cols + k];
                for j in 0..n_rows {
                    let cjk = c_dense[j * n_cols + k];
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
fn exact_schur_matches_dense_reference_when_eliminating_rows() {
    // C is 3x2, so the row block is eliminated (n_rows >= n_cols).
    let c_dense = vec![1.0, 2.0, 3.0, 0.0, 0.0, 4.0];
    let row_diag = vec![5.0, 6.0, 8.0];
    let col_diag = vec![7.0, 9.0];
    let matrix = make_operator(
        &c_dense,
        3,
        2,
        row_diag.clone(),
        col_diag.clone(),
        Grounding::Grounded,
    );
    let inv_diagonal: Vec<f64> = row_diag.iter().map(|d| 1.0 / d).collect();

    let matrix = exact(&matrix, &inv_diagonal, RowSplit::Parallel);
    let expected = dense_exact_schur(&c_dense, 3, 2, &row_diag, &col_diag, true);
    let got = sparse_to_dense(&matrix);
    assert_dense_close(&got, &expected, 1e-12);
}

/// The sequential arm reuses one scatter workspace across rows, so a row that failed to
/// reset it would read the previous row's entries.
#[test]
fn both_row_splits_produce_the_same_complement() {
    let c_dense = vec![1.0, 2.0, 3.0, 0.0, 0.0, 4.0];
    let row_diag = vec![5.0, 6.0, 8.0];
    let col_diag = vec![7.0, 9.0];
    let inv_diagonal: Vec<f64> = row_diag.iter().map(|d| 1.0 / d).collect();

    for grounding in [Grounding::Grounded, Grounding::Floating] {
        let operator = make_operator(
            &c_dense,
            3,
            2,
            row_diag.clone(),
            col_diag.clone(),
            grounding,
        );
        let parallel = exact(&operator, &inv_diagonal, RowSplit::Parallel);
        let sequential = exact(&operator, &inv_diagonal, RowSplit::Sequential);

        assert_eq!(
            sequential.indptr(),
            parallel.indptr(),
            "{grounding:?} indptr"
        );
        assert_eq!(
            sequential.indices(),
            parallel.indices(),
            "{grounding:?} indices"
        );
        assert_eq!(sequential.data(), parallel.data(), "{grounding:?} data");
    }
}

#[test]
fn approximate_schur_is_seed_deterministic_and_laplacian_like() {
    // Diagonals equal the adjacency row/column sums, so the reduced system is a Laplacian.
    let c_dense = vec![1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let matrix = make_operator(
        &c_dense,
        3,
        3,
        vec![6.0, 1.0, 1.0],
        vec![2.0, 3.0, 3.0],
        Grounding::Floating,
    );
    let config = crate::config::ApproxSchurConfig {
        seed: 12345,
        ..Default::default()
    };

    let a = sampled(&matrix, &config);
    let b = sampled(&matrix, &config);

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
    // Every eliminated star has at most two entries including ground, so sampling is exact.
    let c_dense = vec![1.0, 0.0, 2.0, 3.0, 0.0, 4.0];
    let row_diag = vec![3.0, 5.0, 6.0];
    let col_diag = vec![3.5, 7.0];
    let matrix = make_operator(
        &c_dense,
        3,
        2,
        row_diag.clone(),
        col_diag.clone(),
        Grounding::Grounded,
    );

    let sampled_dense = sparse_to_dense(&sampled(&matrix, &Default::default()));
    let expected = dense_exact_schur(&c_dense, 3, 2, &row_diag, &col_diag, true);
    let sampled_principal: Vec<Vec<f64>> = sampled_dense[..expected.len()]
        .iter()
        .map(|row| row[..expected.len()].to_vec())
        .collect();
    assert_dense_close(&sampled_principal, &expected, 1e-12);

    // Surplus becomes an explicit ground vertex, so the augmented matrix is a Laplacian.
    for (i, row) in sampled_dense.iter().enumerate() {
        let row_sum: f64 = row.iter().sum();
        assert!(row_sum.abs() < 1e-12, "row {i} sum is {row_sum}");
    }
}
