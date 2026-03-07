use faer::sparse::SparseRowMat;
use schwarz_precond::SparseMatrix;

/// Build a 4-node path graph Laplacian (0-1-2-3) as a faer sparse CSR matrix.
///
/// ```text
///  1 -1  0  0
/// -1  2 -1  0
///  0 -1  2 -1
///  0  0 -1  1
/// ```
fn path_laplacian_faer() -> SparseRowMat<u32, f64> {
    let nrows = 4usize;
    let ncols = 4usize;
    let row_ptrs = vec![0u32, 2, 5, 8, 10];
    let col_indices = vec![0u32, 1, 0, 1, 2, 1, 2, 3, 2, 3];
    let values = vec![1.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 1.0];

    let symbolic = unsafe {
        faer::sparse::SymbolicSparseRowMat::<u32>::new_unchecked(
            nrows,
            ncols,
            row_ptrs,
            None,
            col_indices,
        )
    };
    SparseRowMat::new(symbolic, values)
}

#[test]
fn faer_to_sparse_matrix_conversion() {
    let mat = path_laplacian_faer();
    let sparse: SparseMatrix = mat.as_ref().into();

    assert_eq!(sparse.n(), 4);
    assert_eq!(sparse.indptr().len(), 5);
    assert_eq!(sparse.nnz(), 10);
}

#[test]
fn faer_matvec_roundtrip() {
    let mat = path_laplacian_faer();
    let sparse: SparseMatrix = mat.as_ref().into();

    // Multiply by [1, 0, 0, 0]:
    //   row 0:  1*1 + 0*(-1)           = 1
    //   row 1:  1*(-1) + 0*2           = -1
    //   row 2:  0                       = 0
    //   row 3:  0                       = 0
    let x = [1.0, 0.0, 0.0, 0.0];
    let mut y = vec![0.0; 4];
    sparse.matvec(&x, &mut y);

    assert_eq!(y, vec![1.0, -1.0, 0.0, 0.0]);
}

/// Converting a non-square matrix should panic.
#[test]
#[should_panic(expected = "square")]
fn faer_non_square_panics() {
    let row_ptrs = vec![0u32, 1, 2, 3];
    let col_indices = vec![0u32, 1, 0];
    let values = vec![1.0, 1.0, 1.0];
    let symbolic = unsafe {
        faer::sparse::SymbolicSparseRowMat::<u32>::new_unchecked(3, 4, row_ptrs, None, col_indices)
    };
    let mat = SparseRowMat::new(symbolic, values);
    let _: SparseMatrix = mat.as_ref().into();
}
