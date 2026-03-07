use within::{
    demean_batch, solve_batch, solve_least_squares, FixedEffectsDesign, SolverParams, WithinError,
};

#[test]
fn test_solve_batch_matches_single_column_solves() {
    let design = FixedEffectsDesign::new(
        vec![vec![0, 1, 0, 1, 2], vec![0, 0, 1, 1, 0]],
        vec![3, 2],
        5,
    )
    .expect("valid design");

    let columns = vec![
        vec![1.0, 2.0, 3.0, 4.0, 5.0],
        vec![2.0, 1.0, 0.0, -1.0, -2.0],
    ];
    let params = SolverParams::default();

    let batch = solve_batch(&design, &columns, &params).expect("solve batch");
    assert_eq!(batch.x.len(), columns.len());
    assert!(batch.time_total >= 0.0);
    assert!(batch.time_setup >= 0.0);

    for (idx, y_col) in columns.iter().enumerate() {
        let single = solve_least_squares(&design, y_col, None, &params).expect("single solve");
        assert_eq!(batch.x[idx].len(), single.x.len());
        assert!(batch.time_solve[idx] >= 0.0);
        for (a, b) in batch.x[idx].iter().zip(single.x.iter()) {
            assert!((a - b).abs() < 1e-9, "x mismatch: {a} vs {b}");
        }
        assert_eq!(batch.converged[idx], single.converged);
        assert_eq!(batch.iterations[idx], single.iterations);
        assert!(
            (batch.final_residual[idx] - single.final_residual).abs() < 1e-12,
            "residual mismatch: {} vs {}",
            batch.final_residual[idx],
            single.final_residual
        );
    }
}

#[test]
fn test_demean_batch_matches_single_column_solves() {
    let design = FixedEffectsDesign::new(
        vec![vec![0, 1, 0, 1, 2], vec![0, 0, 1, 1, 0]],
        vec![3, 2],
        5,
    )
    .expect("valid design");

    let columns = vec![
        vec![1.0, 2.0, 3.0, 4.0, 5.0],
        vec![2.0, 1.0, 0.0, -1.0, -2.0],
    ];
    let params = SolverParams::default();

    let batch = demean_batch(&design, &columns, &params).expect("demean batch");
    assert_eq!(batch.y_demean.len(), columns.len());
    assert!(batch.time_total >= 0.0);
    assert!(batch.time_setup >= 0.0);

    for (idx, y_col) in columns.iter().enumerate() {
        let solve = solve_least_squares(&design, y_col, None, &params).expect("single solve");
        let mut fitted = vec![0.0; y_col.len()];
        design.matvec_d(&solve.x, &mut fitted);
        let expected_demean: Vec<f64> = y_col
            .iter()
            .zip(fitted.iter())
            .map(|(y, f)| y - f)
            .collect();

        assert_eq!(batch.y_demean[idx].len(), y_col.len());
        assert!(batch.y_demean[idx].iter().all(|v| v.is_finite()));
        assert!(batch.time_solve[idx] >= 0.0);

        for (a, b) in batch.y_demean[idx].iter().zip(expected_demean.iter()) {
            assert!((a - b).abs() < 1e-9, "demeaned mismatch: {a} vs {b}");
        }
        assert_eq!(batch.converged[idx], solve.converged);
        assert_eq!(batch.iterations[idx], solve.iterations);
        assert!(
            (batch.final_residual[idx] - solve.final_residual).abs() < 1e-12,
            "residual mismatch: {} vs {}",
            batch.final_residual[idx],
            solve.final_residual
        );
    }
}

#[test]
fn test_demean_batch_rhs_length_mismatch_returns_error() {
    let design = FixedEffectsDesign::new(
        vec![vec![0, 1, 0, 1, 2], vec![0, 0, 1, 1, 0]],
        vec![3, 2],
        5,
    )
    .expect("valid design");
    let params = SolverParams::default();

    let err = demean_batch(&design, &[vec![1.0, 2.0, 3.0]], &params).expect_err("must fail");
    assert!(matches!(
        err,
        WithinError::RhsCountMismatch {
            expected: 5,
            got: 3
        }
    ));
}
