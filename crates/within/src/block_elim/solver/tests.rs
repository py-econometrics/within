use super::*;

use crate::block_elim::csr_matrix::CsrMatrix;
use crate::config::{ApproxCholConfig, SchurMode, DEFAULT_DENSE_SCHUR_THRESHOLD};
use crate::csr_block::CsrBlock;

#[test]
fn test_subtract_mean_empty() {
    let mut data = vec![1.0, 2.0, 3.0];
    subtract_mean(&mut data, 0);
    // Should not modify anything
    assert_eq!(data, vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_subtract_mean_basic() {
    let mut data = vec![2.0, 4.0, 6.0];
    subtract_mean(&mut data, 3);
    // mean = 4.0
    assert!((data[0] - (-2.0)).abs() < 1e-14);
    assert!((data[1] - 0.0).abs() < 1e-14);
    assert!((data[2] - 2.0).abs() < 1e-14);
}

#[test]
fn test_subtract_mean_partial() {
    let mut data = vec![3.0, 5.0, 100.0];
    subtract_mean(&mut data, 2);
    // mean of first 2 = 4.0
    assert!((data[0] - (-1.0)).abs() < 1e-14);
    assert!((data[1] - 1.0).abs() < 1e-14);
    assert_eq!(data[2], 100.0); // unchanged
}

#[test]
fn fold_inverts_the_eliminated_diagonal() {
    // No cross entries, so the fold reads the diagonal alone.
    let matrix = SddmMatrix::from_dense_for_test(
        &[0.0; 6],
        3,
        2,
        vec![5.0, 6.0, 8.0, 7.0, 9.0],
        Grounding::Grounded,
    );
    let eliminated = Eliminated::new(matrix).expect("a positive diagonal must fold");

    assert_eq!(eliminated.inv_diagonal.len(), 3);
    for (&got, &expected) in eliminated
        .inv_diagonal
        .iter()
        .zip([1.0 / 5.0, 1.0 / 6.0, 1.0 / 8.0].iter())
    {
        assert!((got - expected).abs() < 1e-12);
    }
}

#[test]
fn fold_rejects_a_zero_eliminated_diagonal() {
    let matrix = SddmMatrix::from_dense_for_test(
        &[0.0; 6],
        3,
        2,
        vec![5.0, 6.0, 0.0, 8.0, 9.0],
        Grounding::Grounded,
    );

    match Eliminated::new(matrix) {
        Err(BuildError::SingularDiagonal { index: 2, .. }) => {}
        Err(e) => panic!("expected SingularDiagonal at index 2, got: {e}"),
        Ok(_) => panic!("expected SingularDiagonal error, got Ok"),
    }
}

#[test]
fn an_unusable_dense_pivot_is_retried_rather_than_fatal() {
    // The exact attempt only fails when weight spread costs the complement its definiteness.
    let matrix =
        SddmMatrix::laplacian_for_test(&[0.0, 1e-16, 1e2, 1e0, 1e5, 0.0, 0.0, 1e14, 0.0], 3, 3);
    let eliminated = Eliminated::new(matrix).expect("a positive diagonal must fold");
    let exact = schur::exact_for_factor(&eliminated.matrix, &eliminated.inv_diagonal);
    let exact_only = ApproxCholConfig::default()
        .to_approx_chol(DEFAULT_DENSE_SCHUR_THRESHOLD, ExactFailure::Error);
    assert!(
        matches!(
            factor_sparse(&exact, exact_only),
            Err(approx_chol::Error::DenseFactorizationFailed { .. })
        ),
        "the fixture no longer reaches the fall-through"
    );

    for schur in [SchurMode::Exact, SchurMode::Approximate(Default::default())] {
        let config = LocalSolverConfig {
            approx_chol: ApproxCholConfig::default(),
            schur,
            dense_threshold: DEFAULT_DENSE_SCHUR_THRESHOLD,
            scaling: Default::default(),
        };

        assert!(
            eliminated.factor_reduced(&config).is_ok(),
            "{:?}: an unusable pivot must not be fatal",
            config.schur
        );
    }
}

/// Build an eliminated-major CrossTab (`n_rows > n_cols`, as orientation
/// guarantees) whose two cross entries leave three isolated rows, plus its
/// build-time diagonal.
fn make_cross_tab() -> (CrossTab, Vec<f64>) {
    let c_dense = vec![
        1.0, 0.0, // row 0
        0.0, 1.0, // row 1
        0.0, 0.0, // row 2
        0.0, 0.0, // row 3
        0.0, 0.0, // row 4
    ];
    let c = CsrBlock::from_dense_table(&c_dense, 5, 2);
    let ct = c.transpose();
    let diagonal = vec![2.0, 3.0, 1.0, 1.0, 1.0, 2.0, 3.0];
    (CrossTab { c, ct }, diagonal)
}

#[test]
fn block_elim_solver_solves_a_two_block_component() {
    let (cross_tab, diagonals) = make_cross_tab();
    assert_eq!(cross_tab.n_rows(), 5);
    assert_eq!(cross_tab.n_cols(), 2);

    let config = LocalSolverConfig {
        approx_chol: ApproxCholConfig::default(),
        schur: SchurMode::Exact,
        dense_threshold: 0, // disable dense fast path to ensure sparse path is covered
        scaling: Default::default(),
    };
    let component = LocalComponent::general_for_test(cross_tab, diagonals);
    let solver = BlockElimSolver::build(component, &config).expect("block-elim build failed");

    let n_local = solver.n_local();
    assert_eq!(n_local, 7);

    let scratch_sz = solver.scratch_size();
    let mut rhs = vec![0.0; scratch_sz];
    for (i, v) in rhs[..n_local].iter_mut().enumerate() {
        *v = (i as f64 + 1.0) * 0.5;
    }
    let mut sol = vec![0.0; scratch_sz];

    solver
        .solve_local(&mut rhs, &mut sol, true)
        .expect("solve_local should succeed");

    for (i, &v) in sol[..n_local].iter().enumerate() {
        assert!(v.is_finite(), "sol[{i}] = {v} is not finite");
    }
    let sol_norm: f64 = sol[..n_local].iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(sol_norm > 1e-15, "solution is unexpectedly all-zero");
}

#[test]
fn grounded_two_block_solve_is_leak_free() {
    // Grounded sparse factor: pins no-leak (pre-dirtying must not move the result).
    let (cross_tab, diagonals) = make_cross_tab();
    let config = LocalSolverConfig {
        approx_chol: ApproxCholConfig::default(),
        schur: SchurMode::Exact,
        dense_threshold: 0, // force the sparse grounded path → explicit ground vertex
        scaling: Default::default(),
    };
    let component = LocalComponent::general_for_test(cross_tab, diagonals);
    let solver = BlockElimSolver::build(component, &config).expect("block-elim build failed");
    // Original bipartite Gram A the solver inverts (diagonals + the two cross entries).
    let n = solver.n_local();
    let mut a = vec![vec![0.0; n]; n];
    for (i, d) in [2.0, 3.0, 1.0, 1.0, 1.0, 2.0, 3.0].into_iter().enumerate() {
        a[i][i] = d;
    }
    for (i, j) in [(0, 5), (1, 6)] {
        a[i][j] = 1.0;
        a[j][i] = 1.0;
    }

    let r = [0.5, 3.0, -1.25, 0.75, 2.0, 1.0, -2.0];
    let solve_with_dirty = |dirty: f64| {
        let mut rhs = vec![dirty; solver.scratch_size()];
        rhs[..n].copy_from_slice(&r);
        let mut sol = vec![dirty; solver.scratch_size()];
        solver
            .solve_local(&mut rhs, &mut sol, false)
            .expect("solve_local failed");
        sol
    };
    let clean = solve_with_dirty(0.0);
    let dirty = solve_with_dirty(1e6);
    assert_eq!(
        clean[..n],
        dirty[..n],
        "pre-dirtied overlap/scratch slots leaked into the solution",
    );

    for i in 0..n {
        let ax: f64 = (0..n).map(|j| a[i][j] * clean[j]).sum();
        assert!(
            (ax - r[i]).abs() < 1e-9,
            "row {i}: A·x = {ax}, expected {}",
            r[i],
        );
    }
}

#[test]
fn trivial_singleton_component_solves_r_over_d() {
    // Live 1×1 components keep n_keep = 0, so the solve degenerates to x = r/d exactly.
    let config = LocalSolverConfig {
        approx_chol: ApproxCholConfig::default(),
        schur: SchurMode::Exact,
        dense_threshold: 0,
        scaling: Default::default(),
    };
    for (n_rows, n_cols) in [(1usize, 0usize), (0, 1)] {
        let c = CsrBlock::from_dense_table(&[], n_rows, n_cols);
        let ct = c.transpose();
        let diagonal = vec![4.0; n_rows + n_cols];
        let component = LocalComponent::general_for_test(CrossTab { c, ct }, diagonal);
        let solver = BlockElimSolver::build(component, &config).expect("trivial 1×1 build");
        assert_eq!(solver.n_local(), 1);

        let mut rhs = vec![0.0; solver.scratch_size()];
        rhs[0] = 2.0;
        let mut sol = vec![0.0; solver.scratch_size()];
        solver
            .solve_local(&mut rhs, &mut sol, false)
            .expect("trivial solve");
        assert_eq!(
            sol[0], 0.5,
            "n_rows={n_rows}, n_cols={n_cols}: expected r/d"
        );
    }
}

#[test]
fn sampled_sparse_preserves_barely_pd_direction() {
    let surplus = 5e-10;
    let c = CsrBlock::from_dense_table(&[1.0], 1, 1);
    let ct = c.transpose();
    let component = LocalComponent::general_for_test(CrossTab { c, ct }, vec![1.0 + surplus, 1.0]);
    let config = LocalSolverConfig {
        approx_chol: ApproxCholConfig::default(),
        schur: SchurMode::Approximate(crate::config::ApproxSchurConfig::default()),
        dense_threshold: 0,
        scaling: Default::default(),
    };
    let solver = BlockElimSolver::build(component, &config).unwrap();
    let mut rhs = vec![0.0; solver.scratch_size()];
    rhs[..2].copy_from_slice(&[1.0, -1.0]);
    let mut solution = vec![0.0; solver.scratch_size()];
    solver.solve_local(&mut rhs, &mut solution, false).unwrap();

    let ax = [
        (1.0 + surplus) * solution[0] + solution[1],
        solution[0] + solution[1],
    ];
    assert!((ax[0] - 1.0).abs() < 1e-6);
    assert!((ax[1] + 1.0).abs() < 1e-6);
}

#[test]
fn grounded_backend_auxiliary_is_initialized_on_every_solve() {
    let large = 1e7;
    let small = 1e-9;
    let principal = CsrMatrix::new(
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![large, -large, -large, large + small],
        2,
    );
    let explicit_ground_laplacian =
        schur::build_explicit_laplacian(&principal, &[0.0, small], Grounding::Grounded);
    let config = ApproxCholConfig::default().to_approx_chol(
        DEFAULT_DENSE_SCHUR_THRESHOLD,
        ExactFailure::FallBackToApproximate,
    );
    let factor = ReducedFactor::Direct {
        factor: factor_sparse(&explicit_ground_laplacian, config)
            .expect("factorization must succeed"),
        grounding: Grounding::Grounded,
    };
    assert_eq!(factor.input_dimension(), 3);
    // approx-chol declines to ground a surplus below its resolvable pivot scale.
    assert_eq!(factor.solve_dimension(), 3);

    let c = CsrBlock::from_dense_table(&[0.0; 6], 3, 2);
    let cross_tab = CrossTab {
        ct: c.transpose(),
        c,
    };
    let solver = BlockElimSolver::new(cross_tab, vec![1.0; 3], factor, CoordinateMap::default());

    let solve_with_dirty_auxiliary = |dirty: f64| {
        let mut rhs = vec![0.0; solver.scratch_size()];
        rhs[3..5].copy_from_slice(&[1.0, -1.0]);
        rhs[7] = dirty;
        let mut solution = vec![0.0; solver.scratch_size()];
        solver.solve_local(&mut rhs, &mut solution, false).unwrap();
        solution
    };

    let first = solve_with_dirty_auxiliary(3.0);
    let second = solve_with_dirty_auxiliary(-7.0);
    assert_eq!(first[..solver.n_local()], second[..solver.n_local()]);
}

#[test]
fn signed_component_realizes_congruence_transformed_solve() {
    // Stars of ≤2 entries sample deterministically-exact, so all three arms share one oracle.
    let (n_rows, n_cols) = (3usize, 2usize);
    let d: Vec<f64> = vec![-1.0, 4.0, 0.25, 2.0, -0.5];
    let c_hat = [[1.0, 3.0], [2.0, 0.0], [0.5, 1.5]];
    let diag_hat = [4.0, 2.5, 2.0, 4.0, 5.0]; // surplus on q0, q2, r0

    let mut c_raw = vec![0.0; n_rows * n_cols];
    for i in 0..n_rows {
        for j in 0..n_cols {
            c_raw[i * n_cols + j] = c_hat[i][j] / (d[i] * d[n_rows + j]);
        }
    }

    for (label, dense_threshold, schur) in [
        ("dense full minor", 8, SchurMode::Exact),
        ("exact sparse", 0, SchurMode::Exact),
        (
            "sampled sparse",
            0,
            SchurMode::Approximate(crate::config::ApproxSchurConfig::default()),
        ),
    ] {
        let c = CsrBlock::from_dense_table(&c_raw, n_rows, n_cols);
        let ct = c.transpose();
        let diagonals: Vec<f64> = (0..n_rows + n_cols)
            .map(|k| diag_hat[k] / (d[k] * d[k]))
            .collect();
        let config = LocalSolverConfig {
            approx_chol: ApproxCholConfig::default(),
            schur,
            dense_threshold,
            scaling: Default::default(),
        };
        // SDDM factors fold the bipartite negation in: f = d on q, -d on r.
        let factors: Vec<f64> = d
            .iter()
            .enumerate()
            .map(|(i, &v)| if i < n_rows { v } else { -v })
            .collect();
        let component =
            LocalComponent::with_factors_for_test(CrossTab { c, ct }, diagonals, &factors);
        let solver =
            BlockElimSolver::build(component, &config).expect("signed block-elim build failed");

        let n = n_rows + n_cols;
        let r = [0.5, 3.0, -1.25, 1.0, -2.0];
        let mut rhs = vec![0.0; solver.scratch_size()];
        rhs[..n].copy_from_slice(&r);
        let mut sol = vec![0.0; solver.scratch_size()];
        solver
            .solve_local(&mut rhs, &mut sol, false)
            .expect("solve_local failed");

        for i in 0..n {
            let mut ax = 0.0;
            for j in 0..n {
                let a_ij = if i == j {
                    diag_hat[i] / (d[i] * d[i])
                } else if i < n_rows && j >= n_rows {
                    c_raw[i * n_cols + (j - n_rows)]
                } else if j < n_rows && i >= n_rows {
                    c_raw[j * n_cols + (i - n_rows)]
                } else {
                    0.0
                };
                ax += a_ij * sol[j];
            }
            assert!(
                (ax - r[i]).abs() < 1e-9,
                "{label}: row {i}: A·x = {ax}, expected {}",
                r[i]
            );
        }
    }
}

#[test]
fn frustrated_component_solves_exactly_through_cover() {
    // Dense only: it is the one arm whose factor is exact, so the oracle A·x = r is sharp.
    let (n_rows, n_cols) = (2usize, 2usize);
    let c_raw = [1.0, 1.5, 2.0, -1.0];
    let a = [
        [2.5, 0.0, 1.0, 1.5],
        [0.0, 3.0, 2.0, -1.0],
        [1.0, 2.0, 3.4, 0.0],
        [1.5, -1.0, 0.0, 2.9],
    ];

    let c = CsrBlock::from_dense_table(&c_raw, n_rows, n_cols);
    let ct = c.transpose();
    let component = LocalComponent::general_for_test(
        CrossTab { c, ct },
        vec![a[0][0], a[1][1], a[2][2], a[3][3]],
    );
    let config = LocalSolverConfig {
        approx_chol: ApproxCholConfig::default(),
        schur: SchurMode::Exact,
        dense_threshold: 8,
        scaling: Default::default(),
    };
    let solver =
        BlockElimSolver::build(component, &config).expect("covered block-elim build failed");
    let n = n_rows + n_cols;
    // #91: the stored operator stays single-sized; the cover lives inside the reduced factor.
    assert_eq!(solver.n_local(), n, "operator stays single-sized");
    assert_eq!(
        solver.cross_tab.n_local(),
        n,
        "stored cross-tab not doubled"
    );
    assert!(
        matches!(solver.reduced_factor, ReducedFactor::Cover { .. }),
        "frustrated component must reduce through a cover",
    );

    let r = [1.0, -2.0, 0.5, 3.0];
    let mut rhs = vec![0.0; solver.scratch_size()];
    rhs[..n].copy_from_slice(&r);
    let mut sol = vec![0.0; solver.scratch_size()];
    solver
        .solve_local(&mut rhs, &mut sol, false)
        .expect("solve_local failed");

    for i in 0..n {
        let ax: f64 = (0..n).map(|j| a[i][j] * sol[j]).sum();
        assert!(
            (ax - r[i]).abs() < 1e-9,
            "row {i}: A·x = {ax}, expected {}",
            r[i]
        );
    }
}

/// A structurally valid solver built the normal way; the deserialize-validation
/// tests below tamper one field of a clone and assert the round-trip rejects it.
fn valid_solver_for_deser() -> BlockElimSolver {
    let (cross_tab, diagonals) = make_cross_tab();
    let config = LocalSolverConfig {
        approx_chol: ApproxCholConfig::default(),
        schur: SchurMode::Exact,
        dense_threshold: 0,
        scaling: Default::default(),
    };
    let component = LocalComponent::general_for_test(cross_tab, diagonals);
    BlockElimSolver::build(component, &config).expect("block-elim build failed")
}

#[test]
fn valid_solver_round_trips() {
    let solver = valid_solver_for_deser();
    let bytes = postcard::to_stdvec(&solver).expect("serialize");
    let restored: BlockElimSolver = postcard::from_bytes(&bytes).expect("deserialize");
    assert_eq!(restored.n_local(), solver.n_local());
    assert_eq!(restored.scratch_size(), solver.scratch_size());
}

#[test]
fn inv_diag_elim_length_mismatch_is_rejected() {
    let mut bad = valid_solver_for_deser();
    bad.inv_diag_elim.push(0.0);
    let bytes = postcard::to_stdvec(&bad).expect("serialize");
    assert!(postcard::from_bytes::<BlockElimSolver>(&bytes).is_err());
}

#[test]
fn scaled_coordinate_length_mismatch_is_rejected() {
    let mut bad = valid_solver_for_deser();
    bad.coordinates = CoordinateMap::Scaled(vec![1.0; bad.n_internal + 3].into_boxed_slice());
    let bytes = postcard::to_stdvec(&bad).expect("serialize");
    assert!(postcard::from_bytes::<BlockElimSolver>(&bytes).is_err());
}

#[test]
fn kept_block_wider_than_the_reduced_factor_is_rejected() {
    let mut bad = valid_solver_for_deser();
    // Earlier cross-field witnesses stay intact, so the factor's span is the check under test.
    let mut c = bad.cross_tab.c.clone();
    c.ncols += 2;
    let ct = c.transpose();
    bad.cross_tab = Arc::new(CrossTab { c, ct });
    let bytes = postcard::to_stdvec(&bad).expect("serialize");
    assert!(postcard::from_bytes::<BlockElimSolver>(&bytes).is_err());
}
