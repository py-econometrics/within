use rstest::rstest;

use super::*;

use crate::block_elim::csr_matrix::CsrMatrix;
use crate::config::{ApproxCholConfig, SchurMode, DEFAULT_DENSE_SCHUR_THRESHOLD};

impl BlockElimSolver {
    /// Solves `r` with the workspace preset to `stale`.
    pub(crate) fn solve_for_test(&self, r: &[f64], stale: f64) -> Vec<f64> {
        let mut x = r.to_vec();
        let mut workspace = vec![stale; self.workspace_len()];
        self.solve_local(&mut x, &mut workspace, false)
            .expect("solve_local failed");
        x
    }
}

#[test]
fn test_subtract_mean_empty() {
    let mut data: Vec<f64> = Vec::new();
    subtract_mean(&mut data);
    assert!(data.is_empty());
}

#[test]
fn test_subtract_mean_basic() {
    let mut data = vec![2.0, 4.0, 6.0];
    subtract_mean(&mut data);
    // mean = 4.0
    assert!((data[0] - (-2.0)).abs() < 1e-14);
    assert!((data[1] - 0.0).abs() < 1e-14);
    assert!((data[2] - 2.0).abs() < 1e-14);
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

#[rstest]
fn an_unusable_dense_pivot_is_retried_rather_than_fatal(
    #[values(SchurMode::Exact, SchurMode::Approximate(Default::default()))] schur: SchurMode,
) {
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

    let config = LocalSolverConfig {
        approx_chol: ApproxCholConfig::default(),
        schur,
        dense_threshold: DEFAULT_DENSE_SCHUR_THRESHOLD,
        ..Default::default()
    };
    assert!(
        Eliminated::factor_reduced(&eliminated, &config).is_ok(),
        "an unusable pivot must not be fatal"
    );
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
    let diagonal = vec![2.0, 3.0, 1.0, 1.0, 1.0, 2.0, 3.0];
    (CrossTab::from_dense_for_test(&c_dense, 5, 2), diagonal)
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
        ..Default::default()
    };
    let component = LocalComponent::general_for_test(cross_tab, diagonals);
    let solver = BlockElimSolver::build(component, &config).expect("block-elim build failed");

    let n_local = solver.n_local();
    assert_eq!(n_local, 7);

    let r: Vec<f64> = (0..n_local).map(|i| (i as f64 + 1.0) * 0.5).collect();
    let sol = solver.solve_for_test(&r, 0.0);

    for (i, &v) in sol.iter().enumerate() {
        assert!(v.is_finite(), "sol[{i}] = {v} is not finite");
    }
    let sol_norm: f64 = sol.iter().map(|v| v * v).sum::<f64>().sqrt();
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
        ..Default::default()
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
    let clean = solver.solve_for_test(&r, 0.0);
    let dirty = solver.solve_for_test(&r, 1e6);
    assert_eq!(
        clean, dirty,
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

/// Live 1×1 components keep n_keep = 0, so the solve degenerates to x = r/d exactly.
#[rstest]
#[case::row_singleton(1, 0)]
#[case::column_singleton(0, 1)]
fn trivial_singleton_component_solves_r_over_d(#[case] n_rows: usize, #[case] n_cols: usize) {
    let config = LocalSolverConfig {
        approx_chol: ApproxCholConfig::default(),
        schur: SchurMode::Exact,
        dense_threshold: 0,
        ..Default::default()
    };
    let cross_tab = CrossTab::from_dense_for_test(&[], n_rows, n_cols);
    let diagonal = vec![4.0; n_rows + n_cols];
    let component = LocalComponent::general_for_test(cross_tab, diagonal);
    let solver = BlockElimSolver::build(component, &config).expect("trivial 1×1 build");
    assert_eq!(solver.n_local(), 1);

    let sol = solver.solve_for_test(&[2.0], 0.0);
    assert_eq!(sol[0], 0.5, "expected r/d");
}

#[test]
fn sampled_sparse_preserves_barely_pd_direction() {
    let surplus = 5e-10;
    let cross_tab = CrossTab::from_dense_for_test(&[1.0], 1, 1);
    let component = LocalComponent::general_for_test(cross_tab, vec![1.0 + surplus, 1.0]);
    let config = LocalSolverConfig {
        approx_chol: ApproxCholConfig::default(),
        schur: SchurMode::Approximate(crate::config::ApproxSchurConfig::default()),
        dense_threshold: 0,
        ..Default::default()
    };
    let solver = BlockElimSolver::build(component, &config).unwrap();
    let solution = solver.solve_for_test(&[1.0, -1.0], 0.0);

    let ax = [
        (1.0 + surplus) * solution[0] + solution[1],
        solution[0] + solution[1],
    ];
    assert!((ax[0] - 1.0).abs() < 1e-6);
    assert!((ax[1] + 1.0).abs() < 1e-6);
}

fn two_dof_sddm(diagonal: [f64; 2], off_diagonal: f64) -> CsrMatrix {
    let data = vec![diagonal[0], off_diagonal, off_diagonal, diagonal[1]];
    CsrMatrix::new(vec![0, 2, 4], vec![0, 1, 0, 1], data, 2)
}

fn explicit_ground_laplacian() -> CsrMatrix {
    let (large, small) = (1e7, 1e-9);
    let principal = two_dof_sddm([large, large + small], -large);
    schur::build_explicit_laplacian(&principal, &[0.0, small], Grounding::Grounded)
}

#[rstest]
// approx-chol declines to ground a surplus below its resolvable pivot scale.
#[case::explicit_ground(explicit_ground_laplacian(), 3)]
// A resolvable surplus makes approx-chol append a ground vertex of its own.
#[case::backend_ground(two_dof_sddm([2.5, 2.5], -2.0), 2)]
fn stale_ground_slots_past_the_kept_block_never_reach_the_solution(
    #[case] reduced: CsrMatrix,
    #[case] input_dimension: usize,
    #[values(3.0, -7.0)] stale: f64,
) {
    let config = ApproxCholConfig::default().to_approx_chol(
        DEFAULT_DENSE_SCHUR_THRESHOLD,
        ExactFailure::FallBackToApproximate,
    );
    let factor = ReducedFactor::Direct {
        factor: factor_sparse(&reduced, config).expect("factorization must succeed"),
        grounding: Grounding::Grounded,
    };
    assert_eq!(
        (factor.input_dimension(), factor.solve_dimension()),
        (input_dimension, 3)
    );

    let cross_tab = CrossTab::from_dense_for_test(&[0.0; 6], 3, 2);
    let solver = BlockElimSolver::new(cross_tab, vec![1.0; 3], factor, CoordinateMap::default());

    let r = [0.0, 0.0, 0.0, 1.0, -1.0];
    assert_eq!(
        solver.solve_for_test(&r, stale),
        solver.solve_for_test(&r, 0.0)
    );
}

/// Stars of ≤2 entries sample deterministically-exact, so all three arms share one oracle.
#[rstest]
#[case::dense_full_minor(8, SchurMode::Exact)]
#[case::exact_sparse(0, SchurMode::Exact)]
#[case::sampled_sparse(0, SchurMode::Approximate(crate::config::ApproxSchurConfig::default()))]
fn signed_component_realizes_congruence_transformed_solve(
    #[case] dense_threshold: usize,
    #[case] schur: SchurMode,
) {
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

    let cross_tab = CrossTab::from_dense_for_test(&c_raw, n_rows, n_cols);
    let diagonals: Vec<f64> = (0..n_rows + n_cols)
        .map(|k| diag_hat[k] / (d[k] * d[k]))
        .collect();
    let config = LocalSolverConfig {
        approx_chol: ApproxCholConfig::default(),
        schur,
        dense_threshold,
        ..Default::default()
    };
    // SDDM factors fold the bipartite negation in: f = d on q, -d on r.
    let factors: Vec<f64> = d
        .iter()
        .enumerate()
        .map(|(i, &v)| if i < n_rows { v } else { -v })
        .collect();
    let component = LocalComponent::with_factors_for_test(cross_tab, diagonals, &factors);
    let solver =
        BlockElimSolver::build(component, &config).expect("signed block-elim build failed");

    let n = n_rows + n_cols;
    let r = [0.5, 3.0, -1.25, 1.0, -2.0];
    let sol = solver.solve_for_test(&r, 0.0);

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
            "row {i}: A·x = {ax}, expected {}",
            r[i]
        );
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

    let component = LocalComponent::general_for_test(
        CrossTab::from_dense_for_test(&c_raw, n_rows, n_cols),
        vec![a[0][0], a[1][1], a[2][2], a[3][3]],
    );
    let config = LocalSolverConfig {
        approx_chol: ApproxCholConfig::default(),
        schur: SchurMode::Exact,
        dense_threshold: 8,
        ..Default::default()
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
    let sol = solver.solve_for_test(&r, 0.0);

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
        ..Default::default()
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
    assert_eq!(restored.workspace_len(), solver.workspace_len());
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
    bad.cross_tab = Arc::new(CrossTab::eager(c));
    let bytes = postcard::to_stdvec(&bad).expect("serialize");
    assert!(postcard::from_bytes::<BlockElimSolver>(&bytes).is_err());
}
