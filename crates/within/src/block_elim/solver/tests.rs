use super::*;

use crate::block_elim::csr_matrix::CsrMatrix;
use crate::config::{ApproxCholConfig, SchurMode};
use crate::csr_block::CsrBlock;
use crate::domain::BlockDiagonals;

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

/// Build a CrossTab with `n_q < r` so that `eliminate_q == false`, plus its
/// build-time diagonals.
fn make_cross_tab_q_lt_r() -> (CrossTab, BlockDiagonals) {
    let c_dense = vec![
        // row 0
        1.0, 0.0, 0.0, 0.0, 0.0, // row 1
        0.0, 1.0, 0.0, 0.0, 0.0,
    ];
    let c = CsrBlock::from_dense_table(&c_dense, 2, 5);
    let ct = c.transpose();
    let diagonals = BlockDiagonals {
        q: vec![2.0, 3.0],
        r: vec![2.0, 3.0, 1.0, 1.0, 1.0],
    };
    (CrossTab { c, ct }, diagonals)
}

#[test]
fn test_block_elim_solver_eliminate_q_false() {
    let (cross_tab, diagonals) = make_cross_tab_q_lt_r();
    assert_eq!(cross_tab.n_q(), 2);
    assert_eq!(cross_tab.n_r(), 5);

    let config = LocalSolverConfig {
        approx_chol: ApproxCholConfig::default(),
        schur: SchurMode::Exact,
        dense_threshold: 0, // disable dense fast path to ensure sparse path is covered
        scaling: Default::default(),
    };
    let component = LocalComponent::general_for_test(cross_tab, diagonals);
    let solver = BlockElimSolver::build(component, &config).expect("block-elim build failed");

    assert!(
        !solver.eliminate_q,
        "expected eliminate_q=false when n_q < n_r",
    );
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
fn trivial_singleton_component_solves_r_over_d() {
    // Live 1×1 components (positive diagonal, cancelled cross row) keep
    // n_keep = 0: the whole solve must degenerate to x = r/d exactly, in
    // both block orientations.
    let config = LocalSolverConfig {
        approx_chol: ApproxCholConfig::default(),
        schur: SchurMode::Exact,
        dense_threshold: 0,
        scaling: Default::default(),
    };
    for (n_q, n_r) in [(1usize, 0usize), (0, 1)] {
        let c = CsrBlock::from_dense_table(&[], n_q, n_r);
        let ct = c.transpose();
        let diagonals = BlockDiagonals {
            q: vec![4.0; n_q],
            r: vec![4.0; n_r],
        };
        let component = LocalComponent::general_for_test(CrossTab { c, ct }, diagonals);
        let solver = BlockElimSolver::build(component, &config).expect("trivial 1×1 build");
        assert_eq!(solver.n_local(), 1);

        let mut rhs = vec![0.0; solver.scratch_size()];
        rhs[0] = 2.0;
        let mut sol = vec![0.0; solver.scratch_size()];
        solver
            .solve_local(&mut rhs, &mut sol, false)
            .expect("trivial solve");
        assert_eq!(sol[0], 0.5, "n_q={n_q}, n_r={n_r}: expected r/d");
    }
}

#[test]
fn sampled_sparse_preserves_barely_pd_direction() {
    let surplus = 5e-10;
    let c = CsrBlock::from_dense_table(&[1.0], 1, 1);
    let ct = c.transpose();
    let component = LocalComponent::general_for_test(
        CrossTab { c, ct },
        BlockDiagonals {
            q: vec![1.0 + surplus],
            r: vec![1.0],
        },
    );
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
        schur::build_explicit_laplacian(&principal, &[0.0, small], SolveSpace::Grounded);
    let factor = factor_sparse(&explicit_ground_laplacian, ApproxCholConfig::default())
        .expect("factorization must succeed");
    assert_eq!(factor.input_dimension(), 3);
    assert_eq!(factor.factor_dimension(), 4);

    let c = CsrBlock::from_dense_table(&[0.0; 6], 3, 2);
    let cross_tab = CrossTab {
        ct: c.transpose(),
        c,
    };
    let solver = BlockElimSolver::new(
        cross_tab,
        vec![1.0; 3],
        factor,
        true,
        CoordinateMap::default(),
        SolveSpace::Grounded,
    );

    let solve_with_dirty_auxiliary = |dirty: f64| {
        let mut rhs = vec![0.0; solver.scratch_size()];
        rhs[3..5].copy_from_slice(&[1.0, -1.0]);
        rhs[8] = dirty;
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
    // Balanced/scalable signed component, constructed backward from a plain
    // strictly-SDD target Â and a mixed-sign congruence d: the solver gets
    // the raw signed Gram A = D⁻¹·Â·D⁻¹ plus the descriptor and must
    // realize x = D·Â⁻¹·D·r — checked as A·x = r (Â nonsingular, so the
    // pseudo-solve is the exact inverse and every RHS is in range).
    //
    // All three reduction arms realize the identical solve: the surplus
    // sits on the kept block and one eliminated row whose star (with its
    // ground entry) has two entries, and the other eliminated stars are
    // exactly balanced two-entry stars — clique sampling of ≤2-entry stars
    // is deterministic-exact, so the sampled arm admits the exact oracle.
    let (n_q, n_r) = (2usize, 3usize);
    let d: Vec<f64> = vec![2.0, -0.5, -1.0, 4.0, 0.25];
    let c_hat = [[1.0, 2.0, 0.5], [3.0, 0.0, 1.5]];
    let diag_hat = [4.0, 5.0, 4.0, 2.5, 2.0]; // surplus on q0, q1, r1

    let mut c_raw = vec![0.0; n_q * n_r];
    for i in 0..n_q {
        for j in 0..n_r {
            c_raw[i * n_r + j] = c_hat[i][j] / (d[i] * d[n_q + j]);
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
        let c = CsrBlock::from_dense_table(&c_raw, n_q, n_r);
        let ct = c.transpose();
        let diagonals = BlockDiagonals {
            q: (0..n_q).map(|k| diag_hat[k] / (d[k] * d[k])).collect(),
            r: (n_q..n_q + n_r)
                .map(|k| diag_hat[k] / (d[k] * d[k]))
                .collect(),
        };
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
            .map(|(i, &v)| if i < n_q { v } else { -v })
            .collect();
        let component =
            LocalComponent::with_factors_for_test(CrossTab { c, ct }, diagonals, &factors);
        let solver =
            BlockElimSolver::build(component, &config).expect("signed block-elim build failed");

        let n = n_q + n_r;
        let r = [1.0, -2.0, 0.5, 3.0, -1.25];
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
                } else if i < n_q && j >= n_q {
                    c_raw[i * n_r + (j - n_q)]
                } else if j < n_q && i >= n_q {
                    c_raw[j * n_r + (i - n_q)]
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
    // Frustrated 2×2 component (negative 4-cycle), weakly dominant with
    // strict surplus on the kept side (Grounded cover). Dense reduction only:
    // it is the one arm whose factor is exact, so the oracle A·x = r is sharp
    // (A is irreducibly dominant, hence nonsingular, and the cover acts as
    // the exact inverse on the antisymmetric subspace). The sparse arms'
    // approx-chol factor is inexact on the cover's degree-3 reduced graph;
    // they are exercised end-to-end in slopes_routing.rs, where factor error
    // only costs LSMR iterations.
    let (n_q, n_r) = (2usize, 2usize);
    let c_raw = [1.0, 1.5, 2.0, -1.0];
    let a = [
        [2.5, 0.0, 1.0, 1.5],
        [0.0, 3.0, 2.0, -1.0],
        [1.0, 2.0, 3.4, 0.0],
        [1.5, -1.0, 0.0, 2.9],
    ];

    let c = CsrBlock::from_dense_table(&c_raw, n_q, n_r);
    let ct = c.transpose();
    let component = LocalComponent::general_for_test(
        CrossTab { c, ct },
        BlockDiagonals {
            q: vec![a[0][0], a[1][1]],
            r: vec![a[2][2], a[3][3]],
        },
    );
    let config = LocalSolverConfig {
        approx_chol: ApproxCholConfig::default(),
        schur: SchurMode::Exact,
        dense_threshold: 8,
        scaling: Default::default(),
    };
    let solver =
        BlockElimSolver::build(component, &config).expect("covered block-elim build failed");
    let n = n_q + n_r;
    // #91 invariant: the stored operator stays single-sized (not doubled); the
    // cover lives only inside the reduced factor.
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
