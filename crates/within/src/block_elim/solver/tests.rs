use super::*;

use crate::config::ApproxCholConfig;
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
        approx_schur: None,
        dense_threshold: 0, // disable dense fast path to ensure sparse path is covered
        scaling: Default::default(),
    };
    let component = SddmComponent::general_for_test(cross_tab, diagonals);
    let solver = BlockElimSolver::build(component, &config).expect("block-elim build failed");

    assert!(
        !solver.eliminate_q,
        "expected eliminate_q=false when n_q < n_r",
    );
    // n_local = n_q + n_r = 2 + 5 = 7
    assert_eq!(solver.n_local(), 7);
}

#[test]
fn test_block_elim_solver_eliminate_q_false_solve_residual() {
    let (cross_tab, diagonals) = make_cross_tab_q_lt_r();
    let n_local = cross_tab.n_q() + cross_tab.n_r(); // 7

    let config = LocalSolverConfig {
        approx_chol: ApproxCholConfig::default(),
        approx_schur: None,
        dense_threshold: 0,
        scaling: Default::default(),
    };
    let component = SddmComponent::general_for_test(cross_tab, diagonals);
    let solver = BlockElimSolver::build(component, &config).expect("block-elim build failed");
    assert!(!solver.eliminate_q);

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
        approx_schur: None,
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
        let component = SddmComponent::general_for_test(CrossTab { c, ct }, diagonals);
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

    for (label, dense_threshold, approx_schur) in [
        ("dense full minor", 8, None),
        ("exact sparse", 0, None),
        (
            "sampled sparse",
            0,
            Some(crate::config::ApproxSchurConfig::default()),
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
            approx_schur,
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
            SddmComponent::with_factors_for_test(CrossTab { c, ct }, diagonals, &factors);
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
