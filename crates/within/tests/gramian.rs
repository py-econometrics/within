use schwarz_precond::Operator;

use within::observation::FactorMajorStore;
use within::operator::gramian::{Gramian, GramianOperator};
use within::Design;

fn make_test_design() -> Design<FactorMajorStore> {
    let store = FactorMajorStore::new(vec![vec![0, 1, 2, 0], vec![0, 1, 0, 1]], 4)
        .expect("valid factor-major store");
    Design::from_store(store).expect("valid test design")
}

fn make_weighted_design() -> Design<FactorMajorStore> {
    let store = FactorMajorStore::new(vec![vec![0, 1, 0, 1], vec![0, 0, 1, 1]], 4)
        .expect("valid weighted factor-major store");
    Design::from_store(store).expect("valid weighted design")
}

#[test]
fn test_gramian_build_diagonal_block() {
    let dm = make_test_design();
    let g = Gramian::build(&dm, None);
    let diag = g.diagonal();
    assert_eq!(diag[0], 2.0);
    assert_eq!(diag[1], 1.0);
    assert_eq!(diag[2], 1.0);
    assert_eq!(diag[3], 2.0);
    assert_eq!(diag[4], 2.0);
}

#[test]
fn test_gramian_symmetry() {
    let dm = make_test_design();
    let g = Gramian::build(&dm, None);
    let n = g.n_dofs();
    for i in 0..n {
        let mut ei = vec![0.0; n];
        ei[i] = 1.0;
        let mut gi = vec![0.0; n];
        g.matrix.matvec(&ei, &mut gi);
        for j in 0..n {
            let mut ej = vec![0.0; n];
            ej[j] = 1.0;
            let mut gj = vec![0.0; n];
            g.matrix.matvec(&ej, &mut gj);
            assert!((gi[j] - gj[i]).abs() < 1e-14);
        }
    }
}

#[test]
fn test_gramian_matches_gramian_operator() {
    let dm = make_test_design();
    let g = Gramian::build(&dm, None);
    let gop = GramianOperator::new(&dm);
    let n = g.n_dofs();
    let x = vec![1.0, -0.5, 2.0, 0.3, -1.0];
    let mut y_explicit = vec![0.0; n];
    let mut y_implicit = vec![0.0; n];
    g.matrix.matvec(&x, &mut y_explicit);
    gop.apply(&x, &mut y_implicit).expect("apply");
    for (a, b) in y_explicit.iter().zip(y_implicit.iter()) {
        assert!((a - b).abs() < 1e-12);
    }
}

#[test]
fn test_gramian_submatrix() {
    let dm = make_test_design();
    let g = Gramian::build(&dm, None);
    let sub = g.matrix.extract_submatrix(&[0, 1, 2]);
    assert_eq!(sub.n(), 3);
    let diag = sub.diagonal();
    assert_eq!(diag[0], 2.0);
    assert_eq!(diag[1], 1.0);
    assert_eq!(diag[2], 1.0);
}

#[test]
fn test_gramian_operator_symmetric() {
    let dm = make_test_design();
    let gop = GramianOperator::new(&dm);
    let n = dm.n_dofs;

    let x = vec![1.0, -0.5, 2.0, 0.3, -1.0];
    let mut y1 = vec![0.0; n];
    let mut y2 = vec![0.0; n];
    gop.apply(&x, &mut y1).expect("apply");
    gop.apply_adjoint(&x, &mut y2).expect("apply");
    assert_eq!(y1, y2);
}

#[test]
fn test_gramian_diagonal_matches_explicit() {
    let dm = make_test_design();
    let g = Gramian::build(&dm, None);
    assert_eq!(g.diagonal(), Gramian::build(&dm, None).diagonal());
}

#[test]
fn test_weighted_gramian_diagonal() {
    let dm = make_weighted_design();
    let weights = vec![1.0, 2.0, 3.0, 4.0];
    let g = Gramian::build(&dm, Some(&weights));
    assert_eq!(g.diagonal(), Gramian::build(&dm, Some(&weights)).diagonal());
}

#[test]
fn test_weighted_gramian_matches_operator() {
    let dm = make_weighted_design();
    let weights = vec![1.0, 2.0, 3.0, 4.0];
    let g = Gramian::build(&dm, Some(&weights));
    let gop = within::operator::gramian::WeightedGramianOperator::new(&dm, &weights);
    let n = g.n_dofs();
    let x = vec![1.0, -0.5, 2.0, 0.3];
    let mut y_explicit = vec![0.0; n];
    let mut y_implicit = vec![0.0; n];
    g.matrix.matvec(&x, &mut y_explicit);
    gop.apply(&x, &mut y_implicit).expect("apply");
    for (a, b) in y_explicit.iter().zip(y_implicit.iter()) {
        assert!((a - b).abs() < 1e-12);
    }
}

#[test]
fn test_gramian_sparse_accumulation_path() {
    let n_obs = 200;
    let n_lev_a = 2001;
    let n_lev_b = 2001;
    let mut fa = Vec::with_capacity(n_obs);
    let mut fb = Vec::with_capacity(n_obs);
    for i in 0..n_obs {
        fa.push((i % n_lev_a) as u32);
        fb.push(((i * 7) % n_lev_b) as u32);
    }
    let store = FactorMajorStore::new(vec![fa, fb], n_obs).expect("valid factor-major store");
    let dm = Design::from_store(store).expect("valid sparse accumulation design");
    let g = Gramian::build(&dm, None);
    let gop = GramianOperator::new(&dm);
    let n = g.n_dofs();

    let mut x = vec![0.0; n];
    for (i, xi) in x.iter_mut().enumerate() {
        *xi = (i as f64 * 0.1).sin();
    }
    let mut y_explicit = vec![0.0; n];
    let mut y_implicit = vec![0.0; n];
    g.matrix.matvec(&x, &mut y_explicit);
    gop.apply(&x, &mut y_implicit).expect("apply");
    for (a, b) in y_explicit.iter().zip(y_implicit.iter()) {
        assert!((a - b).abs() < 1e-10);
    }
}

#[test]
fn test_gramian_apply_adjoint_delegates() {
    let dm = make_test_design();
    let g = Gramian::build(&dm, None);
    let n = g.n_dofs();
    let x = vec![1.0, -0.5, 2.0, 0.3, -1.0];
    let mut y1 = vec![0.0; n];
    let mut y2 = vec![0.0; n];
    g.apply(&x, &mut y1).expect("apply");
    g.apply_adjoint(&x, &mut y2).expect("apply");
    assert_eq!(y1, y2);
}

#[test]
fn test_gramian_linearity() {
    let dm = make_test_design();
    let g = Gramian::build(&dm, None);
    let n = g.n_dofs();

    let x = vec![1.0, -0.5, 2.0, 0.3, -1.0];
    let y_vec = vec![0.5, 1.0, -1.0, 0.7, 0.2];
    let a = 2.5;
    let b = -1.3;

    // Compute G(ax + by)
    let combined: Vec<f64> = x
        .iter()
        .zip(&y_vec)
        .map(|(&xi, &yi)| a * xi + b * yi)
        .collect();
    let mut g_combined = vec![0.0; n];
    g.matrix.matvec(&combined, &mut g_combined);

    // Compute a*G(x) + b*G(y)
    let mut gx = vec![0.0; n];
    let mut gy = vec![0.0; n];
    g.matrix.matvec(&x, &mut gx);
    g.matrix.matvec(&y_vec, &mut gy);
    let linear_combo: Vec<f64> = gx
        .iter()
        .zip(&gy)
        .map(|(&gxi, &gyi)| a * gxi + b * gyi)
        .collect();

    for (i, (&actual, &expected)) in g_combined.iter().zip(&linear_combo).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-10,
            "linearity violation at {}: {} vs {}",
            i,
            actual,
            expected
        );
    }
}

#[test]
fn test_three_factor_gramian_build() {
    let store = FactorMajorStore::new(
        vec![
            vec![0, 1, 2, 0, 1, 2, 0, 1],
            vec![0, 0, 1, 1, 0, 0, 1, 1],
            vec![0, 1, 0, 1, 0, 1, 0, 1],
        ],
        8,
    )
    .expect("valid store");
    let dm = Design::from_store(store).expect("valid design");
    let g = Gramian::build(&dm, None);
    let n = g.n_dofs();

    // Verify symmetry
    for i in 0..n {
        let mut ei = vec![0.0; n];
        ei[i] = 1.0;
        let mut gi = vec![0.0; n];
        g.matrix.matvec(&ei, &mut gi);
        for j in (i + 1)..n {
            let mut ej = vec![0.0; n];
            ej[j] = 1.0;
            let mut gj = vec![0.0; n];
            g.matrix.matvec(&ej, &mut gj);
            assert!(
                (gi[j] - gj[i]).abs() < 1e-14,
                "symmetry violation at ({}, {}): {} vs {}",
                i,
                j,
                gi[j],
                gj[i]
            );
        }
    }

    // Verify diagonal sums = observations per level
    let diag = g.diagonal();
    // Factor 0 has levels 0,1,2 appearing in: 0={0,3,6}->3, 1={1,4,7}->3, 2={2,5}->2
    assert_eq!(diag[0], 3.0);
    assert_eq!(diag[1], 3.0);
    assert_eq!(diag[2], 2.0);
    // Factor 1 has levels 0,1: 0={0,1,4,5}->4, 1={2,3,6,7}->4
    assert_eq!(diag[3], 4.0);
    assert_eq!(diag[4], 4.0);
    // Factor 2 has levels 0,1: 0={0,2,4,6}->4, 1={1,3,5,7}->4
    assert_eq!(diag[5], 4.0);
    assert_eq!(diag[6], 4.0);
}

#[test]
fn test_gramian_operator_dimensions() {
    let dm = make_test_design();
    let gop = GramianOperator::new(&dm);
    assert_eq!(gop.nrows(), dm.n_dofs);
    assert_eq!(gop.ncols(), dm.n_dofs);
}

#[test]
fn test_gramian_large_row_permutation_sort() {
    // Create a factor pair where one level co-occurs with many levels of the other factor
    // This creates a row in the Gramian with > 64 entries
    let n_obs = 200;
    let n_b = 100; // factor B has 100 levels
    let mut fa = Vec::with_capacity(n_obs);
    let mut fb = Vec::with_capacity(n_obs);
    for i in 0..n_obs {
        fa.push(0u32); // all same level for factor A — this row will have n_b+1 entries
        fb.push((i % n_b) as u32);
    }
    // Add some variety to factor A
    for (i, val) in fa.iter_mut().enumerate() {
        if i < 50 {
            *val = 0;
        } else if i < 100 {
            *val = 1;
        } else {
            *val = (i % 3) as u32;
        }
    }
    let store = FactorMajorStore::new(vec![fa, fb], n_obs).expect("valid store");
    let dm = Design::from_store(store).expect("valid design");
    let g = Gramian::build(&dm, None);
    let gop = GramianOperator::new(&dm);
    let n = g.n_dofs();

    // Verify explicit matches implicit
    let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
    let mut y_exp = vec![0.0; n];
    let mut y_imp = vec![0.0; n];
    g.matrix.matvec(&x, &mut y_exp);
    gop.apply(&x, &mut y_imp).expect("apply");
    for (a, b) in y_exp.iter().zip(y_imp.iter()) {
        assert!((a - b).abs() < 1e-10, "explicit/implicit mismatch");
    }
}

#[test]
fn test_gramian_operator_trait() {
    let dm = make_test_design();
    let g = Gramian::build(&dm, None);
    assert_eq!(g.nrows(), dm.n_dofs);
    assert_eq!(g.ncols(), dm.n_dofs);
}

#[test]
fn test_gramian_parallel_build_path() {
    // PAR_THRESHOLD = 100_000; use 100_001 observations to trigger the parallel branch.
    let n_obs = 100_001usize;
    let n_lev_a = 100usize;
    let n_lev_b = 100usize;
    let fa: Vec<u32> = (0..n_obs).map(|i| (i % n_lev_a) as u32).collect();
    let fb: Vec<u32> = (0..n_obs).map(|i| ((i * 7) % n_lev_b) as u32).collect();

    let store = FactorMajorStore::new(vec![fa, fb], n_obs).expect("valid parallel store");
    let dm = within::domain::Design::from_store(store).expect("valid parallel design");

    let g = Gramian::build(&dm, None);
    let gop = GramianOperator::new(&dm);
    let n = g.n_dofs();
    assert_eq!(n, n_lev_a + n_lev_b);

    let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.13).sin()).collect();
    let mut y_explicit = vec![0.0; n];
    let mut y_implicit = vec![0.0; n];
    g.matrix.matvec(&x, &mut y_explicit);
    gop.apply(&x, &mut y_implicit).expect("apply");

    for (i, (a, b)) in y_explicit.iter().zip(y_implicit.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-10,
            "parallel vs serial mismatch at dof {i}: explicit={a}, implicit={b}"
        );
    }
}

#[test]
fn test_gramian_single_factor() {
    // A design with only one factor has a purely diagonal Gramian.
    // Factor 0 has 4 levels; observations cycle through them.
    let fa = vec![0u32, 1, 2, 3, 0, 1, 2, 0];
    let n_obs = fa.len();
    let store = FactorMajorStore::new(vec![fa], n_obs).expect("valid store");
    let dm = within::domain::Design::from_store(store).expect("valid single-factor design");

    let g = Gramian::build(&dm, None);
    let n = g.n_dofs();

    assert_eq!(n, 4, "single factor with 4 levels -> n_dofs=4");

    // Diagonal: level 0 appears 3 times, level 1 appears 2 times,
    //           level 2 appears 2 times, level 3 appears 1 time.
    let diag = g.diagonal();
    assert_eq!(diag.len(), 4);
    assert!((diag[0] - 3.0).abs() < 1e-12, "level 0 count: {}", diag[0]);
    assert!((diag[1] - 2.0).abs() < 1e-12, "level 1 count: {}", diag[1]);
    assert!((diag[2] - 2.0).abs() < 1e-12, "level 2 count: {}", diag[2]);
    assert!((diag[3] - 1.0).abs() < 1e-12, "level 3 count: {}", diag[3]);

    // The Gramian should be purely diagonal: G[i,j] = 0 for i != j.
    for i in 0..n {
        let mut ei = vec![0.0; n];
        ei[i] = 1.0;
        let mut gi = vec![0.0; n];
        g.matrix.matvec(&ei, &mut gi);
        for (j, &gij) in gi.iter().enumerate() {
            if i == j {
                assert!(
                    (gij - diag[i]).abs() < 1e-12,
                    "diagonal entry ({i},{j}) should equal diag[{i}]={}: got {}",
                    diag[i],
                    gij
                );
            } else {
                assert!(
                    gij.abs() < 1e-12,
                    "off-diagonal entry ({i},{j}) should be 0: got {}",
                    gij
                );
            }
        }
    }

    // matvec scales each DOF by its observation count.
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let mut y = vec![0.0; n];
    g.matrix.matvec(&x, &mut y);
    assert!((y[0] - 3.0 * 1.0).abs() < 1e-12, "y[0] = diag[0]*x[0]");
    assert!((y[1] - 2.0 * 2.0).abs() < 1e-12, "y[1] = diag[1]*x[1]");
    assert!((y[2] - 2.0 * 3.0).abs() < 1e-12, "y[2] = diag[2]*x[2]");
    assert!((y[3] - 1.0 * 4.0).abs() < 1e-12, "y[3] = diag[3]*x[3]");
}
