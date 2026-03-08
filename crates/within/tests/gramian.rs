use schwarz_precond::Operator;

use within::domain::FixedEffectsDesign;
use within::observation::{FactorMajorStore, ObservationWeights};
use within::operator::gramian::{Gramian, GramianOperator};

fn make_test_design() -> FixedEffectsDesign {
    let store = FactorMajorStore::new(
        vec![vec![0, 1, 2, 0], vec![0, 1, 0, 1]],
        ObservationWeights::Unit,
        4,
    )
    .expect("valid factor-major store");
    FixedEffectsDesign::from_store(store).expect("valid test design")
}

fn make_weighted_design(weights: Vec<f64>) -> FixedEffectsDesign {
    let store = FactorMajorStore::new(
        vec![vec![0, 1, 0, 1], vec![0, 0, 1, 1]],
        ObservationWeights::Dense(weights),
        4,
    )
    .expect("valid weighted factor-major store");
    FixedEffectsDesign::from_store(store).expect("valid weighted design")
}

#[test]
fn test_gramian_build_diagonal_block() {
    let dm = make_test_design();
    let g = Gramian::build(&dm);
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
    let g = Gramian::build(&dm);
    let n = g.n_dofs();
    for i in 0..n {
        let mut ei = vec![0.0; n];
        ei[i] = 1.0;
        let mut gi = vec![0.0; n];
        g.matvec(&ei, &mut gi);
        for j in 0..n {
            let mut ej = vec![0.0; n];
            ej[j] = 1.0;
            let mut gj = vec![0.0; n];
            g.matvec(&ej, &mut gj);
            assert!((gi[j] - gj[i]).abs() < 1e-14);
        }
    }
}

#[test]
fn test_gramian_matches_gramian_operator() {
    let dm = make_test_design();
    let g = Gramian::build(&dm);
    let gop = GramianOperator::new(&dm);
    let n = g.n_dofs();
    let x = vec![1.0, -0.5, 2.0, 0.3, -1.0];
    let mut y_explicit = vec![0.0; n];
    let mut y_implicit = vec![0.0; n];
    g.matvec(&x, &mut y_explicit);
    gop.apply(&x, &mut y_implicit);
    for (a, b) in y_explicit.iter().zip(y_implicit.iter()) {
        assert!((a - b).abs() < 1e-12);
    }
}

#[test]
fn test_gramian_submatrix() {
    let dm = make_test_design();
    let g = Gramian::build(&dm);
    let sub = g.extract_submatrix(&[0, 1, 2]);
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
    gop.apply(&x, &mut y1);
    gop.apply_adjoint(&x, &mut y2);
    assert_eq!(y1, y2);
}

#[test]
fn test_gramian_diagonal_matches_explicit() {
    let dm = make_test_design();
    let g = Gramian::build(&dm);
    assert_eq!(g.diagonal(), dm.gramian_diagonal());
}

#[test]
fn test_weighted_gramian_diagonal() {
    let dm = make_weighted_design(vec![1.0, 2.0, 3.0, 4.0]);
    let g = Gramian::build(&dm);
    assert_eq!(g.diagonal(), dm.gramian_diagonal());
}

#[test]
fn test_weighted_gramian_matches_operator() {
    let dm = make_weighted_design(vec![1.0, 2.0, 3.0, 4.0]);
    let g = Gramian::build(&dm);
    let gop = GramianOperator::new(&dm);
    let n = g.n_dofs();
    let x = vec![1.0, -0.5, 2.0, 0.3];
    let mut y_explicit = vec![0.0; n];
    let mut y_implicit = vec![0.0; n];
    g.matvec(&x, &mut y_explicit);
    gop.apply(&x, &mut y_implicit);
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
    let store = FactorMajorStore::new(vec![fa, fb], ObservationWeights::Unit, n_obs)
        .expect("valid factor-major store");
    let dm = FixedEffectsDesign::from_store(store).expect("valid sparse accumulation design");
    let g = Gramian::build(&dm);
    let gop = GramianOperator::new(&dm);
    let n = g.n_dofs();

    let mut x = vec![0.0; n];
    for (i, xi) in x.iter_mut().enumerate() {
        *xi = (i as f64 * 0.1).sin();
    }
    let mut y_explicit = vec![0.0; n];
    let mut y_implicit = vec![0.0; n];
    g.matvec(&x, &mut y_explicit);
    gop.apply(&x, &mut y_implicit);
    for (a, b) in y_explicit.iter().zip(y_implicit.iter()) {
        assert!((a - b).abs() < 1e-10);
    }
}

#[test]
fn test_build_for_pair_matches_full_gramian() {
    let store = FactorMajorStore::new(
        vec![
            vec![0, 1, 2, 0, 1, 2, 0, 1],
            vec![0, 0, 1, 1, 0, 0, 1, 1],
            vec![0, 1, 0, 1, 0, 1, 0, 1],
        ],
        ObservationWeights::Unit,
        8,
    )
    .expect("valid factor-major store");
    let dm = FixedEffectsDesign::from_store(store).expect("valid pairwise design");
    let full = Gramian::build(&dm);
    let n = full.n_dofs();

    let pairs = [(0, 1), (0, 2), (1, 2)];
    let offsets = [0usize, 3, 5];
    let sizes = [3usize, 2, 2];

    for &(q, r) in &pairs {
        let pair_g = Gramian::build_for_pair(&dm, q, r);
        assert_eq!(pair_g.n_dofs(), n);

        let relevant_rows: Vec<usize> = (offsets[q]..offsets[q] + sizes[q])
            .chain(offsets[r]..offsets[r] + sizes[r])
            .collect();

        for &row in &relevant_rows {
            let mut ei = vec![0.0; n];
            ei[row] = 1.0;
            let mut y_full = vec![0.0; n];
            let mut y_pair = vec![0.0; n];
            full.matvec(&ei, &mut y_full);
            pair_g.matvec(&ei, &mut y_pair);

            for &col in &relevant_rows {
                assert!((y_full[col] - y_pair[col]).abs() < 1e-12);
            }
        }

        for row in 0..n {
            if relevant_rows.contains(&row) {
                continue;
            }
            let start = pair_g.matrix.indptr()[row];
            let end = pair_g.matrix.indptr()[row + 1];
            assert_eq!(start, end);
        }
    }
}
