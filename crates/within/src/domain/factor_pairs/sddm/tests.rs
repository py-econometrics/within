use super::*;
use crate::csr_block::CsrBlock;

impl LocalComponent {
    pub(crate) fn plain_for_test(cross_tab: CrossTab, diagonals: BlockDiagonals) -> Self {
        convert(
            cross_tab,
            diagonals,
            ComponentClass::KnownLaplacian,
            &ScalingConfig::default(),
        )
        .expect("plain test component must convert to SDDM")
        .0
    }

    pub(crate) fn general_for_test(cross_tab: CrossTab, diagonals: BlockDiagonals) -> Self {
        convert(
            cross_tab,
            diagonals,
            ComponentClass::General,
            &ScalingConfig::default(),
        )
        .expect("general test component must convert to SDDM")
        .0
    }

    /// Assemble under externally supplied congruence factors: the relaxation's
    /// certified scaling is valid but not unique, so tests that construct a
    /// component backward from known factors pin those factors here.
    pub(crate) fn with_factors_for_test(
        cross_tab: CrossTab,
        diagonals: BlockDiagonals,
        factors: &[f64],
    ) -> Self {
        assemble(
            cross_tab,
            diagonals,
            factors.to_vec(),
            ReductionKind::Direct,
            &ScalingConfig::default(),
        )
        .expect("test factors must fold to SDDM")
        .0
    }
}

fn cross_tab(table: &[f64], n_q: usize, n_r: usize) -> CrossTab {
    let c = CsrBlock::from_dense_table(table, n_q, n_r);
    let ct = c.transpose();
    CrossTab { c, ct }
}

fn assert_sddm(component: &LocalComponent) {
    assert!(component.cross_tab.c.data.iter().all(|value| *value >= 0.0));
    let sums = adjacency_sums(&component.cross_tab);
    for ((&diagonal, &row_sum), &surplus) in component
        .diagonals
        .q
        .iter()
        .zip(sums.q.iter())
        .zip(component.ground_edges.q.iter())
        .chain(
            component
                .diagonals
                .r
                .iter()
                .zip(sums.r.iter())
                .zip(component.ground_edges.r.iter()),
        )
    {
        assert!(diagonal >= row_sum);
        assert!((diagonal - row_sum - surplus).abs() <= 1e-12 * diagonal);
    }
}

#[test]
fn known_laplacian_has_canonical_coordinates_and_no_ground() {
    let (component, uncertified) = convert(
        cross_tab(&[2.0, 1.0, 0.0, 3.0], 2, 2),
        BlockDiagonals {
            q: vec![3.0, 3.0],
            r: vec![2.0, 4.0],
        },
        ComponentClass::KnownLaplacian,
        &ScalingConfig::default(),
    )
    .unwrap();
    assert_eq!(component.reduction.solve_space(), SolveSpace::Floating);
    assert!(matches!(component.coordinates, CoordinateMap::Canonical));
    assert!(uncertified.is_none());
    assert_sddm(&component);
}

#[test]
fn known_laplacian_claim_is_checked() {
    // Structural surplus contradicts the Laplacian claim — a broken plain
    // accumulator must fail loudly, not misclassify.
    let result = convert(
        cross_tab(&[2.0, 1.0, 0.0, 3.0], 2, 2),
        BlockDiagonals {
            q: vec![4.0, 3.0],
            r: vec![2.0, 4.0],
        },
        ComponentClass::KnownLaplacian,
        &ScalingConfig::default(),
    );
    assert!(matches!(result, Err(NotScalable)));
}

#[test]
fn frustrated_component_stores_single_signed_operator() {
    let (component, uncertified) = convert(
        cross_tab(&[1.0, 1.0, 1.0, -1.0], 2, 2),
        BlockDiagonals {
            q: vec![2.0, 2.0],
            r: vec![2.0, 2.0],
        },
        ComponentClass::General,
        &ScalingConfig::default(),
    )
    .unwrap();
    assert!(uncertified.is_none());
    // The operator stays single-sized and signed (M[1,1] = −1): the cover is
    // deferred to factor time via the Cover reduction marker.
    assert_eq!(component.reduction, Reduction::Cover);
    assert_eq!(component.cross_tab.c.nrows, 2);
    assert_eq!(component.cross_tab.c.ncols, 2);
    let mut dense = [[0.0; 2]; 2];
    for (i, row) in dense.iter_mut().enumerate() {
        for (j, value) in component.cross_tab.c.row(i) {
            row[j] = value;
        }
    }
    assert_eq!(dense, [[1.0, 1.0], [1.0, -1.0]]);
    // Magnitude dominance with zero surplus: the operator is Signed (the cover
    // self-grounds). Its congruence is exactly the canonical bipartite sign flip
    // (`+1` on q, `−1` on r), so no explicit factor map is stored.
    assert_eq!(component.reduction.solve_space(), SolveSpace::Signed);
    assert!(component.ground_edges.q.iter().all(|&s| s == 0.0));
    assert!(component.ground_edges.r.iter().all(|&s| s == 0.0));
    assert!(matches!(component.coordinates, CoordinateMap::Canonical));
}

#[test]
fn scalable_signed_component_produces_valid_grounded_sddm() {
    let d = [2.0, -0.5, -1.0, 4.0, 0.25];
    let c_hat = [[1.0, 2.0, 0.5], [3.0, 0.0, 1.5]];
    let diag_hat = [4.0, 5.0, 4.5, 2.5, 2.375];
    let mut raw = vec![0.0; 6];
    for i in 0..2 {
        for j in 0..3 {
            raw[i * 3 + j] = c_hat[i][j] / (d[i] * d[2 + j]);
        }
    }
    let (component, uncertified) = convert(
        cross_tab(&raw, 2, 3),
        BlockDiagonals {
            q: (0..2).map(|i| diag_hat[i] / (d[i] * d[i])).collect(),
            r: (2..5).map(|i| diag_hat[i] / (d[i] * d[i])).collect(),
        },
        ComponentClass::General,
        &ScalingConfig::default(),
    )
    .unwrap();
    assert_eq!(component.reduction.solve_space(), SolveSpace::Grounded);
    assert!(uncertified.is_none());
    assert_sddm(&component);
}

#[test]
fn singular_signed_boundary_remains_floating() {
    let (component, _) = convert(
        cross_tab(&[0.5, -1.0], 2, 1),
        BlockDiagonals {
            q: vec![0.25, 1.0],
            r: vec![2.0],
        },
        ComponentClass::General,
        &ScalingConfig::default(),
    )
    .unwrap();
    assert_eq!(component.reduction.solve_space(), SolveSpace::Floating);
    assert_sddm(&component);
}

#[test]
fn large_rescaled_singular_boundary_remains_floating() {
    let n_r = 20_000usize;
    let q_factor = 1.3;
    let r_factors: Vec<f64> = (0..n_r).map(|j| 0.7 + 0.03 * (j % 17) as f64).collect();
    let weights: Vec<f64> = (0..n_r).map(|j| 1.0 + 0.01 * (j % 23) as f64).collect();
    let c = CsrBlock {
        indptr: vec![0, n_r as u32],
        indices: (0..n_r as u32).collect(),
        data: weights
            .iter()
            .zip(&r_factors)
            .map(|(&weight, &factor)| -weight / (q_factor * factor))
            .collect(),
        nrows: 1,
        ncols: n_r,
    };
    let cross_tab = CrossTab {
        ct: c.transpose(),
        c,
    };
    let diagonals = BlockDiagonals {
        q: vec![weights.iter().sum::<f64>() / q_factor.powi(2)],
        r: weights
            .iter()
            .zip(&r_factors)
            .map(|(&weight, &factor)| weight / factor.powi(2))
            .collect(),
    };
    let factors: Vec<f64> = std::iter::once(q_factor).chain(r_factors).collect();

    let (component, _) = assemble(
        cross_tab,
        diagonals,
        factors,
        ReductionKind::Direct,
        &ScalingConfig::default(),
    )
    .unwrap();

    assert_eq!(component.reduction.solve_space(), SolveSpace::Floating);
    assert_sddm(&component);
}

#[test]
fn non_scalable_component_errors_under_error_mode() {
    let result = convert(
        cross_tab(&[1.0, -1.0, 2.0, -2.0], 2, 2),
        BlockDiagonals {
            q: vec![1.0, 2.0],
            r: vec![1.0, 2.0],
        },
        ComponentClass::General,
        &ScalingConfig {
            on_failure: ScalingFailure::Error,
            ..Default::default()
        },
    );
    assert!(matches!(result, Err(NotScalable)));
}

#[test]
fn non_scalable_component_warns_and_clamps_under_warn_mode() {
    let config = ScalingConfig {
        max_sweeps: 32,
        ..Default::default()
    };
    assert_eq!(config.on_failure, ScalingFailure::Warn);
    let (component, uncertified) = convert(
        cross_tab(&[1.0, -1.0, 2.0, -2.0], 2, 2),
        BlockDiagonals {
            q: vec![1.0, 2.0],
            r: vec![1.0, 2.0],
        },
        ComponentClass::General,
        &config,
    )
    .unwrap();
    let uncertified = uncertified.expect("no dominant scaling exists");
    assert_eq!(uncertified.sweeps, 32);
    assert!(uncertified.violation > config.tolerance);
    // Clamping restores the SDDM invariant, so the hand-over is usable.
    assert_sddm(&component);
}

#[test]
fn barely_pd_surplus_is_structural() {
    let surplus = 5e-10;
    let (component, _) = convert(
        cross_tab(&[1.0], 1, 1),
        BlockDiagonals {
            q: vec![1.0 + surplus],
            r: vec![1.0],
        },
        ComponentClass::General,
        &ScalingConfig::default(),
    )
    .unwrap();
    assert_eq!(component.reduction.solve_space(), SolveSpace::Grounded);
    assert!((component.ground_edges.q[0] - surplus).abs() < 1e-15);
    assert_sddm(&component);
}

#[test]
fn large_barely_pd_surplus_is_not_absorbed_by_validation_slack() {
    let n = 1_000_000;
    let total_diagonal = 1.0;
    let structural_surplus = 5e-12;

    assert!(structural_surplus > FLOATING_CLASSIFICATION_BUDGET.tolerance(n, total_diagonal));
    assert!(structural_surplus <= LAPLACIAN_VALIDATION_BUDGET.tolerance(n, total_diagonal));
}
