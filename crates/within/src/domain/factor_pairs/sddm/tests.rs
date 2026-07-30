use super::*;
use crate::csr_block::CsrBlock;

impl LocalComponent {
    pub(crate) fn plain_for_test(cross_tab: CrossTab, diagonal: Vec<f64>) -> Self {
        convert(
            cross_tab,
            diagonal,
            ComponentClass::KnownLaplacian,
            &ScalingConfig::default(),
        )
        .expect("plain test component must convert to SDDM")
        .0
    }

    pub(crate) fn general_for_test(cross_tab: CrossTab, diagonal: Vec<f64>) -> Self {
        let globals = (0..cross_tab.n_local() as u32).collect();
        let (cross_tab, diagonal, _) = super::orient_for_elimination(cross_tab, diagonal, globals);
        convert(
            cross_tab,
            diagonal,
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
        diagonal: Vec<f64>,
        factors: &[f64],
    ) -> Self {
        assemble(
            cross_tab,
            diagonal,
            factors.to_vec(),
            MatrixForm::Laplacian,
            &ScalingConfig::default(),
        )
        .expect("test factors must fold to SDDM")
        .0
    }
}

fn cross_tab(table: &[f64], n_rows: usize, n_cols: usize) -> CrossTab {
    let c = CsrBlock::from_dense_table(table, n_rows, n_cols);
    let ct = c.transpose();
    CrossTab { c, ct }
}

fn assert_sddm(component: &LocalComponent) {
    assert!(component
        .matrix
        .cross_tab
        .c
        .data
        .iter()
        .all(|value| *value >= 0.0));
    let sums = adjacency_sums(&component.matrix.cross_tab);
    for ((&diagonal, &row_sum), &surplus) in component
        .matrix
        .diagonal
        .iter()
        .zip(sums.iter())
        .zip(component.matrix.ground_edges.iter())
    {
        assert!(diagonal >= row_sum);
        assert!((diagonal - row_sum - surplus).abs() <= 1e-12 * diagonal);
    }
}

#[test]
fn known_laplacian_has_canonical_coordinates_and_no_ground() {
    let (component, uncertified) = convert(
        cross_tab(&[2.0, 1.0, 0.0, 3.0], 2, 2),
        vec![3.0, 3.0, 2.0, 4.0],
        ComponentClass::KnownLaplacian,
        &ScalingConfig::default(),
    )
    .unwrap();
    assert_eq!(component.form, MatrixForm::Laplacian);
    assert_eq!(component.matrix.grounding, Grounding::Floating);
    assert!(matches!(component.coordinates, CoordinateMap::Canonical));
    assert!(uncertified.is_none());
    assert_sddm(&component);
}

#[test]
fn known_laplacian_claim_is_checked() {
    // Structural surplus contradicts the Laplacian claim, so it must fail loudly.
    let result = convert(
        cross_tab(&[2.0, 1.0, 0.0, 3.0], 2, 2),
        vec![4.0, 3.0, 2.0, 4.0],
        ComponentClass::KnownLaplacian,
        &ScalingConfig::default(),
    );
    assert!(matches!(result, Err(NotScalable)));
}

#[test]
fn frustrated_component_stores_single_signed_operator() {
    let (component, uncertified) = convert(
        cross_tab(&[1.0, 1.0, 1.0, -1.0], 2, 2),
        vec![2.0, 2.0, 2.0, 2.0],
        ComponentClass::General,
        &ScalingConfig::default(),
    )
    .unwrap();
    assert!(uncertified.is_none());
    // The matrix stays single-sized and signed; the cover is deferred to factor time.
    assert_eq!(component.form, MatrixForm::SignedPendingCover);
    assert_eq!(component.matrix.grounding, Grounding::Floating);
    assert_eq!(component.matrix.cross_tab.c.nrows, 2);
    assert_eq!(component.matrix.cross_tab.c.ncols, 2);
    let mut dense = [[0.0; 2]; 2];
    for (i, row) in dense.iter_mut().enumerate() {
        for (j, value) in component.matrix.cross_tab.c.row(i) {
            row[j] = value;
        }
    }
    assert_eq!(dense, [[1.0, 1.0], [1.0, -1.0]]);
    // Zero surplus and the canonical bipartite sign flip, so no explicit factor map is stored.
    assert!(component.matrix.ground_edges.iter().all(|&s| s == 0.0));
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
        (0..5).map(|i| diag_hat[i] / (d[i] * d[i])).collect(),
        ComponentClass::General,
        &ScalingConfig::default(),
    )
    .unwrap();
    assert_eq!(component.form, MatrixForm::Laplacian);
    assert_eq!(component.matrix.grounding, Grounding::Grounded);
    assert!(uncertified.is_none());
    assert_sddm(&component);
}

#[test]
fn singular_signed_boundary_remains_floating() {
    let (component, _) = convert(
        cross_tab(&[0.5, -1.0], 2, 1),
        vec![0.25, 1.0, 2.0],
        ComponentClass::General,
        &ScalingConfig::default(),
    )
    .unwrap();
    assert_eq!(component.form, MatrixForm::Laplacian);
    assert_eq!(component.matrix.grounding, Grounding::Floating);
    assert_sddm(&component);
}

#[test]
fn large_rescaled_singular_boundary_remains_floating() {
    let n_cols = 20_000usize;
    let row_factor = 1.3;
    let r_factors: Vec<f64> = (0..n_cols).map(|j| 0.7 + 0.03 * (j % 17) as f64).collect();
    let weights: Vec<f64> = (0..n_cols).map(|j| 1.0 + 0.01 * (j % 23) as f64).collect();
    let c = CsrBlock {
        indptr: vec![0, n_cols as u32],
        indices: (0..n_cols as u32).collect(),
        data: weights
            .iter()
            .zip(&r_factors)
            .map(|(&weight, &factor)| -weight / (row_factor * factor))
            .collect(),
        nrows: 1,
        ncols: n_cols,
    };
    let cross_tab = CrossTab {
        ct: c.transpose(),
        c,
    };
    let diagonal: Vec<f64> = std::iter::once(weights.iter().sum::<f64>() / row_factor.powi(2))
        .chain(
            weights
                .iter()
                .zip(&r_factors)
                .map(|(&weight, &factor)| weight / factor.powi(2)),
        )
        .collect();
    let factors: Vec<f64> = std::iter::once(row_factor).chain(r_factors).collect();

    let (component, _) = assemble(
        cross_tab,
        diagonal,
        factors,
        MatrixForm::Laplacian,
        &ScalingConfig::default(),
    )
    .unwrap();

    assert_eq!(component.form, MatrixForm::Laplacian);
    assert_eq!(component.matrix.grounding, Grounding::Floating);
    assert_sddm(&component);
}

#[test]
fn non_scalable_component_errors_under_error_mode() {
    let result = convert(
        cross_tab(&[1.0, -1.0, 2.0, -2.0], 2, 2),
        vec![1.0, 2.0, 1.0, 2.0],
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
        vec![1.0, 2.0, 1.0, 2.0],
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
        vec![1.0 + surplus, 1.0],
        ComponentClass::General,
        &ScalingConfig::default(),
    )
    .unwrap();
    assert_eq!(component.form, MatrixForm::Laplacian);
    assert_eq!(component.matrix.grounding, Grounding::Grounded);
    assert!((component.matrix.ground_edges[0] - surplus).abs() < 1e-15);
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
