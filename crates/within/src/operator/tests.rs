// ===========================================================================
// design tests
// ===========================================================================

mod design_tests {
    use crate::domain::FixedEffectsDesign;
    use crate::observation::{FactorMajorStore, ObservationWeights};
    use crate::operator::DesignOperator;
    use schwarz_precond::Operator;

    fn make_test_design() -> FixedEffectsDesign {
        let store = FactorMajorStore::new(
            vec![vec![0, 1, 2, 0, 1], vec![0, 1, 2, 3, 0]],
            ObservationWeights::Unit,
            5,
        )
        .expect("valid factor-major store");
        FixedEffectsDesign::from_store(store, &[3, 4]).expect("valid test design")
    }

    #[test]
    fn test_design_operator_dimensions() {
        let schema = make_test_design();
        let op = DesignOperator::new(&schema);
        assert_eq!(op.nrows(), 5);
        assert_eq!(op.ncols(), 7);
    }

    #[test]
    fn test_design_operator_adjoint() {
        let schema = make_test_design();
        let op = DesignOperator::new(&schema);

        let x = vec![1.0, -0.5, 2.0, 0.3, -1.0, 0.7, 1.5];
        let r = vec![0.1, 0.2, -0.3, 0.4, -0.5];

        let mut dx = vec![0.0; 5];
        op.apply(&x, &mut dx);
        let lhs: f64 = dx.iter().zip(r.iter()).map(|(a, b)| a * b).sum();

        let mut dtr = vec![0.0; 7];
        op.apply_adjoint(&r, &mut dtr);
        let rhs: f64 = x.iter().zip(dtr.iter()).map(|(a, b)| a * b).sum();

        assert!((lhs - rhs).abs() < 1e-12);
    }

    #[test]
    fn test_matvec_d() {
        let schema = make_test_design();
        let op = DesignOperator::new(&schema);
        let x = vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 40.0];
        let mut y = vec![0.0; 5];
        op.apply(&x, &mut y);
        assert_eq!(y, vec![11.0, 22.0, 33.0, 41.0, 12.0]);
    }

    #[test]
    fn test_rmatvec_dt() {
        let schema = make_test_design();
        let op = DesignOperator::new(&schema);
        let r = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut x = vec![0.0; 7];
        op.apply_adjoint(&r, &mut x);
        assert_eq!(x, vec![5.0, 7.0, 3.0, 6.0, 2.0, 3.0, 4.0]);
    }
}

// ===========================================================================
// csr_block tests
// ===========================================================================

mod csr_block_tests {
    use crate::operator::csr_block::CsrBlock;

    fn sample_block() -> CsrBlock {
        // 3x4 matrix:
        //  [1 0 2 0]
        //  [0 3 0 4]
        //  [5 0 0 6]
        CsrBlock {
            indptr: vec![0, 2, 4, 6],
            indices: vec![0, 2, 1, 3, 0, 3],
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            nrows: 3,
            ncols: 4,
        }
    }

    #[test]
    fn test_nnz() {
        assert_eq!(sample_block().nnz(), 6);
    }

    #[test]
    fn test_transpose_dimensions() {
        let a = sample_block();
        let at = a.transpose();
        assert_eq!(at.nrows, 4);
        assert_eq!(at.ncols, 3);
        assert_eq!(at.nnz(), a.nnz());
    }

    #[test]
    fn test_transpose_roundtrip() {
        let a = sample_block();
        let att = a.transpose().transpose();
        assert_eq!(att.nrows, a.nrows);
        assert_eq!(att.ncols, a.ncols);
        assert_eq!(att.indptr, a.indptr);
        assert_eq!(att.indices, a.indices);
        assert_eq!(att.data, a.data);
    }

    #[test]
    fn test_transpose_values() {
        let a = sample_block();
        let at = a.transpose();
        // A^T should be 4x3:
        //  [1 0 5]
        //  [0 3 0]
        //  [2 0 0]
        //  [0 4 6]
        assert_eq!(at.indptr, vec![0, 2, 3, 4, 6]);
        assert_eq!(at.indices, vec![0, 2, 1, 0, 1, 2]);
        assert_eq!(at.data, vec![1.0, 5.0, 3.0, 2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_from_dense_table() {
        let table = vec![1.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 4.0, 5.0, 0.0, 0.0, 6.0];
        let a = CsrBlock::from_dense_table(&table, 3, 4);
        let expected = sample_block();
        assert_eq!(a.indptr, expected.indptr);
        assert_eq!(a.indices, expected.indices);
        assert_eq!(a.data, expected.data);
        assert_eq!(a.nrows, expected.nrows);
        assert_eq!(a.ncols, expected.ncols);
    }

    #[test]
    fn test_spmv_diag_add() {
        let a = sample_block();
        let d = vec![2.0, 3.0, 1.0, 0.5];
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let mut y = vec![0.0; 3];
        a.spmv_diag_add(&d, &x, &mut y);
        // row 0: 1*(2*1) + 2*(1*3) = 2+6 = 8
        // row 1: 3*(3*2) + 4*(0.5*4) = 18+8 = 26
        // row 2: 5*(2*1) + 6*(0.5*4) = 10+12 = 22
        assert_eq!(y, vec![8.0, 26.0, 22.0]);
    }

    #[test]
    fn test_empty_block() {
        let a = CsrBlock {
            indptr: vec![0, 0, 0],
            indices: vec![],
            data: vec![],
            nrows: 2,
            ncols: 3,
        };
        assert_eq!(a.nnz(), 0);
        let at = a.transpose();
        assert_eq!(at.nrows, 3);
        assert_eq!(at.ncols, 2);
        assert_eq!(at.nnz(), 0);
    }

    #[test]
    fn test_from_dense_table_all_zeros() {
        let table = vec![0.0; 6];
        let a = CsrBlock::from_dense_table(&table, 2, 3);
        assert_eq!(a.nnz(), 0);
        assert_eq!(a.indptr, vec![0, 0, 0]);
    }
}

// ===========================================================================
// residual_update tests
// ===========================================================================

mod residual_update_tests {
    use crate::domain::FixedEffectsDesign;
    use crate::observation::{FactorMajorStore, ObservationWeights};
    use crate::operator::gramian::Gramian;
    use crate::operator::residual_update::{
        DofObservationIndex, ObservationSpaceUpdater, SparseGramianUpdater,
    };
    use schwarz_precond::{OperatorResidualUpdater, ResidualUpdater};

    // -------------------------------------------------------------------
    // DofObservationIndex tests
    // -------------------------------------------------------------------

    // 2 factors, 5 observations:
    //   factor 0: levels [0, 1, 2, 0, 1]  (3 levels, offset 0)
    //   factor 1: levels [0, 1, 2, 3, 0]  (4 levels, offset 3)
    // DOFs: 0..3 for factor 0, 3..7 for factor 1
    fn make_dof_index_design() -> FixedEffectsDesign {
        let store = FactorMajorStore::new(
            vec![vec![0, 1, 2, 0, 1], vec![0, 1, 2, 3, 0]],
            ObservationWeights::Unit,
            5,
        )
        .expect("valid factor-major store");
        FixedEffectsDesign::from_store(store, &[3, 4]).expect("valid test design")
    }

    #[test]
    fn test_basic_dof_observation_mapping() {
        let design = make_dof_index_design();
        let idx = DofObservationIndex::build(&design);

        assert_eq!(idx.n_dofs(), 7);

        // factor 0, level 0 (DOF 0): obs 0, 3
        let mut dof0: Vec<u32> = idx.obs_for_dof(0).to_vec();
        dof0.sort();
        assert_eq!(dof0, vec![0, 3]);

        // factor 0, level 1 (DOF 1): obs 1, 4
        let mut dof1: Vec<u32> = idx.obs_for_dof(1).to_vec();
        dof1.sort();
        assert_eq!(dof1, vec![1, 4]);

        // factor 0, level 2 (DOF 2): obs 2
        assert_eq!(idx.obs_for_dof(2), &[2]);

        // factor 1, level 0 (DOF 3): obs 0, 4
        let mut dof3: Vec<u32> = idx.obs_for_dof(3).to_vec();
        dof3.sort();
        assert_eq!(dof3, vec![0, 4]);

        // factor 1, level 1 (DOF 4): obs 1
        assert_eq!(idx.obs_for_dof(4), &[1]);

        // factor 1, level 2 (DOF 5): obs 2
        assert_eq!(idx.obs_for_dof(5), &[2]);

        // factor 1, level 3 (DOF 6): obs 3
        assert_eq!(idx.obs_for_dof(6), &[3]);
    }

    #[test]
    fn test_total_entries_equals_n_obs_times_n_factors() {
        let design = make_dof_index_design();
        let idx = DofObservationIndex::build(&design);

        let total: usize = (0..idx.n_dofs())
            .map(|d| idx.obs_for_dof(d as u32).len())
            .sum();
        // Each observation contributes one entry per factor
        assert_eq!(total, 5 * 2);
    }

    #[test]
    fn test_single_factor_single_level() {
        // 1 factor, 3 observations, all same level
        let store = FactorMajorStore::new(vec![vec![0, 0, 0]], ObservationWeights::Unit, 3)
            .expect("valid factor-major store");
        let design =
            FixedEffectsDesign::from_store(store, &[1]).expect("valid single-factor design");
        let idx = DofObservationIndex::build(&design);

        assert_eq!(idx.n_dofs(), 1);
        let mut obs: Vec<u32> = idx.obs_for_dof(0).to_vec();
        obs.sort();
        assert_eq!(obs, vec![0, 1, 2]);
    }

    #[test]
    fn test_dof_with_zero_observations() {
        // 1 factor, 2 observations using levels 0 and 2 — level 1 has no observations
        let store = FactorMajorStore::new(vec![vec![0, 2]], ObservationWeights::Unit, 2)
            .expect("valid factor-major store");
        let design =
            FixedEffectsDesign::from_store(store, &[3]).expect("valid sparse-level design");
        let idx = DofObservationIndex::build(&design);

        assert_eq!(idx.n_dofs(), 3);
        assert_eq!(idx.obs_for_dof(0), &[0]);
        assert_eq!(idx.obs_for_dof(1), &[] as &[u32]); // empty
        assert_eq!(idx.obs_for_dof(2), &[1]);
    }

    // -------------------------------------------------------------------
    // ObservationSpaceUpdater tests
    // -------------------------------------------------------------------

    /// Helper: build a design, explicit Gramian, and both updaters.
    /// Returns (design, gramian, obs_updater).
    fn make_test_setup() -> (FixedEffectsDesign, Gramian) {
        // 2 factors, 5 observations
        // factor 0: [0, 1, 2, 0, 1] (3 levels)
        // factor 1: [0, 1, 2, 3, 0] (4 levels)
        // n_dofs = 7
        let store = FactorMajorStore::new(
            vec![vec![0, 1, 2, 0, 1], vec![0, 1, 2, 3, 0]],
            ObservationWeights::Unit,
            5,
        )
        .expect("valid factor-major store");
        let design =
            FixedEffectsDesign::from_store(store, &[3, 4]).expect("valid fixed-effects design");
        let gramian = Gramian::build(&design);
        (design, gramian)
    }

    #[test]
    fn test_obs_updater_matches_operator_updater_single_step() {
        let (design, gramian) = make_test_setup();
        let n_dofs = design.n_dofs;

        let mut obs_updater = ObservationSpaceUpdater::new(&design);
        let mut op_updater = OperatorResidualUpdater::new(&gramian, n_dofs);

        // Initial residual
        let r_original = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        op_updater.reset(&r_original);

        let mut r_obs = r_original.clone();
        let mut r_op = r_original.clone();

        // Correction on subdomain {0, 1, 3, 4} (factor0 levels 0,1 + factor1 levels 0,1)
        let global_indices: Vec<u32> = vec![0, 1, 3, 4];
        let correction = vec![0.5, -0.3, 0.2, 0.1];

        obs_updater.update(&global_indices, &correction, &mut r_obs);
        op_updater.update(&global_indices, &correction, &mut r_op);

        for i in 0..n_dofs {
            assert!(
                (r_obs[i] - r_op[i]).abs() < 1e-12,
                "mismatch at DOF {i}: obs={}, op={}",
                r_obs[i],
                r_op[i],
            );
        }
    }

    #[test]
    fn test_obs_updater_matches_operator_updater_two_steps() {
        let (design, gramian) = make_test_setup();
        let n_dofs = design.n_dofs;

        let mut obs_updater = ObservationSpaceUpdater::new(&design);
        let mut op_updater = OperatorResidualUpdater::new(&gramian, n_dofs);

        let r_original = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0];
        op_updater.reset(&r_original);

        let mut r_obs = r_original.clone();
        let mut r_op = r_original.clone();

        // First subdomain correction
        let gi1: Vec<u32> = vec![0, 1, 3, 4];
        let c1 = vec![0.5, -0.3, 0.2, 0.1];
        obs_updater.update(&gi1, &c1, &mut r_obs);
        op_updater.update(&gi1, &c1, &mut r_op);

        for i in 0..n_dofs {
            assert!(
                (r_obs[i] - r_op[i]).abs() < 1e-12,
                "step 1 mismatch at DOF {i}: obs={}, op={}",
                r_obs[i],
                r_op[i],
            );
        }

        // Second subdomain correction
        let gi2: Vec<u32> = vec![2, 5, 6];
        let c2 = vec![1.0, -0.5, 0.8];
        obs_updater.update(&gi2, &c2, &mut r_obs);
        op_updater.update(&gi2, &c2, &mut r_op);

        for i in 0..n_dofs {
            assert!(
                (r_obs[i] - r_op[i]).abs() < 1e-12,
                "step 2 mismatch at DOF {i}: obs={}, op={}",
                r_obs[i],
                r_op[i],
            );
        }
    }

    #[test]
    fn test_obs_updater_zero_correction_is_noop() {
        let (design, _) = make_test_setup();
        let mut updater = ObservationSpaceUpdater::new(&design);

        let r_original = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let mut r_work = r_original.clone();

        let gi: Vec<u32> = vec![0, 1, 3];
        let correction = vec![0.0, 0.0, 0.0];
        updater.update(&gi, &correction, &mut r_work);

        assert_eq!(r_work, r_original);
    }

    #[test]
    fn test_obs_updater_single_dof_correction() {
        let (design, gramian) = make_test_setup();
        let n_dofs = design.n_dofs;

        let mut obs_updater = ObservationSpaceUpdater::new(&design);
        let mut op_updater = OperatorResidualUpdater::new(&gramian, n_dofs);

        let r_original = vec![1.0; n_dofs];
        op_updater.reset(&r_original);

        let mut r_obs = r_original.clone();
        let mut r_op = r_original.clone();

        // Single DOF correction
        let gi: Vec<u32> = vec![0];
        let correction = vec![1.0];
        obs_updater.update(&gi, &correction, &mut r_obs);
        op_updater.update(&gi, &correction, &mut r_op);

        for i in 0..n_dofs {
            assert!(
                (r_obs[i] - r_op[i]).abs() < 1e-12,
                "single-DOF mismatch at {i}: obs={}, op={}",
                r_obs[i],
                r_op[i],
            );
        }
    }

    #[test]
    fn test_obs_updater_weighted_design() {
        use crate::domain::WeightedDesign;

        let fl = vec![vec![0u32, 1, 0, 1], vec![0, 0, 1, 1]];
        let weights = vec![1.0, 2.0, 3.0, 4.0];
        let n_levels = vec![2, 2];

        let store = FactorMajorStore::new(fl, ObservationWeights::Dense(weights), 4)
            .expect("valid weighted store");
        let design = WeightedDesign::from_store(store, &n_levels).expect("valid weighted design");
        let gramian = Gramian::build(&design);
        let n_dofs = design.n_dofs; // 4

        let mut obs_updater = ObservationSpaceUpdater::new(&design);
        let mut op_updater = OperatorResidualUpdater::new(&gramian, n_dofs);

        let r_original = vec![5.0, 3.0, 7.0, 1.0];
        op_updater.reset(&r_original);

        let mut r_obs = r_original.clone();
        let mut r_op = r_original.clone();

        let gi: Vec<u32> = vec![0, 2];
        let correction = vec![0.5, -0.3];
        obs_updater.update(&gi, &correction, &mut r_obs);
        op_updater.update(&gi, &correction, &mut r_op);

        for i in 0..n_dofs {
            assert!(
                (r_obs[i] - r_op[i]).abs() < 1e-12,
                "weighted mismatch at {i}: obs={}, op={}",
                r_obs[i],
                r_op[i],
            );
        }
    }

    // -----------------------------------------------------------------------
    // SparseGramianUpdater tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_sparse_gramian_updater_matches_obs_updater_single_step() {
        let (design, gramian) = make_test_setup();
        let n_dofs = design.n_dofs;

        let mut obs_updater = ObservationSpaceUpdater::new(&design);
        let mut sparse_updater = SparseGramianUpdater::new(gramian.matrix);

        let r_original = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let mut r_obs = r_original.clone();
        let mut r_sparse = r_original;

        let global_indices: Vec<u32> = vec![0, 1, 3, 4];
        let correction = vec![0.5, -0.3, 0.2, 0.1];

        obs_updater.update(&global_indices, &correction, &mut r_obs);
        sparse_updater.update(&global_indices, &correction, &mut r_sparse);

        for i in 0..n_dofs {
            assert!(
                (r_obs[i] - r_sparse[i]).abs() < 1e-12,
                "mismatch at DOF {i}: obs={}, sparse={}",
                r_obs[i],
                r_sparse[i],
            );
        }
    }

    #[test]
    fn test_sparse_gramian_updater_matches_obs_updater_two_steps() {
        let (design, gramian) = make_test_setup();
        let n_dofs = design.n_dofs;

        let mut obs_updater = ObservationSpaceUpdater::new(&design);
        let mut sparse_updater = SparseGramianUpdater::new(gramian.matrix);

        let r_original = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0];
        let mut r_obs = r_original.clone();
        let mut r_sparse = r_original;

        // First subdomain correction
        let gi1: Vec<u32> = vec![0, 1, 3, 4];
        let c1 = vec![0.5, -0.3, 0.2, 0.1];
        obs_updater.update(&gi1, &c1, &mut r_obs);
        sparse_updater.update(&gi1, &c1, &mut r_sparse);

        for i in 0..n_dofs {
            assert!(
                (r_obs[i] - r_sparse[i]).abs() < 1e-12,
                "step 1 mismatch at DOF {i}: obs={}, sparse={}",
                r_obs[i],
                r_sparse[i],
            );
        }

        // Second subdomain correction
        let gi2: Vec<u32> = vec![2, 5, 6];
        let c2 = vec![1.0, -0.5, 0.8];
        obs_updater.update(&gi2, &c2, &mut r_obs);
        sparse_updater.update(&gi2, &c2, &mut r_sparse);

        for i in 0..n_dofs {
            assert!(
                (r_obs[i] - r_sparse[i]).abs() < 1e-12,
                "step 2 mismatch at DOF {i}: obs={}, sparse={}",
                r_obs[i],
                r_sparse[i],
            );
        }
    }

    #[test]
    fn test_sparse_gramian_updater_zero_correction_is_noop() {
        let (_, gramian) = make_test_setup();
        let mut updater = SparseGramianUpdater::new(gramian.matrix);

        let r_original = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let mut r_work = r_original.clone();

        let gi: Vec<u32> = vec![0, 1, 3];
        let correction = vec![0.0, 0.0, 0.0];
        updater.update(&gi, &correction, &mut r_work);

        assert_eq!(r_work, r_original);
    }

    #[test]
    fn test_sparse_gramian_updater_single_dof_correction() {
        let (design, gramian) = make_test_setup();
        let n_dofs = design.n_dofs;

        let mut obs_updater = ObservationSpaceUpdater::new(&design);
        let mut sparse_updater = SparseGramianUpdater::new(gramian.matrix);

        let r_original = vec![1.0; n_dofs];
        let mut r_obs = r_original.clone();
        let mut r_sparse = r_original;

        let gi: Vec<u32> = vec![0];
        let correction = vec![1.0];
        obs_updater.update(&gi, &correction, &mut r_obs);
        sparse_updater.update(&gi, &correction, &mut r_sparse);

        for i in 0..n_dofs {
            assert!(
                (r_obs[i] - r_sparse[i]).abs() < 1e-12,
                "single-DOF mismatch at {i}: obs={}, sparse={}",
                r_obs[i],
                r_sparse[i],
            );
        }
    }

    #[test]
    fn test_sparse_gramian_updater_weighted_design() {
        use crate::domain::WeightedDesign;

        let fl = vec![vec![0u32, 1, 0, 1], vec![0, 0, 1, 1]];
        let weights = vec![1.0, 2.0, 3.0, 4.0];
        let n_levels = vec![2, 2];

        let store = FactorMajorStore::new(fl, ObservationWeights::Dense(weights), 4)
            .expect("valid weighted store");
        let design = WeightedDesign::from_store(store, &n_levels).expect("valid weighted design");
        let gramian = Gramian::build(&design);
        let n_dofs = design.n_dofs;

        let mut obs_updater = ObservationSpaceUpdater::new(&design);
        let mut sparse_updater = SparseGramianUpdater::new(gramian.matrix);

        let r_original = vec![5.0, 3.0, 7.0, 1.0];
        let mut r_obs = r_original.clone();
        let mut r_sparse = r_original;

        let gi: Vec<u32> = vec![0, 2];
        let correction = vec![0.5, -0.3];
        obs_updater.update(&gi, &correction, &mut r_obs);
        sparse_updater.update(&gi, &correction, &mut r_sparse);

        for i in 0..n_dofs {
            assert!(
                (r_obs[i] - r_sparse[i]).abs() < 1e-12,
                "weighted mismatch at {i}: obs={}, sparse={}",
                r_obs[i],
                r_sparse[i],
            );
        }
    }
}

// ===========================================================================
// schur_complement tests
// ===========================================================================

mod schur_complement_tests {
    use crate::operator::csr_block::CsrBlock;
    use crate::operator::gramian::CrossTab;
    use crate::operator::schur_complement::{
        ApproxSchurComplement, ExactSchurComplement, SchurComplement,
    };
    use schwarz_precond::SparseMatrix;

    fn make_cross_tab(
        c_dense: &[f64],
        n_q: usize,
        n_r: usize,
        diag_q: Vec<f64>,
        diag_r: Vec<f64>,
    ) -> CrossTab {
        let c = CsrBlock::from_dense_table(c_dense, n_q, n_r);
        let ct = c.transpose();
        CrossTab {
            c,
            ct,
            diag_q,
            diag_r,
        }
    }

    fn sparse_to_dense(matrix: &SparseMatrix) -> Vec<Vec<f64>> {
        let n = matrix.n();
        let mut dense = vec![vec![0.0; n]; n];
        for (i, row) in dense.iter_mut().enumerate().take(n) {
            let start = matrix.indptr()[i] as usize;
            let end = matrix.indptr()[i + 1] as usize;
            for idx in start..end {
                let j = matrix.indices()[idx] as usize;
                row[j] = matrix.data()[idx];
            }
        }
        dense
    }

    fn dense_exact_schur(
        c_dense: &[f64],
        n_q: usize,
        n_r: usize,
        diag_q: &[f64],
        diag_r: &[f64],
        eliminate_q: bool,
    ) -> Vec<Vec<f64>> {
        if eliminate_q {
            let mut s = vec![vec![0.0; n_r]; n_r];
            for i in 0..n_r {
                s[i][i] = diag_r[i];
            }
            for k in 0..n_q {
                let inv = if diag_q[k] > 0.0 {
                    1.0 / diag_q[k]
                } else {
                    0.0
                };
                for i in 0..n_r {
                    let cki = c_dense[k * n_r + i];
                    for j in 0..n_r {
                        let ckj = c_dense[k * n_r + j];
                        s[i][j] -= cki * inv * ckj;
                    }
                }
            }
            s
        } else {
            let mut s = vec![vec![0.0; n_q]; n_q];
            for i in 0..n_q {
                s[i][i] = diag_q[i];
            }
            for k in 0..n_r {
                let inv = if diag_r[k] > 0.0 {
                    1.0 / diag_r[k]
                } else {
                    0.0
                };
                for i in 0..n_q {
                    let cik = c_dense[i * n_r + k];
                    for j in 0..n_q {
                        let cjk = c_dense[j * n_r + k];
                        s[i][j] -= cik * inv * cjk;
                    }
                }
            }
            s
        }
    }

    fn assert_dense_close(lhs: &[Vec<f64>], rhs: &[Vec<f64>], tol: f64) {
        assert_eq!(lhs.len(), rhs.len(), "row count mismatch");
        for i in 0..lhs.len() {
            assert_eq!(lhs[i].len(), rhs[i].len(), "col count mismatch on row {i}");
            for j in 0..lhs[i].len() {
                assert!(
                    (lhs[i][j] - rhs[i][j]).abs() <= tol,
                    "mismatch at ({i}, {j}): lhs={}, rhs={}",
                    lhs[i][j],
                    rhs[i][j]
                );
            }
        }
    }

    #[test]
    fn exact_schur_matches_dense_reference_when_eliminating_q() {
        // C is 3x2, so q-block is eliminated (n_q >= n_r).
        let c_dense = vec![1.0, 2.0, 3.0, 0.0, 0.0, 4.0];
        let diag_q = vec![5.0, 6.0, 8.0];
        let diag_r = vec![7.0, 9.0];
        let cross_tab = make_cross_tab(&c_dense, 3, 2, diag_q.clone(), diag_r.clone());

        let result = ExactSchurComplement.compute(&cross_tab);

        assert!(result.elimination.eliminate_q);
        assert_eq!(result.elimination.inv_diag_elim.len(), 3);
        for (&got, &expected) in result
            .elimination
            .inv_diag_elim
            .iter()
            .zip([1.0 / 5.0, 1.0 / 6.0, 1.0 / 8.0].iter())
        {
            assert!((got - expected).abs() < 1e-12);
        }

        let expected = dense_exact_schur(&c_dense, 3, 2, &diag_q, &diag_r, true);
        let got = sparse_to_dense(&result.matrix);
        assert_dense_close(&got, &expected, 1e-12);
    }

    #[test]
    fn exact_schur_handles_zero_eliminated_diagonal_when_eliminating_r() {
        // C is 2x3, so r-block is eliminated (n_q < n_r). Last eliminated
        // diagonal is zero, so its inverse contribution should be 0.
        let c_dense = vec![2.0, 0.0, 1.0, 0.0, 3.0, 4.0];
        let diag_q = vec![8.0, 9.0];
        let diag_r = vec![5.0, 6.0, 0.0];
        let cross_tab = make_cross_tab(&c_dense, 2, 3, diag_q.clone(), diag_r.clone());

        let result = ExactSchurComplement.compute(&cross_tab);

        assert!(!result.elimination.eliminate_q);
        assert_eq!(result.elimination.inv_diag_elim.len(), 3);
        assert!((result.elimination.inv_diag_elim[0] - 1.0 / 5.0).abs() < 1e-12);
        assert!((result.elimination.inv_diag_elim[1] - 1.0 / 6.0).abs() < 1e-12);
        assert_eq!(result.elimination.inv_diag_elim[2], 0.0);

        let expected = dense_exact_schur(&c_dense, 2, 3, &diag_q, &diag_r, false);
        let got = sparse_to_dense(&result.matrix);
        assert_dense_close(&got, &expected, 1e-12);
    }

    #[test]
    fn exact_dense_schur_matches_sparse_exact() {
        let c_dense = vec![1.0, 2.0, 3.0, 0.0, 0.0, 4.0];
        let cross_tab = make_cross_tab(&c_dense, 3, 2, vec![5.0, 6.0, 8.0], vec![7.0, 9.0]);

        let sparse = ExactSchurComplement.compute(&cross_tab);
        let dense = ExactSchurComplement.compute_dense(&cross_tab);

        assert_eq!(dense.n, sparse.matrix.n());
        assert_eq!(
            dense.elimination.eliminate_q,
            sparse.elimination.eliminate_q
        );
        assert_eq!(
            dense.elimination.inv_diag_elim.len(),
            sparse.elimination.inv_diag_elim.len()
        );
        for (&a, &b) in dense
            .elimination
            .inv_diag_elim
            .iter()
            .zip(sparse.elimination.inv_diag_elim.iter())
        {
            assert!((a - b).abs() < 1e-15);
        }

        let got_dense: Vec<Vec<f64>> = (0..dense.n)
            .map(|i| dense.matrix[i * dense.n..(i + 1) * dense.n].to_vec())
            .collect();
        let got_sparse = sparse_to_dense(&sparse.matrix);
        assert_dense_close(&got_dense, &got_sparse, 1e-12);
    }

    #[test]
    fn exact_dense_anchored_matches_full_dense_minor() {
        let c_dense = vec![1.0, 2.0, 3.0, 0.0, 0.0, 4.0];
        let cross_tab = make_cross_tab(&c_dense, 3, 2, vec![5.0, 6.0, 8.0], vec![7.0, 9.0]);

        let full = ExactSchurComplement.compute_dense(&cross_tab);
        let anchored = ExactSchurComplement.compute_dense_anchored(&cross_tab);
        assert_eq!(full.n, anchored.n);

        let m = full.n.saturating_sub(1);
        let mut full_minor = vec![0.0; m * m];
        for i in 0..m {
            for j in 0..m {
                full_minor[i * m + j] = full.matrix[i * full.n + j];
            }
        }

        assert_eq!(anchored.anchored_minor.len(), full_minor.len());
        for (&a, &b) in anchored.anchored_minor.iter().zip(full_minor.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn approximate_schur_is_seed_deterministic_and_laplacian_like() {
        // Degree-3 star in eliminated block gives nontrivial sampled edges.
        let c_dense = vec![1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let cross_tab = make_cross_tab(&c_dense, 3, 3, vec![10.0, 4.0, 5.0], vec![2.0, 3.0, 4.0]);
        let approx = ApproxSchurComplement::new(crate::config::ApproxSchurConfig { seed: 12345 });

        let a = approx.compute(&cross_tab);
        let b = approx.compute(&cross_tab);

        assert_eq!(a.elimination.eliminate_q, b.elimination.eliminate_q);
        assert_eq!(a.elimination.inv_diag_elim, b.elimination.inv_diag_elim);
        assert_eq!(a.matrix.indptr(), b.matrix.indptr());
        assert_eq!(a.matrix.indices(), b.matrix.indices());
        assert_eq!(a.matrix.data(), b.matrix.data());

        let dense = sparse_to_dense(&a.matrix);
        for (i, row) in dense.iter().enumerate() {
            let mut row_sum = 0.0;
            for (j, &value) in row.iter().enumerate() {
                row_sum += value;
                assert!(
                    (value - dense[j][i]).abs() <= 1e-12,
                    "matrix not symmetric at ({i}, {j})"
                );
                if i != j {
                    assert!(value <= 1e-12, "off-diagonal should be non-positive");
                }
            }
            assert!(row_sum.abs() <= 1e-10, "row {i} sum is not near zero");
            assert!(row[i] >= -1e-12, "diagonal should be non-negative");
        }
    }
}

// ===========================================================================
// schwarz tests
// ===========================================================================

mod schwarz_tests {
    use std::cmp::Ordering;
    use std::hint::black_box;
    use std::time::Instant;

    use crate::config::LocalSolverConfig;
    use crate::config::{ApproxSchurConfig, DEFAULT_DENSE_SCHUR_THRESHOLD};
    use crate::domain::{build_local_domains, FixedEffectsDesign, Subdomain};
    use crate::observation::{FactorMajorStore, ObservationWeights};
    use crate::operator::csr_block::CsrBlock;
    use crate::operator::gramian::CrossTab;
    use crate::operator::local_solver::BlockElimSolver;
    use crate::operator::local_solver::FeLocalSolver;
    use crate::operator::schwarz::{
        build_additive, build_entry, build_multiplicative_obs, build_multiplicative_sparse,
        build_reduced_schur_factor, build_schwarz, compute_first_block_size, DomainSource,
    };
    use approx_chol::Config;
    use schwarz_precond::{LocalSolver, Operator};

    fn make_test_data() -> (FixedEffectsDesign, Vec<(Subdomain, CrossTab)>) {
        let store = FactorMajorStore::new(
            vec![vec![0, 1, 0, 1, 2], vec![0, 0, 1, 1, 0]],
            ObservationWeights::Unit,
            5,
        )
        .expect("valid factor-major store");
        let design =
            FixedEffectsDesign::from_store(store, &[3, 2]).expect("valid fixed-effects design");
        let domain_pairs = build_local_domains(&design, None);
        (design, domain_pairs)
    }

    fn synthetic_cross_tab(n_keep: usize, elim_ratio: usize) -> CrossTab {
        let n_q = n_keep * elim_ratio;
        let n_r = n_keep;
        let mut table = vec![0.0; n_q * n_r];

        for i in 0..n_q {
            let j0 = i % n_r;
            let j1 = (i + 1) % n_r;
            let j2 = (i.wrapping_mul(7).wrapping_add(3)) % n_r;
            table[i * n_r + j0] += 1.0;
            table[i * n_r + j1] += 0.8;
            table[i * n_r + j2] += 0.6;
        }

        let mut diag_q = vec![0.0; n_q];
        let mut diag_r = vec![0.0; n_r];
        for i in 0..n_q {
            let row = &table[i * n_r..(i + 1) * n_r];
            let mut s = 0.0;
            for (j, &w) in row.iter().enumerate() {
                s += w;
                diag_r[j] += w;
            }
            diag_q[i] = s;
        }

        let c = CsrBlock::from_dense_table(&table, n_q, n_r);
        let ct = c.transpose();
        CrossTab {
            c,
            ct,
            diag_q,
            diag_r,
        }
    }

    fn benchmark_build_path(
        cross_tab: &CrossTab,
        approx_schur: Option<ApproxSchurConfig>,
        dense_threshold: usize,
        iters: usize,
    ) -> f64 {
        let approx_chol = Config {
            split_merge: Some(8),
            seed: 42,
        };
        let mut samples = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            let reduced =
                build_reduced_schur_factor(cross_tab, approx_chol, approx_schur, dense_threshold)
                    .expect("reduced Schur build failed");
            black_box(reduced.factor);
            samples.push(t0.elapsed().as_secs_f64() * 1e6);
        }
        samples.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        samples[samples.len() / 2]
    }

    fn build_local_solver_for_bench(
        n_keep: usize,
        approx_schur: Option<ApproxSchurConfig>,
        dense_threshold: usize,
    ) -> BlockElimSolver {
        let cross_tab = synthetic_cross_tab(n_keep, 8);
        let approx_chol = Config {
            split_merge: Some(8),
            seed: 42,
        };
        let reduced =
            build_reduced_schur_factor(&cross_tab, approx_chol, approx_schur, dense_threshold)
                .expect("reduced Schur build failed");
        BlockElimSolver::new(
            cross_tab,
            reduced.elimination.inv_diag_elim,
            reduced.factor,
            reduced.elimination.eliminate_q,
        )
    }

    fn benchmark_local_solve_path(solver: &BlockElimSolver, iters: usize) -> f64 {
        let n_local = solver.n_local();
        let scratch = solver.scratch_size();
        let mut rhs_template = vec![0.0; n_local];
        for (i, v) in rhs_template.iter_mut().enumerate() {
            *v = ((i.wrapping_mul(13) % 31) as f64 - 15.0) * 0.1;
        }

        let mut rhs = vec![0.0; scratch];
        let mut sol = vec![0.0; scratch];
        let t0 = Instant::now();
        let mut checksum = 0.0;
        for _ in 0..iters {
            rhs[..n_local].copy_from_slice(&rhs_template);
            solver
                .solve_local(&mut rhs, &mut sol)
                .expect("benchmark local solve");
            checksum += sol[0];
        }
        black_box(checksum);
        (t0.elapsed().as_secs_f64() * 1e6) / iters as f64
    }

    #[test]
    fn test_build_schwarz() {
        let (design, domain_pairs) = make_test_data();
        let config = LocalSolverConfig::default();
        let schwarz = build_additive::<FactorMajorStore>(
            DomainSource::FromParts(domain_pairs),
            design.n_dofs,
            &config,
        )
        .expect("build schwarz with explicit domains");
        assert!(!schwarz.subdomains().is_empty());

        let r = vec![1.0; design.n_dofs];
        let mut z = vec![0.0; design.n_dofs];
        schwarz.apply(&r, &mut z);
        assert!(z.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_build_default() {
        let (design, _) = make_test_data();
        let config = LocalSolverConfig::default();
        let schwarz = build_schwarz(&design, &config).expect("build default schwarz");
        assert!(!schwarz.subdomains().is_empty());
    }

    #[test]
    fn test_first_block_size_computation() {
        let store = FactorMajorStore::new(
            vec![vec![0, 1, 0, 1], vec![0, 0, 1, 1]],
            ObservationWeights::Unit,
            4,
        )
        .expect("valid factor-major store");
        let design = FixedEffectsDesign::from_store(store, &[2, 2]).expect("valid design");
        let domain_pairs = build_local_domains(&design, None);

        assert!(!domain_pairs.is_empty());
        let fbs = compute_first_block_size(&design, &domain_pairs[0].0);
        assert_eq!(fbs, 2);
    }

    #[test]
    fn test_exact_schur_uses_dense_fast_path_for_tiny_reduced_system() {
        let (_, mut domain_pairs) = make_test_data();
        let (domain, cross_tab) = domain_pairs.swap_remove(0);

        let config = LocalSolverConfig::SchurComplement {
            approx_chol: Config::default(),
            approx_schur: None,
            dense_threshold: DEFAULT_DENSE_SCHUR_THRESHOLD,
        };
        let entry =
            build_entry(domain, cross_tab, &config).expect("exact Schur entry build failed");

        match entry.solver {
            FeLocalSolver::SchurComplement(solver) => {
                assert!(solver.uses_dense_reduced_factor());
            }
            FeLocalSolver::FullSddm { .. } => panic!("expected SchurComplement solver"),
        }
    }

    #[test]
    fn test_approximate_schur_uses_dense_fast_path_for_tiny_reduced_system() {
        let (_, mut domain_pairs) = make_test_data();
        let (domain, cross_tab) = domain_pairs.swap_remove(0);

        let config = LocalSolverConfig::SchurComplement {
            approx_chol: Config::default(),
            approx_schur: Some(ApproxSchurConfig { seed: 7 }),
            dense_threshold: DEFAULT_DENSE_SCHUR_THRESHOLD,
        };
        let entry =
            build_entry(domain, cross_tab, &config).expect("approximate Schur entry build failed");

        match entry.solver {
            FeLocalSolver::SchurComplement(solver) => {
                assert!(solver.uses_dense_reduced_factor());
            }
            FeLocalSolver::FullSddm { .. } => panic!("expected SchurComplement solver"),
        }
    }

    #[test]
    fn test_dense_threshold_zero_disables_dense_fast_path() {
        let (_, mut domain_pairs) = make_test_data();
        let (domain, cross_tab) = domain_pairs.swap_remove(0);

        let config = LocalSolverConfig::SchurComplement {
            approx_chol: Config::default(),
            approx_schur: None,
            dense_threshold: 0,
        };
        let entry =
            build_entry(domain, cross_tab, &config).expect("exact Schur entry build failed");

        match entry.solver {
            FeLocalSolver::SchurComplement(solver) => {
                assert!(!solver.uses_dense_reduced_factor());
            }
            FeLocalSolver::FullSddm { .. } => panic!("expected SchurComplement solver"),
        }
    }

    #[test]
    #[ignore]
    fn bench_isolated_schur_dense_vs_sparse_paths() {
        let sizes = [4usize, 8, 12, 16, 20, 24, 28, 32, 40, 48, 64];
        println!(
            "{:>5} | {:>11} {:>11} {:>7} | {:>11} {:>11} {:>7} | {:>10} {:>10} {:>7}",
            "n_keep",
            "exact_dense",
            "exact_sparse",
            "ratio",
            "approx_dense",
            "approx_sparse",
            "ratio",
            "solve_dense",
            "solve_sparse",
            "ratio"
        );
        println!("{}", "-".repeat(118));

        for &n_keep in &sizes {
            let cross_tab = synthetic_cross_tab(n_keep, 8);
            let build_iters = if n_keep <= 32 { 100 } else { 40 };

            let exact_dense = benchmark_build_path(&cross_tab, None, usize::MAX, build_iters);
            let exact_sparse = benchmark_build_path(&cross_tab, None, 0, build_iters);
            let approx_dense = benchmark_build_path(
                &cross_tab,
                Some(ApproxSchurConfig { seed: 42 }),
                usize::MAX,
                build_iters,
            );
            let approx_sparse = benchmark_build_path(
                &cross_tab,
                Some(ApproxSchurConfig { seed: 42 }),
                0,
                build_iters,
            );

            let solve_iters = if n_keep <= 32 { 8_000 } else { 3_000 };
            let solver_dense = build_local_solver_for_bench(n_keep, None, usize::MAX);
            let solver_sparse = build_local_solver_for_bench(n_keep, None, 0);
            let solve_dense = benchmark_local_solve_path(&solver_dense, solve_iters);
            let solve_sparse = benchmark_local_solve_path(&solver_sparse, solve_iters);

            println!(
                "{:>5} | {:>11.2} {:>11.2} {:>7.2} | {:>11.2} {:>11.2} {:>7.2} | {:>10.3} {:>10.3} {:>7.2}",
                n_keep,
                exact_dense,
                exact_sparse,
                exact_dense / exact_sparse,
                approx_dense,
                approx_sparse,
                approx_dense / approx_sparse,
                solve_dense,
                solve_sparse,
                solve_dense / solve_sparse
            );
        }

        println!(
            "\nDefault dense threshold currently: {}",
            DEFAULT_DENSE_SCHUR_THRESHOLD
        );
    }

    #[test]
    fn test_gramian_multiplicative_schwarz_matches_obs_schwarz() {
        let store = FactorMajorStore::new(
            vec![vec![0, 1, 0, 1, 2], vec![0, 0, 1, 1, 0]],
            ObservationWeights::Unit,
            5,
        )
        .expect("valid factor-major store");
        let design = FixedEffectsDesign::from_store(store, &[3, 2]).expect("valid design");

        let config = LocalSolverConfig::default();
        let gramian = crate::operator::gramian::Gramian::build(&design);

        let domain_pairs = build_local_domains(&design, None);

        let obs_schwarz =
            build_multiplicative_obs(DomainSource::FromDesign(&design), &design, &config).unwrap();
        let gram_schwarz = build_multiplicative_sparse(
            DomainSource::<FactorMajorStore>::FromParts(domain_pairs),
            &gramian,
            design.n_dofs,
            &config,
        )
        .unwrap();

        let r = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut z_obs = vec![0.0; 5];
        let mut z_gram = vec![0.0; 5];

        obs_schwarz.apply(&r, &mut z_obs);
        gram_schwarz.apply(&r, &mut z_gram);

        for i in 0..5 {
            assert!(
                (z_obs[i] - z_gram[i]).abs() < 1e-12,
                "mismatch at DOF {i}: obs={}, gram={}",
                z_obs[i],
                z_gram[i],
            );
        }
    }
}
