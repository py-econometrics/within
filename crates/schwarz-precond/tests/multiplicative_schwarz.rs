mod common;

use schwarz_precond::domain::PartitionWeights;
use schwarz_precond::{
    MultiplicativeSchwarzPreconditioner, Operator, OperatorResidualUpdater, SubdomainCore,
    SubdomainEntry,
};

use common::{DiagLocalSolver, DiagOperator, TridiagOperator};

#[test]
fn test_multiplicative_single_subdomain_exact() {
    let a = DiagOperator {
        values: vec![2.0, 3.0, 4.0],
    };
    let solver = DiagLocalSolver::new(&[2.0, 3.0, 4.0]);
    let core = SubdomainCore::uniform(vec![0, 1, 2]);
    let entry = SubdomainEntry::new(core, solver);

    let updater = OperatorResidualUpdater::new(&a, 3);
    let prec = MultiplicativeSchwarzPreconditioner::new(vec![entry], updater, 3, false)
        .expect("valid multiplicative preconditioner");

    let r = vec![6.0, 9.0, 12.0];
    let mut z = vec![0.0; 3];
    prec.apply(&r, &mut z);

    assert!((z[0] - 3.0).abs() < 1e-12);
    assert!((z[1] - 3.0).abs() < 1e-12);
    assert!((z[2] - 3.0).abs() < 1e-12);
}

#[test]
fn test_multiplicative_two_nonoverlapping_subdomains() {
    let a = DiagOperator {
        values: vec![2.0, 3.0, 1.0, 5.0],
    };

    let solver0 = DiagLocalSolver::new(&[2.0, 3.0]);
    let core0 = SubdomainCore::uniform(vec![0, 1]);
    let entry0 = SubdomainEntry::new(core0, solver0);

    let solver1 = DiagLocalSolver::new(&[1.0, 5.0]);
    let core1 = SubdomainCore::uniform(vec![2, 3]);
    let entry1 = SubdomainEntry::new(core1, solver1);

    let updater = OperatorResidualUpdater::new(&a, 4);
    let prec = MultiplicativeSchwarzPreconditioner::new(vec![entry0, entry1], updater, 4, false)
        .expect("valid multiplicative preconditioner");

    let r = vec![4.0, 9.0, 3.0, 10.0];
    let mut z = vec![0.0; 4];
    prec.apply(&r, &mut z);

    assert!((z[0] - 2.0).abs() < 1e-12);
    assert!((z[1] - 3.0).abs() < 1e-12);
    assert!((z[2] - 3.0).abs() < 1e-12);
    assert!((z[3] - 2.0).abs() < 1e-12);
}

#[test]
fn test_multiplicative_overlapping_residual_update() {
    let a = DiagOperator {
        values: vec![2.0, 3.0, 4.0],
    };

    let solver0 = DiagLocalSolver::new(&[2.0, 3.0]);
    let core0 = SubdomainCore {
        global_indices: vec![0, 1],
        partition_weights: PartitionWeights::NonUniform(vec![1.0, 0.5]),
    };
    let entry0 = SubdomainEntry::new(core0, solver0);

    let solver1 = DiagLocalSolver::new(&[3.0, 4.0]);
    let core1 = SubdomainCore {
        global_indices: vec![1, 2],
        partition_weights: PartitionWeights::NonUniform(vec![0.5, 1.0]),
    };
    let entry1 = SubdomainEntry::new(core1, solver1);

    let updater = OperatorResidualUpdater::new(&a, 3);
    let prec = MultiplicativeSchwarzPreconditioner::new(vec![entry0, entry1], updater, 3, false)
        .expect("valid multiplicative preconditioner");

    let r = vec![2.0, 6.0, 8.0];
    let mut z = vec![0.0; 3];
    prec.apply(&r, &mut z);

    for &v in &z {
        assert!(v.is_finite(), "z contains non-finite value: {}", v);
    }

    let mut residual = vec![0.0; 3];
    a.apply(&z, &mut residual);
    let resid_norm: f64 = (0..3)
        .map(|i| (r[i] - residual[i]).powi(2))
        .sum::<f64>()
        .sqrt();
    let r_norm: f64 = r.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(
        resid_norm < r_norm,
        "Preconditioner did not reduce residual: {} >= {}",
        resid_norm,
        r_norm
    );
}

#[test]
fn test_symmetric_multiplicative_reduces_residual_more() {
    let a = DiagOperator {
        values: vec![2.0, 3.0, 4.0, 5.0],
    };

    let make_entries = || {
        let s0 = DiagLocalSolver::new(&[2.0, 3.0]);
        let c0 = SubdomainCore {
            global_indices: vec![0, 1],
            partition_weights: PartitionWeights::NonUniform(vec![1.0, 0.5]),
        };
        let s1 = DiagLocalSolver::new(&[3.0, 4.0]);
        let c1 = SubdomainCore {
            global_indices: vec![1, 2],
            partition_weights: PartitionWeights::NonUniform(vec![0.5, 0.5]),
        };
        let s2 = DiagLocalSolver::new(&[4.0, 5.0]);
        let c2 = SubdomainCore {
            global_indices: vec![2, 3],
            partition_weights: PartitionWeights::NonUniform(vec![0.5, 1.0]),
        };
        vec![
            SubdomainEntry::new(c0, s0),
            SubdomainEntry::new(c1, s1),
            SubdomainEntry::new(c2, s2),
        ]
    };

    let r = vec![4.0, 9.0, 12.0, 15.0];

    let updater_fwd = OperatorResidualUpdater::new(&a, 4);
    let prec_fwd = MultiplicativeSchwarzPreconditioner::new(make_entries(), updater_fwd, 4, false)
        .expect("valid forward multiplicative preconditioner");
    let mut z_fwd = vec![0.0; 4];
    prec_fwd.apply(&r, &mut z_fwd);

    let updater_sym = OperatorResidualUpdater::new(&a, 4);
    let prec_sym = MultiplicativeSchwarzPreconditioner::new(make_entries(), updater_sym, 4, true)
        .expect("valid symmetric multiplicative preconditioner");
    let mut z_sym = vec![0.0; 4];
    prec_sym.apply(&r, &mut z_sym);

    let resid_norm = |z: &[f64]| -> f64 {
        let mut az = vec![0.0; 4];
        a.apply(z, &mut az);
        (0..4).map(|i| (r[i] - az[i]).powi(2)).sum::<f64>().sqrt()
    };

    let rn_fwd = resid_norm(&z_fwd);
    let rn_sym = resid_norm(&z_sym);
    assert!(
        rn_sym <= rn_fwd + 1e-12,
        "Symmetric should not be worse: sym={} > fwd={}",
        rn_sym,
        rn_fwd
    );
}

#[test]
fn test_multiplicative_with_tridiag() {
    let n = 6;
    let a = TridiagOperator::new(n, 3.0);
    let make_solver = |diag_val: f64| DiagLocalSolver::new(&[diag_val, diag_val]);

    let e0 = SubdomainEntry::new(SubdomainCore::uniform(vec![0, 1]), make_solver(3.0));
    let e1 = SubdomainEntry::new(SubdomainCore::uniform(vec![2, 3]), make_solver(3.0));
    let e2 = SubdomainEntry::new(SubdomainCore::uniform(vec![4, 5]), make_solver(3.0));

    let updater = OperatorResidualUpdater::new(&a, n);
    let prec = MultiplicativeSchwarzPreconditioner::new(vec![e0, e1, e2], updater, n, true)
        .expect("valid multiplicative preconditioner");

    let r = vec![1.0; n];
    let mut z = vec![0.0; n];
    prec.apply(&r, &mut z);

    for (i, &v) in z.iter().enumerate() {
        assert!(v.is_finite(), "z[{}] = {} is not finite", i, v);
    }
    let z_norm: f64 = z.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(z_norm > 1e-14, "z is zero");

    let mut az = vec![0.0; n];
    a.apply(&z, &mut az);
    let resid_norm: f64 = (0..n).map(|i| (r[i] - az[i]).powi(2)).sum::<f64>().sqrt();
    let r_norm: f64 = r.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(
        resid_norm < r_norm,
        "No residual reduction: {} >= {}",
        resid_norm,
        r_norm
    );
}
