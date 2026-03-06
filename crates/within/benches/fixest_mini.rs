use std::time::Duration;

use approx_chol::Config;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use schwarz_precond::Operator;
use within::config::{LocalSolverConfig, OperatorRepr, SolverMethod, SolverParams};
use within::domain::WeightedDesign;
use within::observation::{FactorMajorStore, ObservationWeights};
use within::operator::gramian::{Gramian, GramianOperator};
use within::orchestrate::solve_normal_equations;

const MINI_MAXITER: usize = 20;
const MINI_TOL: f64 = 1e-6;

#[derive(Clone, Copy)]
enum FixestType {
    Simple,
    Difficult,
}

#[derive(Clone, Copy)]
struct Case {
    n_obs: usize,
    dgp_type: FixestType,
    n_fe: usize,
}

impl Case {
    fn label(&self) -> String {
        let kind = match self.dgp_type {
            FixestType::Simple => "simple",
            FixestType::Difficult => "difficult",
        };
        format!("n={} {} {}FE", self.n_obs, kind, self.n_fe)
    }
}

fn mini_cases() -> [Case; 6] {
    [
        Case {
            n_obs: 25_000,
            dgp_type: FixestType::Simple,
            n_fe: 2,
        },
        Case {
            n_obs: 25_000,
            dgp_type: FixestType::Difficult,
            n_fe: 2,
        },
        Case {
            n_obs: 50_000,
            dgp_type: FixestType::Simple,
            n_fe: 3,
        },
        Case {
            n_obs: 50_000,
            dgp_type: FixestType::Difficult,
            n_fe: 3,
        },
        Case {
            n_obs: 100_000,
            dgp_type: FixestType::Simple,
            n_fe: 2,
        },
        Case {
            n_obs: 100_000,
            dgp_type: FixestType::Simple,
            n_fe: 3,
        },
    ]
}

fn generate_fixest_like_case(
    case: Case,
    seed: u64,
) -> (WeightedDesign<FactorMajorStore>, Vec<f64>) {
    let mut rng = SmallRng::seed_from_u64(seed);
    let n_years = 10usize;
    let n_indiv_per_firm = 23usize;

    let n_indiv = ((case.n_obs as f64 / n_years as f64).round() as usize).max(1);
    let n_firm = ((n_indiv as f64 / n_indiv_per_firm as f64).round() as usize).max(1);

    let mut indiv_id = Vec::with_capacity(case.n_obs);
    let mut year = Vec::with_capacity(case.n_obs);
    let mut firm_id = Vec::with_capacity(case.n_obs);

    for i in 0..case.n_obs {
        indiv_id.push((i / n_years) as u32);
        year.push((i % n_years) as u32);
        let firm = match case.dgp_type {
            FixestType::Simple => rng.random_range(0..n_firm) as u32,
            FixestType::Difficult => (i % n_firm) as u32,
        };
        firm_id.push(firm);
    }

    let (factor_levels, n_levels): (Vec<Vec<u32>>, Vec<usize>) = if case.n_fe == 2 {
        (vec![indiv_id, year], vec![n_indiv, n_years])
    } else {
        (
            vec![indiv_id, year, firm_id],
            vec![n_indiv, n_years, n_firm],
        )
    };

    let store = FactorMajorStore::new(factor_levels, ObservationWeights::Unit, case.n_obs)
        .expect("valid factor-major store");
    let design = WeightedDesign::from_store(store, &n_levels).expect("valid design");

    let mut x_true = vec![0.0; design.n_dofs];
    for x in &mut x_true {
        *x = rng.random_range(-1.0..1.0);
    }

    let mut y = vec![0.0; case.n_obs];
    design.matvec_d(&x_true, &mut y);
    for yi in &mut y {
        *yi += 0.1 * rng.random_range(-1.0..1.0);
    }

    (design, y)
}

fn run_cg_1l(design: &WeightedDesign<FactorMajorStore>, y: &[f64], ac2: bool) {
    let approx_chol = if ac2 {
        Config {
            seed: 42,
            split_merge: Some(2),
        }
    } else {
        Config {
            seed: 42,
            split_merge: None,
        }
    };

    let params = SolverParams {
        method: SolverMethod::Cg {
            preconditioner: Some(LocalSolverConfig::SchurComplement {
                approx_chol,
                approx_schur: Some(within::ApproxSchurConfig::default()),
                dense_threshold: within::DEFAULT_DENSE_SCHUR_THRESHOLD,
            }),
            operator: OperatorRepr::Implicit,
        },
        tol: MINI_TOL,
        maxiter: MINI_MAXITER,
    };

    let mut rhs = vec![0.0; design.n_dofs];
    design.rmatvec_wdt(y, &mut rhs);
    let result = solve_normal_equations(design, &rhs, &params).expect("normal-equation solve");
    assert!(result.converged, "solver did not converge");
    assert!(result.final_residual.is_finite(), "non-finite residual");
    assert!(
        result.iterations < MINI_MAXITER,
        "solver hit max iterations"
    );
}

fn bench_fixest_mini(c: &mut Criterion) {
    let mut group = c.benchmark_group("fixest_mini_cg_1l");
    group.sample_size(10);
    group.sampling_mode(SamplingMode::Flat);
    group.warm_up_time(Duration::from_millis(50));
    group.measurement_time(Duration::from_millis(100));

    for case in mini_cases() {
        let label = case.label();
        let (design, y) = generate_fixest_like_case(case, 42);

        group.bench_function(BenchmarkId::new("CG-1L-AC", &label), |b| {
            b.iter(|| run_cg_1l(&design, &y, false));
        });
        group.bench_function(BenchmarkId::new("CG-1L-AC2", &label), |b| {
            b.iter(|| run_cg_1l(&design, &y, true));
        });
    }

    group.finish();
}

fn matvec_cases() -> [Case; 4] {
    [
        Case {
            n_obs: 500_000,
            dgp_type: FixestType::Difficult,
            n_fe: 2,
        },
        Case {
            n_obs: 500_000,
            dgp_type: FixestType::Difficult,
            n_fe: 3,
        },
        Case {
            n_obs: 2_000_000,
            dgp_type: FixestType::Difficult,
            n_fe: 2,
        },
        Case {
            n_obs: 2_000_000,
            dgp_type: FixestType::Difficult,
            n_fe: 3,
        },
    ]
}

fn bench_matvec(c: &mut Criterion) {
    let mut group = c.benchmark_group("matvec_explicit_vs_implicit");
    group.sample_size(20);
    group.sampling_mode(SamplingMode::Flat);
    group.warm_up_time(Duration::from_millis(100));
    group.measurement_time(Duration::from_millis(500));

    for case in matvec_cases() {
        let label = case.label();
        let (design, _y) = generate_fixest_like_case(case, 42);
        let n_dofs = design.n_dofs;

        let mut rng = SmallRng::seed_from_u64(123);
        let x: Vec<f64> = (0..n_dofs).map(|_| rng.random_range(-1.0..1.0)).collect();

        // Implicit: D^T W D through observation space
        let implicit_op = GramianOperator::new(&design);

        // Explicit: CSR SpMV
        let explicit_g = Gramian::build(&design);

        // Sanity check: both produce the same result
        let mut y_impl = vec![0.0; n_dofs];
        let mut y_expl = vec![0.0; n_dofs];
        implicit_op.apply(&x, &mut y_impl);
        explicit_g.apply(&x, &mut y_expl);
        for (a, b) in y_impl.iter().zip(y_expl.iter()) {
            assert!((a - b).abs() < 1e-10 * a.abs().max(1.0));
        }

        group.bench_function(BenchmarkId::new("implicit", &label), |b| {
            let mut y = vec![0.0; n_dofs];
            b.iter(|| implicit_op.apply(&x, &mut y));
        });

        group.bench_function(BenchmarkId::new("explicit", &label), |b| {
            let mut y = vec![0.0; n_dofs];
            b.iter(|| explicit_g.apply(&x, &mut y));
        });
    }

    group.finish();
}

criterion_group!(benches, bench_fixest_mini, bench_matvec);
criterion_main!(benches);
