use std::time::Duration;

use approx_chol::Config;
use criterion::measurement::WallTime;
use criterion::{
    criterion_group, criterion_main, BenchmarkGroup, BenchmarkId, Criterion, SamplingMode,
};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use within::config::{GmresPrecond, LocalSolverConfig, OperatorRepr, SolverMethod, SolverParams};
use within::domain::WeightedDesign;
use within::observation::{FactorMajorStore, ObservationWeights};
use within::orchestrate::solve_least_squares;

const SMOKE_MAXITER: usize = 20;
const SMOKE_TOL: f64 = 1e-6;
const SMOKE_GMRES_RESTART: usize = 30;

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

fn smoke_cases() -> [Case; 8] {
    [
        Case {
            n_obs: 1_000_000,
            dgp_type: FixestType::Simple,
            n_fe: 2,
        },
        Case {
            n_obs: 1_000_000,
            dgp_type: FixestType::Difficult,
            n_fe: 2,
        },
        Case {
            n_obs: 1_000_000,
            dgp_type: FixestType::Simple,
            n_fe: 3,
        },
        Case {
            n_obs: 1_000_000,
            dgp_type: FixestType::Difficult,
            n_fe: 3,
        },
        Case {
            n_obs: 2_000_000,
            dgp_type: FixestType::Simple,
            n_fe: 2,
        },
        Case {
            n_obs: 2_000_000,
            dgp_type: FixestType::Difficult,
            n_fe: 2,
        },
        Case {
            n_obs: 2_000_000,
            dgp_type: FixestType::Simple,
            n_fe: 3,
        },
        Case {
            n_obs: 2_000_000,
            dgp_type: FixestType::Difficult,
            n_fe: 3,
        },
    ]
}

fn configure_smoke_group<'a>(c: &'a mut Criterion, name: &str) -> BenchmarkGroup<'a, WallTime> {
    let mut group = c.benchmark_group(name);
    group.sample_size(10);
    group.sampling_mode(SamplingMode::Flat);
    group.warm_up_time(Duration::from_millis(100));
    group.measurement_time(Duration::from_millis(200));
    group
}

fn one_level_local_solver(ac2: bool) -> LocalSolverConfig {
    let approx_chol = if ac2 {
        Config {
            seed: 42,
            split_merge: Some(2),
            ..Default::default()
        }
    } else {
        Config {
            seed: 42,
            ..Default::default()
        }
    };
    LocalSolverConfig::SchurComplement {
        approx_chol,
        approx_schur: Some(within::ApproxSchurConfig::default()),
        dense_threshold: within::DEFAULT_DENSE_SCHUR_THRESHOLD,
    }
}

fn run_smoke(
    design: &WeightedDesign<FactorMajorStore>,
    y: &[f64],
    params: &SolverParams,
    label: &str,
) {
    let result = solve_least_squares(design, y, params).expect("least-squares solve");
    assert!(result.converged, "{label}: solver did not converge");
    assert!(
        result.final_residual.is_finite(),
        "{label}: non-finite residual"
    );
    assert!(
        result.iterations < SMOKE_MAXITER,
        "{label}: solver hit max iterations"
    );
}

fn run_cg_one_level(design: &WeightedDesign<FactorMajorStore>, y: &[f64], ac2: bool) {
    let cfg = one_level_local_solver(ac2);
    let params = SolverParams {
        method: SolverMethod::Cg {
            preconditioner: Some(cfg),
            operator: OperatorRepr::Implicit,
        },
        tol: SMOKE_TOL,
        maxiter: SMOKE_MAXITER,
    };
    let label = if ac2 { "CG(1L,AC2)" } else { "CG(1L,AC)" };
    run_smoke(design, y, &params, label);
}

fn run_gmres_multiplicative_one_level(
    design: &WeightedDesign<FactorMajorStore>,
    y: &[f64],
    ac2: bool,
    operator: OperatorRepr,
) {
    let cfg = one_level_local_solver(ac2);
    let params = SolverParams {
        method: SolverMethod::Gmres {
            preconditioner: Some(GmresPrecond::Multiplicative(cfg)),
            operator,
            restart: SMOKE_GMRES_RESTART,
        },
        tol: SMOKE_TOL,
        maxiter: SMOKE_MAXITER,
    };
    let label = if ac2 {
        "GMRES(M1L,AC2)"
    } else {
        "GMRES(M1L,AC)"
    };
    run_smoke(design, y, &params, label);
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

fn bench_fixest_smoke_cg_1l(c: &mut Criterion) {
    let mut group = configure_smoke_group(c, "fixest_smoke_cg_1l");

    for case in smoke_cases() {
        let label = case.label();
        let (design, y) = generate_fixest_like_case(case, 42);

        group.bench_function(BenchmarkId::new("CG-1L-AC", &label), |b| {
            b.iter(|| run_cg_one_level(&design, &y, false));
        });
        group.bench_function(BenchmarkId::new("CG-1L-AC2", &label), |b| {
            b.iter(|| run_cg_one_level(&design, &y, true));
        });
    }

    group.finish();
}

fn bench_fixest_smoke_other_1l(c: &mut Criterion) {
    let mut group = configure_smoke_group(c, "fixest_smoke_other_1l");

    for case in smoke_cases() {
        let label = case.label();
        let (design, y) = generate_fixest_like_case(case, 42);

        group.bench_function(BenchmarkId::new("GMRES-M1L-AC-impl", &label), |b| {
            b.iter(|| {
                run_gmres_multiplicative_one_level(&design, &y, false, OperatorRepr::Implicit)
            });
        });
        group.bench_function(BenchmarkId::new("GMRES-M1L-AC2-impl", &label), |b| {
            b.iter(|| {
                run_gmres_multiplicative_one_level(&design, &y, true, OperatorRepr::Implicit)
            });
        });
        group.bench_function(BenchmarkId::new("GMRES-M1L-AC", &label), |b| {
            b.iter(|| {
                run_gmres_multiplicative_one_level(&design, &y, false, OperatorRepr::Explicit)
            });
        });
        group.bench_function(BenchmarkId::new("GMRES-M1L-AC2", &label), |b| {
            b.iter(|| {
                run_gmres_multiplicative_one_level(&design, &y, true, OperatorRepr::Explicit)
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_fixest_smoke_cg_1l,
    bench_fixest_smoke_other_1l
);
criterion_main!(benches);
