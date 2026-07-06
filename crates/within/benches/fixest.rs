use std::time::Duration;

use criterion::measurement::WallTime;
use criterion::{
    criterion_group, criterion_main, BenchmarkGroup, BenchmarkId, Criterion, SamplingMode,
};
use within::config::{
    ApproxCholConfig, LocalSolverConfig, LsmrOptions, PreconditionerConfig, ReductionStrategy,
};
use within::{Design, Solver};

#[path = "shared/fixest_dgp.rs"]
mod fixest_dgp;
use fixest_dgp::generate_fixest_like_case;

// ===========================================================================
// Shared types and helpers
// ===========================================================================

const MAXITER: usize = 200;
const TOL: f64 = 1e-6;

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

    fn generate(&self, seed: u64) -> (Design<'static>, Vec<f64>) {
        let difficult = matches!(self.dgp_type, FixestType::Difficult);
        generate_fixest_like_case(self.n_obs, self.n_fe, difficult, seed)
    }
}

fn one_level_local_solver(ac2: bool) -> LocalSolverConfig {
    let mut cfg = LocalSolverConfig::default();
    if ac2 {
        cfg.approx_chol = ApproxCholConfig {
            split_merge: Some(2),
            ..Default::default()
        };
    } else {
        cfg.approx_chol = ApproxCholConfig::default();
    }
    cfg
}

fn configure_group<'a>(
    c: &'a mut Criterion,
    name: &str,
    sample_size: usize,
    measurement_ms: u64,
) -> BenchmarkGroup<'a, WallTime> {
    let mut group = c.benchmark_group(name);
    group.sample_size(sample_size);
    group.measurement_time(Duration::from_millis(measurement_ms));
    group.sampling_mode(SamplingMode::Flat);
    group
}

fn run_smoke(
    group: &mut BenchmarkGroup<'_, WallTime>,
    label: &str,
    design: &Design<'_>,
    y: &[f64],
) {
    group.bench_function(BenchmarkId::new(label, ""), |b| {
        b.iter(|| run_lsmr_one_level(design, y, false))
    });
}

fn run_lsmr_one_level(design: &Design<'_>, y: &[f64], ac2: bool) {
    let params = LsmrOptions {
        tol: TOL,
        maxiter: MAXITER,
        ..Default::default()
    };
    let cfg = one_level_local_solver(ac2);
    let precond = PreconditionerConfig::Additive {
        local_solver: cfg,
        reduction: ReductionStrategy::Auto,
    };
    let solver = Solver::new(design.clone(), None, &precond).expect("solver build");
    let _ = solver.solve(y, &params).expect("solve");
}

fn smoke_cases() -> [Case; 8] {
    [
        Case {
            n_obs: 100_000,
            dgp_type: FixestType::Simple,
            n_fe: 2,
        },
        Case {
            n_obs: 100_000,
            dgp_type: FixestType::Difficult,
            n_fe: 2,
        },
        Case {
            n_obs: 100_000,
            dgp_type: FixestType::Simple,
            n_fe: 3,
        },
        Case {
            n_obs: 100_000,
            dgp_type: FixestType::Difficult,
            n_fe: 3,
        },
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
    ]
}

fn bench_fixest_smoke_lsmr_1l(c: &mut Criterion) {
    let mut group = configure_group(c, "fixest_smoke_lsmr_1l", 100, 200);
    for case in smoke_cases() {
        let label = case.label();
        let (design, y) = case.generate(42);
        group.bench_function(BenchmarkId::new("LSMR-AC", &label), |b| {
            b.iter(|| run_lsmr_one_level(&design, &y, false));
        });
        group.bench_function(BenchmarkId::new("LSMR-AC2", &label), |b| {
            b.iter(|| run_lsmr_one_level(&design, &y, true));
        });
    }
    group.finish();
}

fn mini_cases() -> [Case; 6] {
    [
        Case {
            n_obs: 10_000,
            dgp_type: FixestType::Simple,
            n_fe: 2,
        },
        Case {
            n_obs: 10_000,
            dgp_type: FixestType::Difficult,
            n_fe: 2,
        },
        Case {
            n_obs: 10_000,
            dgp_type: FixestType::Simple,
            n_fe: 3,
        },
        Case {
            n_obs: 10_000,
            dgp_type: FixestType::Difficult,
            n_fe: 3,
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
    ]
}

fn bench_fixest_mini(c: &mut Criterion) {
    let mut group = configure_group(c, "fixest_mini_lsmr_1l", 50, 100);
    for case in mini_cases() {
        let label = case.label();
        let (design, y) = case.generate(42);
        run_smoke(&mut group, &format!("LSMR-{label}"), &design, &y);
    }
    group.finish();
}

criterion_group!(
    name = smoke_benches;
    config = Criterion::default();
    targets = bench_fixest_smoke_lsmr_1l,
);
criterion_group!(mini_benches, bench_fixest_mini);
criterion_main!(smoke_benches, mini_benches);
