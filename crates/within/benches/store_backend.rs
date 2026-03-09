//! Benchmark: ArrayStore (zero-copy, C vs F order) vs FactorMajorStore (copy).

use std::time::Duration;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode};
use ndarray::{Array2, ShapeBuilder};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use within::config::{KrylovMethod, LocalSolverConfig, OperatorRepr, Preconditioner, SolverParams};
use within::domain::WeightedDesign;
use within::observation::{ArrayStore, FactorMajorStore, ObservationWeights};
use within::Solver;

const TOL: f64 = 1e-6;
const MAXITER: usize = 20;

struct Problem {
    /// C-contiguous (n_obs, n_factors) category array.
    categories_c: Array2<u32>,
    /// F-contiguous (n_obs, n_factors) category array (same data).
    categories_f: Array2<u32>,
    y: Vec<f64>,
    params: SolverParams,
    preconditioner: Option<Preconditioner>,
    label: String,
}

fn generate_problem(n_obs: usize, n_lev: &[usize], seed: u64) -> Problem {
    let n_factors = n_lev.len();
    let mut rng = SmallRng::seed_from_u64(seed);

    let mut categories_c = Array2::<u32>::zeros((n_obs, n_factors));
    for i in 0..n_obs {
        for (q, &nl) in n_lev.iter().enumerate() {
            categories_c[[i, q]] = rng.random_range(0..nl as u32);
        }
    }

    // F-contiguous copy of the same data.
    let categories_f = {
        let mut f = Array2::<u32>::zeros(categories_c.dim().f());
        f.assign(&categories_c);
        f
    };
    assert!(!categories_f.is_standard_layout(), "should be F-order");

    let y: Vec<f64> = (0..n_obs).map(|_| rng.random::<f64>()).collect();

    let params = SolverParams {
        krylov: KrylovMethod::Cg,
        operator: OperatorRepr::Implicit,
        tol: TOL,
        maxiter: MAXITER,
    };
    let preconditioner = Some(Preconditioner::Additive(LocalSolverConfig::solver_default()));

    let label = format!(
        "{}FE {} n={}",
        n_factors,
        n_lev
            .iter()
            .map(|n| format!("{n}"))
            .collect::<Vec<_>>()
            .join("x"),
        n_obs,
    );

    Problem {
        categories_c,
        categories_f,
        y,
        params,
        preconditioner,
        label,
    }
}

fn bench_store_backends(c: &mut Criterion) {
    let mut group = c.benchmark_group("store_backend");
    group.sample_size(10);
    group.sampling_mode(SamplingMode::Flat);
    group.warm_up_time(Duration::from_millis(100));
    group.measurement_time(Duration::from_millis(500));

    let cases: Vec<(usize, Vec<usize>, u64)> = vec![
        (100_000, vec![500, 500], 42),
        (500_000, vec![1000, 1000], 43),
        (1_000_000, vec![2000, 2000], 44),
        (1_000_000, vec![500, 500, 500], 45),
        (2_000_000, vec![5000, 5000], 46),
    ];

    for (n_obs, n_lev, seed) in &cases {
        let p = generate_problem(*n_obs, n_lev, *seed);
        let precond_ref = p.preconditioner.as_ref();

        // FactorMajorStore: copy columns from C-order array, contiguous factor_column.
        group.bench_function(BenchmarkId::new("FactorMajor", &p.label), |b| {
            b.iter(|| {
                let factor_levels: Vec<Vec<u32>> = (0..p.categories_c.ncols())
                    .map(|q| p.categories_c.column(q).to_vec())
                    .collect();
                let store =
                    FactorMajorStore::new(factor_levels, ObservationWeights::Unit, *n_obs).unwrap();
                let design = WeightedDesign::from_store(store).unwrap();
                let solver = Solver::from_design(design, &p.params, precond_ref).unwrap();
                let r = solver.solve(&p.y).unwrap();
                assert!(r.converged);
            });
        });

        // ArrayStore C-order: zero-copy, strided columns.
        group.bench_function(BenchmarkId::new("Array(C)", &p.label), |b| {
            b.iter(|| {
                let store =
                    ArrayStore::new(p.categories_c.view(), ObservationWeights::Unit).unwrap();
                let design = WeightedDesign::from_store(store).unwrap();
                let solver = Solver::from_design(design, &p.params, precond_ref).unwrap();
                let r = solver.solve(&p.y).unwrap();
                assert!(r.converged);
            });
        });

        // ArrayStore F-order: zero-copy, contiguous columns.
        group.bench_function(BenchmarkId::new("Array(F)", &p.label), |b| {
            b.iter(|| {
                let store =
                    ArrayStore::new(p.categories_f.view(), ObservationWeights::Unit).unwrap();
                let design = WeightedDesign::from_store(store).unwrap();
                let solver = Solver::from_design(design, &p.params, precond_ref).unwrap();
                let r = solver.solve(&p.y).unwrap();
                assert!(r.converged);
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_store_backends);
criterion_main!(benches);
