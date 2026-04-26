# within

Solves linear fixed-effects models **y = D x + e** where D is a sparse
categorical design matrix. The normal equations **G x = D^T W y** (with
G = D^T W D) are solved via preconditioned CG or right-preconditioned GMRES
with additive or multiplicative Schwarz preconditioners backed by approximate
Cholesky local solvers. Designed for econometric panel data with
millions of observations and multiple high-dimensional fixed effects.

## Install

```
cargo add within
```

## Quick example

```rust
use ndarray::Array2;
use within::{solve, SolverParams, KrylovMethod, Preconditioner, LocalSolverConfig};

// Two factors: 100 levels each, 10 000 observations
let n_obs = 10_000usize;
let mut categories = Array2::<u32>::zeros((n_obs, 2));
for i in 0..n_obs {
    categories[[i, 0]] = (i % 100) as u32;
    categories[[i, 1]] = (i / 100) as u32;
}
let y: Vec<f64> = (0..n_obs).map(|i| i as f64 * 0.01).collect();

// Solve with the default solver: CG + additive Schwarz + implicit operator
let result = solve(categories.view(), &y, None, &SolverParams::default(), None)
    .expect("solve should succeed");
assert!(result.converged);
println!("CG converged in {} iterations", result.iterations);

// GMRES with multiplicative Schwarz
let params = SolverParams {
    krylov: KrylovMethod::Gmres { restart: 30 },
    ..SolverParams::default()
};
let precond = Preconditioner::Multiplicative(LocalSolverConfig::default());
let result = solve(categories.view(), &y, None, &params, Some(&precond))
    .expect("solve should succeed");
assert!(result.converged);
println!("GMRES converged in {} iterations", result.iterations);
```

## Feature flags

| Feature   | Default | Effect                                                                 |
|-----------|---------|------------------------------------------------------------------------|
| `ndarray` | yes     | Enables `from_array` constructors on observation stores for interop with `ndarray::ArrayView2`. |

## Architecture

The crate is organized in four layers:

1. **`observation`** — Per-observation factor levels and weights via
   `FactorMajorStore` and the `ObservationStore` trait.

2. **`domain`** — Domain decomposition. `WeightedDesign` wraps a store with
   factor metadata; `build_local_domains` constructs factor-pair subdomains
   with partition-of-unity weights for the Schwarz preconditioner.

3. **`operator`** — Linear algebra primitives. `Gramian` (explicit CSR) and
   `GramianOperator` (implicit D^T W D) for the normal equations; Schwarz
   preconditioner builders that wire approximate Cholesky local solvers into
   the generic `schwarz-precond` framework.

4. **`orchestrate`** — End-to-end solve entry points (`solve`, `solve_batch`)
   with typed configuration (`SolverParams`, `KrylovMethod`, `Preconditioner`,
   `OperatorRepr`).

## License

MIT

## References

- Correia, Sergio. "A feasible estimator for linear models with multi-way fixed effects." *Preprint* at http://scorreia.com/research/hdfe.pdf (2016).
- Gao, Y., Kyng, R. & Spielman, D. A. (2025). AC(k): Robust Solution of Laplacian Equations by Randomized Approximate Cholesky Factorization. *SIAM Journal on Scientific Computing*.
- Toselli & Widlund (2005). *Domain Decomposition Methods — Algorithms and Theory*. Springer.
- Xu, J. (1992). Iterative Methods by Space Decomposition and Subspace Correction. *SIAM Review*, 34(4), 581--613.
