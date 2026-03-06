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
use within::{
    solve, GmresPrecond, LocalSolverConfig, OperatorRepr, SolverMethod, SolverParams,
};

// Two factors: 100 levels each, 10 000 observations
let factor_0: Vec<u32> = (0..10_000).map(|i| (i % 100) as u32).collect();
let factor_1: Vec<u32> = (0..10_000).map(|i| (i / 100) as u32).collect();
let y: Vec<f64> = (0..10_000).map(|i| i as f64 * 0.01).collect();

// Solve with the default solver: CG + additive Schwarz + implicit operator
let result = solve(
    &[factor_0.clone(), factor_1.clone()],
    &[100, 100],
    &y,
    &SolverParams::default(),
).expect("solve should succeed");
assert!(result.converged);
println!("default CG converged in {} iterations", result.iterations);

// Solve with GMRES + multiplicative Schwarz on the implicit operator
let params = SolverParams {
    method: SolverMethod::Gmres {
        preconditioner: Some(GmresPrecond::Multiplicative(LocalSolverConfig::gmres_default())),
        operator: OperatorRepr::Implicit,
        restart: 30,
    },
    tol: 1e-8,
    maxiter: 1000,
};
let result = solve(
    &[factor_0, factor_1],
    &[100, 100],
    &y,
    &params,
).expect("solve should succeed");
assert!(result.converged);
println!("GMRES converged in {} iterations", result.iterations);
```

## Feature flags

| Feature   | Default | Effect                                                                 |
|-----------|---------|------------------------------------------------------------------------|
| `ndarray` | yes     | Enables `from_array` constructors on observation stores for interop with `ndarray::ArrayView2`. |

To build without `ndarray`:

```
cargo add within --no-default-features
```

## Architecture

The crate is organized in four layers:

1. **`observation`** -- Storage backends for per-observation factor levels and
   weights. Three implementations (`FactorMajorStore`, `RowMajorStore`,
   `CompressedStore`) all satisfy the `ObservationStore` trait, allowing
   zero-cost generic dispatch.

2. **`domain`** -- Domain decomposition. `WeightedDesign` wraps a store with
   factor metadata; `build_local_domains` constructs factor-pair subdomains
   with partition-of-unity weights for the Schwarz preconditioner.

3. **`operator`** -- Linear algebra primitives. `Gramian` (explicit CSR) and
   `GramianOperator` (implicit D^T W D) for the normal equations; Schwarz
   preconditioner builders that wire approximate Cholesky local solvers into
   the generic `schwarz-precond` framework.

4. **`orchestrate`** -- End-to-end solve entry points (`solve`,
   `solve_weighted`, `solve_normal_equations`) with
   typed configuration (`SolverParams`, `SolverMethod`, `GmresPrecond`,
   `OperatorRepr`).

## License

MIT

## References

- Correia, S. (2017). Linear Models with High-Dimensional Fixed Effects: An
  Efficient and Feasible Estimator. *Working paper*.
  https://hdl.handle.net/10.2139/ssrn.3129010

- Gaure, S. (2013). OLS with Multiple High Dimensional Category Variables.
  *Computational Statistics & Data Analysis*, 66, 2--17.
  https://doi.org/10.1016/j.csda.2013.03.024

- Spielman, D. A. & Teng, S.-H. (2014). Nearly Linear Time Algorithms for
  Preconditioning and Solving Symmetric, Diagonally Dominant Linear Systems.
  *SIAM Journal on Matrix Analysis and Applications*, 35(3), 835--885.
  https://doi.org/10.1137/090771430
