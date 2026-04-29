# within

`within` provides high-performance solvers for projecting out high-dimensional fixed effects from regression problems.

By the Frisch-Waugh-Lovell theorem, estimating a regression of the form *y = Xβ + Dα + ε* reduces to a sequence of least-squares projections, one for y and one for each column of X, followed by a cheap regression fit on the resulting residuals. The projection step of solving the normal equations *D'Dx = D'z* is the computational bottleneck, which is the problem `within` is designed to solve.

`within`'s solvers are tailored to the structure of fixed effects problems, which can be represented as a graph (as first noted by Correia, 2016). Concretely, `within` uses iterative methods (preconditioned CG, right-preconditioned GMRES) with domain decomposition (Schwarz) preconditioners, backed by approximate Cholesky local solvers (Gao et al, 2025).

## Installation

You can install Python bindings from PyPi by running 

```bash
pip install within_py
```

## Python Quickstart

`within`'s main user-facing function is `solve`. Provide a 2-D `uint32` array of category codes (one column per fixed-effect factor) and a response vector `y`. The solver finds x in the normal equations **D'D x = D'y**, where D is the sparse categorical design matrix.

```python
from within import solve, solve_batch, CG, GMRES, AdditiveSchwarz, MultiplicativeSchwarz
import numpy as np

np.random.seed(1)
n = 100_000
fe = np.asfortranarray(np.column_stack([
    np.random.randint(0, 500, n).astype(np.uint32),
    np.random.randint(0, 200, n).astype(np.uint32),
]))
y = np.random.randn(n)

# Default: additive Schwarz + CG (recommended for most problems)
result = solve(fe, y)

# Multiplicative Schwarz + GMRES (fewer iterations, less parallelism)
result = solve(fe, y, config=GMRES(), preconditioner=MultiplicativeSchwarz())

# Weighted solve
result = solve(fe, y, weights=np.ones(n))
```

## R quickstart

Requires R and a Rust toolchain (`cargo` on `PATH`).

From the repository root, use `devtools` to install R dependencies and build the
package:

```r
install.packages("devtools")
devtools::install_deps("withinr/", dependencies = TRUE)
devtools::load_all("withinr/")
```

Example (FWL with two-way fixed effects):

```r
set.seed(42)
n <- 1000
n_firms <- 50L
n_years <- 20L

# 0-based fixed-effect ids (required by within)
firm <- rep(0:(n_firms - 1L), each = n_years)
year <- rep(0:(n_years - 1L), times = n_firms)
categories <- matrix(as.integer(c(firm, year)), nrow = n, ncol = 2)

beta <- 1.5
firm_fe <- rnorm(n_firms, sd = 3)[firm + 1L]
year_fe <- rnorm(n_years, sd = 1)[year + 1L]
x <- rnorm(n) + 0.3 * firm_fe
y <- beta * x + firm_fe + year_fe + rnorm(n, sd = 0.5)

res <- within_solve_batch(categories, cbind(y, x))
y_tilde <- res$demeaned[, 1]
x_tilde <- res$demeaned[, 2]
beta_hat <- sum(x_tilde * y_tilde) / sum(x_tilde^2)

print(beta_hat)
print(res$converged)
```

### Solver methods

| Class | Description |
|---|---|
| `CG(tol=1e-8, maxiter=1000, operator=Implicit)` | Conjugate gradient on the normal equations. Default solver. |
| `GMRES(tol=1e-8, maxiter=1000, restart=30, operator=Implicit)` | Right-preconditioned GMRES. |

### Preconditioners

| Class | Description |
|---|---|
| `AdditiveSchwarz(local_solver?)` | Additive one-level Schwarz. Works with CG and GMRES. |
| `MultiplicativeSchwarz(local_solver?)` | Multiplicative one-level Schwarz. GMRES only. |
| `Preconditioner.Off` | Disable preconditioning. |

Pass `None` (the default) to use additive Schwarz with the default local solver.

### Local solver configuration (advanced)

| Class | Description |
|---|---|
| `SchurComplement(approx_chol?, approx_schur?, dense_threshold=24)` | Schur complement reduction with approximate Cholesky on the reduced system. Default local solver. |
| `FullSddm(approx_chol?)` | Full bipartite SDDM factorized via approximate Cholesky. |
| `ApproxCholConfig(seed=0, split=1)` | Approximate Cholesky parameters. |
| `ApproxSchurConfig(seed=0, split=1)` | Approximate Schur complement sampling parameters. |

### Result types

**`SolveResult`**: `x` (coefficients), `demeaned` (residuals), `converged`, `iterations`, `residual`, `time_total`, `time_setup`, `time_solve`.

**`BatchSolveResult`**: Same fields, with `converged`, `iterations`, `residual`, and `time_solve` as lists (one entry per RHS).

## Rust API

```rust
use ndarray::Array2;
use within::{solve, SolverParams, KrylovMethod, Preconditioner, LocalSolverConfig};

let categories = /* Array2<u32> of shape (n_obs, n_factors) */;
let y: &[f64] = /* response vector */;

// Default: CG + additive Schwarz
let r = solve(categories.view(), &y, None, &SolverParams::default(), None)?;
assert!(r.converged);

// GMRES + multiplicative Schwarz
let params = SolverParams {
    krylov: KrylovMethod::Gmres { restart: 30 },
    ..SolverParams::default()
};
let precond = Preconditioner::Multiplicative(LocalSolverConfig::default());
let r = solve(categories.view(), &y, None, &params, Some(&precond))?;
```

Persistent solver — build once, solve many:

```rust
use within::Solver;

let solver = Solver::new(categories.view(), None, &SolverParams::default(), None)?;
let r1 = solver.solve(&y)?;
let r2 = solver.solve(&another_y)?;  // reuses preconditioner
```

| Type | Variants / Fields |
|---|---|
| `SolverParams` | `krylov: KrylovMethod`, `operator: OperatorRepr`, `tol: f64`, `maxiter: usize` |
| `KrylovMethod` | `Cg` (default), `Gmres { restart }` |
| `OperatorRepr` | `Implicit` (default, matrix-free D'WD), `Explicit` (CSR Gramian) |
| `Preconditioner` | `Additive(LocalSolverConfig)`, `Multiplicative(LocalSolverConfig)` |
| `LocalSolverConfig` | `SchurComplement { approx_chol, approx_schur, dense_threshold }`, `FullSddm { approx_chol }` |

### Lower-level types

The crate exposes its internals for advanced use:

| Module | Key types |
|---|---|
| `observation` | `FactorMajorStore`, `ArrayStore`, `ObservationStore` trait |
| `domain` | `WeightedDesign`, `FixedEffectsDesign`, `Subdomain` |
| `operator` | `Gramian` (CSR), `GramianOperator` (implicit), `DesignOperator`, `build_schwarz`, `FeSchwarz` |
| `solver` | `Solver<S: ObservationStore>` — generic persistent solver |

### Feature flags

| Feature | Default | Effect |
|---|---|---|
| `ndarray` | yes | Enables `from_array` constructors for `ndarray::ArrayView2` interop. |

## Project structure

```
crates/
  schwarz-precond/   Generic domain decomposition library (traits, solvers, Schwarz preconditioners)
  within/            Core fixed-effects solver (observation stores, domains, operators, orchestration)
  within-py/         PyO3 bridge (cdylib → within._within)
python/within/       Python package re-exporting the Rust extension
benchmarks/          Python benchmark framework
```

## Development

Uses [pixi](https://pixi.sh) as the task runner.

```bash
pixi run develop          # Build Rust extension (release mode)
pixi run test             # Rebuild + pytest
cargo test --workspace    # Rust tests only
cargo bench -p within     # Criterion benchmarks
pixi run bench run all    # Python benchmarks
```

Rust changes require rebuilding before running Python code (`pixi run develop`).

## License

MIT

## References

- Correia, Sergio. "A feasible estimator for linear models with multi-way fixed effects." *Preprint* at http://scorreia.com/research/hdfe.pdf (2016).
- Gao, Y., Kyng, R. & Spielman, D. A. (2025). AC(k): Robust Solution of Laplacian Equations by Randomized Approximate Cholesky Factorization. *SIAM Journal on Scientific Computing*.
- Toselli & Widlund (2005). *Domain Decomposition Methods — Algorithms and Theory*. Springer.
- Xu, J. (1992). Iterative Methods by Space Decomposition and Subspace Correction. *SIAM Review*, 34(4), 581--613.
