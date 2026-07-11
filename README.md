# within

`within` provides high-performance solvers for projecting out high-dimensional fixed effects from regression problems.

By the Frisch-Waugh-Lovell theorem, estimating a regression of the form *y = Xβ + Dα + ε* reduces to a sequence of least-squares projections, one for y and one for each column of X, followed by a cheap regression fit on the resulting residuals. The projection step of solving the normal equations *D'Dx = D'z* is the computational bottleneck, which is the problem `within` is designed to solve.

`within`'s solvers are tailored to the structure of fixed effects problems, which can be represented as a graph (as first noted by Correia, 2016). Concretely, `within` uses modified LSMR with a domain decomposition (Schwarz) preconditioner, backed by approximate Cholesky local solvers (Gao et al, 2025).

## Installation

You can install Python bindings from PyPi by running 

```bash
pip install within_py
```

## Python Quickstart

`within`'s main user-facing function is `solve`. Provide a 2-D `uint32` array of category codes (one column per fixed-effect factor) and a response vector `y`. The solver finds x in the normal equations **D'D x = D'y**, where D is the sparse categorical design matrix.

```python
import numpy as np
from within import solve, solve_batch, LsmrOptions, PreconditionerConfig

np.random.seed(1)
n = 100_000
fe = np.asfortranarray(np.column_stack([
    np.random.randint(0, 500, n).astype(np.uint32),
    np.random.randint(0, 200, n).astype(np.uint32),
]))
y = np.random.randn(n)

# Default: additive Schwarz + LSMR
result = solve(fe, y)

# Custom tolerance / iteration cap
result = solve(fe, y, options=LsmrOptions(tol=1e-10, maxiter=2000))

# Weighted solve
result = solve(fe, y, weights=np.ones(n))

# Opt into diagonal/Jacobi preconditioning
result = solve(fe, y, preconditioner=PreconditionerConfig.Diagonal)
```

## R quickstart

Requires R and a Rust toolchain (`cargo` on `PATH`).

From the repository root, use `devtools` to install R dependencies and build the
package:

```r
install.packages("devtools")
Sys.setenv(NOT_CRAN = "true")
devtools::install_deps("withinr/", dependencies = TRUE)
devtools::load_all("withinr/")
```

Example (FWL with two-way fixed effects):

```r
set.seed(42)
n <- 1000
n_firms <- 50L
n_years <- 20L

# 1-based fixed-effect ids in R
firm <- rep(seq_len(n_firms), each = n_years)
year <- rep(seq_len(n_years), times = n_firms)
categories <- cbind(firm, year)

beta <- 1.5
firm_fe <- rnorm(n_firms, sd = 3)[firm]
year_fe <- rnorm(n_years, sd = 1)[year]
x <- rnorm(n) + 0.3 * firm_fe
y <- beta * x + firm_fe + year_fe + rnorm(n, sd = 0.5)

res <- withinr::solve_batch(categories, cbind(y, x))
y_tilde <- res$demeaned[, 1]
x_tilde <- res$demeaned[, 2]
beta_hat <- sum(x_tilde * y_tilde) / sum(x_tilde^2)

print(beta_hat)
print(res$converged)
```

| Function | Description |
|---|---|
| `solve(categories, y, options?, weights?, preconditioner?)` | Solve a single right-hand side. Returns a list shaped like `SolveResult`. |
| `solve_batch(categories, Y, options?, weights?, preconditioner?)` | Solve multiple RHS vectors in parallel. `Y` has shape `(n_obs, k)`. |

For repeated solves with the same design matrix, `Solver` builds the preconditioner once and reuses it. In R, the solver is an environment with methods.

```r
solver <- withinr::Solver(categories)
r <- solver$solve(y)
r <- solver$solve_batch(cbind(y, x))

precond <- solver$preconditioner()
payload <- precond$serialize()
solver2 <- withinr::Solver(categories, preconditioner = withinr::Preconditioner(payload))
```

| Property / Method | Description |
|---|---|
| `Solver(categories, weights?, preconditioner?)` | Build solver. Factorizes the preconditioner at construction. |
| `$solve(y, options?)` | Solve a single RHS with the given LSMR tuning. |
| `$solve_batch(Y, options?)` | Solve multiple RHS columns in parallel. |
| `$preconditioner()` | Return the built `Preconditioner`, or `NULL`. Reuse via `Solver(categories, preconditioner=p)`. |


### Solver configuration

| Class | Description |
|---|---|
| `LsmrOptions(tol=1e-8, maxiter=1000, local_size=None)` / `LsmrOptions(tol = 1e-8, maxiter = 1000L, local_size = NULL)` | Modified LSMR. `local_size` enables windowed reorthogonalization. |

### Preconditioner (5-form Union)

The `preconditioner` argument accepts any of:

| Form | Meaning |
|---|---|
| `None` / `NULL` (default) | Library default — Additive Schwarz with sensible defaults. |
| `PreconditionerConfig.Off` / `PreconditionerConfig$Off` | Explicit identity — solve unpreconditioned. |
| `PreconditionerConfig.Additive` / `PreconditionerConfig$Additive` | Additive Schwarz shortcut, equivalent to the default. |
| `PreconditionerConfig.Diagonal` / `PreconditionerConfig$Diagonal` | Diagonal/Jacobi preconditioner using `diag(D^T W D)^{-1}`. |
| `AdditiveSchwarz(local_solver?, reduction?)` | Tuned Schwarz config — import from `within.config`. |
| `Preconditioner` instance | Reuse a previously-built preconditioner across solvers. |

### Local solver configuration (advanced — `within.config`)

| Class | Description |
|---|---|
| `LocalSolverConfig(approx_chol?, approx_schur?, dense_threshold=24)` | Schur reduction + approximate Cholesky. Omit `approx_schur` for the library-default approximate variant; pass `approx_schur=None` to request an exact Schur (slower, used for validation). |
| `ApproxCholConfig(seed=0, split_merge=None)` | Approximate Cholesky parameters. |
| `ApproxSchurConfig(seed=0, split=1)` | Approximate Schur complement sampling parameters. |
| `ReductionStrategy` enum | `Auto` (default), `AtomicScatter`, `ParallelReduction`. |

### Result types

**`SolveResult`**: `x` (coefficients), `demeaned` (residuals), `converged`, `iterations`, `residual`, `time_total`, `time_setup`, `time_solve`.

**`BatchSolveResult`**: Same fields, with `converged`, `iterations`, `residual`, and `time_solve` as lists (one entry per RHS).

## Rust API

```rust
use ndarray::Array2;
use within::{solve, LsmrOptions, PreconditionerConfig};
use within::config::{LocalSolverConfig, ReductionStrategy};

let categories = /* Array2<u32> of shape (n_obs, n_factors) */;
let y: &[f64] = /* response vector */;

// Default: LSMR + additive Schwarz (None → library default)
let r = solve(categories.view(), &y, None, &LsmrOptions::default(), None)?;
assert!(r.converged);

// Tighter tolerance with an explicit additive preconditioner
let lsmr = LsmrOptions { tol: 1e-10, ..LsmrOptions::default() };
let precond = PreconditionerConfig::Additive {
    local_solver: LocalSolverConfig::default(),
    reduction: ReductionStrategy::default(),
};
let r = solve(categories.view(), &y, None, &lsmr, &precond)?;

// Opt into diagonal/Jacobi preconditioning
let diagonal = PreconditionerConfig::Diagonal;
let r = solve(categories.view(), &y, None, &lsmr, &diagonal)?;
```

Persistent solver — build once, solve many:

```rust
use within::Solver;

let solver = Solver::new(categories.view(), None, None)?;
let r1 = solver.solve(&y, &LsmrOptions::default())?;
let r2 = solver.solve(&another_y, &LsmrOptions::default())?;  // reuses preconditioner
```

Two-channel preconditioner signaling: `Option<&PreconditionerConfig>` where
`None` is the library default and `Some(PreconditionerConfig::Off)` is the
explicit identity preconditioner.

| Type | Variants / Fields |
|---|---|
| `LsmrOptions` | `{ tol: f64, maxiter: usize, local_size: Option<usize> }` |
| `PreconditionerConfig` | `Off` \| `Additive { local_solver: LocalSolverConfig, reduction: ReductionStrategy }` \| `Diagonal` (`#[non_exhaustive]`) |
| `LocalSolverConfig` | `{ approx_chol, approx_schur, dense_threshold }` |
| `Preconditioner` | Opaque built handle — reuse via `Solver::new(.., precond)` (owned or `&`) |

### Lower-level access

| Module | Visibility | Key types |
|---|---|---|
| `within::config` | public | `LsmrOptions`, `PreconditionerConfig`, `LocalSolverConfig`, `ApproxCholConfig`, `ApproxSchurConfig`, `ReductionStrategy` |
| `within::observation` | public | `Store` trait, `FactorMajorStore`, `ArrayStore`, `FactorMeta` |
| `within::error` | public | `WithinError`, `BuildError`, `SolveError` |
| `domain` / `operator` / `solver` / `orchestrate` | `pub(crate)` | implementation layers — public items are re-exported at the crate root |

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
  within-r/          Workspace mirror for the extendr bridge
python/within/       Python package re-exporting the Rust extension
withinr/             R package using the published within 0.2.0 crate
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
