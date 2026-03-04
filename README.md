# within

High-performance fixed-effects solver for econometric panel data. Solves
**y = D x + e** where D is a sparse categorical design matrix, using
iterative methods (LSMR, preconditioned CG, right-preconditioned GMRES) with
domain decomposition (Schwarz) preconditioners backed by approximate Cholesky
local solvers.

Designed for millions of observations with multiple high-dimensional fixed
effects.

## Python quickstart

Requires Python >= 3.11.

```bash
pip install within
```

```python
import within
import numpy as np

# Two factors with 500 levels each, 100k observations
categories = [
    np.random.randint(0, 500, size=100_000),
    np.random.randint(0, 500, size=100_000),
]
y = np.random.randn(100_000)

# Solve with default LSMR
result = within.solve(categories, [500, 500], y, within.LSMR())
print(f"converged={result.converged}  iters={result.iterations}  residual={result.residual:.2e}")

# Solve with preconditioned CG (additive Schwarz)
result = within.solve(categories, [500, 500], y, within.CG(preconditioner=within.OneLevelSchwarz()))
print(f"converged={result.converged}  iters={result.iterations}")
```

### Solver methods

| Class | Description |
|---|---|
| `LSMR(tol, maxiter, conlim)` | Handles singular Gramians natively. Good default. |
| `CG(tol, maxiter, preconditioner)` | Preconditioned conjugate gradient. |
| `GMRES(preconditioner, tol, maxiter, restart)` | Right-preconditioned GMRES. Requires a multiplicative Schwarz preconditioner. |

### Preconditioners

| Class | Description |
|---|---|
| `OneLevelSchwarz(...)` | Additive one-level Schwarz. Use with `CG`. |
| `MultiplicativeOneLevelSchwarz(...)` | Multiplicative one-level Schwarz. Use with `CG` or `GMRES`. |

## Rust quickstart

```bash
cargo add within
```

See [`crates/within/README.md`](crates/within/README.md) for Rust API usage and architecture details.

## Project structure

```
crates/
  schwarz-precond/   Generic domain decomposition library (traits, solvers, Schwarz preconditioners)
  within/            Core fixed-effects solver (observation stores, domains, operators, orchestration)
  within-py/         PyO3 bridge (cdylib → within._within)
python/within/       Python package re-exporting the Rust extension
benchmarks/          Python benchmark framework (18 suites)
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

- Correia, S. (2017). Linear Models with High-Dimensional Fixed Effects: An
  Efficient and Feasible Estimator.
  https://hdl.handle.net/10.2139/ssrn.3129010

- Gaure, S. (2013). OLS with Multiple High Dimensional Category Variables.
  *Computational Statistics & Data Analysis*, 66, 2--17.

- Spielman, D. A. & Teng, S.-H. (2014). Nearly Linear Time Algorithms for
  Preconditioning and Solving Symmetric, Diagonally Dominant Linear Systems.
  *SIAM Journal on Matrix Analysis and Applications*, 35(3), 835--885.

- Toselli & Widlund (2005). *Domain Decomposition Methods — Algorithms and Theory*. Springer.
