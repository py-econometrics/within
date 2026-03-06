# within

High-performance fixed-effects solver for econometric panel data. Solves
**y = D x** where D is a sparse categorical design matrix, using
iterative methods (preconditioned CG and right-preconditioned GMRES) with
domain decomposition (Schwarz) preconditioners backed by approximate Cholesky
local solvers.

Designed for millions of observations with multiple high-dimensional fixed
effects.

## Python quickstart

Requires Python >= 3.9.

```bash
pip install within
```

```python
from within import solve, CG, GMRES, MultiplicativeSchwarz
import numpy as np

# Two factors with 500 levels each, 100k observations
categories = [
    np.random.randint(0, 500, size=100_000),
    np.random.randint(0, 500, size=100_000),
]
y = np.random.randn(100_000)

# Solve with Schwarz-preconditioned CG (default)
result = solve(categories, y)
print(f"converged={result.converged}  iters={result.iterations}  residual={result.residual:.2e}")

# Solve with unpreconditioned CG
result = solve(categories, y, CG(preconditioner=None))
print(f"converged={result.converged}  iters={result.iterations}")

# Solve with GMRES + multiplicative Schwarz
result = solve(categories, y, GMRES(preconditioner=MultiplicativeSchwarz()))
print(f"converged={result.converged}  iters={result.iterations}")

# Weighted solve
weights = np.random.exponential(1.0, size=100_000)
result = solve(categories, y, weights=weights)
print(f"converged={result.converged}  iters={result.iterations}")
```

### Solver methods

| Class | Description |
|---|---|
| `CG(tol, maxiter, preconditioner, operator)` | Conjugate gradient on the normal equations. Default is additive Schwarz with the implicit operator. |
| `GMRES(tol, maxiter, restart, preconditioner, operator)` | Right-preconditioned GMRES on the normal equations. |

### Preconditioners

| Class | Description |
|---|---|
| `AdditiveSchwarz(...)` | Additive one-level Schwarz. Use with `CG` or `GMRES`. |
| `MultiplicativeSchwarz(...)` | Multiplicative one-level Schwarz. Use with `GMRES` only. |

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

- Gao, Y., Kyng, R. & Spielman, D. A. (2025). AC(k): Robust Solution of
  Laplacian Equations by Randomized Approximate Cholesky Factorization.
  *SIAM Journal on Scientific Computing*.

- Xu, J. (1992). Iterative Methods by Space Decomposition and Subspace
  Correction. *SIAM Review*, 34(4), 581--613.

- Toselli & Widlund (2005). *Domain Decomposition Methods — Algorithms and Theory*. Springer.
