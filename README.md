# within

High-performance fixed-effects solver for econometric panel data. Solves
**y = D x** where D is a sparse categorical design matrix, using
iterative methods (preconditioned CG, right-preconditioned GMRES, LSMR) with
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
from within import solve, CG, LSMR
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

# Solve with LSMR (avoids preconditioner computation for simple problems)
result = solve(categories, y, LSMR())
print(f"converged={result.converged}  iters={result.iterations}")

# Weighted solve
weights = np.random.exponential(1.0, size=100_000)
result = solve(categories, y, weights=weights)
print(f"converged={result.converged}  iters={result.iterations}")
```

### Solver methods

| Class | Description |
|---|---|
| `CG(tol, maxiter, preconditioner)` | Preconditioned conjugate gradient. Default with Schwarz preconditioner. |
| `LSMR(tol, maxiter, conlim)` | Handles singular Gramians natively. |
| `GMRES(tol, maxiter, restart, preconditioner)` | Right-preconditioned GMRES. Requires a multiplicative Schwarz preconditioner. |

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

- Correia, Sergio. "A feasible estimator for linear models with multi-way fixed effects." *Preprint* at http://scorreia.com/research/hdfe.pdf (2016).
- Gao, Y., Kyng, R. & Spielman, D. A. (2025). AC(k): Robust Solution of
  Laplacian Equations by Randomized Approximate Cholesky Factorization.
  *SIAM Journal on Scientific Computing*.
- Gaure, Simen. "OLS with multiple high dimensional category variables." *Computational Statistics & Data Analysis* 66 (2013): 8-18.
- Guimaraes, Paulo, and Pedro Portugal. "A simple feasible procedure to fit models with high-dimensional fixed effects." *The Stata Journal* 10.4 (2010): 628-649.
- Toselli & Widlund (2005). *Domain Decomposition Methods — Algorithms and Theory*. Springer.
- Xu, J. (1992). Iterative Methods by Space Decomposition and Subspace
  Correction. *SIAM Review*, 34(4), 581--613.
