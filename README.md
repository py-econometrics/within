# within

`within `provides high-performance solvers for regression problems with high-dimensional fixed effects. 

Intuitively, `within`'s solvers are fast because they are tailored to the structure of fixed effects problems, which can be represented as a graph (as first noted by Correia, 2016), and make use of innovations in solvers for graph-structured linear systems (Gao et al, 2025).

More concretely, `within` solves problems of the form **y = D x**,  where D is a sparse one-hot encoded design matrix of fixed effects, and uses iterative methods (preconditioned CG, right-preconditioned GMRES, LSMR) with domain decomposition (Schwarz) preconditioners backed by approximate Cholesky local solvers.

Obtaining very fast solvers for this problem is useful in econometric applications as it allows to efficiently fit regression problems with high-dimensional fixed effects via the Frisch-Waugh-Lovell theorem. By the FWL theorem, estimating a regression `y = Xβ + Dα + ε` reduces to projecting out the fixed effects from both y and each column of X, and then to running OLS on the residuals. Each projection requires solving the least-squares problem `D'Dx = D'z`, which is exactly the problem `within` is designed to solve fast.

## Installation

Requires Python >= 3.11.

```bash
pip install within
```

## Python Quickstart

In this section, we show how to run a fixed effects regression via `within` and the *Frisch-Waugh-Lovell* theorem. 
We project out the fixed effects from y and each column of X, and then run ordinary least squares on the demeand residuals. 

```python
from within import solve
import numpy as np

np.random.seed(42)
n = 100_000

beta_true = np.array([1.0, -2.0, 0.5])
X = np.random.randn(n, 3)
categories = [np.random.randint(0, 500, size=n), np.random.randint(0, 200, size=n)]
alpha = np.random.randn(500)[categories[0]]
gamma = np.random.randn(200)[categories[1]]
y = X @ beta_true + alpha + gamma + 0.1 * np.random.randn(n)

# FWL step 1: project out fixed effects from y and X
cols = np.column_stack([y, X])
result = solve(categories, cols)
print(f"converged={result.converged}  iters={result.iterations}")
fe = result.x[categories[0]] + result.x[500:][categories[1]]
y_tilde, X_tilde = (cols - fe)[:, 0], (cols - fe)[:, 1:]

# FWL step 2: OLS on demeaned data
beta_hat = np.linalg.lstsq(X_tilde, y_tilde, rcond=None)[0]
print(f"True β:      {beta_true}")
print(f"Estimated β: {np.round(beta_hat, 4)}")
```

### API reference

```python
from within import CG, LSMR

# Schwarz-preconditioned CG (default)
result = solve(categories, cols, CG())

# LSMR (avoids preconditioner computation for simple problems)
result = solve(categories, cols, LSMR())

# Weighted solve
weights = np.random.exponential(1.0, size=n)
result = solve(categories, cols, weights=weights)
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
  within-py/         PyO3 bridge 
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
