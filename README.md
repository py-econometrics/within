# within

`within` provides high-performance solvers for projecting out high-dimensional fixed effects from regression problems.

 By the Frisch-Waugh-Lovell theorem, estimating a regression of the form *y = Xβ + Dα + ε* reduces to a sequence of least-squares projections, one for y and one for each column of X, followed by a cheap regression fit on the resulting residuals. The projection step of solving the normal equations *D'Dx = D'z* is the computational bottleneck, which is the problem `within` is designed to solve.

`within`'s solvers are tailored to the structure of fixed effects problems, which can be represented as a graph (as first noted by Correia, 2016), and make use of innovations in solvers for graph-structured linear systems (Gao et al, 2025). Concretely, `within` uses iterative methods (preconditioned CG, right-preconditioned GMRES, LSMR) with domain decomposition (Schwarz) preconditioners, backed by approximate Cholesky local solvers.

## Installation

Requires Python >= 3.11.

```bash
pip install within
```

## Python Quickstart

`within`'s main user facing function is `solve`. To use it, you provide an integer 2D array of fixed effects `x` and a 1D array `y` to solve the linear system **y = D x**, where D is a large sparse matrix. For fixed effects problems, `x` is a matrix of integer encodings of fixed effects, and y any regression column. 

```python
from within import solve, CG, LSMR
import numpy as np

# set up a dgp
np.random.seed(42)
n = 100_000

beta_true = np.array([1.0, -2.0, 0.5])
X = np.random.randn(n, 3)
fixed_effects = [np.random.randint(0, 500, size=n), np.random.randint(0, 200, size=n)]
alpha = np.random.randn(500)[fixed_effects[0]]
gamma = np.random.randn(200)[fixed_effects[1]]
y = X @ beta_true + alpha + gamma + 0.1 * np.random.randn(n)
weights = np.random.exponential(1.0, size=n)

# Schwarz-preconditioned CG (default)
result = solve(fixed_effects, y, CG())
print(result.x)
# LSMR (avoids preconditioner computation for simple problems)
result = solve(fixed_effects, y, LSMR())
print(result.x)
# Weighted solve
result = solve(fixed_effects, y, weights=weights)
print(result.x)
```

Next, we show how to run a fixed effects regression via `within` and the *Frisch-Waugh-Lovell* theorem. 
We project out the fixed effects from y and each column of X, and then run ordinary least squares on the demeand residuals. 

```python
# FWL step 1 and 2: solve for fixed-effect coefficients and compute residuals
all_cols = np.column_stack([y, X])
residuals = np.empty_like(all_cols)
for j in range(all_cols.shape[1]):
    result = solve(fixed_effects, all_cols[:, j])
    fitted = result.x[fixed_effects[0]] + result.x[500:][fixed_effects[1]]
    residuals[:, j] = all_cols[:, j] - fitted
y_tilde, X_tilde = residuals[:, 0], residuals[:, 1:]

# FWL step 3: OLS on demeaned data
beta_hat = np.linalg.lstsq(X_tilde, y_tilde, rcond=None)[0]
print(f"True β:      {beta_true}")
print(f"Estimated β: {np.round(beta_hat, 4)}")
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
