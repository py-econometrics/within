# within-py

`within-py` provides Python bindings to high-performance solvers for projecting out high-dimensional fixed effects from regression problems from the [`within` Rust crate](https://crates.io/crates/within).

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
from within import PreconditionerConfig, solve, solve_batch
import numpy as np

np.random.seed(1)
n = 100_000
fe = np.asfortranarray(np.column_stack([
    np.random.randint(0, 500, n).astype(np.uint32),
    np.random.randint(0, 200, n).astype(np.uint32),
]))
y = np.random.randn(n)

result = solve(fe, y)                          # Schwarz-preconditioned LSMR
result = solve(fe, y, weights=np.ones(n))      # weighted solve
result = solve(fe, y, preconditioner=PreconditionerConfig.Diagonal)
```

### FWL regression example

```python
beta_true = np.array([1.0, -2.0, 0.5])
X = np.random.randn(n, 3)
y = X @ beta_true + np.random.randn(n)

result = solve_batch(fe, np.column_stack([y, X]))
y_tilde, X_tilde = result.demeaned[:, 0], result.demeaned[:, 1:]
beta_hat = np.linalg.lstsq(X_tilde, y_tilde, rcond=None)[0]
print(np.round(beta_hat, 4))  # [ 0.9982 -2.006   0.5005]
```

## Python API

### High-level functions

| Function | Description |
|---|---|
| `solve(design, y, weights?, options?, preconditioner?)` | Solve a single right-hand side. Returns `SolveResult`. |
| `solve_batch(design, Y, weights?, options?, preconditioner?)` | Solve multiple RHS vectors in parallel. `Y` has shape `(n_obs, k)`. Returns `BatchSolveResult`. |

`design` is a 2-D `uint32` array of category codes with shape `(n_obs, n_factors)`, or a list of `Effect` terms for varying slopes. A `UserWarning` is emitted when a C-contiguous array is passed — if the data is already sorted by the largest factor, `np.asfortranarray(design)` gives faster solves; unsorted input is copied internally either way.

### Persistent solver

For repeated solves with the same design matrix, `Solver` builds the preconditioner once and reuses it.

```python
from within import Solver

solver = Solver(fe)
r = solver.solve(y)                            # reuses preconditioner
r = solver.solve_batch(np.column_stack([y, X]))

precond = solver.preconditioner                # picklable property
solver2 = Solver(fe, preconditioner=precond)   # skip re-factorization
```

| Property / Method | Description |
|---|---|
| `Solver(design, weights?, preconditioner?)` | Build solver. Factorizes the preconditioner at construction. |
| `.solve(y, options?)` | Solve a single RHS with the given LSMR tuning. Returns `SolveResult`. |
| `.solve_batch(Y, options?)` | Solve multiple RHS columns in parallel. Returns `BatchSolveResult`. |
| `.preconditioner` | Return the built `Preconditioner` (picklable), or `None`. Reuse via `Solver(..., preconditioner=p)`. |


### Solver configuration

| Class | Description |
|---|---|
| `LsmrOptions(tol=1e-8, maxiter=1000, local_size=None)` | Modified LSMR. `local_size` enables windowed reorthogonalization. |

### Preconditioners

| Class | Description |
|---|---|
| `PreconditionerConfig.Off` | Disable preconditioning. |
| `PreconditionerConfig.Additive` | Additive Schwarz shortcut (equivalent to `None`). |
| `PreconditionerConfig.additive(local_solver?, reduction?)` | Tuned additive Schwarz configuration. Advanced argument types are available from `within.config`. |
| `PreconditionerConfig.Diagonal` | Diagonal/Jacobi preconditioner using `diag(D^T W D)^{-1}`. |
| `AdditiveSchwarz(local_solver?, reduction?)` | Existing tuned Schwarz configuration API — import from `within.config`. |
| `Preconditioner` (built) | Reuse a previously-built preconditioner across solvers. |

Pass `None` (the default) to use additive Schwarz with the default local solver.
`PreconditionerConfig` instances compare by value. A built preconditioner exposes
the configuration used to construct it as `.config`, including after pickling,
so cached preconditioners can be checked with `precond.config == requested_config`.

### Local solver configuration (advanced)

| Class | Description |
|---|---|
| `LocalSolverConfig(approx_chol?, schur?, dense_threshold?, scaling?)` | Schur complement reduction with approximate Cholesky on the reduced system. Default local solver. Import from `within.config`. Omit `schur` for the library default (approximate); pass `Schur.exact()` for the exact complement. |
| `Schur.approximate(config?)` / `Schur.exact()` | Schur-reduction mode for `LocalSolverConfig`. |
| `ApproxCholConfig(seed=0, split_merge=None)` | Approximate Cholesky parameters. |
| `ApproxSchurConfig(seed=0, split=1)` | Approximate Schur complement sampling parameters. |

### Result types

**`SolveResult`**: `x` (coefficients), `demeaned` (residuals), `converged`, `iterations`, `residual`, `unidentified`, `layout`, `time_total`, `time_setup`, `time_solve`.

**`BatchSolveResult`**: Same fields, with `converged`, `iterations`, `residual`, and `time_solve` as lists (one entry per RHS).
