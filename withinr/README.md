# withinr

R bindings for the [within](https://github.com/py-econometrics/within) fixed-effects solver.

`withinr` exposes the Rust `within` crate through [extendr](https://extendr.github.io/extendr/extendr_api/). Development builds target the workspace `within` crate on `main`, including unreleased solver improvements; offline/CRAN-style builds use the vendored copy of those same local Rust sources. The solver surface is modified LSMR with additive Schwarz, diagonal, or identity preconditioning.

## Quickstart (local development)

Requires R and a Rust toolchain (`cargo` on `PATH`).

```r
install.packages(c("rextendr", "devtools"))
Sys.setenv(NOT_CRAN = "true") # or WITHINR_DEV = "true"
devtools::load_all("withinr")
```

For command-line installation from the repository root:

```bash
NOT_CRAN=true R CMD INSTALL withinr
```

## API

R category matrices are 1-based. Each column is one fixed-effect factor, and `withinr` converts values to the 0-based indices used by Rust.

```r
within_solve(categories, y, options = NULL, weights = NULL, preconditioner = NULL)
within_solve_batch(categories, Y, options = NULL, weights = NULL, preconditioner = NULL)
```

The one-shot entry points carry a `within_` prefix (unlike Python's
`within.solve`, which is namespaced) so that attaching the package does not
mask `base::solve()`.

`options` is `NULL` or `LsmrOptions(tol = 1e-8, maxiter = 1000L, local_size = NULL)`.

`preconditioner` accepts `NULL`, `PreconditionerConfig$Additive`, `PreconditionerConfig$Off`, `PreconditionerConfig$Diagonal`, `AdditiveSchwarz(...)`, or a built `Preconditioner` returned by a persistent solver.

Both solve functions return lists with `x`, `demeaned`, `converged`, `iterations`, `residual`, and timing fields. In batch results, `x` and `demeaned` are matrices with one column per right-hand side.

## Example

```r
library(withinr)

set.seed(42)
n <- 5000L
n_f1 <- 50L
n_f2 <- 30L
n_f3 <- 20L

f1 <- sample.int(n_f1, n, replace = TRUE)
f2 <- sample.int(n_f2, n, replace = TRUE)
f3 <- sample.int(n_f3, n, replace = TRUE)
categories <- cbind(f1, f2, f3)

alpha1 <- rnorm(n_f1)
alpha2 <- rnorm(n_f2)
alpha3 <- rnorm(n_f3)
beta <- c(1, 2, 3)
X <- matrix(rnorm(n * 3), ncol = 3)
y <- X %*% beta + alpha1[f1] + alpha2[f2] + alpha3[f3] + rnorm(n, sd = 0.5)

res <- within_solve_batch(categories, cbind(y, X))
y_tilde <- res$demeaned[, 1]
X_tilde <- res$demeaned[, 2:4]
qr.solve(X_tilde, y_tilde)
```

## Persistent Solver

For repeated solves with the same design matrix, build a solver once and reuse its preconditioner.

```r
solver <- Solver(categories)
r <- solver$solve(y)
b <- solver$solve_batch(cbind(y, X), options = LsmrOptions(tol = 1e-10))

precond <- solver$preconditioner()
print(precond) # Preconditioner(Additive, n=...)

payload <- precond$serialize()
precond2 <- Preconditioner(payload)
solver2 <- Solver(categories, preconditioner = precond2)
```

## Advanced Configuration

```r
schwarz <- AdditiveSchwarz(
  local_solver = LocalSolverConfig(
    approx_chol = ApproxCholConfig(seed = 1, split_merge = 2L),
    approx_schur = ApproxSchurConfig(seed = 1, split = 1L),
    dense_threshold = 24L
  ),
  reduction = ReductionStrategy$Auto
)

res <- within_solve(categories, y, preconditioner = schwarz)
```

Passing `approx_schur = NULL` to `LocalSolverConfig()` requests exact Schur complements; omitting it uses the library-default approximate Schur configuration.

## CRAN / Offline Packaging

The package crate keeps `within = "0.2.0"` in `src/rust/Cargo.toml` as the compatibility requirement. Build wiring overrides that requirement:

- `NOT_CRAN=true` or `WITHINR_DEV=true` builds patch `within` to `../../crates/within` from this workspace.
- Offline builds unpack `src/rust/vendor.tar.xz`; `src/rust/vendor-config.toml` patches `within` to `vendor/within`.

Regenerate the vendored crate archive after Rust dependency or local crate changes:

```r
rextendr::vendor_crates(path = "withinr")
```

Then replace `vendor/within` and `vendor/schwarz-precond` in the archive with the local workspace crates so offline builds match development builds.

## Tests

```r
Sys.setenv(NOT_CRAN = "true")
devtools::load_all("withinr")
source("withinr/tests/run_tests.R")
withinr_run_tests()
```
