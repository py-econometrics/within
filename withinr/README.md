# withinr

R bindings for the [within](https://github.com/py-econometrics/within) fixed-effects solver.

`withinr` exposes fast iterative Krylov solvers (CG, GMRES) with Schwarz
domain-decomposition preconditioners for absorbing high-dimensional fixed
effects. The heavy lifting happens in compiled Rust code via
[extendr](https://extendr.github.io/extendr/extendr_api/).

> **Status:** v1 API is implemented (`solve`, `solve_batch`). No persistent
> `Solver` class yet — that is deferred to a future release.

---

## Prerequisites

| Requirement | Minimum | Notes |
|---|---|---|
| R | >= 4.2 | |
| Rust toolchain | stable (`rustc`, `cargo`) | install via [rustup](https://rustup.rs) |
| [rextendr](https://extendr.github.io/rextendr/) | >= 0.3 | `install.packages("rextendr")` |
| [devtools](https://devtools.r-lib.org/) | >= 2.4 | `install.packages("devtools")` |

---

## Local development

### Build loop

```r
# 1. Compile Rust and regenerate R wrappers
rextendr::document(pkg = "withinr")

# 2. Load the package into the current session (no install)
devtools::load_all("withinr")
```

### Install / check

```bash
# Install into your local R library
R CMD INSTALL withinr

# Full CRAN-style check
R CMD check withinr
```

### Run tests

```r
devtools::test("withinr")
```

---

## v1 API

### `solve()`

Solve fixed-effects normal equations for a single response vector.

```r
solve(
  categories,
  y,
  weights        = NULL,
  method         = c("cg", "gmres"),
  tol            = 1e-8,
  maxiter        = 1000L,
  restart        = 30L,
  preconditioner = c("additive", "multiplicative", "off")
)
```

**Arguments**

| Argument | Type | Description |
|---|---|---|
| `categories` | integer matrix `(n_obs x n_factors)` | Factor assignments. **1-based** in R; converted internally to 0-based Rust indices. |
| `y` | numeric vector `(n_obs)` | Response vector. |
| `weights` | numeric vector `(n_obs)` or `NULL` | Observation weights. `NULL` = unit weights. |
| `method` | `"cg"` or `"gmres"` | Krylov solver. CG requires a symmetric preconditioner. |
| `tol` | numeric | Convergence tolerance on the relative residual norm. |
| `maxiter` | integer | Maximum Krylov iterations. |
| `restart` | integer | GMRES restart parameter (ignored for CG). |
| `preconditioner` | `"additive"`, `"multiplicative"`, or `"off"` | Schwarz preconditioner variant. `"multiplicative"` requires `method = "gmres"`. |

**Returns** a named list:

| Field | Type | Description |
|---|---|---|
| `coefficients` | numeric vector `(n_dofs)` | Fixed-effect coefficient estimates. Ordered: all levels of factor 1, then factor 2, etc. |
| `demeaned` | numeric vector `(n_obs)` | Response after subtracting estimated fixed effects. |
| `converged` | logical | Did the solver meet the tolerance? |
| `iterations` | integer | Krylov iterations performed. |
| `residual` | numeric | Final relative residual norm. |
| `time_total` | numeric | Wall-clock seconds (setup + solve). |
| `time_setup` | numeric | Wall-clock seconds for operator/preconditioner construction. |
| `time_solve` | numeric | Wall-clock seconds for the iterative solve. |

**Example**

```r
library(withinr)

# Two factors, 4 observations (1-based indices)
categories <- matrix(c(1L, 1L, 2L, 2L,
                        1L, 2L, 1L, 2L), ncol = 2)
y <- c(1.0, 2.0, 3.0, 4.0)

result <- solve(categories, y)
result$coefficients
result$converged
#> [1] TRUE
```

### `solve_batch()`

Solve for multiple response vectors sharing the same panel structure. Builds
the operator and preconditioner once, then solves each column in parallel.

```r
solve_batch(
  categories,
  Y,
  weights        = NULL,
  method         = c("cg", "gmres"),
  tol            = 1e-8,
  maxiter        = 1000L,
  restart        = 30L,
  preconditioner = c("additive", "multiplicative", "off")
)
```

**Arguments** are the same as `solve()`, except `Y` is a numeric matrix
`(n_obs x k)` where each column is a response vector.

**Returns** a named list:

| Field | Type | Description |
|---|---|---|
| `coefficients` | numeric matrix `(n_dofs x k)` | Coefficient estimates, one column per RHS. |
| `demeaned` | numeric matrix `(n_obs x k)` | Demeaned responses. |
| `converged` | logical vector `(k)` | Per-RHS convergence flags. |
| `iterations` | integer vector `(k)` | Krylov iterations per RHS. |
| `residual` | numeric vector `(k)` | Final residual per RHS. |
| `time_total` | numeric | Wall-clock seconds for the entire batch (shared setup included). |
| `time_solve` | numeric vector `(k)` | Per-RHS solve time. |

**Example**

```r
categories <- matrix(c(1L, 1L, 2L, 2L,
                        1L, 2L, 1L, 2L), ncol = 2)
Y <- cbind(c(1.0, 2.0, 3.0, 4.0),
           c(4.0, 3.0, 2.0, 1.0))

result <- solve_batch(categories, Y)
dim(result$coefficients)
#> [1] 4 2
result$converged
#> [1] TRUE TRUE
```

---

## CRAN / offline packaging

### 1. Vendor dependencies

```r
rextendr::vendor_pkgs(path = "withinr")
```

This creates `src/rust/vendor.tar.xz` and `src/rust/vendor-config.toml`.
The `Makevars` will automatically unpack and use them when building on CRAN
(where network access is unavailable).

Re-run whenever you update Rust dependencies in `src/rust/Cargo.toml`.

### 2. Verify

```bash
R CMD build withinr
R CMD check withinr_0.1.0.tar.gz --as-cran
```

---

## Release flow

1. Publish the new `within` crate version to crates.io.
2. Bump the `within` version in `withinr/src/rust/Cargo.toml`.
3. Re-vendor: `rextendr::vendor_pkgs(path = "withinr")`.
4. `rextendr::document(pkg = "withinr")` to regenerate wrappers.
5. `R CMD check` and submit to CRAN.

---

## Troubleshooting

**Wrapper regeneration issues**
If R wrappers get out of sync with Rust exports, regenerate them:

```r
rextendr::document(pkg = "withinr")
```

**Stale build artifacts**
If you hit cryptic linker or symbol errors after changing Rust code:

```r
rextendr::clean(path = "withinr")
rextendr::document(pkg = "withinr")
```

**Dependency / version mismatch**
If you bump the `within` crate version, update `src/rust/Cargo.toml` and
run `cargo check` to catch breakage early:

```bash
cd withinr/src/rust && cargo check
```

**Rust toolchain not found**
Ensure `rustc` and `cargo` are on your `PATH`:

```bash
rustc --version
cargo --version
```

If missing, install via [rustup](https://rustup.rs).
