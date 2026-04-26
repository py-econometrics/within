# withinr

R bindings for the [within](https://github.com/py-econometrics/within) fixed-effects solver.

`withinr` exposes fast iterative Krylov solvers (CG, GMRES) with Schwarz
domain-decomposition preconditioners for absorbing high-dimensional fixed
effects. The heavy lifting happens in compiled Rust code via
[extendr](https://extendr.github.io/extendr/extendr_api/).

---

## Prerequisites

| Requirement | Minimum | Notes |
|---|---|---|
| R | >= 4.2 | |
| Rust toolchain | stable (`rustc`, `cargo`) | install via [rustup](https://rustup.rs) |
| [rextendr](https://extendr.github.io/rextendr/) | >= 0.3 | `install.packages("rextendr")` |
| [devtools](https://devtools.r-lib.org/) | >= 2.4 | `install.packages("devtools")` |

---

## Quickstart (local development)

From the repository root:

```bash
cd /path/to/within
```

### 1. Install Rust (once)

macOS / Linux:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
rustc --version
cargo --version
```

Windows (PowerShell):

```powershell
winget install Rustlang.Rustup
rustup default stable
rustc --version
cargo --version
```

### 2. Install required R packages (once)

```r
install.packages(c("rextendr", "devtools", "testthat"))
```

### 3. Build and load `withinr`

Run the standard local dev loop (networked Rust build allowed):

```r
Sys.setenv(NOT_CRAN = "true") # or WITHINR_DEV = "true"
rextendr::document(pkg = "withinr")
devtools::load_all("withinr")
devtools::test("withinr")
```

### 4. Optional: command-line install instead of `load_all()`

```bash
NOT_CRAN=true R CMD INSTALL withinr
```

---

## API

### `solve()`and `solve_batch()`

Solve fixed-effects normal equations for a single response vector. Both share identical APIs - the only difference is that `solve` requires an input vector for y, while `solve_batch` also accepts matrices. 

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
| `y` | numeric vector `(n_obs)` or `solve`, vector or matrix for `solve_batch`| Response vector / covariates. |
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

**Example** — 3-way FE regression with 3 covariates

```r
library(withinr)

# DGP: y = X * beta + alpha_f1 + alpha_f2 + alpha_f3 + eps
set.seed(42)
n <- 5000L
n_f1 <- 50L; n_f2 <- 30L; n_f3 <- 20L

# Fixed-effect assignments (1-based for R)
f1 <- sample.int(n_f1, n, replace = TRUE)
f2 <- sample.int(n_f2, n, replace = TRUE)
f3 <- sample.int(n_f3, n, replace = TRUE)
categories <- cbind(f1, f2, f3)

# True FE levels, covariates, and coefficients
alpha1 <- rnorm(n_f1)
alpha2 <- rnorm(n_f2)
alpha3 <- rnorm(n_f3)
beta <- c(1, 2, 3)
X <- matrix(rnorm(n * 3), ncol = 3)

# Response
eps <- rnorm(n, sd = 0.5)
y <- X %*% beta + eps + alpha1[f1] + alpha2[f2] + alpha3[f3]

# Demean y and X together via solve_batch, then OLS on demeaned data
res <- withinr::solve_batch(categories, cbind(y, X))
y_tilde <- res$demeaned[, 1]
X_tilde <- res$demeaned[, 2:4]

beta_hat <- qr.solve(X_tilde, y_tilde)
beta_hat
#> [1] 1.003 1.998 3.005   (approx)
```

---

## CRAN / offline packaging

### 1. Vendor dependencies

```r
rextendr::vendor_crates(path = "withinr")
```

This creates `src/rust/vendor.tar.xz` and `src/rust/vendor-config.toml`.
The `Makevars` will automatically unpack and use them when building on CRAN
(where network access is unavailable).
Leave `NOT_CRAN` / `WITHINR_DEV` unset for this strict offline path.

Re-run whenever you update Rust dependencies in `src/rust/Cargo.toml`.

### 2. Verify

```bash
R CMD build withinr
R CMD check withinr_0.1.0.tar.gz --as-cran
```

---

## Release flow

When updating `withinr` after a new `within` crate release:

1. Bump the `within` version in `withinr/src/rust/Cargo.toml`.
2. Re-vendor: `rextendr::vendor_crates(path = "withinr")`.
3. Regenerate wrappers and docs:
   ```r
   rextendr::document(pkg = "withinr")  # regenerates R/extendr-wrappers.R + man/
   ```
4. Build, check, submit:
   ```bash
   R CMD build withinr
   R CMD check withinr_<version>.tar.gz --as-cran
   ```

---
