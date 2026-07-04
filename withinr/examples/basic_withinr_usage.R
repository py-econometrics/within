# Basic withinr usage.
#
# Run from the repository root after installing the package:
#   NOT_CRAN=true R CMD INSTALL withinr
#   Rscript withinr/examples/basic_withinr_usage.R

if (!requireNamespace("withinr", quietly = TRUE)) {
  stop(
    "Package 'withinr' is not installed. Run: NOT_CRAN=true R CMD INSTALL withinr",
    call. = FALSE
  )
}

set.seed(42)

n <- 2000L
n_firm <- 120L
n_worker <- 300L
n_year <- 8L

firm <- sample.int(n_firm, n, replace = TRUE)
worker <- sample.int(n_worker, n, replace = TRUE)
year <- sample.int(n_year, n, replace = TRUE)

# withinr expects a 1-based integer matrix: rows are observations, columns are
# fixed-effect dimensions.
categories <- cbind(firm = firm, worker = worker, year = year)

x1 <- rnorm(n)
x2 <- rnorm(n)
firm_fe <- rnorm(n_firm)
worker_fe <- rnorm(n_worker)
year_fe <- rnorm(n_year)

y <- 1.5 * x1 - 0.75 * x2 +
  firm_fe[firm] + worker_fe[worker] + year_fe[year] +
  rnorm(n, sd = 0.2)

X <- cbind(x1 = x1, x2 = x2)
weights <- runif(n, min = 0.5, max = 2.0)

options <- withinr::LsmrOptions(
  tol = 1e-10,
  maxiter = 4000L,
  local_size = 4L
)

# 1. One-shot demeaning of y.
fit_y <- withinr::solve(
  categories,
  y,
  options = options,
  weights = weights,
  preconditioner = withinr::PreconditionerConfig$Additive
)

cat("Single RHS converged:", fit_y$converged, "\n")
cat("Iterations:", fit_y$iterations, "\n")
cat("Residual:", signif(fit_y$residual, 4), "\n\n")

# 2. Batch demeaning: demean y and all regressors with one shared setup.
batch <- withinr::solve_batch(
  categories,
  cbind(y = y, X),
  options = options,
  weights = weights,
  preconditioner = withinr::PreconditionerConfig$Additive
)

y_tilde <- batch$demeaned[, 1]
X_tilde <- batch$demeaned[, 2:3]

coef <- qr.solve(X_tilde, y_tilde)
cat("OLS after absorbing fixed effects:\n")
print(coef)
cat("\n")

# 3. Persistent solver: build the design/preconditioner once and reuse it.
solver <- withinr::Solver(categories, weights = weights)
preconditioner <- solver$preconditioner()

cat("Preconditioner metadata:\n")
cat("  variant:", preconditioner$variant, "\n")
cat("  build_time_seconds:", signif(preconditioner$build_time_seconds, 4), "\n\n")

again <- solver$solve(y, options = options)
stopifnot(isTRUE(all.equal(again$demeaned, fit_y$demeaned, tolerance = 1e-6)))

# 4. Serialize and reuse a prebuilt preconditioner.
payload <- preconditioner$serialize()
preconditioner2 <- withinr::Preconditioner(payload)
solver2 <- withinr::Solver(categories, weights = weights, preconditioner = preconditioner2)
reused <- solver2$solve(y, options = options)
stopifnot(isTRUE(all.equal(reused$demeaned, fit_y$demeaned, tolerance = 1e-6)))

cat("Serialized preconditioner bytes:", length(payload), "\n")
cat("Reuse check: OK\n")
