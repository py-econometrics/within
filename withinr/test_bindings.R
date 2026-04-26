# Smoke test for withinr bindings
# DGP: y_i = alpha_{f1(i)} + alpha_{f2(i)} + alpha_{f3(i)} + X_i * beta + eps_i
#   - 3-way fixed effects
#   - 3 covariates with true coefficients beta = c(1, 2, 3)

devtools::load_all("withinr")

cat("=== withinr binding test ===\n\n")

# --- DGP ---
set.seed(42)
n <- 5000L
n_f1 <- 50L; n_f2 <- 30L; n_f3 <- 20L

# Fixed-effect assignments (1-based for R)
f1 <- sample.int(n_f1, n, replace = TRUE)
f2 <- sample.int(n_f2, n, replace = TRUE)
f3 <- sample.int(n_f3, n, replace = TRUE)
categories <- cbind(f1, f2, f3)

# True fixed-effect levels
alpha1 <- rnorm(n_f1)
alpha2 <- rnorm(n_f2)
alpha3 <- rnorm(n_f3)

# Covariates and true coefficients
beta <- c(1, 2, 3)
X <- matrix(rnorm(n * 3), ncol = 3)

# Response
eps <- rnorm(n, sd = 0.5)
y <- X %*% beta + eps + alpha1[f1] + alpha2[f2] + alpha3[f3]

cat("n =", n, "| factors:", n_f1, n_f2, n_f3,
    "| total FE levels:", n_f1 + n_f2 + n_f3, "\n\n")

# --- Demean y and X to absorb fixed effects ---
cat("--- demean y and X (3 covariates) ---\n")
res <- withinr::solve_batch(categories, cbind(y, X))

# Extract demeaned variables
y_tilde <- res$demeaned[, 1]
X_tilde <- res$demeaned[, 2:4]

# --- OLS on demeaned data to recover beta ---
cat("--- OLS on demeaned data ---\n")
beta_hat <- qr.solve(X_tilde, y_tilde)
