# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Two factors, 4 observations (balanced 2x2 design)
cats_2x2 <- matrix(c(1L, 1L, 2L, 2L,
                      1L, 2L, 1L, 2L), ncol = 2)
y_simple <- c(1.0, 2.0, 3.0, 4.0)

# ---------------------------------------------------------------------------
# solve() — smoke tests
# ---------------------------------------------------------------------------

test_that("solve returns expected list structure", {
  result <- solve(cats_2x2, y_simple)

  expect_type(result, "list")
  expect_named(result, c("coefficients", "demeaned", "converged",
                          "iterations", "residual",
                          "time_total", "time_setup", "time_solve"))

  expect_true(result$converged)
  expect_type(result$coefficients, "double")
  expect_type(result$demeaned, "double")
  expect_length(result$demeaned, length(y_simple))
  expect_true(result$iterations >= 0L)
  expect_true(result$residual >= 0)
  expect_true(result$time_total >= 0)
})

test_that("solve works with GMRES method", {
  result <- solve(cats_2x2, y_simple, method = "gmres")
  expect_true(result$converged)
})

test_that("solve works with weights", {
  w <- c(1.0, 2.0, 1.0, 2.0)
  result <- solve(cats_2x2, y_simple, weights = w)
  expect_true(result$converged)
  expect_length(result$demeaned, 4L)
})

test_that("solve works with preconditioner = 'off'", {
  result <- solve(cats_2x2, y_simple, preconditioner = "off")
  expect_true(result$converged)
})

# ---------------------------------------------------------------------------
# solve() — parity / correctness
# ---------------------------------------------------------------------------

test_that("demeaned values sum to approximately zero within each factor level", {
  result <- solve(cats_2x2, y_simple)
  d <- result$demeaned

  # For a balanced 2-way design, within-group means should cancel
  # Factor 1: groups {1,2} and {3,4}
  expect_equal(d[1] + d[2], 0, tolerance = 1e-6)
  expect_equal(d[3] + d[4], 0, tolerance = 1e-6)
  # Factor 2: groups {1,3} and {2,4}
  expect_equal(d[1] + d[3], 0, tolerance = 1e-6)
  expect_equal(d[2] + d[4], 0, tolerance = 1e-6)
})

test_that("y equals coefficients applied plus demeaned", {
  result <- solve(cats_2x2, y_simple)
  # Reconstruct: y_hat[i] = coef[factor1[i]] + coef[n_levels_f1 + factor2[i]]
  n1 <- 2L  # levels in factor 1
  coef <- result$coefficients
  y_hat <- coef[cats_2x2[, 1]] + coef[n1 + cats_2x2[, 2]]
  expect_equal(y_hat + result$demeaned, y_simple, tolerance = 1e-6)
})

# ---------------------------------------------------------------------------
# solve_batch() — smoke tests
# ---------------------------------------------------------------------------

test_that("solve_batch returns expected structure", {
  Y <- cbind(y_simple, rev(y_simple))
  result <- solve_batch(cats_2x2, Y)

  expect_type(result, "list")
  expect_named(result, c("coefficients", "demeaned", "converged",
                          "iterations", "residual",
                          "time_total", "time_solve"))

  expect_true(is.matrix(result$coefficients))
  expect_true(is.matrix(result$demeaned))
  expect_equal(ncol(result$coefficients), 2L)
  expect_equal(ncol(result$demeaned), 2L)
  expect_equal(nrow(result$demeaned), 4L)
  expect_length(result$converged, 2L)
  expect_true(all(result$converged))
})

test_that("solve_batch results match individual solves", {
  Y <- cbind(y_simple, rev(y_simple))
  batch <- solve_batch(cats_2x2, Y)

  single1 <- solve(cats_2x2, Y[, 1])
  single2 <- solve(cats_2x2, Y[, 2])

  expect_equal(batch$coefficients[, 1], single1$coefficients, tolerance = 1e-6)
  expect_equal(batch$coefficients[, 2], single2$coefficients, tolerance = 1e-6)
  expect_equal(batch$demeaned[, 1], single1$demeaned, tolerance = 1e-6)
  expect_equal(batch$demeaned[, 2], single2$demeaned, tolerance = 1e-6)
})

# ---------------------------------------------------------------------------
# Input validation / error behaviour
# ---------------------------------------------------------------------------

test_that("solve rejects non-matrix categories", {
  expect_error(solve(c(1L, 2L), y_simple), "must be a matrix")
})

test_that("solve_batch rejects non-matrix Y", {
  expect_error(solve_batch(cats_2x2, y_simple), "must be a matrix")
})

test_that("solve rejects NA in categories", {
  bad <- cats_2x2
  bad[1, 1] <- NA_integer_
  expect_error(solve(bad, y_simple))
})

test_that("solve rejects 0-based categories", {
  bad <- cats_2x2 - 1L
  expect_error(solve(bad, y_simple))
})

test_that("solve rejects CG with multiplicative preconditioner", {
  expect_error(
    solve(cats_2x2, y_simple, method = "cg", preconditioner = "multiplicative")
  )
})

test_that("GMRES with multiplicative preconditioner succeeds", {
  result <- solve(cats_2x2, y_simple,
                  method = "gmres", preconditioner = "multiplicative")
  expect_true(result$converged)
})

test_that("solve coerces numeric categories to integer", {
  cats_dbl <- matrix(c(1, 1, 2, 2, 1, 2, 1, 2), ncol = 2)
  result <- solve(cats_dbl, y_simple)
  expect_true(result$converged)
})
