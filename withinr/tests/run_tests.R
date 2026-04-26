assert_true <- function(cond, msg) {
  if (!isTRUE(cond)) stop(msg, call. = FALSE)
}

assert_equal <- function(a, b, tol = 0, msg = "Values are not equal.") {
  ok <- isTRUE(all.equal(a, b, tolerance = tol, check.attributes = FALSE))
  if (!ok) stop(msg, call. = FALSE)
}

assert_error <- function(expr, contains = NULL, msg = "Expected an error, but none occurred.") {
  got <- tryCatch(
    {
      force(expr)
      NULL
    },
    error = function(e) e
  )
  if (is.null(got)) stop(msg, call. = FALSE)
  if (!is.null(contains) && !grepl(contains, conditionMessage(got), fixed = TRUE)) {
    stop(
      sprintf("Expected error containing '%s', got: %s", contains, conditionMessage(got)),
      call. = FALSE
    )
  }
}

withinr_run_tests <- function(verbose = TRUE) {
  if (!requireNamespace("withinr", quietly = TRUE)) {
    stop("Package 'withinr' must be installed or loaded first.", call. = FALSE)
  }

  # Two factors, 4 observations (balanced 2x2 design)
  cats_2x2 <- matrix(c(1L, 1L, 2L, 2L,
                       1L, 2L, 1L, 2L), ncol = 2)
  y_simple <- c(1.0, 2.0, 3.0, 4.0)

  # solve() smoke
  r <- withinr::solve(cats_2x2, y_simple)
  assert_true(is.list(r), "solve() did not return a list.")
  assert_equal(
    names(r),
    c("coefficients", "demeaned", "converged", "iterations", "residual", "time_total", "time_setup", "time_solve"),
    msg = "solve() returned unexpected list fields."
  )
  assert_true(isTRUE(r$converged), "solve() did not converge on smoke case.")
  assert_true(is.double(r$coefficients), "solve() coefficients are not double.")
  assert_true(is.double(r$demeaned), "solve() demeaned output is not double.")
  assert_true(length(r$demeaned) == length(y_simple), "solve() demeaned length mismatch.")
  assert_true(r$iterations >= 0L, "solve() iterations is negative.")
  assert_true(r$residual >= 0, "solve() residual is negative.")
  assert_true(r$time_total >= 0, "solve() time_total is negative.")

  # solve() variants
  assert_true(withinr::solve(cats_2x2, y_simple, method = "gmres")$converged, "GMRES solve failed.")
  assert_true(withinr::solve(cats_2x2, y_simple, preconditioner = "off")$converged, "Unpreconditioned solve failed.")
  w <- c(1.0, 2.0, 1.0, 2.0)
  assert_true(withinr::solve(cats_2x2, y_simple, weights = w)$converged, "Weighted solve failed.")

  # solve() correctness checks
  d <- withinr::solve(cats_2x2, y_simple)$demeaned
  assert_equal(d[1] + d[2], 0, tol = 1e-6, msg = "Factor-1 group mean not centered (group 1).")
  assert_equal(d[3] + d[4], 0, tol = 1e-6, msg = "Factor-1 group mean not centered (group 2).")
  assert_equal(d[1] + d[3], 0, tol = 1e-6, msg = "Factor-2 group mean not centered (group 1).")
  assert_equal(d[2] + d[4], 0, tol = 1e-6, msg = "Factor-2 group mean not centered (group 2).")

  r2 <- withinr::solve(cats_2x2, y_simple)
  n1 <- 2L
  y_hat <- r2$coefficients[cats_2x2[, 1]] + r2$coefficients[n1 + cats_2x2[, 2]]
  assert_equal(y_hat + r2$demeaned, y_simple, tol = 1e-6, msg = "Reconstruction y != y_hat + demeaned.")

  # solve_batch() smoke
  Y <- cbind(y_simple, rev(y_simple))
  b <- withinr::solve_batch(cats_2x2, Y)
  assert_true(is.matrix(b$coefficients), "solve_batch() coefficients are not matrix.")
  assert_true(is.matrix(b$demeaned), "solve_batch() demeaned are not matrix.")
  assert_true(ncol(b$coefficients) == 2L, "solve_batch() coefficient column count mismatch.")
  assert_true(ncol(b$demeaned) == 2L, "solve_batch() demeaned column count mismatch.")
  assert_true(nrow(b$demeaned) == 4L, "solve_batch() demeaned row count mismatch.")
  assert_true(length(b$converged) == 2L, "solve_batch() converged vector length mismatch.")
  assert_true(all(b$converged), "solve_batch() did not converge for all RHS.")

  s1 <- withinr::solve(cats_2x2, Y[, 1])
  s2 <- withinr::solve(cats_2x2, Y[, 2])
  assert_equal(b$coefficients[, 1], s1$coefficients, tol = 1e-6, msg = "Batch coefficients RHS1 mismatch.")
  assert_equal(b$coefficients[, 2], s2$coefficients, tol = 1e-6, msg = "Batch coefficients RHS2 mismatch.")
  assert_equal(b$demeaned[, 1], s1$demeaned, tol = 1e-6, msg = "Batch demeaned RHS1 mismatch.")
  assert_equal(b$demeaned[, 2], s2$demeaned, tol = 1e-6, msg = "Batch demeaned RHS2 mismatch.")

  # validation/error behavior
  assert_error(withinr::solve(c(1L, 2L), y_simple), contains = "must be a matrix")
  assert_error(withinr::solve_batch(cats_2x2, y_simple), contains = "must be a matrix")

  bad <- cats_2x2
  bad[1, 1] <- NA_integer_
  assert_error(withinr::solve(bad, y_simple))

  bad0 <- cats_2x2 - 1L
  assert_error(withinr::solve(bad0, y_simple))

  assert_error(withinr::solve(cats_2x2, y_simple, method = "cg", preconditioner = "multiplicative"))
  assert_true(
    withinr::solve(cats_2x2, y_simple, method = "gmres", preconditioner = "multiplicative")$converged,
    "GMRES + multiplicative preconditioner failed."
  )

  cats_dbl <- matrix(c(1, 1, 2, 2, 1, 2, 1, 2), ncol = 2)
  assert_true(withinr::solve(cats_dbl, y_simple)$converged, "Numeric categories coercion case failed.")

  if (verbose) message("withinr manual tests: OK")
  invisible(TRUE)
}

if (identical(environment(), globalenv())) {
  withinr_run_tests(verbose = TRUE)
}
