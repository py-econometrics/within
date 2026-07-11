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

assert_all_converged <- function(result, msg) {
  assert_true(all(result$converged), msg)
}

withinr_run_tests <- function(verbose = TRUE) {
  if (!requireNamespace("withinr", quietly = TRUE)) {
    stop("Package 'withinr' must be installed or loaded first.", call. = FALSE)
  }

  cats_2x2 <- matrix(c(1L, 1L, 2L, 2L,
                       1L, 2L, 1L, 2L), ncol = 2)
  y_simple <- c(1.0, 2.0, 3.0, 4.0)

  # solve() smoke
  r <- withinr::within_solve(cats_2x2, y_simple)
  assert_true(is.list(r), "solve() did not return a list.")
  assert_equal(
    names(r),
    c("x", "demeaned", "converged", "iterations", "residual", "time_total", "time_setup", "time_solve"),
    msg = "solve() returned unexpected list fields."
  )
  assert_true(isTRUE(r$converged), "solve() did not converge on smoke case.")
  assert_true(is.double(r$x), "solve() x is not double.")
  assert_true(is.double(r$demeaned), "solve() demeaned output is not double.")
  assert_true(length(r$demeaned) == length(y_simple), "solve() demeaned length mismatch.")
  assert_true(r$iterations >= 0L, "solve() iterations is negative.")
  assert_true(r$residual >= 0, "solve() residual is negative.")
  assert_true(r$time_total >= 0, "solve() time_total is negative.")

  opts <- withinr::LsmrOptions(tol = 1e-10, maxiter = 2000L, local_size = 2L)
  assert_true(withinr::within_solve(cats_2x2, y_simple, options = opts)$converged, "Custom options failed.")
  assert_true(
    withinr::within_solve(cats_2x2, y_simple, preconditioner = withinr::PreconditionerConfig$Off)$converged,
    "Unpreconditioned solve failed."
  )
  assert_true(
    withinr::within_solve(cats_2x2, y_simple, preconditioner = withinr::PreconditionerConfig$Diagonal)$converged,
    "Diagonal preconditioner solve failed."
  )
  schwarz <- withinr::AdditiveSchwarz(
    local_solver = withinr::LocalSolverConfig(
      approx_chol = withinr::ApproxCholConfig(split_merge = 2L),
      approx_schur = withinr::ApproxSchurConfig(split = 1L)
    ),
    reduction = withinr::ReductionStrategy$Auto
  )
  assert_true(withinr::within_solve(cats_2x2, y_simple, preconditioner = schwarz)$converged, "AdditiveSchwarz config failed.")

  w <- c(1.0, 2.0, 1.0, 2.0)
  assert_true(withinr::within_solve(cats_2x2, y_simple, weights = w)$converged, "Weighted solve failed.")

  # solve() correctness checks
  d <- withinr::within_solve(cats_2x2, y_simple)$demeaned
  assert_equal(d[1] + d[2], 0, tol = 1e-6, msg = "Factor-1 group mean not centered (group 1).")
  assert_equal(d[3] + d[4], 0, tol = 1e-6, msg = "Factor-1 group mean not centered (group 2).")
  assert_equal(d[1] + d[3], 0, tol = 1e-6, msg = "Factor-2 group mean not centered (group 1).")
  assert_equal(d[2] + d[4], 0, tol = 1e-6, msg = "Factor-2 group mean not centered (group 2).")

  r2 <- withinr::within_solve(cats_2x2, y_simple)
  n1 <- 2L
  y_hat <- r2$x[cats_2x2[, 1]] + r2$x[n1 + cats_2x2[, 2]]
  assert_equal(y_hat + r2$demeaned, y_simple, tol = 1e-6, msg = "Reconstruction y != y_hat + demeaned.")

  # solve_batch() smoke
  Y <- cbind(y_simple, rev(y_simple))
  b <- withinr::within_solve_batch(cats_2x2, Y)
  assert_true(is.matrix(b$x), "solve_batch() x is not matrix.")
  assert_true(is.matrix(b$demeaned), "solve_batch() demeaned are not matrix.")
  assert_true(ncol(b$x) == 2L, "solve_batch() x column count mismatch.")
  assert_true(ncol(b$demeaned) == 2L, "solve_batch() demeaned column count mismatch.")
  assert_true(nrow(b$demeaned) == 4L, "solve_batch() demeaned row count mismatch.")
  assert_true(length(b$converged) == 2L, "solve_batch() converged vector length mismatch.")
  assert_true(all(b$converged), "solve_batch() did not converge for all RHS.")

  s1 <- withinr::within_solve(cats_2x2, Y[, 1])
  s2 <- withinr::within_solve(cats_2x2, Y[, 2])
  assert_equal(b$x[, 1], s1$x, tol = 1e-6, msg = "Batch x RHS1 mismatch.")
  assert_equal(b$x[, 2], s2$x, tol = 1e-6, msg = "Batch x RHS2 mismatch.")
  assert_equal(b$demeaned[, 1], s1$demeaned, tol = 1e-6, msg = "Batch demeaned RHS1 mismatch.")
  assert_equal(b$demeaned[, 2], s2$demeaned, tol = 1e-6, msg = "Batch demeaned RHS2 mismatch.")

  # Persistent Solver and Preconditioner reuse
  solver <- withinr::Solver(cats_2x2)
  assert_true(inherits(solver, "within_solver"), "Solver() did not return a solver object.")
  assert_equal(solver$n_obs, 4L, msg = "solver$n_obs mismatch.")
  assert_true(solver$n_dofs >= 4L, "solver$n_dofs is too small.")
  p <- solver$preconditioner()
  assert_true(inherits(p, "within_preconditioner"), "solver$preconditioner() did not return a preconditioner.")
  assert_true(
    grepl("Preconditioner(Additive", paste(capture.output(print(p)), collapse = ""), fixed = TRUE),
    "preconditioner print does not report the Additive variant."
  )
  assert_equal(length(p$apply(rep(1, p$ncols))), p$nrows, msg = "preconditioner apply length mismatch.")

  solver_diag <- withinr::Solver(cats_2x2, preconditioner = withinr::PreconditionerConfig$Diagonal)
  p_diag <- solver_diag$preconditioner()
  assert_true(
    grepl("Preconditioner(Diagonal", paste(capture.output(print(p_diag)), collapse = ""), fixed = TRUE),
    "diagonal preconditioner print does not report the Diagonal variant."
  )

  r_persistent <- solver$solve(y_simple)
  assert_equal(r_persistent$demeaned, s1$demeaned, tol = 1e-6, msg = "Persistent solve mismatch.")
  b_persistent <- solver$solve_batch(Y)
  assert_equal(b_persistent$demeaned, b$demeaned, tol = 1e-6, msg = "Persistent batch mismatch.")

  bytes <- p$serialize()
  p2 <- withinr::Preconditioner(bytes)
  assert_true(
    grepl("Preconditioner(Additive", paste(capture.output(print(p2)), collapse = ""), fixed = TRUE),
    "deserialized preconditioner print does not report the Additive variant."
  )
  solver2 <- withinr::Solver(cats_2x2, preconditioner = p2)
  assert_equal(solver2$solve(y_simple)$demeaned, s1$demeaned, tol = 1e-6, msg = "Preconditioner reuse mismatch.")

  # validation/error behavior
  assert_error(withinr::within_solve(c(1L, 2L), y_simple), contains = "must be a matrix")
  assert_error(withinr::within_solve_batch(cats_2x2, y_simple), contains = "must be a matrix")

  bad <- cats_2x2
  bad[1, 1] <- NA_integer_
  assert_error(withinr::within_solve(bad, y_simple), contains = "must not contain NA")

  bad0 <- cats_2x2 - 1L
  assert_error(withinr::within_solve(bad0, y_simple), contains = "1-based")

  cats_dbl <- matrix(c(1, 1, 2, 2, 1, 2, 1, 2), ncol = 2)
  assert_true(withinr::within_solve(cats_dbl, y_simple)$converged, "Numeric categories coercion case failed.")

  # Larger deterministic design: preconditioner correctness and caller-order invariance
  set.seed(20260704)
  n <- 180L
  cats <- cbind(
    sample.int(29L, n, replace = TRUE),
    sample.int(17L, n, replace = TRUE),
    sample.int(11L, n, replace = TRUE)
  )
  y <- rnorm(n)
  y_alt <- 0.25 * y + rnorm(n)
  Y_big <- cbind(y, y_alt)
  weights <- runif(n, min = 0.5, max = 2.0)
  tight <- withinr::LsmrOptions(tol = 1e-10, maxiter = 4000L, local_size = 4L)

  add_big <- withinr::within_solve(cats, y, options = tight, weights = weights)
  diag_big <- withinr::within_solve(
    cats,
    y,
    options = tight,
    weights = weights,
    preconditioner = withinr::PreconditionerConfig$Diagonal
  )
  off_big <- withinr::within_solve(
    cats,
    y,
    options = tight,
    weights = weights,
    preconditioner = withinr::PreconditionerConfig$Off
  )
  assert_all_converged(add_big, "Additive preconditioner failed on larger design.")
  assert_all_converged(diag_big, "Diagonal preconditioner failed on larger design.")
  assert_all_converged(off_big, "Unpreconditioned solve failed on larger design.")
  assert_equal(
    diag_big$demeaned,
    add_big$demeaned,
    tol = 1e-6,
    msg = "Diagonal and additive demeaned outputs disagree."
  )
  assert_equal(
    off_big$demeaned,
    add_big$demeaned,
    tol = 1e-6,
    msg = "Unpreconditioned and additive demeaned outputs disagree."
  )

  solver_big <- withinr::Solver(cats, weights = weights)
  pre_big <- solver_big$preconditioner()
  reuse_big <- withinr::within_solve(cats, y, options = tight, weights = weights, preconditioner = pre_big)
  assert_equal(reuse_big$demeaned, add_big$demeaned, tol = 1e-6, msg = "One-shot preconditioner reuse mismatch.")

  batch_reuse <- withinr::within_solve_batch(cats, Y_big, options = tight, weights = weights, preconditioner = pre_big)
  batch_fresh <- withinr::within_solve_batch(cats, Y_big, options = tight, weights = weights)
  assert_all_converged(batch_reuse, "Prebuilt batch solve failed on larger design.")
  assert_equal(
    batch_reuse$demeaned,
    batch_fresh$demeaned,
    tol = 1e-6,
    msg = "Batch preconditioner reuse mismatch."
  )

  order_dominant <- order(cats[, 1], cats[, 2], cats[, 3])
  sorted <- withinr::within_solve(
    cats[order_dominant, , drop = FALSE],
    y[order_dominant],
    options = tight,
    weights = weights[order_dominant]
  )
  assert_equal(
    add_big$demeaned[order_dominant],
    sorted$demeaned,
    tol = 1e-6,
    msg = "Unsorted input did not return demeaned values in caller order."
  )

  sorted_batch <- withinr::within_solve_batch(
    cats[order_dominant, , drop = FALSE],
    Y_big[order_dominant, , drop = FALSE],
    options = tight,
    weights = weights[order_dominant]
  )
  assert_equal(
    batch_fresh$demeaned[order_dominant, , drop = FALSE],
    sorted_batch$demeaned,
    tol = 1e-6,
    msg = "Unsorted batch input did not return demeaned values in caller order."
  )

  if (verbose) message("withinr manual tests: OK")
  invisible(TRUE)
}

if (identical(environment(), globalenv())) {
  withinr_run_tests(verbose = TRUE)
}
