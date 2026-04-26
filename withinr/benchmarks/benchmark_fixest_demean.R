# Benchmark withinr::solve_batch() vs fixest::demean() on the "difficult data" DGP.
#
# DGP provenance:
# - taken from fixest benchmarks
# - which are adapted from the authors of FixedEffectsModels.jl
#
# This script is fully in-memory (no disk writes).

generate_difficult_data <- function(pows = 4:7, seed = 1L) {
  set.seed(seed)
  datasets <- vector("list", length(pows))
  names(datasets) <- paste0("n=", 10^pows)

  for (i in seq_along(pows)) {
    pow <- pows[[i]]
    n <- 10^pow
    nb_indiv <- n / 20
    nb_firm <- round(n / 160)
    nb_year <- round(n^0.3)

    id_indiv <- sample.int(nb_indiv, n, replace = TRUE)
    id_firm <- pmin(sample.int(21, n, replace = TRUE) - 1L + pmax(1, id_indiv %/% 8 - 10), nb_firm)
    id_year <- sample.int(nb_year, n, replace = TRUE)

    x1 <- 5 * cos(id_indiv) + 5 * sin(id_firm) + 5 * sin(id_year) + runif(n)
    x2 <- cos(id_indiv) + sin(id_firm) + sin(id_year) + stats::rnorm(n)
    y <- 3 * x1 + 5 * x2 + cos(id_indiv) + cos(id_firm)^2 + sin(id_year) + stats::rnorm(n)

    datasets[[i]] <- data.frame(
      id_indiv = id_indiv,
      id_firm = id_firm,
      id_year = id_year,
      x1 = x1,
      x2 = x2,
      y = y
    )
  }

  datasets
}

benchmark_withinr_vs_fixest_difficult <- function(
    pows = 4:7,
    reps = 3L,
    seed = 1L,
    method = "cg",
    tol = 1e-8,
    maxiter = 1000L,
    restart = 30L,
    preconditioner = "additive",
    verbose = TRUE) {
  if (!requireNamespace("fixest", quietly = TRUE)) {
    stop("Package 'fixest' is required. Install with install.packages('fixest').", call. = FALSE)
  }
  if (!requireNamespace("withinr", quietly = TRUE)) {
    stop("Package 'withinr' is required. Install/load withinr first.", call. = FALSE)
  }

  if (length(pows) == 0L) {
    stop("`pows` must be non-empty.", call. = FALSE)
  }
  if (reps < 1L) {
    stop("`reps` must be >= 1.", call. = FALSE)
  }

  datasets <- generate_difficult_data(pows = pows, seed = seed)
  out <- vector("list", length(datasets))

  for (i in seq_along(datasets)) {
    df <- datasets[[i]]
    n <- nrow(df)
    cats <- as.matrix(df[, c("id_indiv", "id_firm", "id_year")])
    Y <- as.matrix(df[, c("y", "x1", "x2")])
    fes <- df[, c("id_indiv", "id_firm", "id_year")]

    if (verbose) {
      message(sprintf("Benchmarking n=%s (replicates=%d) ...", format(n, big.mark = ","), reps))
    }

    t_within <- numeric(reps)
    t_fixest <- numeric(reps)
    demeaned_within <- NULL
    demeaned_fixest <- NULL

    for (r in seq_len(reps)) {
      t_within[r] <- system.time({
        wr <- withinr::solve_batch(
          categories = cats,
          Y = Y,
          method = method,
          tol = tol,
          maxiter = as.integer(maxiter),
          restart = as.integer(restart),
          preconditioner = preconditioner
        )
      })[["elapsed"]]
      demeaned_within <- wr$demeaned

      t_fixest[r] <- system.time({
        demeaned_fixest <- fixest::demean(
          X = Y,
          f = fes,
          na.rm = TRUE,
          as.matrix = TRUE
        )
      })[["elapsed"]]
    }

    beta_within <- qr.solve(demeaned_within[, 2:3, drop = FALSE], demeaned_within[, 1])
    beta_fixest <- qr.solve(demeaned_fixest[, 2:3, drop = FALSE], demeaned_fixest[, 1])

    out[[i]] <- data.frame(
      pow = pows[[i]],
      n = n,
      withinr_elapsed_mean = mean(t_within),
      withinr_elapsed_median = stats::median(t_within),
      fixest_elapsed_mean = mean(t_fixest),
      fixest_elapsed_median = stats::median(t_fixest),
      speedup_fixest_over_withinr = stats::median(t_fixest) / stats::median(t_within),
      withinr_beta_x1 = unname(beta_within[1]),
      withinr_beta_x2 = unname(beta_within[2]),
      fixest_beta_x1 = unname(beta_fixest[1]),
      fixest_beta_x2 = unname(beta_fixest[2]),
      abs_beta_diff_x1 = abs(unname(beta_within[1]) - unname(beta_fixest[1])),
      abs_beta_diff_x2 = abs(unname(beta_within[2]) - unname(beta_fixest[2])),
      max_abs_demean_diff = max(abs(demeaned_within - demeaned_fixest)),
      max_abs_beta_diff = max(abs(beta_within - beta_fixest))
    )
  }

  result <- do.call(rbind, out)
  rownames(result) <- NULL
  result
}

format_benchmark_withinr_fixest <- function(bench, digits = 6L) {
  if (!is.data.frame(bench)) {
    stop("`bench` must be a data.frame returned by benchmark_withinr_vs_fixest_difficult().",
         call. = FALSE)
  }

  runtime <- data.frame(
    n = bench$n,
    withinr_sec = round(bench$withinr_elapsed_median, digits),
    fixest_sec = round(bench$fixest_elapsed_median, digits)
  )

  estimates <- data.frame(
    n = bench$n,
    withinr_x1 = round(bench$withinr_beta_x1, digits),
    fixest_x1 = round(bench$fixest_beta_x1, digits),
    withinr_x2 = round(bench$withinr_beta_x2, digits),
    fixest_x2 = round(bench$fixest_beta_x2, digits)
  )

  list(runtime = runtime, estimates = estimates, raw = bench)
}

print_benchmark_withinr_fixest <- function(bench, digits = 6L) {
  tabs <- format_benchmark_withinr_fixest(bench, digits = digits)

  cat("\n== Runtime (median seconds) ==\n")
  print(tabs$runtime, row.names = FALSE)

  cat("\n== Point Estimates ==\n")
  print(tabs$estimates, row.names = FALSE)

  invisible(tabs)
}

# Example:
# devtools::load_all("withinr")
# source("withinr/benchmarks/benchmark_fixest_demean.R")
# bench <- benchmark_withinr_vs_fixest_difficult(pows = 4:7, reps = 3)
# print_benchmark_withinr_fixest(bench)
