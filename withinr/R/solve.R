.default_approx_schur <- function() {
  ApproxSchurConfig()
}

as_nullable_integer <- function(x) {
  if (is.null(x)) {
    NULL
  } else {
    as.integer(x)
  }
}

as_lsmr_options <- function(options) {
  if (is.null(options)) {
    return(LsmrOptions())
  }
  if (!inherits(options, "within_lsmr_options")) {
    stop("`options` must be created by LsmrOptions(...) or NULL", call. = FALSE)
  }
  options
}

as_preconditioner_arg <- function(preconditioner) {
  if (inherits(preconditioner, "within_preconditioner")) {
    return(preconditioner$ptr)
  }
  preconditioner
}

validate_categories <- function(categories) {
  if (!is.matrix(categories)) {
    stop("`categories` must be a matrix", call. = FALSE)
  }
  if (!is.integer(categories)) {
    categories <- matrix(
      as.integer(categories),
      nrow = nrow(categories),
      ncol = ncol(categories)
    )
  }
  categories
}

validate_weights <- function(weights) {
  if (is.null(weights)) {
    NULL
  } else {
    as.double(weights)
  }
}

#' LSMR solver options
#'
#' @param tol Positive finite convergence tolerance.
#' @param maxiter Maximum number of LSMR iterations.
#' @param local_size Optional reorthogonalization window size, or `NULL`.
#' @return A solver options object accepted by [solve()], [solve_batch()], and
#'   persistent solver methods.
#' @export
LsmrOptions <- function(tol = 1e-8, maxiter = 1000L, local_size = NULL) {
  if (!is.finite(tol) || tol <= 0) {
    stop("`tol` must be a positive finite number", call. = FALSE)
  }
  maxiter <- as.integer(maxiter)
  if (length(maxiter) != 1L || is.na(maxiter) || maxiter < 1L) {
    stop("`maxiter` must be >= 1", call. = FALSE)
  }
  if (!is.null(local_size)) {
    local_size <- as.integer(local_size)
    if (length(local_size) != 1L || is.na(local_size) || local_size < 1L) {
      stop("`local_size` must be NULL or >= 1", call. = FALSE)
    }
  }
  structure(
    list(tol = as.double(tol), maxiter = maxiter, local_size = local_size),
    class = "within_lsmr_options"
  )
}

#' @rdname LsmrOptions
#' @export
lsmr_options <- LsmrOptions

#' Preconditioner shortcut values
#'
#' Use `PreconditionerConfig$Additive`, `PreconditionerConfig$Off`, or
#' `PreconditionerConfig$Diagonal` as the `preconditioner` argument.
#'
#' @export
PreconditionerConfig <- structure(
  list(Additive = "additive", Off = "off", Diagonal = "diagonal"),
  class = "within_preconditioner_config"
)

#' Reduction strategy shortcut values
#'
#' Use these values inside [AdditiveSchwarz()].
#'
#' @export
ReductionStrategy <- structure(
  list(
    Auto = "auto",
    AtomicScatter = "atomic_scatter",
    ParallelReduction = "parallel_reduction"
  ),
  class = "within_reduction_strategy"
)

#' Approximate Cholesky configuration
#'
#' @param seed Non-negative integer random seed.
#' @param split_merge Optional positive integer split/merge parameter, or `NULL`.
#' @return A local-solver configuration object.
#' @export
ApproxCholConfig <- function(seed = 0, split_merge = NULL) {
  seed <- as.numeric(seed)
  if (length(seed) != 1L || is.na(seed) || seed < 0 || seed != floor(seed)) {
    stop("`seed` must be a non-negative integer", call. = FALSE)
  }
  if (!is.null(split_merge)) {
    split_merge <- as.integer(split_merge)
    if (length(split_merge) != 1L || is.na(split_merge) || split_merge < 1L) {
      stop("`split_merge` must be NULL or >= 1", call. = FALSE)
    }
  }
  structure(
    list(seed = seed, split_merge = split_merge),
    class = "within_approx_chol_config"
  )
}

#' Approximate Schur complement configuration
#'
#' @param seed Non-negative integer random seed.
#' @param split Positive integer edge split factor.
#' @return A local-solver configuration object.
#' @export
ApproxSchurConfig <- function(seed = 0, split = 1L) {
  seed <- as.numeric(seed)
  split <- as.integer(split)
  if (length(seed) != 1L || is.na(seed) || seed < 0 || seed != floor(seed)) {
    stop("`seed` must be a non-negative integer", call. = FALSE)
  }
  if (length(split) != 1L || is.na(split) || split < 1L) {
    stop("`split` must be >= 1", call. = FALSE)
  }
  structure(
    list(seed = seed, split = split),
    class = "within_approx_schur_config"
  )
}

#' Local solver configuration for additive Schwarz subdomains
#'
#' @param approx_chol `NULL` for the library default, or an
#'   [ApproxCholConfig()] object.
#' @param approx_schur Omitted for the library default approximate Schur,
#'   `NULL` for exact Schur, or an [ApproxSchurConfig()] object.
#' @param dense_threshold Optional non-negative integer dense Schur threshold.
#' @return A local-solver configuration object accepted by [AdditiveSchwarz()].
#' @export
LocalSolverConfig <- function(approx_chol = NULL,
                              approx_schur = .default_approx_schur(),
                              dense_threshold = NULL) {
  if (!is.null(approx_chol) && !inherits(approx_chol, "within_approx_chol_config")) {
    stop("`approx_chol` must be created by ApproxCholConfig(...) or NULL", call. = FALSE)
  }
  if (!is.null(approx_schur) && !inherits(approx_schur, "within_approx_schur_config")) {
    stop("`approx_schur` must be created by ApproxSchurConfig(...) or NULL", call. = FALSE)
  }
  if (!is.null(dense_threshold)) {
    dense_threshold <- as.integer(dense_threshold)
    if (length(dense_threshold) != 1L || is.na(dense_threshold) || dense_threshold < 0L) {
      stop("`dense_threshold` must be NULL or >= 0", call. = FALSE)
    }
  }
  structure(
    list(
      approx_chol = approx_chol,
      approx_schur = approx_schur,
      dense_threshold = dense_threshold
    ),
    class = "within_local_solver_config"
  )
}

#' Additive Schwarz preconditioner configuration
#'
#' @param local_solver `NULL` for the library default, or a
#'   [LocalSolverConfig()] object.
#' @param reduction One of `ReductionStrategy$Auto`,
#'   `ReductionStrategy$AtomicScatter`, or `ReductionStrategy$ParallelReduction`.
#' @return A preconditioner configuration object.
#' @export
AdditiveSchwarz <- function(local_solver = NULL,
                            reduction = ReductionStrategy$Auto) {
  if (!is.null(local_solver) && !inherits(local_solver, "within_local_solver_config")) {
    stop("`local_solver` must be created by LocalSolverConfig(...) or NULL", call. = FALSE)
  }
  if (!is.character(reduction) || length(reduction) != 1L) {
    stop("`reduction` must be a ReductionStrategy value", call. = FALSE)
  }
  structure(
    list(local_solver = local_solver, reduction = reduction),
    class = "within_additive_schwarz"
  )
}

#' Solve fixed-effects normal equations
#'
#' Computes fixed-effect coefficients by solving the normal equations
#' \eqn{D^T W D x = D^T W y} where \eqn{D} is the dummy-variable design
#' matrix implied by `categories` and \eqn{W} is the diagonal weight matrix.
#'
#' @param categories Integer matrix of shape `(n_obs, n_factors)`. Each column
#'   contains **1-based** factor level assignments. Values must be positive
#'   integers with no `NA`s.
#' @param y Numeric vector of length `n_obs`.
#' @param options `NULL` for default [LsmrOptions()] or an options object.
#' @param weights Numeric vector of length `n_obs` or `NULL`.
#' @param preconditioner `NULL`, a `PreconditionerConfig` value,
#'   [AdditiveSchwarz()], or a built [Preconditioner()] object.
#' @return A named list with fields `x`, `demeaned`, `converged`,
#'   `iterations`, `residual`, `time_total`, `time_setup`, and `time_solve`.
#' @export
solve <- function(categories,
                  y,
                  options = NULL,
                  weights = NULL,
                  preconditioner = NULL) {
  categories <- validate_categories(categories)
  options <- as_lsmr_options(options)
  y <- as.double(y)
  weights <- validate_weights(weights)
  preconditioner <- as_preconditioner_arg(preconditioner)

  solve_impl(
    categories,
    y,
    weights,
    options$tol,
    options$maxiter,
    as_nullable_integer(options$local_size),
    preconditioner
  )
}

#' Solve fixed-effects normal equations for multiple response vectors
#'
#' Builds the operator and preconditioner once, then solves for each column
#' of `Y` in parallel.
#'
#' @inheritParams solve
#' @param Y Numeric matrix of shape `(n_obs, k)`.
#' @return A named list with matrix fields `x` and `demeaned`, plus
#'   `converged`, `iterations`, `residual`, `time_solve`, and `time_total`.
#' @export
solve_batch <- function(categories,
                        Y,
                        options = NULL,
                        weights = NULL,
                        preconditioner = NULL) {
  categories <- validate_categories(categories)
  options <- as_lsmr_options(options)
  if (!is.matrix(Y)) {
    stop("`Y` must be a matrix", call. = FALSE)
  }
  Y <- matrix(as.double(Y), nrow = nrow(Y), ncol = ncol(Y))
  weights <- validate_weights(weights)
  preconditioner <- as_preconditioner_arg(preconditioner)

  solve_batch_impl(
    categories,
    Y,
    weights,
    options$tol,
    options$maxiter,
    as_nullable_integer(options$local_size),
    preconditioner
  )
}

#' Persistent fixed-effects solver
#'
#' Builds the preconditioner once and reuses it for repeated solves with the
#' same design matrix.
#'
#' @inheritParams solve
#' @return A `within_solver` object with `$solve()`, `$solve_batch()`, and
#'   `$preconditioner()` methods, plus `$n_dofs` and `$n_obs` fields.
#' @export
Solver <- function(categories, weights = NULL, preconditioner = NULL) {
  categories <- validate_categories(categories)
  weights <- validate_weights(weights)
  preconditioner <- as_preconditioner_arg(preconditioner)
  ptr <- solver_new_impl(categories, weights, preconditioner)

  solver <- new.env(parent = emptyenv())
  solver$ptr <- ptr
  solver$n_dofs <- solver_n_dofs_impl(ptr)
  solver$n_obs <- solver_n_obs_impl(ptr)
  solver$solve <- function(y, options = NULL) {
    options <- as_lsmr_options(options)
    solver_solve_impl(
      ptr,
      as.double(y),
      options$tol,
      options$maxiter,
      as_nullable_integer(options$local_size)
    )
  }
  solver$solve_batch <- function(Y, options = NULL) {
    options <- as_lsmr_options(options)
    if (!is.matrix(Y)) {
      stop("`Y` must be a matrix", call. = FALSE)
    }
    Y <- matrix(as.double(Y), nrow = nrow(Y), ncol = ncol(Y))
    solver_solve_batch_impl(
      ptr,
      Y,
      options$tol,
      options$maxiter,
      as_nullable_integer(options$local_size)
    )
  }
  solver$preconditioner <- function() {
    ptr <- solver_preconditioner_impl(solver$ptr)
    if (is.null(ptr)) {
      NULL
    } else {
      new_preconditioner(ptr)
    }
  }
  class(solver) <- "within_solver"
  solver
}

new_preconditioner <- function(ptr) {
  preconditioner <- new.env(parent = emptyenv())
  preconditioner$ptr <- ptr
  preconditioner$nrows <- preconditioner_nrows_impl(ptr)
  preconditioner$ncols <- preconditioner_ncols_impl(ptr)
  preconditioner$variant <- preconditioner_variant_impl(ptr)
  build_time_seconds <- preconditioner_build_time_seconds_impl(ptr)
  preconditioner$build_time_seconds <- if (is.null(build_time_seconds)) {
    NA_real_
  } else {
    as.double(build_time_seconds)
  }
  preconditioner$apply <- function(x) {
    preconditioner_apply_impl(ptr, as.double(x))
  }
  preconditioner$serialize <- function() {
    preconditioner_serialize_impl(ptr)
  }
  class(preconditioner) <- "within_preconditioner"
  preconditioner
}

#' Built preconditioner handle
#'
#' Deserializes a preconditioner from raw bytes produced by
#' `solver$preconditioner()$serialize()`.
#'
#' @param data Raw vector containing serialized preconditioner bytes.
#' @return A `within_preconditioner` object.
#' @export
Preconditioner <- function(data) {
  if (!is.raw(data)) {
    stop("`data` must be a raw vector", call. = FALSE)
  }
  new_preconditioner(preconditioner_deserialize_impl(data))
}

#' @export
print.within_solver <- function(x, ...) {
  cat(sprintf("<within_solver: n_obs=%d, n_dofs=%d>\n", x$n_obs, x$n_dofs))
  invisible(x)
}

#' @export
print.within_preconditioner <- function(x, ...) {
  build_time <- if (is.na(x$build_time_seconds)) {
    "NA"
  } else {
    sprintf("%.6g", x$build_time_seconds)
  }
  cat(sprintf(
    "<within_preconditioner: variant=%s, nrows=%d, ncols=%d, build_time_seconds=%s>\n",
    x$variant,
    x$nrows,
    x$ncols,
    build_time
  ))
  invisible(x)
}
