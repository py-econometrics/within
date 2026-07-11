.default_approx_schur <- function() {
  ApproxSchurConfig()
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

#' Modified LSMR solver options
#'
#' Uses Modified Golub-Kahan bidiagonalization to solve the least-squares
#' problem directly. The preconditioner approximates `A^T A` and is applied as
#' one `M^{-1}` solve per iteration.
#'
#' Values are validated by the Rust bridge when the options are used.
#'
#' @param tol Positive finite convergence tolerance. Default `1e-8`.
#' @param maxiter Maximum number of LSMR iterations. Default `1000L`.
#' @param local_size Optional window size for modified Gram-Schmidt
#'   reorthogonalization, or `NULL` to use the short recurrence.
#' @return A solver options object accepted by [within_solve()],
#'   [within_solve_batch()], and persistent solver methods.
#' @export
LsmrOptions <- function(tol = 1e-8, maxiter = 1000L, local_size = NULL) {
  structure(
    list(tol = tol, maxiter = maxiter, local_size = local_size),
    class = "within_lsmr_options"
  )
}

#' Preconditioner shortcut values
#'
#' Use `PreconditionerConfig$Additive`, `PreconditionerConfig$Off`, or
#' `PreconditionerConfig$Diagonal` as the `preconditioner` argument.
#' `Additive` builds the default additive Schwarz preconditioner, `Off`
#' disables preconditioning, and `Diagonal` uses diagonal/Jacobi scaling.
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
#' @param preconditioner Controls preconditioning. Five input forms are
#'   accepted: `NULL` builds the default additive Schwarz preconditioner,
#'   `PreconditionerConfig$Off` disables preconditioning,
#'   `PreconditionerConfig$Diagonal` uses diagonal/Jacobi scaling,
#'   [AdditiveSchwarz()] overrides local-solver and reduction settings, and a
#'   built [Preconditioner()] object reuses an existing factorization.
#' @return A named list with fields `x`, `demeaned`, `converged`,
#'   `iterations`, `residual`, `time_total`, `time_setup`, and `time_solve`.
#' @export
within_solve <- function(categories,
                         y,
                         options = NULL,
                         weights = NULL,
                         preconditioner = NULL) {
  categories <- validate_categories(categories)
  y <- as.double(y)
  weights <- validate_weights(weights)

  solve_impl(categories, y, options, weights, preconditioner)
}

#' Solve fixed-effects normal equations for multiple response vectors
#'
#' Builds the operator and preconditioner once, then solves for each column
#' of `Y` in parallel.
#'
#' @inheritParams within_solve
#' @param Y Numeric matrix of shape `(n_obs, k)`.
#' @return A named list with matrix fields `x` and `demeaned`, plus
#'   `converged`, `iterations`, `residual`, `time_solve`, and `time_total`.
#' @export
within_solve_batch <- function(categories,
                               Y,
                               options = NULL,
                               weights = NULL,
                               preconditioner = NULL) {
  categories <- validate_categories(categories)
  if (!is.matrix(Y)) {
    stop("`Y` must be a matrix", call. = FALSE)
  }
  Y <- matrix(as.double(Y), nrow = nrow(Y), ncol = ncol(Y))
  weights <- validate_weights(weights)

  solve_batch_impl(categories, Y, options, weights, preconditioner)
}

#' Persistent fixed-effects solver
#'
#' Builds the preconditioner once and reuses it for repeated solves with the
#' same design matrix.
#'
#' @inheritParams within_solve
#' @return A `within_solver` object with `$solve()`, `$solve_batch()`, and
#'   `$preconditioner()` methods, plus `$n_dofs` and `$n_obs` fields.
#' @export
Solver <- function(categories, weights = NULL, preconditioner = NULL) {
  categories <- validate_categories(categories)
  weights <- validate_weights(weights)
  ptr <- solver_new_impl(categories, weights, preconditioner)

  solver <- new.env(parent = emptyenv())
  solver$ptr <- ptr
  solver$n_dofs <- solver_n_dofs_impl(ptr)
  solver$n_obs <- solver_n_obs_impl(ptr)
  solver$solve <- function(y, options = NULL) {
    solver_solve_impl(ptr, as.double(y), options)
  }
  solver$solve_batch <- function(Y, options = NULL) {
    if (!is.matrix(Y)) {
      stop("`Y` must be a matrix", call. = FALSE)
    }
    Y <- matrix(as.double(Y), nrow = nrow(Y), ncol = ncol(Y))
    solver_solve_batch_impl(ptr, Y, options)
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
#' @return A `within_preconditioner` object with `$apply()` and `$serialize()`
#'   methods and `$nrows` and `$ncols` fields.
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

# Matches the Python __repr__: Preconditioner(<variant>, n=<nrows>).
#' @export
print.within_preconditioner <- function(x, ...) {
  cat(sprintf(
    "Preconditioner(%s, n=%d)\n",
    preconditioner_variant_impl(x$ptr),
    x$nrows
  ))
  invisible(x)
}
