#' Solve Fixed-Effects Demeaning
#'
#' Demean a response vector by removing fixed effects defined by a categorical
#' factor matrix, using a preconditioned Krylov solver implemented in Rust.
#'
#' @param categories An integer matrix of dimension `(n_obs, n_factors)` where
#'   each column contains 0-based factor levels. Values must be non-negative.
#' @param y A numeric vector of length `n_obs`.
#' @param method Character string: `"cg"` (conjugate gradient, default) or
#'   `"gmres"`.
#' @param tol Convergence tolerance (default `1e-8`).
#' @param maxiter Maximum number of iterations (default `1000`).
#' @param restart GMRES restart parameter (default `30`; ignored for CG).
#' @param preconditioner Character string: `"additive"` (default),
#'   `"multiplicative"`, or `"off"`. Multiplicative requires `method = "gmres"`.
#' @param weights Optional numeric vector of observation weights (length
#'   `n_obs`).
#'
#' @return A named list with components:
#'   \describe{
#'     \item{x}{Numeric vector of estimated fixed-effect coefficients.}
#'     \item{demeaned}{Numeric vector of demeaned response values.}
#'     \item{converged}{Logical; did the solver converge?}
#'     \item{iterations}{Integer; number of iterations used.}
#'     \item{residual}{Numeric; final residual norm.}
#'     \item{time_total}{Numeric; total wall time in seconds.}
#'     \item{time_setup}{Numeric; preconditioner setup time in seconds.}
#'     \item{time_solve}{Numeric; iterative solve time in seconds.}
#'   }
#'
#' @export
solve <- function(categories, 
                  y,
                  method = NULL,
                  tol = NULL,
                  maxiter = NULL,
                  restart = NULL,
                  preconditioner = NULL,
                  weights = NULL) {
  if (!is.matrix(categories)) {
    stop("`categories` must be a matrix.", call. = FALSE)
  }
  if (!is.integer(categories)) {
    nr <- nrow(categories)
    nc <- ncol(categories)
    storage.mode(categories) <- "integer"
    dim(categories) <- c(nr, nc)
  }
  if (!is.numeric(y)) {
    stop("`y` must be numeric.", call. = FALSE)
  }
  if (nrow(categories) != length(y)) {
    stop("`nrow(categories)` must equal `length(y)`.", call. = FALSE)
  }
  if (!is.null(weights)) {
    weights <- as.double(weights)
    if (length(weights) != length(y)) {
      stop("`length(weights)` must equal `length(y)`.", call. = FALSE)
    }
  }
  if (!is.null(maxiter)) maxiter <- as.integer(maxiter)

  .Call(
    wrap__solve,
    categories,
    y,
    method,
    tol,
    maxiter,
    restart,
    preconditioner,
    weights
  )
}

#' Batch Solve Fixed-Effects Demeaning
#'
#' Demean multiple response vectors simultaneously, reusing the preconditioner.
#' Each column of `Y` is treated as a separate response vector.
#'
#' @param categories An integer matrix of dimension `(n_obs, n_factors)` where
#'   each column contains 0-based factor levels.
#' @param Y A numeric matrix of dimension `(n_obs, k)` where each column is a
#'   response vector.
#' @param method,tol,maxiter,restart,preconditioner,weights See
#'   [solve()].
#'
#' @return A named list with components:
#'   \describe{
#'     \item{x}{Numeric matrix of estimated coefficients (n_dofs x k).}
#'     \item{demeaned}{Numeric matrix of demeaned values (n_obs x k).}
#'     \item{converged}{Logical vector of length k.}
#'     \item{iterations}{Integer vector of length k.}
#'     \item{residual}{Numeric vector of final residuals (length k).}
#'     \item{time_solve}{Numeric vector of per-column solve times (length k).}
#'     \item{time_total}{Numeric scalar; total wall time in seconds.}
#'   }
#'
#' @export
solve_batch <- function(categories, Y,
                        method = NULL,
                        tol = NULL,
                        maxiter = NULL,
                        restart = NULL,
                        preconditioner = NULL,
                        weights = NULL) {
  if (!is.matrix(categories)) {
    stop("`categories` must be a matrix.", call. = FALSE)
  }
  if (!is.integer(categories)) {
    nr <- nrow(categories)
    nc <- ncol(categories)
    storage.mode(categories) <- "integer"
    dim(categories) <- c(nr, nc)
  }
  if (!is.matrix(Y) || !is.numeric(Y)) {
    stop("`Y` must be a numeric matrix.", call. = FALSE)
  }
  if (nrow(categories) != nrow(Y)) {
    stop("`nrow(categories)` must equal `nrow(Y)`.", call. = FALSE)
  }
  if (!is.null(weights)) {
    weights <- as.double(weights)
    if (length(weights) != nrow(Y)) {
      stop("`length(weights)` must equal `nrow(Y)`.", call. = FALSE)
    }
  }
  if (!is.null(maxiter)) maxiter <- as.integer(maxiter)

  .Call(
    wrap__solve_batch,
    categories,
    Y,
    method,
    tol,
    maxiter,
    restart,
    preconditioner,
    weights
  )
}
