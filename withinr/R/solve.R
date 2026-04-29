#' Solve fixed-effects normal equations
#'
#' Computes fixed-effect coefficients by solving the normal equations
#' \eqn{D^T W D x = D^T W y} where \eqn{D} is the dummy-variable design
#' matrix implied by `categories` and \eqn{W} is the diagonal weight matrix.
#'
#' @param categories Integer matrix of shape `(n_obs, n_factors)`. Each column
#'   contains **1-based** factor level assignments. Values must be positive
#'   integers with no `NA`s.
#' @param y Numeric vector of length `n_obs`. The response variable.
#' @param weights Numeric vector of length `n_obs` or `NULL` (default).
#'   Observation weights. `NULL` means unit weights (unweighted).
#' @param method Character, one of `"cg"` (default) or `"gmres"`.
#'   `"cg"` uses Conjugate Gradient (requires symmetric preconditioner).
#'   `"gmres"` uses GMRES (supports all preconditioners).
#' @param tol Convergence tolerance on the relative residual norm.
#'   Default `1e-8`.
#' @param maxiter Maximum number of Krylov iterations. Default `1000L`.
#' @param restart GMRES restart parameter (ignored for CG). Default `30L`.
#' @param preconditioner Character, one of `"additive"` (default),
#'   `"multiplicative"`, or `"off"`. Schwarz preconditioner variant.
#'   `"multiplicative"` requires `method = "gmres"`.
#'
#' @return A named list with components:
#' \describe{
#'   \item{coefficients}{Numeric vector of fixed-effect coefficient estimates.
#'     Ordered: all levels of factor 1, then factor 2, etc.}
#'   \item{demeaned}{Numeric vector. Response after subtracting estimated
#'     fixed effects (\eqn{y - D x}).}
#'   \item{converged}{Logical. Did the solver meet the tolerance?}
#'   \item{iterations}{Integer. Number of Krylov iterations performed.}
#'   \item{residual}{Numeric. Final relative residual norm.}
#'   \item{time_total}{Numeric. Wall-clock seconds (setup + solve).}
#'   \item{time_setup}{Numeric. Wall-clock seconds for operator/preconditioner
#'     construction.}
#'   \item{time_solve}{Numeric. Wall-clock seconds for the iterative solve.}
#' }
#'
#' @examples
#' \dontrun{
#' # Two factors, 4 observations (1-based indices)
#' categories <- matrix(c(1L, 1L, 2L, 2L,
#'                         1L, 2L, 1L, 2L), ncol = 2)
#' y <- c(1.0, 2.0, 3.0, 4.0)
#' result <- solve(categories, y)
#' result$coefficients
#' result$converged
#' }
#'
#' @export
solve <- function(categories,
                  y,
                  weights = NULL,
                  method = c("cg", "gmres"),
                  tol = 1e-8,
                  maxiter = 1000L,
                  restart = 30L,
                  preconditioner = c("additive", "multiplicative", "off")) {
  method <- match.arg(method)
  preconditioner <- match.arg(preconditioner)

  if (!is.matrix(categories)) {
    stop("`categories` must be a matrix", call. = FALSE)
  }
  if (!is.integer(categories)) {
    categories <- matrix(as.integer(categories),
                         nrow = nrow(categories),
                         ncol = ncol(categories))
  }

  y <- as.double(y)
  if (!is.null(weights)) {
    weights <- as.double(weights)
  }

  solve_impl(
    categories, y, weights, method, tol,
    as.integer(maxiter), as.integer(restart), preconditioner
  )
}

#' Solve fixed-effects normal equations for multiple response vectors
#'
#' Builds the operator and preconditioner once, then solves for each column
#' of `Y` in parallel. More efficient than calling [solve()] in a loop.
#'
#' @inheritParams solve
#' @param Y Numeric matrix of shape `(n_obs, k)`. Each column is a separate
#'   response vector.
#'
#' @return A named list with components:
#' \describe{
#'   \item{coefficients}{Numeric matrix `(n_dofs, k)`. Coefficient estimates,
#'     one column per RHS.}
#'   \item{demeaned}{Numeric matrix `(n_obs, k)`. Demeaned responses.}
#'   \item{converged}{Logical vector of length `k`. Per-RHS convergence flags.}
#'   \item{iterations}{Integer vector of length `k`. Krylov iterations per RHS.}
#'   \item{residual}{Numeric vector of length `k`. Final residual per RHS.}
#'   \item{time_total}{Numeric scalar. Wall-clock seconds for the entire batch.}
#'   \item{time_solve}{Numeric vector of length `k`. Per-RHS solve time.}
#' }
#'
#' @examples
#' \dontrun{
#' categories <- matrix(c(1L, 1L, 2L, 2L,
#'                         1L, 2L, 1L, 2L), ncol = 2)
#' Y <- cbind(c(1.0, 2.0, 3.0, 4.0),
#'            c(4.0, 3.0, 2.0, 1.0))
#' result <- solve_batch(categories, Y)
#' dim(result$coefficients)
#' result$converged
#' }
#'
#' @export
solve_batch <- function(categories,
                        Y,
                        weights = NULL,
                        method = c("cg", "gmres"),
                        tol = 1e-8,
                        maxiter = 1000L,
                        restart = 30L,
                        preconditioner = c("additive", "multiplicative", "off")) {
  method <- match.arg(method)
  preconditioner <- match.arg(preconditioner)

  if (!is.matrix(categories)) {
    stop("`categories` must be a matrix", call. = FALSE)
  }
  if (!is.integer(categories)) {
    categories <- matrix(as.integer(categories),
                         nrow = nrow(categories),
                         ncol = ncol(categories))
  }
  if (!is.matrix(Y)) {
    stop("`Y` must be a matrix", call. = FALSE)
  }

  Y <- matrix(as.double(Y), nrow = nrow(Y), ncol = ncol(Y))
  if (!is.null(weights)) {
    weights <- as.double(weights)
  }

  raw <- solve_batch_impl(
    categories, Y, weights, method, tol,
    as.integer(maxiter), as.integer(restart), preconditioner
  )

  # Reshape flat coefficient and demeaned vectors into matrices
  n_dofs <- raw$n_dofs
  n_obs  <- raw$n_obs
  n_rhs  <- raw$n_rhs

  raw$coefficients <- matrix(raw$coefficients, nrow = n_dofs, ncol = n_rhs)
  raw$demeaned     <- matrix(raw$demeaned,     nrow = n_obs,  ncol = n_rhs)

  # Drop internal dimension fields
  raw$n_dofs <- NULL
  raw$n_obs  <- NULL
  raw$n_rhs  <- NULL

  raw
}
