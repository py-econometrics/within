//! Right-preconditioned GMRES(m) with restarts.
//!
//! Solves `A x = b` using right preconditioning: the Krylov subspace is
//! built for `A M⁻¹`, then the solution is recovered as `x = M⁻¹ y`.
//! This avoids changing the residual norm (left preconditioning would),
//! so the convergence check uses the true residual `‖b - A x‖`.
//!
//! Implementation uses Modified Gram-Schmidt orthogonalisation with
//! Givens rotations to maintain the upper Hessenberg least-squares
//! problem. All Krylov storage (basis vectors, Hessenberg matrix,
//! rotation coefficients) is allocated once and reused across restarts.

use super::{dot, vec_norm};
use crate::{Operator, SolveError};

// ---------------------------------------------------------------------------
// Column basis storage
// ---------------------------------------------------------------------------

/// Column-major flat storage for basis vectors.
///
/// Stores up to `capacity` vectors of dimension `n` in a single contiguous
/// allocation. Avoids per-restart/per-iteration heap allocation.
struct ColumnBasis {
    data: Vec<f64>,
    /// Dimension of each vector.
    n: usize,
    len: usize,
}

impl ColumnBasis {
    /// Create with room for `capacity` vectors of length `n`.
    fn new(capacity: usize, n: usize) -> Self {
        Self {
            data: vec![0.0; capacity * n],
            n,
            len: 0,
        }
    }

    /// Return column `j` as a shared slice.
    #[inline]
    fn col(&self, j: usize) -> &[f64] {
        debug_assert!(j < self.len);
        let start = j * self.n;
        &self.data[start..start + self.n]
    }

    /// Return the last column as a mutable slice.
    ///
    /// This is the safe alternative to `col_mut` for the common pattern of
    /// accessing the column that was just pushed via `push_zeroed` or `push_from`.
    #[inline]
    fn last_col_mut(&mut self) -> &mut [f64] {
        debug_assert!(self.len > 0);
        let j = self.len - 1;
        let start = j * self.n;
        &mut self.data[start..start + self.n]
    }

    /// Append `src` as the next column, incrementing `len`.
    fn push_from(&mut self, src: &[f64]) {
        debug_assert_eq!(src.len(), self.n);
        let start = self.len * self.n;
        self.data[start..start + self.n].copy_from_slice(src);
        self.len += 1;
    }

    /// Append a zero-filled column, incrementing `len`.
    fn push_zeroed(&mut self) {
        let start = self.len * self.n;
        self.data[start..start + self.n].fill(0.0);
        self.len += 1;
    }

    /// Reset the number of stored vectors to zero (does not deallocate).
    fn clear(&mut self) {
        self.len = 0;
    }
}

// ---------------------------------------------------------------------------
// GMRES result
// ---------------------------------------------------------------------------

/// Result of a GMRES solve.
#[must_use]
pub struct GmresResult {
    /// Solution vector.
    pub x: Vec<f64>,
    /// Whether GMRES converged within the tolerance.
    pub converged: bool,
    /// Total number of iterations performed (across all restarts).
    pub iterations: usize,
    /// Final residual norm.
    pub residual_norm: f64,
}

// ---------------------------------------------------------------------------
// Flat storage helpers
// ---------------------------------------------------------------------------

/// Flat column-major storage for the upper Hessenberg matrix.
///
/// Column `j` has `j + 2` active entries (rows `0..=j+1`).
/// Storage is allocated for `max_cols` columns, each with `max_cols + 1` rows.
struct HessenbergMatrix {
    data: Vec<f64>,
    max_cols: usize,
    stride: usize,
}

impl HessenbergMatrix {
    fn new(max_cols: usize) -> Self {
        let stride = max_cols + 1;
        Self {
            data: vec![0.0; max_cols * stride],
            max_cols,
            stride,
        }
    }

    #[inline]
    fn get(&self, row: usize, col: usize) -> f64 {
        debug_assert!(col < self.max_cols && row < self.stride);
        self.data[col * self.stride + row]
    }

    #[inline]
    fn set(&mut self, row: usize, col: usize, val: f64) {
        debug_assert!(col < self.max_cols && row < self.stride);
        self.data[col * self.stride + row] = val;
    }

    fn clear(&mut self) {
        self.data.fill(0.0);
    }
}

// ---------------------------------------------------------------------------
// Arnoldi cycle
// ---------------------------------------------------------------------------

/// Owned restart-cycle buffers for GMRES, allocated once and reused across restarts.
struct KrylovBuffers {
    v_basis: ColumnBasis,
    z_basis: ColumnBasis,
    h: HessenbergMatrix,
    cs: Vec<f64>,
    sn: Vec<f64>,
    g: Vec<f64>,
    w: Vec<f64>,
}

impl KrylovBuffers {
    fn new(restart: usize, n: usize) -> Self {
        Self {
            v_basis: ColumnBasis::new(restart + 1, n),
            z_basis: ColumnBasis::new(restart, n),
            h: HessenbergMatrix::new(restart),
            cs: vec![0.0; restart],
            sn: vec![0.0; restart],
            g: vec![0.0; restart + 1],
            w: vec![0.0; n],
        }
    }

    /// Clear Krylov storage, set g[0] = beta, normalise the residual into v_0,
    /// and push it onto the V basis.
    fn clear_for_restart(&mut self, beta: f64, r: &mut [f64]) {
        self.v_basis.clear();
        self.z_basis.clear();
        self.h.clear();
        self.cs.fill(0.0);
        self.sn.fill(0.0);
        self.g.fill(0.0);
        self.g[0] = beta;

        let inv_beta = 1.0 / beta;
        for val in r.iter_mut() {
            *val *= inv_beta;
        }
        self.v_basis.push_from(r);
    }
}

enum ArnoldiOutcome {
    Converged { residual: f64 },
    MaxIter { residual: f64 },
    Breakdown,
    RestartNeeded,
}

/// Modified Gram-Schmidt orthogonalisation of `w` against columns `0..=j` of `v_basis`.
///
/// Updates Hessenberg column `j` with the projection coefficients and sets
/// `h[j+1, j]` to the resulting norm of `w`.
///
/// Returns `(h_jp1_j, local_h_max)` where `local_h_max` is the maximum
/// absolute projection coefficient seen during the loop.
fn modified_gram_schmidt(
    w: &mut [f64],
    v_basis: &ColumnBasis,
    h: &mut HessenbergMatrix,
    j: usize,
) -> (f64, f64) {
    let mut local_h_max: f64 = 0.0;
    for i in 0..=j {
        let hij = dot(w, v_basis.col(i));
        h.set(i, j, hij);
        local_h_max = local_h_max.max(hij.abs());
        let v_i = v_basis.col(i);
        for (wk, &vi) in w.iter_mut().zip(v_i) {
            *wk -= hij * vi;
        }
    }
    let h_jp1_j = vec_norm(w);
    h.set(j + 1, j, h_jp1_j);
    (h_jp1_j, local_h_max)
}

/// Apply previously computed Givens rotations `0..j` to Hessenberg column `j`.
fn apply_previous_givens(h: &mut HessenbergMatrix, cs: &[f64], sn: &[f64], j: usize) {
    for i in 0..j {
        let h_ij = h.get(i, j);
        let h_i1j = h.get(i + 1, j);
        let temp = cs[i] * h_ij + sn[i] * h_i1j;
        h.set(i + 1, j, -sn[i] * h_ij + cs[i] * h_i1j);
        h.set(i, j, temp);
    }
}

/// Run one Arnoldi cycle (inner GMRES loop).
///
/// Returns `(outcome, j)` where `j` is the number of Arnoldi steps completed
/// in this cycle.  The caller is responsible for solving the upper-triangular
/// system and updating `x` once, regardless of the outcome.
fn arnoldi_cycle<A: Operator + ?Sized, M: Operator + ?Sized>(
    operator: &A,
    preconditioner: Option<&M>,
    bufs: &mut KrylovBuffers,
    abs_tol: f64,
    iters_this_cycle: usize,
    total_iters: &mut usize,
    maxiter: usize,
) -> Result<(ArnoldiOutcome, usize), SolveError> {
    let mut j = 0;
    let mut h_norms_max: f64 = 0.0;

    while j < iters_this_cycle {
        // z_j = M^{-1} v_j
        {
            let v_j = bufs.v_basis.col(j);
            bufs.z_basis.push_zeroed();
            let z_j = bufs.z_basis.last_col_mut();
            match preconditioner {
                Some(m) => m.apply(v_j, z_j)?,
                None => z_j.copy_from_slice(v_j),
            }
        }

        // w = A z_j
        {
            let z_j = bufs.z_basis.col(j);
            operator.apply(z_j, &mut bufs.w)?;
        }

        // Modified Gram-Schmidt orthogonalisation
        let (h_jp1_j, local_h_max) =
            modified_gram_schmidt(&mut bufs.w, &bufs.v_basis, &mut bufs.h, j);
        h_norms_max = h_norms_max.max(local_h_max).max(h_jp1_j);

        // Apply previous Givens rotations to the new column
        apply_previous_givens(&mut bufs.h, &bufs.cs, &bufs.sn, j);

        // Compute new Givens rotation for row (j, j+1)
        let (c, s) = givens_rotation(bufs.h.get(j, j), bufs.h.get(j + 1, j));
        bufs.cs[j] = c;
        bufs.sn[j] = s;

        // Apply to h column j
        let h_jj = bufs.h.get(j, j);
        let h_j1j = bufs.h.get(j + 1, j);
        bufs.h.set(j, j, c * h_jj + s * h_j1j);
        bufs.h.set(j + 1, j, 0.0);

        // Apply to g
        let temp = c * bufs.g[j] + s * bufs.g[j + 1];
        bufs.g[j + 1] = -s * bufs.g[j] + c * bufs.g[j + 1];
        bufs.g[j] = temp;

        j += 1;
        *total_iters += 1;

        let residual = bufs.g[j].abs();

        if residual <= abs_tol {
            return Ok((ArnoldiOutcome::Converged { residual }, j));
        }

        if *total_iters >= maxiter {
            return Ok((ArnoldiOutcome::MaxIter { residual }, j));
        }

        // Extend basis if not at last iteration
        if h_jp1_j > 1e-14 * h_norms_max {
            // Normalise w into next basis vector
            let inv = 1.0 / h_jp1_j;
            for val in bufs.w.iter_mut() {
                *val *= inv;
            }
            bufs.v_basis.push_from(&bufs.w);
        } else {
            // Lucky breakdown: Krylov subspace exhausted
            return Ok((ArnoldiOutcome::Breakdown, j));
        }
    }

    Ok((ArnoldiOutcome::RestartNeeded, j))
}

// ---------------------------------------------------------------------------
// GMRES helpers
// ---------------------------------------------------------------------------

/// Compute the residual `r = b - A * x` and return its Euclidean norm.
fn compute_residual<A: Operator + ?Sized>(
    operator: &A,
    x: &[f64],
    b: &[f64],
    r: &mut [f64],
) -> Result<f64, SolveError> {
    operator.apply(x, r)?;
    for (ri, &bi) in r.iter_mut().zip(b) {
        *ri = bi - *ri;
    }
    Ok(vec_norm(r))
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

/// Unpreconditioned GMRES(m) with restarts.
///
/// Solves `A x = b` using Arnoldi iteration with Modified Gram-Schmidt
/// orthogonalisation and Givens rotations.
///
/// `restart`: Krylov subspace dimension before restart (m in GMRES(m)).
pub fn gmres<A: Operator + ?Sized>(
    operator: &A,
    b: &[f64],
    tol: f64,
    maxiter: usize,
    restart: usize,
) -> Result<GmresResult, SolveError> {
    pgmres_impl::<A, A>(operator, b, None, tol, maxiter, restart)
}

/// Right-preconditioned GMRES(m) with restarts.
///
/// Solves `A x = b` using right preconditioning: `A M^{-1} (M x) = b`.
///
/// `restart`: Krylov subspace dimension before restart (m in GMRES(m)).
pub fn pgmres<A: Operator + ?Sized, M: Operator + ?Sized>(
    operator: &A,
    b: &[f64],
    preconditioner: &M,
    tol: f64,
    maxiter: usize,
    restart: usize,
) -> Result<GmresResult, SolveError> {
    pgmres_impl(operator, b, Some(preconditioner), tol, maxiter, restart)
}

fn pgmres_impl<A: Operator + ?Sized, M: Operator + ?Sized>(
    operator: &A,
    b: &[f64],
    preconditioner: Option<&M>,
    tol: f64,
    maxiter: usize,
    restart: usize,
) -> Result<GmresResult, SolveError> {
    let n = operator.ncols();
    debug_assert_eq!(b.len(), n);

    let b_norm = vec_norm(b);
    if b_norm < f64::EPSILON {
        return Ok(GmresResult {
            x: vec![0.0; n],
            converged: true,
            iterations: 0,
            residual_norm: 0.0,
        });
    }

    let abs_tol = tol * b_norm;
    let m = restart;

    let mut x = vec![0.0; n];
    let mut total_iters = 0;

    // Working vector reused across restarts
    let mut r = vec![0.0; n];

    // Pre-allocate Krylov storage once
    let mut bufs = KrylovBuffers::new(m, n);

    loop {
        let beta = compute_residual(operator, &x, b, &mut r)?;

        if beta <= abs_tol {
            return Ok(GmresResult {
                x,
                converged: true,
                iterations: total_iters,
                residual_norm: beta,
            });
        }

        if total_iters >= maxiter {
            return Ok(GmresResult {
                x,
                converged: false,
                iterations: total_iters,
                residual_norm: beta,
            });
        }

        bufs.clear_for_restart(beta, &mut r);

        let iters_this_cycle = (maxiter - total_iters).min(m);

        let (outcome, j) = arnoldi_cycle(
            operator,
            preconditioner,
            &mut bufs,
            abs_tol,
            iters_this_cycle,
            &mut total_iters,
            maxiter,
        )?;

        // Solve upper triangular system and update x -- done ONCE per cycle
        let y = solve_upper_triangular(&bufs.h, &bufs.g, j);
        update_solution(&mut x, &bufs.z_basis, &y, j);

        match outcome {
            ArnoldiOutcome::Converged { residual } => {
                return Ok(GmresResult {
                    x,
                    converged: true,
                    iterations: total_iters,
                    residual_norm: residual,
                });
            }
            ArnoldiOutcome::MaxIter { residual } => {
                return Ok(GmresResult {
                    x,
                    converged: false,
                    iterations: total_iters,
                    residual_norm: residual,
                });
            }
            ArnoldiOutcome::Breakdown => {
                // Lucky breakdown: recompute actual residual
                let res = compute_residual(operator, &x, b, &mut r)?;
                return Ok(GmresResult {
                    x,
                    converged: res <= abs_tol,
                    iterations: total_iters,
                    residual_norm: res,
                });
            }
            ArnoldiOutcome::RestartNeeded => {
                // Loop continues with updated x
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Compute Givens rotation coefficients (c, s) such that
/// [c  s] [a]   [r]
/// [-s c] [b] = [0]
///
/// Uses `f64::hypot` for numerically stable computation of
/// `r = sqrt(a² + b²)`, avoiding overflow/underflow for extreme ratios.
#[inline]
fn givens_rotation(a: f64, b: f64) -> (f64, f64) {
    let r = f64::hypot(a, b);
    if r == 0.0 {
        // Identity rotation
        (1.0, 0.0)
    } else {
        (a / r, b / r)
    }
}

/// Solve the upper triangular system R y = g[0..k]
/// where R is stored in the Hessenberg matrix (after Givens rotations).
#[inline]
fn solve_upper_triangular(h: &HessenbergMatrix, g: &[f64], k: usize) -> Vec<f64> {
    let mut y = vec![0.0; k];
    for i in (0..k).rev() {
        let mut sum = g[i];
        for (j, yj) in y.iter().enumerate().take(k).skip(i + 1) {
            sum -= h.get(i, j) * yj;
        }
        let diag = h.get(i, i);
        if diag.abs() < 1e-14 * (1.0 + sum.abs()) {
            // Near-singular diagonal: intentional best-effort recovery.
            // In a preconditioned system this can occur when the Krylov
            // subspace nearly stagnates. Setting y[i] = 0 keeps the
            // update bounded; the GMRES outer loop will detect lack of
            // convergence via the residual norm check.
            y[i] = 0.0;
        } else {
            y[i] = sum / diag;
        }
    }
    y
}

/// Update x += Z * y
#[inline]
fn update_solution(x: &mut [f64], z_basis: &ColumnBasis, y: &[f64], k: usize) {
    for (j, &yj) in y.iter().enumerate().take(k) {
        let z_j = z_basis.col(j);
        for (xi, &zi) in x.iter_mut().zip(z_j.iter()) {
            *xi += yj * zi;
        }
    }
}
